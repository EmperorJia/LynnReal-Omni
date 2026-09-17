"""Per-stage timing for the Flash graphs, in the release's measurement convention.

The release reports ``DiT`` (three denoiser forwards), ``video decoder`` and
``generation incl. decode/RGB/staging`` -- everything from the first denoiser forward to the
decoded frames, excluding loading, compilation warmup and file encoding. This module measures
the same three quantities inside ComfyUI by wrapping the H3 model, its video VAE and its audio
Vae with CUDA events, and by timestamping the first denoiser forward of each request.

``LYNNREAL_TIMING=1`` arms it; the report is logged as one line
``LYNNREAL_BENCH {"dit_ms": ..., ...}`` when the video decode of a request finishes. Nothing is
printed (and no events are recorded) when the variable is unset.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch

TIMING = os.environ.get("LYNNREAL_TIMING") == "1"
PROFILE = os.environ.get("LYNNREAL_PROFILE") == "1"

_PENDING: list = []          # [(stage, start_event, end_event)]
_TOTALS: dict = {}           # stage -> accumulated ms
_STARTED_AT: float | None = None
_INSTALLED = False
_PROFILED = 0
_STEPS: list = []            # per-forward ms, in call order (the denoiser step list)
_SHAPES: dict = {}           # packed row counts seen during this request


def _record(stage: str, start, end) -> None:
    _PENDING.append((stage, start, end))


def _wrap(module_cls, name: str, stage: str, opens_request: bool, closes_request: bool) -> None:
    original = getattr(module_cls, name)

    def wrapper(self, *args, **kwargs):
        global _STARTED_AT, _PROFILED
        if opens_request and not _TOTALS and _STARTED_AT is None:
            _STARTED_AT = time.perf_counter()
        # Profile one steady-state denoiser forward (the second request's first step).
        if PROFILE and opens_request:
            _PROFILED += 1
            if _PROFILED == 4:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    result = original(self, *args, **kwargs)
                with open("/tmp/lynnreal_profile.txt", "w", encoding="utf-8") as handle:
                    handle.write("=== CUDA time ===\n")
                    handle.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))
                    handle.write("\n=== CPU time ===\n")
                    handle.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35))
                logging.info("LynnReal: profiled one denoiser forward -> /tmp/lynnreal_profile.txt")
                return result
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return original(self, *args, **kwargs)
        finally:
            end.record()
            _record(stage, start, end)
            if stage == "dit_ms":
                # Kept per forward as well as summed: a request that streams weights, or that
                # re-tunes a kernel for a new shape, pays for it in step 1 only, and the total
                # hides that. The list is what tells the two apart.
                _STEPS.append((start, end))
            if closes_request:
                log_report()

    setattr(module_cls, name, wrapper)


def install() -> None:
    """Arm the instrumentation once. Must be called after ComfyUI imported its H3 modules."""
    global _INSTALLED
    if not TIMING or _INSTALLED:
        return
    try:
        import comfy.ldm.minimax.audio_vae as audio_vae
        import comfy.ldm.minimax.model as h3_model
        import comfy.ldm.minimax.vae as h3_vae
    except Exception as error:  # pragma: no cover - depends on the ComfyUI build
        logging.warning("LynnReal: timing unavailable (%s).", error)
        return

    _wrap(h3_model.MiniMaxH3Model, "forward", "dit_ms", opens_request=True, closes_request=False)
    _wrap(h3_vae.MiniMaxH3VideoVAE, "decode", "video_decode_ms", opens_request=False,
          closes_request=True)
    if hasattr(audio_vae, "MiniMaxH3AudioVAE"):
        _wrap(audio_vae.MiniMaxH3AudioVAE, "decode", "audio_decode_ms", opens_request=False,
              closes_request=False)
    _INSTALLED = True
    logging.info("LynnReal: Flash timing armed (LYNNREAL_TIMING=1); one LYNNREAL_BENCH line per "
                 "request.")


def report() -> dict:
    """Drain the accumulated events and return one request's numbers."""
    global _STARTED_AT
    if not _PENDING:
        return {}
    torch.cuda.synchronize()
    for stage, start, end in _PENDING:
        _TOTALS[stage] = _TOTALS.get(stage, 0.0) + start.elapsed_time(end)
    _PENDING.clear()
    steps = [round(start.elapsed_time(end), 1) for start, end in _STEPS]
    _STEPS.clear()
    wall = None
    if _STARTED_AT is not None:
        wall = time.perf_counter() - _STARTED_AT
    payload = {
        "dit_ms": round(_TOTALS.get("dit_ms", 0.0), 1),
        "video_decode_ms": round(_TOTALS.get("video_decode_ms", 0.0), 1),
        "audio_decode_ms": round(_TOTALS.get("audio_decode_ms", 0.0), 1),
        "generate_wall_s": round(wall, 3) if wall is not None else None,
    }
    dit = _TOTALS.get("dit_ms", 0.0)
    if dit:
        # the release always runs three denoiser forwards for Flash
        payload["dit_per_step_ms"] = round(dit / 3.0, 1)
    if steps:
        payload["dit_steps_ms"] = steps
    if _SHAPES:
        payload.update(_SHAPES)
        _SHAPES.clear()
    if os.environ.get("LYNNREAL_FA3_DEBUG") == "1" or os.environ.get("LYNNREAL_FA3_TIMING") == "1":
        try:
            from . import attention_fa3

            payload.update(attention_fa3.flush_timing())
        except Exception:  # pragma: no cover - diagnostics only
            pass
    if os.environ.get("LYNNREAL_FAST_TIMING") == "1":
        try:
            from . import fast_blocks

            payload.update(fast_blocks.flush_timing())
        except Exception:  # pragma: no cover - diagnostics only
            pass
    try:
        from . import int8_fast

        int8 = int8_fast.stats()
        if int8.get("calls"):
            payload["int8_calls"] = int8["calls"]
    except Exception:  # pragma: no cover - diagnostics only
        pass
    _TOTALS.clear()
    _STARTED_AT = None
    return payload


def log_report() -> None:
    payload = report()
    if payload:
        logging.info("LYNNREAL_BENCH %s", json.dumps(payload))


def note_shapes(full_rows: int, compressed_rows=None, signature=None) -> None:
    """Record the packed geometry of the request the next bench line describes.

    ``signature`` is the packed layout's own identity: it moves when the text-token count
    moves, which is the case where a *new* packed shape reaches the kernels. Carrying it in the
    report is what tells a shape change apart from a warm-state change.
    """
    _SHAPES["rows_full"] = int(full_rows)
    if compressed_rows is not None:
        _SHAPES["rows_compressed"] = int(compressed_rows)
    if signature is not None:
        _SHAPES["layout_signature"] = (list(signature) if isinstance(signature, tuple)
                                       else signature)
