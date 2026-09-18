"""Run the Flash DiT blocks with the release's fused normalization / gated residual.

ComfyUI's ``DiTBlock.forward`` normalizes, then walks ``mod_segments`` doing a gather plus a
multiply and an add for every segment; the gate side does another gather plus ``addcmul_``.
At 1344x768 the video segment carries ~37k rows, so those segment loops cost ~0.78 s per step
against the release's ~0.2 s, which the release gets from two Triton kernels
(``_rms_adaln`` and ``_gated_residual``; vendored in ``lynnreal_kernels.py``).

This module supplies the same block arithmetic through those kernels. It is enabled only after
a probe compares it against the stock block on live weights, and it degrades to the stock path
on any failure (no Triton, unsupported dtype, numerical mismatch), so a different GPU only
loses the speedup, never correctness.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

import comfy.model_management
try:  # ComfyUI's DynamicVRAM (comfy-aimdo) memory planner
    import comfy.model_prefetch as _model_prefetch
except Exception:  # pragma: no cover - older ComfyUI
    _model_prefetch = None

from . import lynnreal_kernels as kernels

_FAILURE: str | None = None
_PROBED = False
_ANNOUNCED = False
_TIMING: dict = {}


def _timed() -> bool:
    return os.environ.get("LYNNREAL_FAST_TIMING") == "1"


class _Span:
    """CUDA-event span around one piece of the block, accumulated for the bench report."""

    __slots__ = ("stage", "start", "end")

    def __init__(self, stage):
        self.stage = stage
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        _TIMING.setdefault(self.stage, []).append((self.start, self.end))
        return False


def flush_timing() -> dict:
    """Drain the per-stage block timers (LYNNREAL_FAST_TIMING=1)."""
    if not _TIMING:
        return {}
    torch.cuda.synchronize()
    result = {f"block_{stage}_ms": round(sum(s.elapsed_time(e) for s, e in spans), 1)
              for stage, spans in _TIMING.items()}
    _TIMING.clear()
    return result


def enabled() -> bool:
    return os.environ.get("LYNNREAL_FAST_BLOCKS", "1") != "0"


def usable() -> bool:
    return enabled() and _FAILURE is None and kernels.TRITON_AVAILABLE


def _disable(reason: str) -> None:
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = reason
        logging.warning("LynnReal: fused DiT blocks disabled (%s); using ComfyUI's block math.",
                        reason)


def mod_indices(mod_segments, sequence: int, device) -> torch.Tensor:
    """Per-token table row, the layout the fused kernels index with."""
    indices = torch.empty(sequence, dtype=torch.long, device=device)
    for start, stop, row in mod_segments:
        if isinstance(row, torch.Tensor):
            indices[start:stop] = row.to(torch.long)
        else:
            indices[start:stop] = int(row)
    return indices


class _NormView:
    """What the vendored kernel wrappers read off a norm module: ``weight`` and ``eps``.

    ComfyUI keeps block weights off the compute device until the block is prefetched, so the
    weight is cast the same way ``comfy.ldm.minimax.model.Attention`` casts its q/k norms.
    """

    __slots__ = ("weight", "eps")

    def __init__(self, module, device):
        self.weight = comfy.model_management.cast_to(module.weight, device=device)
        self.eps = module.eps


def block_forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options, attention=None):
    """``DiTBlock.forward`` with the two fused elementwise kernels."""
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        logging.info("LynnReal: fused DiT blocks running (first block has %d rows, %d segments, "
                     "table %s).", x.shape[0], len(mod_segments), tuple(t_emb.shape))
    table = torch.cat(block.adaln_proj(t_emb), dim=-1)
    indices = mod_indices(mod_segments, x.shape[0], x.device)
    call = block.attn if attention is None else attention
    norm1 = _NormView(block.norm1, x.device)
    norm2 = _NormView(block.norm2, x.device)

    # Our kernels allocate their own outputs; tell comfy-aimdo's malloc graph to look away, or it
    # tries to record those allocations and fails with "aimdo memory compile error" under
    # DynamicVRAM. The context is a no-op when the malloc graph is not active.
    pause = (_model_prefetch.pause_malloc_graph() if _model_prefetch is not None
             else nullcontext())
    with pause:
        return _block_body(block, x, t_emb, table, indices, rope_freqs, transformer_options,
                           call, norm1, norm2)


def _block_body(block, x, t_emb, table, indices, rope_freqs, transformer_options, call, norm1,
                norm2):
    timed = _timed()
    with (_Span("norm") if timed else _noop()):
        h = kernels.rms_adaln(x, norm1, table, indices, 0, 1)
    with (_Span("attn") if timed else _noop()):
        update = call(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    with (_Span("gate") if timed else _noop()):
        x = kernels.gated_residual(x, update, table, indices, 2)
    with (_Span("norm") if timed else _noop()):
        h = kernels.rms_adaln(x, norm2, table, indices, 3, 4)
    with (_Span("mlp") if timed else _noop()):
        h = _mlp(block, h)
    with (_Span("gate") if timed else _noop()):
        return kernels.gated_residual(x, h, table, indices, 5)


# comfy-kitchen's INT8 GEMM addresses its activation/accumulator buffers with 32-bit offsets.
# Keep a wide margin below 2**31 so no single call can walk off the end (the observed limit is
# somewhere under 2**31 elements; the release's own kernels switch to int64 here).
_INT32_SAFE_ELEMENTS = 1 << 30


def _mlp(block, h):
    """``block.mlp`` in row chunks when the intermediate would overflow int32 indexing.

    The fused activation+quantize kernel behind ``linear_input_act`` addresses the
    ``2 * ffn`` intermediate with 32-bit offsets, so a 15 s clip (91.7k rows x 28,672 =
    2.6e9 elements) runs off the end and faults the context -- stock ComfyUI dies the same way
    at 15 s while 10 s (1.8e9) is fine. Splitting the rows keeps every call inside the range.
    """
    fc1 = getattr(block.mlp, "fc1", None)
    intermediate = getattr(fc1, "out_features", 0)
    rows = h.shape[0]
    if intermediate <= 0 or rows * intermediate < _INT32_SAFE_ELEMENTS:
        return block.mlp(h)
    chunk = max(1, _INT32_SAFE_ELEMENTS // intermediate)
    logging.info("LynnReal: MLP row-chunked at %d rows (intermediate %d x %d would overflow "
                 "int32 indexing).", chunk, rows, intermediate)
    return torch.cat([block.mlp(part) for part in h.split(chunk, dim=0)], dim=0)


def check_stock_mlp(block, rows: int) -> None:
    """Refuse a shape ComfyUI's own INT8 MLP cannot address, before it faults the context.

    A 15 s clip packs ~92k rows, so ``fc1`` would produce a 2.6e9-element intermediate that
    comfy-kitchen indexes with 32-bit offsets; the resulting illegal access takes the whole CUDA
    context (and, on this node, the GPU) down. Failing the prompt with a clear message is the
    safe outcome when the fused blocks that chunk it are switched off.
    """
    intermediate = getattr(getattr(getattr(block, "mlp", None), "fc1", None), "out_features", 0)
    # the observed failure point, not the safer chunking margin: a 5 s clip (1.1e9) runs fine on
    # the stock path, a 15 s one (2.6e9) faults
    if intermediate and rows * intermediate >= (1 << 31):
        raise ValueError(
            "This clip needs {} x {} = {:.1f}e9 elements in the DiT MLP, past the INT8 kernels' "
            "32-bit indexing. Enable the fused DiT blocks (remove LYNNREAL_FAST_BLOCKS=0) so the "
            "GEMMs are row-chunked.".format(rows, intermediate, rows * intermediate / 1e9))


def needs_chunking(block, rows: int) -> bool:
    """Would the stock MLP walk past 32-bit indexing for this many rows?"""
    intermediate = getattr(getattr(getattr(block, "mlp", None), "fc1", None), "out_features", 0)
    return bool(intermediate) and rows * intermediate >= (1 << 31)


def stock_block_forward(block, x, t_emb, mod_segments, rope_freqs, transformer_options,
                        attention=None):
    """ComfyUI's own block math, but with the MLP row-chunked.

    Used for clips long enough to overflow the INT8 MLP when the fused kernels are unavailable
    (for instance under DynamicVRAM, whose planner rejects them). Every op is ComfyUI's, so the
    arithmetic and the allocator pattern are exactly the stock ones apart from the split.
    """
    import comfy.ldm.minimax.model as minimax

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    call = block.attn if attention is None else attention
    h = minimax._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
    # `_mod_gate(x, gate, other, segments)` -- the gate row comes first, the fresh block output
    # second (that is the order `DiTBlock.forward` uses)
    x = minimax._mod_gate(
        x, gate_msa, call(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
        mod_segments)
    h = minimax._mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return minimax._mod_gate(x, gate_mlp, _mlp(block, h), mod_segments)


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_DEBUGGED = False


def debug_compare(block, x, t_emb, mod_segments, rope_freqs, transformer_options, attention,
                  reference):
    """Stage-by-stage comparison against the stock block on live data (LYNNREAL_FAST_DEBUG=1)."""
    global _DEBUGGED
    if _DEBUGGED:
        return
    _DEBUGGED = True
    try:
        import comfy.ldm.minimax.model as minimax

        table = torch.cat(block.adaln_proj(t_emb), dim=-1)
        indices = mod_indices(mod_segments, x.shape[0], x.device)
        call = block.attn if attention is None else attention
        norm1, norm2 = _NormView(block.norm1, x.device), _NormView(block.norm2, x.device)

        h_ref = minimax._mod_scale_shift(block.norm1(x), table_chunk(block, t_emb, 0),
                                         table_chunk(block, t_emb, 1), mod_segments)
        h_got = kernels.rms_adaln(x, norm1, table, indices, 0, 1)
        attn_ref = call(h_ref, rope_freqs=rope_freqs, transformer_options=transformer_options)
        attn_got = call(h_got, rope_freqs=rope_freqs, transformer_options=transformer_options)
        x_ref = minimax._mod_gate(x.clone(), table_chunk(block, t_emb, 2), attn_ref, mod_segments)
        x_got = kernels.gated_residual(x, attn_got, table, indices, 2)
        m_ref = block.mlp(h_ref)
        m_got = block.mlp(h_got)
        logging.info(
            "LynnReal: fused-block debug | x %.3f | norm+mod %.5f | attn %.5f | gate %.5f | "
            "mlp %.5f | block %.5f (relative %.5f)",
            x.abs().max().item(), (h_ref.float() - h_got.float()).abs().max().item(),
            (attn_ref.float() - attn_got.float()).abs().max().item(),
            (x_ref.float() - x_got.float()).abs().max().item(),
            (m_ref.float() - m_got.float()).abs().max().item(),
            (reference.float() - block_forward(block, x, t_emb, mod_segments, rope_freqs,
                                               transformer_options, attention).float()).abs().max().item(),
            float((reference.float() - block_forward(block, x, t_emb, mod_segments, rope_freqs,
                                                     transformer_options, attention).float()).abs().mean()
                  / reference.float().abs().mean().clamp_min(1e-6)))
    except Exception as error:  # pragma: no cover - diagnostics only
        logging.warning("LynnReal: fused-block debug failed (%s)", error)


def table_chunk(block, t_emb, index):
    return block.adaln_proj(t_emb)[index]


def probe(block, device=None) -> None:
    """Compare the fused block against ComfyUI's on this GPU and weights; disable on mismatch."""
    global _PROBED
    if _PROBED or not usable():
        return
    _PROBED = True
    # The guard below needs a real device: `malloc_graph_enabled` asks
    # `is_device_cuda(device)`, which is False for None, so passing the caller's default would
    # silently skip the disable and leave the fused kernels inside the malloc graph. The block's
    # own parameters may still be staged on the host; the kernels need the compute device, and
    # every weight read goes through comfy's casting ops anyway.
    device = device or comfy.model_management.get_torch_device()
    # comfy-aimdo's malloc *graph* plans the model's allocations ahead of time and its planner
    # rejects the pattern our two kernels produce ("aimdo memory compile error"; pausing the
    # graph around the block does not help). DynamicVRAM itself is fine -- only the graph is not
    # -- so this checks the graph, not `aimdo_enabled`: launching with `--disable-comfy-compiler`
    # (which keeps DynamicVRAM and drops the graph) lets the fused blocks run.
    try:
        import comfy.model_prefetch as _prefetch

        if _prefetch.malloc_graph_enabled(device):
            _disable("ComfyUI's comfy-aimdo malloc graph is active and rejects the fused kernels' "
                     "allocations; launch with --disable-comfy-compiler to keep DynamicVRAM and "
                     "run the fused blocks")
            return
    except Exception:  # pragma: no cover - older ComfyUI
        pass
    if os.environ.get("LYNNREAL_FAST_PROBE") == "0":
        logging.info("LynnReal: fused DiT blocks enabled without the synthetic probe "
                     "(LYNNREAL_FAST_PROBE=0); validate against a reference video instead.")
        return
    try:
        dtype = next(block.parameters()).dtype
        hidden = block.norm1.weight.shape[0]
        sequence = 64
        generator = torch.Generator(device=device).manual_seed(20260915)
        x = torch.randn(sequence, hidden, device=device, dtype=torch.float32,
                        generator=generator).to(dtype) * 1e-4
        t_emb = torch.randn(1, block.adaln_proj.linear.in_features, device=device,
                            dtype=torch.float32, generator=generator).to(dtype)
        segments = [(0, sequence, 0)]
        with torch.inference_mode():
            # the stock block rewrites its input in place, so the reference needs its own copy
            reference = block(x.clone(), t_emb, segments, None, transformer_options={},
                              attention=None)
            fused = block_forward(block, x, t_emb, segments, None, {}, None)
        torch.cuda.synchronize()
        difference = (fused.float() - reference.float()).abs()
        scale = reference.float().abs().mean().clamp_min(1e-3)
        relative = float(difference.mean() / scale)
        if not torch.isfinite(fused).all() or relative > 0.02:
            raise RuntimeError(f"relative error {relative:.4f} (max abs {difference.max().item():.4f})")
        logging.info("LynnReal: fused DiT blocks verified on this GPU (relative error %.4f, "
                     "max abs %.4f).", relative, difference.max().item())
    except Exception as error:
        _disable(f"{type(error).__name__}: {error}")
