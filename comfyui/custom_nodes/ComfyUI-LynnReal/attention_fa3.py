"""FlashAttention 3 for the LynnReal Flash graph, injected per model from the node pack.

The release's Flash launcher runs its diffusion transformer on FlashAttention 3 (``auto``
resolves to ``_flash_3`` on Hopper); ComfyUI's own selection has no FA3 path and falls back to
FA2. Measured on this H100 with the H3 shapes (S=32256, 56 heads, head_dim 128, bf16) FA3
takes 41 ms against FA2's 75 ms, and 2.4 ms against 5.0 ms on the compressed sequence.

Two seams are used, both instance-scoped so that nothing else in the process changes:

* the DiT: ComfyUI's H3 block wrapper forwards ``args.get("attention")`` into
  ``DiTBlock.forward(..., attention=...)``, so the Flash compression node hands each block a
  callable that repeats ``Attention.forward`` with FA3 in place of ``optimized_attention``.
* the video VAE: ``LynnRealH3VAELoader`` builds the VAE object itself, so it can mark that
  object's attention blocks and the class shim only acts on marked instances.

Any failure (unsupported dtype/geometry, OOM, missing package) disables the path for the rest
of the process and falls back to the stock implementation. ``LYNNREAL_FA3=0`` opts out.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext

import torch

import comfy.model_management
import comfy.quant_ops

_FAILURE: str | None = None
_VAE_FAILURE: str | None = None
_PROBED = False
_VAE_SHIM_INSTALLED = False
_DEBUGGED = 0
_TIMING: dict = {"calls": 0}
_STAGES: dict = {}
_STAGE_TIMING = os.environ.get("LYNNREAL_FA3_TIMING") == "1"
# comfy-kitchen's CUDA backend is disabled by ComfyUI on this torch build, but its
# normalisation+RoPE kernel is 4.8x faster than the Triton one at 1344x768 (271 ms against
# 1291 ms per request) while the Triton kernel stays the faster INT8 GEMM. Default to CUDA for
# that single call; `LYNNREAL_ROPE_BACKEND=triton` (or `off`) reverts, and any failure here
# falls back silently to whatever the registry would have chosen.
_rope_setting = os.environ.get("LYNNREAL_ROPE_BACKEND")
_ROPE_BACKEND = "cuda" if not _rope_setting else _rope_setting.lower()
if _ROPE_BACKEND in ("off", "none"):
    _ROPE_BACKEND = None


def _rope_context():
    if not _ROPE_BACKEND:
        return nullcontext()
    try:
        import comfy_kitchen as ck

        ck.registry.enable(_ROPE_BACKEND)
        # keep the Triton kernel on the INT8 GEMMs (it is the faster one there) and let the
        # explicit context below be the only thing that reaches for the CUDA backend
        ck.registry.set_priority(["triton", "cuda", "eager"])
        return ck.registry.use_backend(_ROPE_BACKEND)
    except Exception as error:
        logging.warning("LynnReal: could not use the %s backend for the RoPE call (%s).",
                        _ROPE_BACKEND, error)
        return nullcontext()


_ROPE_PATCHED = False


def install_rope_patch() -> None:
    """Route comfy-kitchen's normalisation+RoPE kernel to the fast backend for every H3 graph.

    ComfyUI's H3 attention calls ``ck.rms_rope_split_half_`` directly, so the backend choice is
    the registry's. On this cu126 build ComfyUI disables the CUDA backend, leaving Triton, which
    measures 10.2 ms per call against the CUDA kernel's 2.1 ms (1291 ms against 271 ms per
    request; 1344x768, 42 blocks x 3 steps). Wrapping the function keeps the fast backend for
    that one operator in every environment -- including the FA2/SDPA environments where our FA3
    path never runs -- and costs nothing when the CUDA backend is unavailable.
    """
    global _ROPE_PATCHED
    if _ROPE_PATCHED or not _ROPE_BACKEND:
        return
    _ROPE_PATCHED = True
    # comfy-aimdo's malloc graph records every allocation inside a model call; switching this
    # kernel's backend mid-call changes that pattern and the planner then fails ("aimdo memory
    # compile error", reproduced with a 15 s clip). Leave the registry's choice alone whenever
    # the graph is active -- the same condition that disables the fused blocks.
    try:
        import comfy.cli_args
        import comfy.memory_management as _memory

        if getattr(_memory, "aimdo_enabled", False) and not getattr(
                comfy.cli_args.args, "disable_comfy_compiler", False):
            logging.info("LynnReal: leaving comfy-kitchen's RoPE kernel on the default backend "
                         "(comfy-aimdo's malloc graph is active).")
            return
    except Exception:  # pragma: no cover - older ComfyUI
        pass
    try:
        import comfy_kitchen as ck
    except Exception:
        return
    for name in ("rms_rope_split_half_", "rms_rope_split_half"):
        original = getattr(ck, name, None)
        if original is None or getattr(original, "_lynnreal_rope_patch", False):
            continue

        def wrapper(*args, __original=original, **kwargs):
            with _rope_context():
                return __original(*args, **kwargs)

        wrapper._lynnreal_rope_patch = True
        setattr(ck, name, wrapper)
    logging.info("LynnReal: comfy-kitchen's RoPE kernel will run on the %s backend.",
                 _ROPE_BACKEND)


class _Stage:
    """CUDA-event span around one piece of the attention call (LYNNREAL_FA3_TIMING=1)."""

    __slots__ = ("name", "start", "end")

    def __init__(self, name):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        _STAGES.setdefault(self.name, []).append((self.start, self.end))
        return False


class _NoStage:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def enabled() -> bool:
    return os.environ.get("LYNNREAL_FA3", "1") != "0"


def vae_enabled() -> bool:
    """FA3 in the video decoder; the small per-tile attention is measured separately."""
    # measured slower than the stock path at the Light VAE's tile geometry (4.4 s against 3.2 s
    # per 5 s clip before the tile batching work), so it stays off unless asked for
    return enabled() and os.environ.get("LYNNREAL_FA3_VAE", "0") == "1"


def vae_usable() -> bool:
    return usable() and vae_enabled() and _VAE_FAILURE is None


def usable() -> bool:
    return enabled() and _FAILURE is None


def _disable(where: str, error: BaseException) -> None:
    global _FAILURE, _VAE_FAILURE
    if where == "VAE":
        # a VAE-only failure must not take the DiT path down with it
        if _VAE_FAILURE is None:
            _VAE_FAILURE = f"{where}: {type(error).__name__}: {error}"
            logging.warning("LynnReal: FlashAttention 3 disabled for the video VAE (%s); the DiT "
                            "keeps it.", _VAE_FAILURE)
        return
    if _FAILURE is None:
        _FAILURE = f"{where}: {type(error).__name__}: {error}"
        logging.warning("LynnReal: FlashAttention 3 disabled (%s); using ComfyUI's attention.",
                        _FAILURE)


def _fa3(q, k, v):
    global _DEBUGGED
    from flash_attn_interface import flash_attn_func

    # FA3 (like the release's dispatch) takes [batch, seq, heads, head_dim]. ComfyUI's own
    # attention entry points use [batch, heads, seq, dim]; both call sites below build the FA3
    # layout themselves, because a swapped order reads past the buffer and faults the context.
    # batch 1 for the DiT's packed sequence, one row per spatial tile in the batched VAE decode
    if q.dim() != 4 or q.shape[0] < 1 or q.shape[-1] % 8:
        raise ValueError(f"unexpected FA3 layout {tuple(q.shape)}")
    if os.environ.get("LYNNREAL_FA3_DEBUG") == "1" and _DEBUGGED < 12 and q.shape[1] > 512:
        _DEBUGGED += 1
        logging.info("LynnReal: FA3 call %s seq%%8=%d | %s | strides %s | k%s | v%s", q.shape,
                     q.shape[1] % 8, q.dtype, q.stride(), tuple(k.shape), tuple(v.shape))
    # The Hopper kernels need the sequence length to be a multiple of 8; the packed H3 sequence
    # (37837 rows at 1344x768) is not. Zero-padding both sides is numerically harmless -- the
    # extra keys score 0, so they add one unit each to a softmax denominator of ~38k, and the
    # measured error against math SDPA (0.0023 relative L2) stays at FA3's own bf16 level.
    rows = q.shape[1]
    pad = (-rows) % 8
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    timed = os.environ.get("LYNNREAL_FA3_DEBUG") == "1"
    if timed:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    out = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
    if timed:
        end.record()
        _TIMING.setdefault("events", []).append((start, end))
        _TIMING["calls"] += 1
    return out[:, :rows].contiguous() if pad else out


def flush_timing() -> dict:
    """Drain the per-call FA3 timers (debug mode only): how many calls and how much CUDA time."""
    events = _TIMING.pop("events", [])
    calls = _TIMING.pop("calls", 0)
    _TIMING["calls"] = 0
    result = {}
    if events:
        torch.cuda.synchronize()
        total = sum(start.elapsed_time(end) for start, end in events)
        result.update(fa3_calls=calls, fa3_ms=round(total, 1))
    if _STAGES:
        torch.cuda.synchronize()
        for name, spans in _STAGES.items():
            result[f"attn_{name}_ms"] = round(sum(s.elapsed_time(e) for s, e in spans), 1)
        _STAGES.clear()
    return result


def probe_once() -> None:
    """Numerically validate FA3 against math SDPA once, like the release launcher does."""
    global _PROBED
    if _PROBED or not usable():
        return
    _PROBED = True
    try:
        generator = torch.Generator(device="cuda").manual_seed(91827)
        worst = 0.0
        for dtype in (torch.bfloat16, torch.float16):
            for dim in (64, 96, 128):
                # multiples of 8: the probe measures the kernel, not the padding the packed H3
                # sequence needs (padding is only harmless at tens of thousands of rows)
                q = torch.randn(1, 128, 4, dim, device="cuda", dtype=dtype, generator=generator)
                k = torch.randn(1, 80, 4, dim, device="cuda", dtype=dtype, generator=generator)
                v = torch.randn(1, 80, 4, dim, device="cuda", dtype=dtype, generator=generator)
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                        v.transpose(1, 2).float()).transpose(1, 2)
                actual = _fa3(q, k, v).float()
                torch.cuda.synchronize()
                relative = float((actual - reference).square().mean().sqrt()
                                 / reference.square().mean().sqrt().clamp_min(1e-8))
                worst = max(worst, relative)
        if worst > 0.02:
            raise RuntimeError(f"attention numerical check failed ({worst:.4f})")
        logging.info("LynnReal: FlashAttention 3 enabled for the Flash model (worst relative L2 "
                     "against math SDPA %.4f).", worst)
    except Exception as error:
        _disable("probe", error)


def make_dit_attention(attention_module):
    """Callable for ``DiTBlock.forward(..., attention=...)`` bound to one Attention module."""

    def attention(x, rope_freqs=None, transformer_options={}):
        if not usable():
            return attention_module(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        try:
            return _dit_forward(attention_module, x, rope_freqs)
        except Exception as error:
            _disable("DiT", error)
            return attention_module(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

    return attention


def _dit_forward(module, x, rope_freqs):
    """``comfy.ldm.minimax.model.Attention.forward`` with FA3 as the attention call."""
    s = x.shape[0]
    timed = _STAGE_TIMING
    with (_Stage("qkv") if timed else _NoStage()):
        inner = module.heads * module.head_dim
        # One 15 s clip packs ~92k rows, so the fused QKV projection would address 1.97e9
        # elements -- close enough to the INT8 kernels' 32-bit limit to be worth splitting. The
        # rows are independent, so chunking is exact.
        if s * inner * 3 >= (1 << 30):
            chunk = max(1, (1 << 30) // (inner * 3))
            chunks = [module.qkv_proj(part) for part in x.split(chunk, dim=0)]
            projected = torch.cat(chunks, dim=0)
        else:
            projected = module.qkv_proj(x)
        q, k, v = projected.split(inner, dim=-1)
    v = v.view(s, module.heads, module.head_dim)
    with (_Stage("rope") if timed else _NoStage()):
        if rope_freqs is not None:
            q = q.view(1, s, module.heads, module.head_dim)
            k = k.view(1, s, module.heads, module.head_dim)
            qw = comfy.model_management.cast_to(module.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(module.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=module.q_norm.eps, rot_dim=rot)
            else:
                with _rope_context():
                    q, k = comfy.quant_ops.ck.rms_rope_split_half_(
                        q, k, rope_freqs, qw, kw, epsilon=module.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = module.q_norm(q.view(s, module.heads, module.head_dim))
            k = module.k_norm(k.view(s, module.heads, module.head_dim))

    q = q.unsqueeze(0)          # [S, H, D] -> [1, S, H, D]
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    expected = (1, s, module.heads, module.head_dim)
    if tuple(q.shape) != expected:
        raise ValueError(f"DiT attention layout {tuple(q.shape)} != {expected}")
    # `optimized_attention(..., skip_output_reshape=False)` answers [rows, heads * head_dim];
    # the quantized out_proj reads exactly that many columns.
    with (_Stage("flash") if timed else _NoStage()):
        attended = _fa3(q, k, v).squeeze(0).reshape(s, module.heads * module.head_dim)
    with (_Stage("out_proj") if timed else _NoStage()):
        return module.out_proj(attended)


def install_vae(vae_module, model) -> int:
    """Enable FA3 for one VAE object's attention blocks. Returns how many were marked."""
    global _VAE_SHIM_INSTALLED
    if not vae_usable():
        return 0
    if not _VAE_SHIM_INSTALLED:
        original = vae_module.Attention.forward

        def forward(self, x, rotary_pos_emb=None):
            if not getattr(self, "_lynnreal_fa3", False):
                return original(self, x, rotary_pos_emb)
            try:
                return _vae_forward(self, x, rotary_pos_emb)
            except Exception as error:
                self._lynnreal_fa3 = False
                _disable("VAE", error)
                return original(self, x, rotary_pos_emb)

        vae_module.Attention.forward = forward
        _VAE_SHIM_INSTALLED = True

    marked = 0
    for module in model.modules():
        if isinstance(module, vae_module.Attention):
            module._lynnreal_fa3 = True
            marked += 1
    return marked


def _vae_forward(self, x, rotary_pos_emb=None):
    """``comfy.ldm.minimax.vae.Attention.forward`` with FA3 as the attention call."""
    batch_size, seq_len, _ = x.shape
    qkv = self.to_qkv(x).view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    query = comfy.rmsnorm.rms_norm(query, self.norm_q.weight, self.norm_q.eps)
    key = comfy.rmsnorm.rms_norm(key, self.norm_k.weight, self.norm_k.eps)
    if rotary_pos_emb is not None:
        rot = rotary_pos_emb.shape[-3] * 2
        query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
            query[..., :rot], key[..., :rot], rotary_pos_emb)

    out = _fa3(query, key, value).reshape(batch_size, seq_len, self.heads * self.dim_head)
    out = out.nan_to_num_(0.0)
    return self.to_out(out)
