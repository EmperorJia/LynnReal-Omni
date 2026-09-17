"""Triton kernels the Flash DiT block needs, vendored from the LynnReal release.

Source: ``model/kernels.py`` of the LynnReal release (``_rms_adaln_kernel``,
``_gated_residual_safe_kernel``). They are the two fusions that make the release's DiT block
cheaper than ComfyUI's elementwise path:

* ``_rms_adaln`` folds RMSNorm, the AdaLN table lookup, the ``1 + scale`` multiply and the
  ``+ shift`` add into one pass over the activation (ComfyUI does norm, then a per-segment
  gather + multiply + add);
* ``_gated_residual`` folds the gate lookup and the ``x += gate * update`` add into one pass.

Both kernels reproduce the released BF16 path's intermediate rounding, so they are *more*
faithful to the reference than a plain torch rewrite would be.

Portability: plain Triton (no Hopper-only features), JIT-compiled per architecture, but
``triton.language.extra.cuda.libdevice`` makes the module CUDA-only. ``_swiglu``'s quantized
variants are not vendored here because ComfyUI already folds the activation into its INT8 GEMM.
Every entry point is guarded and reports failure to the caller, which falls back to ComfyUI's
own block math.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - depends on the installed Triton
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TRITON_ERROR = None
except Exception as error:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    TRITON_ERROR = error


if TRITON_AVAILABLE:

    @triton.jit
    def _rms_adaln_kernel(
        source, weight, table, indices, output,
        sequence: tl.constexpr,
        hidden: tl.constexpr,
        table_width: tl.constexpr,
        shift_slot: tl.constexpr,
        scale_slot: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, block)
        mask = columns < hidden
        token = row % sequence
        table_row = tl.load(indices + token)
        values = tl.load(source + row * hidden + columns, mask=mask, other=0.0).to(tl.float32)
        norm = values * tl.rsqrt(tl.sum(values * values, axis=0) / hidden + eps)
        norm_weight = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
        # Reproduce the released BF16 path's intermediate rounding.
        norm = (norm * norm_weight).to(tl.bfloat16).to(tl.float32)
        shift = tl.load(
            table + table_row * table_width + shift_slot * hidden + columns,
            mask=mask, other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            table + table_row * table_width + scale_slot * hidden + columns,
            mask=mask, other=0.0,
        ).to(tl.float32)
        scaled = (norm * (1.0 + scale).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
        tl.store(output + row * hidden + columns, scaled + shift, mask=mask)

    @triton.jit
    def _gated_residual_safe_kernel(
        residual, update, table, indices, output, elements,
        sequence: tl.constexpr,
        hidden: tl.constexpr,
        table_width: tl.constexpr,
        gate_slot: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        mask = offsets < elements
        row = offsets // hidden
        columns = offsets % hidden
        token = row % sequence
        table_row = tl.load(indices + token, mask, other=0)
        old = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            table + table_row * table_width + gate_slot * hidden + columns,
            mask=mask, other=0.0,
        ).to(tl.float32)
        # ComfyUI's `_mod_gate` uses `addcmul_`, which accumulates in fp32 and rounds once; the
        # release's kernel rounds the product to bf16 first (it reproduces their reference
        # pipeline instead). Rounding here would shift the sampling trajectory, so keep fp32.
        tl.store(output + offsets, old + gate * delta, mask=mask)


def _require() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError(f"Triton is unavailable: {TRITON_ERROR}")


def rms_adaln(source, norm, table, indices, shift_slot, scale_slot):
    """``rmsnorm(source) * (1 + scale) + shift`` with the table row taken per token."""
    _require()
    source = source.contiguous()
    table = table.contiguous()
    indices = indices.reshape(-1).contiguous()
    hidden = source.shape[-1]
    sequence = source.shape[-2]
    output = torch.empty_like(source)
    _rms_adaln_kernel[(source.numel() // hidden,)](
        source, norm.weight, table, indices, output,
        sequence, hidden, table.shape[-1], shift_slot, scale_slot,
        float(norm.eps), triton.next_power_of_2(hidden), num_warps=8, enable_fp_fusion=False,
    )
    return output


def gated_residual(residual, update, table, indices, gate_slot):
    """``residual + gate * update`` with the gate row taken per token."""
    _require()
    residual = residual.contiguous()
    update = update.contiguous()
    table = table.contiguous()
    indices = indices.reshape(-1).contiguous()
    output = torch.empty_like(residual)
    block = 1024
    _gated_residual_safe_kernel[(triton.cdiv(residual.numel(), block),)](
        residual, update, table, indices, output, residual.numel(),
        residual.shape[-2], residual.shape[-1], table.shape[-1], gate_slot,
        block, num_warps=4, enable_fp_fusion=False,
    )
    return output
