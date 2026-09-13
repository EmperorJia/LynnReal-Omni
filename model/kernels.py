"""Selected dense inference kernels; provenance in log/audit/int8_kernel_sources.json."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import nearbyint


@triton.jit
def _quantize_rows_kernel(source, quantized, scales, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    column = tl.arange(0, block)
    value = tl.load(source + row * width + column, column < width, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0), 1e-8) / 127.0
    integer = nearbyint(tl.minimum(tl.maximum(value / scale, -127.0), 127.0)).to(tl.int8)
    tl.store(quantized + row * width + column, integer, column < width)
    tl.store(scales + row, scale)


def quantize_rows(value):
    rows, width = value.shape
    output = torch.empty_like(value, dtype=torch.int8)
    scales = torch.empty(rows, 1, device=value.device, dtype=torch.float32)
    _quantize_rows_kernel[(rows,)](value, output, scales, width, triton.next_power_of_2(width), num_warps=8)
    return output, scales


@triton.jit
def _int8_epilogue_kernel(accumulator, activation_scale, weight_scale, bias, output,
                         size: tl.constexpr, width: tl.constexpr, has_bias: tl.constexpr, block: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = offset < size
    row, column = offset // width, offset % width
    value = tl.load(accumulator + offset, mask, other=0).to(tl.float32)
    value = value * tl.load(activation_scale + row, mask, other=0)
    value = value * tl.load(weight_scale + column, mask, other=0)
    if has_bias:
        value = value + tl.load(bias + column, mask, other=0).to(tl.float32)
    tl.store(output + offset, value, mask)


def int8_epilogue(accumulator, activation_scale, weight_scale, bias):
    output = torch.empty_like(accumulator, dtype=torch.bfloat16)
    _int8_epilogue_kernel[(triton.cdiv(output.numel(), 1024),)](
        accumulator, activation_scale, weight_scale, bias, output, output.numel(),
        output.shape[-1], bias is not None, 1024, enable_fp_fusion=False)
    return output

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
def _rms_adaln_quant_int8_kernel(
    source, weight, table, indices, quantized, scales,
    sequence: tl.constexpr,
    hidden: tl.constexpr,
    table_width: tl.constexpr,
    shift_slot: tl.constexpr,
    scale_slot: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    update=None, residual_out=None,
    WITH_RESIDUAL: tl.constexpr = False,
):
    """Fuse H3 RMSNorm, AdaLN and symmetric per-row INT8 quantization."""
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, block)
    mask = columns < hidden
    token = row % sequence
    table_row = tl.load(indices + token)
    values = tl.load(source + row * hidden + columns, mask=mask, other=0.0).to(tl.float32)
    if WITH_RESIDUAL:
        delta = tl.load(update + row * hidden + columns, mask, other=0).to(tl.float32)
        gate = tl.load(table + table_row * table_width + 2 * hidden + columns, mask, other=0).to(tl.float32)
        values = (values + (gate * delta).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out + row * hidden + columns, values, mask)
    norm = values * tl.rsqrt(tl.sum(values * values, axis=0) / hidden + eps)
    norm_weight = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    norm = (norm * norm_weight).to(tl.bfloat16).to(tl.float32)
    shift = tl.load(
        table + table_row * table_width + shift_slot * hidden + columns,
        mask=mask, other=0.0,
    ).to(tl.float32)
    scale_mod = tl.load(
        table + table_row * table_width + scale_slot * hidden + columns,
        mask=mask, other=0.0,
    ).to(tl.float32)
    value = (
        (norm * (1.0 + scale_mod).to(tl.bfloat16)).to(tl.bfloat16)
        + shift.to(tl.bfloat16)
    ).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.where(mask, tl.abs(value), 0.0), axis=0), 1.0e-8)
    quant_scale = maximum / 127.0
    tl.store(scales + row, quant_scale)
    tl.store(
        quantized + row * hidden + columns,
        nearbyint(tl.maximum(tl.minimum(value / quant_scale, 127.0), -127.0)).to(tl.int8),
        mask=mask,
    )


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
    table_row = tl.load(indices + token, mask=mask, other=0)
    old = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    delta = tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        table + table_row * table_width + gate_slot * hidden + columns,
        mask=mask, other=0.0,
    ).to(tl.float32)
    gated = (gate * delta).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, old + gated, mask=mask)


@triton.jit
def _swiglu_kernel(projected, output, elements, inner: tl.constexpr, block: tl.constexpr):
    # Video-reference FFNs can contain more than 2**31 input elements.
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = offsets < elements
    row = offsets // inner
    column = offsets % inner
    value = tl.load(projected + row * (2 * inner) + column, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(projected + row * (2 * inner) + inner + column, mask=mask, other=0.0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, value * activated, mask=mask)


@triton.jit
def _swiglu_quant_int8_kernel(projected, quantized, scales, inner: tl.constexpr, block: tl.constexpr):
    """Fuse SwiGLU and the FFN-down dynamic INT8 activation quantizer."""
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, block)
    mask = columns < inner
    value = tl.load(projected + row * (2 * inner) + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(projected + row * (2 * inner) + inner + columns, mask=mask, other=0.0).to(tl.float32)
    activated = (value * (gate * tl.sigmoid(gate)).to(tl.bfloat16)).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.where(mask, tl.abs(activated), 0.0), axis=0), 1.0e-8)
    scale = maximum / 127.0
    tl.store(scales + row, scale)
    tl.store(
        quantized + row * inner + columns,
        nearbyint(tl.maximum(tl.minimum(activated / scale, 127.0), -127.0)).to(tl.int8),
        mask=mask,
    )


@triton.jit
def _qk_norm_full_rope_paired_kernel(
    query, key, query_weight, key_weight, cos, sin, query_out, key_out,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    query_token_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    """Fuse full-dimension RoPE without loading every Q/K element twice."""
    row = tl.program_id(0).to(tl.int64)
    token = (row // heads) % sequence
    head = row % heads
    batch = row // (sequence * heads)
    query_base = batch * sequence * query_token_stride + token * query_token_stride + head * query_head_stride
    key_base = batch * sequence * key_token_stride + token * key_token_stride + head * key_head_stride
    half = head_dim // 2
    dims = tl.arange(0, block)
    mask = dims < half
    right = dims + half

    q_left = tl.load(query + query_base + dims, mask=mask, other=0.0).to(tl.float32)
    q_right = tl.load(query + query_base + right, mask=mask, other=0.0).to(tl.float32)
    k_left = tl.load(key + key_base + dims, mask=mask, other=0.0).to(tl.float32)
    k_right = tl.load(key + key_base + right, mask=mask, other=0.0).to(tl.float32)
    q_rstd = tl.rsqrt(
        tl.sum(q_left * q_left + q_right * q_right, axis=0) / head_dim + eps
    )
    k_rstd = tl.rsqrt(
        tl.sum(k_left * k_left + k_right * k_right, axis=0) / head_dim + eps
    )
    q_left = (
        q_left * q_rstd
        * tl.load(query_weight + dims, mask=mask, other=0.0).to(tl.float32)
    ).to(tl.bfloat16).to(tl.float32)
    q_right = (
        q_right * q_rstd
        * tl.load(query_weight + right, mask=mask, other=0.0).to(tl.float32)
    ).to(tl.bfloat16).to(tl.float32)
    k_left = (
        k_left * k_rstd
        * tl.load(key_weight + dims, mask=mask, other=0.0).to(tl.float32)
    ).to(tl.bfloat16).to(tl.float32)
    k_right = (
        k_right * k_rstd
        * tl.load(key_weight + right, mask=mask, other=0.0).to(tl.float32)
    ).to(tl.bfloat16).to(tl.float32)

    cos_left = tl.load(cos + token * head_dim + dims, mask=mask, other=1.0).to(tl.bfloat16).to(tl.float32)
    sin_left = tl.load(sin + token * head_dim + dims, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)
    cos_right = tl.load(cos + token * head_dim + right, mask=mask, other=1.0).to(tl.bfloat16).to(tl.float32)
    sin_right = tl.load(sin + token * head_dim + right, mask=mask, other=0.0).to(tl.bfloat16).to(tl.float32)
    # Native eager H3 rounds each product to BF16 before their sum.
    q_out_left = ((q_left * cos_left).to(tl.bfloat16).to(tl.float32)
                  - (q_right * sin_left).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    q_out_right = ((q_right * cos_right).to(tl.bfloat16).to(tl.float32)
                   + (q_left * sin_right).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    k_out_left = ((k_left * cos_left).to(tl.bfloat16).to(tl.float32)
                  - (k_right * sin_left).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    k_out_right = ((k_right * cos_right).to(tl.bfloat16).to(tl.float32)
                   + (k_left * sin_right).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    output_base = row * head_dim
    tl.store(query_out + output_base + dims, q_out_left, mask=mask)
    tl.store(query_out + output_base + right, q_out_right, mask=mask)
    tl.store(key_out + output_base + dims, k_out_left, mask=mask)
    tl.store(key_out + output_base + right, k_out_right, mask=mask)


def _rms_adaln(source, norm, table, indices, shift_slot, scale_slot):
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


def _rms_adaln_quant_int8(source, norm, table, indices, shift_slot, scale_slot):
    source = source.contiguous()
    table = table.contiguous()
    indices = indices.reshape(-1).contiguous()
    hidden = source.shape[-1]
    sequence = source.shape[-2]
    rows = source.numel() // hidden
    output = torch.empty((rows, hidden), device=source.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=source.device, dtype=torch.float32)
    _rms_adaln_quant_int8_kernel[(rows,)](
        source, norm.weight, table, indices, output, scales,
        sequence, hidden, table.shape[-1], shift_slot, scale_slot,
        float(norm.eps), triton.next_power_of_2(hidden), num_warps=8, enable_fp_fusion=False,
    )
    return output, scales


def _gated_residual(residual, update, table, indices, gate_slot):
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


def _swiglu(projected):
    projected = projected.contiguous()
    inner = projected.shape[-1] // 2
    output = torch.empty((*projected.shape[:-1], inner), device=projected.device, dtype=projected.dtype)
    block = 256
    _swiglu_kernel[(triton.cdiv(output.numel(), block),)](
        projected, output, output.numel(), inner, block, num_warps=8,
    )
    return output


def _swiglu_quant_int8(projected):
    projected = projected.contiguous()
    inner = projected.shape[-1] // 2
    rows = projected.numel() // projected.shape[-1]
    output = torch.empty((rows, inner), device=projected.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=projected.device, dtype=torch.float32)
    block = triton.next_power_of_2(inner)
    _swiglu_quant_int8_kernel[(rows,)](
        projected, output, scales, inner, block, num_warps=8,
    )
    return output, scales


# Preserve the four-warp reduction tree while executing one warp (128-wide heads).
@triton.jit
def _sum128(ptr,base):
    lane=tl.arange(0,32)
    x0=tl.load(ptr+base+lane).to(tl.float32)
    x1=tl.load(ptr+base+lane+32).to(tl.float32)
    x2=tl.load(ptr+base+lane+64).to(tl.float32)
    x3=tl.load(ptr+base+lane+96).to(tl.float32)
    s0=tl.sum(x0*x0,0)
    s1=tl.sum(x1*x1,0)
    s2=tl.sum(x2*x2,0)
    s3=tl.sum(x3*x3,0)
    return (s0+s2)+(s1+s3)

@triton.jit
def _qk_norm_rope_kernel(
    query, key, query_weight, key_weight, cos, sin, query_out, key_out,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    query_token_stride: tl.constexpr,
    query_head_stride: tl.constexpr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    ordered_reduction: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    token = (row // heads) % sequence
    head = row % heads
    query_base = (row // (sequence * heads)) * sequence * query_token_stride + token * query_token_stride + head * query_head_stride
    key_base = (row // (sequence * heads)) * sequence * key_token_stride + token * key_token_stride + head * key_head_stride
    dims = tl.arange(0, block)
    mask = dims < head_dim
    q = tl.load(query + query_base + dims, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(key + key_base + dims, mask=mask, other=0.0).to(tl.float32)
    if ordered_reduction:
        tl.static_assert(head_dim == 128)
        q_rstd = tl.rsqrt(_sum128(query, query_base) / head_dim + eps)
        k_rstd = tl.rsqrt(_sum128(key, key_base) / head_dim + eps)
    else:
        q_rstd = tl.rsqrt(tl.sum(q * q, axis=0) / head_dim + eps)
        k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) / head_dim + eps)
    q *= q_rstd
    k *= k_rstd
    q *= tl.load(query_weight + dims, mask=mask, other=0.0).to(tl.float32)
    k *= tl.load(key_weight + dims, mask=mask, other=0.0).to(tl.float32)
    # RMSNorm returns BF16 before RoPE in the reference path.
    q = q.to(tl.bfloat16).to(tl.float32)
    k = k.to(tl.bfloat16).to(tl.float32)
    rotary_mask = dims < rotary_dim
    half = rotary_dim // 2
    paired = tl.where(dims < half, dims + half, dims - half)
    q_pair = tl.load(query + query_base + paired, mask=rotary_mask, other=0.0).to(tl.float32)
    k_pair = tl.load(key + key_base + paired, mask=rotary_mask, other=0.0).to(tl.float32)
    q_pair *= q_rstd
    k_pair *= k_rstd
    q_pair *= tl.load(query_weight + paired, mask=rotary_mask, other=0.0).to(tl.float32)
    k_pair *= tl.load(key_weight + paired, mask=rotary_mask, other=0.0).to(tl.float32)
    q_pair = q_pair.to(tl.bfloat16).to(tl.float32)
    k_pair = k_pair.to(tl.bfloat16).to(tl.float32)
    cosine = tl.load(cos + token * rotary_dim + dims, mask=rotary_mask, other=1.0).to(tl.bfloat16).to(tl.float32)
    sine = tl.load(sin + token * rotary_dim + dims, mask=rotary_mask, other=0.0).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(dims < half, -1.0, 1.0)
    q_rot = ((q * cosine).to(tl.bfloat16).to(tl.float32)
             + (sign * q_pair * sine).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    k_rot = ((k * cosine).to(tl.bfloat16).to(tl.float32)
             + (sign * k_pair * sine).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(query_out + row * head_dim + dims, tl.where(rotary_mask, q_rot, q), mask=mask)
    tl.store(key_out + row * head_dim + dims, tl.where(rotary_mask, k_rot, k), mask=mask)


def residual_rms_quant(source, update, norm, table, indices):
    """Keep residual BF16 rounding while removing its separate read for FFN norm."""
    source, update = source.contiguous(), update.contiguous()
    table, indices = table.contiguous(), indices.reshape(-1).contiguous()
    hidden = source.shape[-1]
    rows = source.numel() // hidden
    residual = torch.empty_like(source)
    quantized = torch.empty((rows, hidden), device=source.device, dtype=torch.int8)
    scales = torch.empty((rows, 1), device=source.device, dtype=torch.float32)
    _rms_adaln_quant_int8_kernel[(rows,)](
        source, norm.weight, table, indices, quantized, scales,
        source.shape[-2], hidden, table.shape[-1], 3, 4,
        float(norm.eps), triton.next_power_of_2(hidden), update, residual, True,
        num_warps=8, enable_fp_fusion=False)
    return residual, quantized, scales
