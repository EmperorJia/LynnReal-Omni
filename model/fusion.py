"""Fuse dense math inside H3 blocks while preserving surrounding layout hooks."""
from types import MethodType
import os
import warnings
import torch
import torch.nn.functional as F
import triton
from diffusers.models.attention_dispatch import dispatch_attention_fn
from . import kernels as k


class FusedAttention:
    def __init__(self, original, ordered_qk=False):
        self.ordered_qk = ordered_qk
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if rotary_emb is None or rotary_emb[0].ndim != 2:
            raise ValueError("fused attention requires a padless rotary layout")
        if isinstance(hidden_states, tuple):
            quantized, scale, shape = hidden_states
            qkv = attn.to_qkv.forward_quantized(quantized, scale, shape[:-1])
            query, key, value = qkv.chunk(3, dim=-1)
        elif attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query, key, value = [x.unflatten(-1, (attn.heads, -1)) for x in (query, key, value)]
        rotary_dim = rotary_emb[0].shape[-1]
        if rotary_dim > query.shape[-1] or rotary_dim % 2:
            raise ValueError("rotary dimensions must be even and fit the attention head")
        q_out = torch.empty_like(query, memory_format=torch.contiguous_format)
        k_out = torch.empty_like(key, memory_format=torch.contiguous_format)
        common = (query, key, attn.norm_q.weight, attn.norm_k.weight, *rotary_emb, q_out, k_out,
                  query.shape[1], query.shape[2], query.stride(1), query.stride(2),
                  key.stride(1), key.stride(2), query.shape[-1])
        grid = (query.numel() // query.shape[-1],)
        if rotary_dim == query.shape[-1]:
            k._qk_norm_full_rope_paired_kernel[grid](*common, float(attn.norm_q.eps),
                triton.next_power_of_2(query.shape[-1] // 2), num_warps=4, enable_fp_fusion=False)
        else:
            ordered = self.ordered_qk and query.shape[-1] == 128
            args = (*common, rotary_dim, float(attn.norm_q.eps), triton.next_power_of_2(query.shape[-1]))
            try:
                k._qk_norm_rope_kernel[grid](*args, ordered_reduction=ordered,
                    num_warps=1 if ordered else 4, enable_fp_fusion=False)
            except Exception as error:
                from .acceleration import optional_kernel_failure
                if not ordered or not optional_kernel_failure(error):
                    raise
                warnings.warn(f'Ordered Q/K kernel unavailable; using reference reduction: {error}')
                self.ordered_qk = False
                k._qk_norm_rope_kernel[grid](*args, num_warps=4, enable_fp_fusion=False)
        output = dispatch_attention_fn(q_out, k_out, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False, backend=self._attention_backend,
            parallel_config=self._parallel_config)
        return attn.to_out[1](attn.to_out[0](output.flatten(2, 3).type_as(q_out)))


def fused_block(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None, adaln_table=None):
    table = adaln_table
    cache = getattr(self, "_lynnreal_adaln_cache", None)
    key = self._lynnreal_cache_key[0] if cache is not None else None
    if table is None and key is not None:
        table = cache.get(key)
    if table is None:
        linear = self.adaln_proj.linear
        table = linear(F.silu(temb).to(linear.weight.dtype)).view(-1, 6 * self.adaln_proj.hidden_size)
        if key is not None:
            if len(cache) >= 16:
                cache.pop(next(iter(cache)))
            cache[key] = table
    use_quant = self._lynnreal_fused_quant
    if use_quant and hasattr(self.attn.to_qkv, "forward_quantized"):
        quantized, scales = k._rms_adaln_quant_int8(hidden_states, self.norm1, table, adaln_indices, 0, 1)
        attention_input = quantized, scales, hidden_states.shape
    else:
        attention_input = k._rms_adaln(hidden_states, self.norm1, table, adaln_indices, 0, 1)
    attention_output = self.attn(attention_input, rotary_emb, attention_mask)
    up, down = self.ff.net[0].proj, self.ff.net[2]
    if use_quant and hasattr(up, "forward_quantized"):
        if self._lynnreal_residual_quant:
            hidden_states, quantized, scales = k.residual_rms_quant(
                hidden_states, attention_output, self.norm2, table, adaln_indices)
        else:
            hidden_states = k._gated_residual(hidden_states, attention_output, table, adaln_indices, 2)
            quantized, scales = k._rms_adaln_quant_int8(hidden_states, self.norm2, table, adaln_indices, 3, 4)
        projected = up.forward_quantized(quantized, scales, hidden_states.shape[:-1])
    else:
        hidden_states = k._gated_residual(hidden_states, attention_output, table, adaln_indices, 2)
        projected = up(k._rms_adaln(hidden_states, self.norm2, table, adaln_indices, 3, 4))
    if use_quant and hasattr(down, "forward_quantized"):
        quantized, scales = k._swiglu_quant_int8(projected)
        output = down.forward_quantized(quantized, scales, projected.shape[:-1])
    else:
        output = down(k._swiglu(projected))
    return k._gated_residual(hidden_states, output, table, adaln_indices, 5)


def guarded_block(self, *args, **kwargs):
    try:
        return fused_block(self, *args, **kwargs)
    except Exception as error:
        from .acceleration import optional_kernel_failure
        if not optional_kernel_failure(error):
            raise
        warnings.warn(f'Optional block fusion unavailable; using native block: {error}', RuntimeWarning)
        self.forward = self._lynnreal_native_forward
        self.attn.set_processor(self._lynnreal_native_processor)
        from .int8 import Int8Linear
        for layer in self.modules():
            if isinstance(layer, Int8Linear):
                layer.fused_ops = False
        return self.forward(*args, **kwargs)


def enable_fusion(transformer, fused_quant=False):
    residual_quant = fused_quant and torch.cuda.get_device_capability()[0] == 9
    ordered_qk = residual_quant and os.environ.get("LYNNREAL_LONG_KERNELS", "1") != "0"
    if fused_quant:
        from .int8 import Int8Linear
        for module in transformer.modules():
            if isinstance(module, Int8Linear):
                module.fused_ops = True
    for block in transformer.transformer_blocks:
        if any(getattr(block, name, 1) != 1 for name in ("h3_attention_token_stride", "h3_ffn_token_stride", "h3_depth_span")):
            raise ValueError("fusion requires dense attention and FFN math within each block")
        block._lynnreal_fused_quant = fused_quant
        block._lynnreal_residual_quant = residual_quant
        if not hasattr(block, '_lynnreal_native_forward'):
            block._lynnreal_native_forward = block.forward
            block._lynnreal_native_processor = block.attn.processor
        block.forward = MethodType(guarded_block, block)
        block.attn.set_processor(FusedAttention(block.attn.processor, ordered_qk))
    return {"rmsnorm_adaln": True, "qk_norm_rope": True, "swiglu": True,
            "qk_output_allocation": "contiguous_without_copy", "residual_block": 1024,
            "gated_residual": True, "activation_quantization_fused": fused_quant,
            "residual_ffn_norm_fused": residual_quant,
            "qk_ordered_reduction": ordered_qk,
            "int8_epilogue_fused": fused_quant,
            "blocks": len(transformer.transformer_blocks), "token_dropping": False}


def cache_time_modulation(transformer):
    """Cache exact AdaLN tables for immutable inference weights, keyed by all input times."""
    key = [None]
    for block in transformer.transformer_blocks:
        if not hasattr(block, "_lynnreal_fused_quant"):
            raise ValueError("time modulation caching requires fused blocks")
        block._lynnreal_adaln_cache = {}
        block._lynnreal_cache_key = key

    def select_key(module, args, kwargs):
        if module.training or torch.is_grad_enabled():
            raise ValueError("cached time modulation requires frozen inference")
        times = kwargs["timestep"]
        key[0] = (tuple(times.shape), str(times.dtype), tuple(times.detach().cpu().flatten().tolist()))

    transformer.register_forward_pre_hook(select_key, with_kwargs=True)
    return {"kind": "exact AdaLN tables", "key": "complete input noise-time tensor",
            "maximum_entries_per_block": 16, "requires_immutable_weights": True}
