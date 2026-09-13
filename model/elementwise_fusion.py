"""Fuse elementwise BF16 work while retaining native normalization and attention."""
from types import MethodType
import torch.nn.functional as F
from .kernels import _gated_residual, _swiglu


def block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    linear = self.adaln_proj.linear
    width = self.adaln_proj.hidden_size
    table = linear(F.silu(temb).to(linear.weight.dtype)).view(-1, 6 * width)
    shift, scale, _, ff_shift, ff_scale, _ = table.chunk(6, dim=-1)
    normalized = self.norm1(hidden_states)
    normalized = normalized * (1 + scale.index_select(0, adaln_indices)) + shift.index_select(0, adaln_indices)
    update = self.attn(normalized, rotary_emb, attention_mask)
    hidden_states = _gated_residual(hidden_states, update, table, adaln_indices, 2)
    normalized = self.norm2(hidden_states)
    normalized = normalized * (1 + ff_scale.index_select(0, adaln_indices)) + ff_shift.index_select(0, adaln_indices)
    projected = self.ff.net[0].proj(normalized)
    update = self.ff.net[2](_swiglu(projected))
    return _gated_residual(hidden_states, update, table, adaln_indices, 5)


def enable_elementwise_fusion(transformer, fused_quant=False):
    if fused_quant or transformer.training:
        raise ValueError('elementwise fusion requires an evaluation-mode BF16 model')
    for block in transformer.transformer_blocks:
        block.forward = MethodType(block_forward, block)
    return dict(kind='elementwise', swiglu=True, gated_residual=True,
                normalization='native', attention_processor='native',
                blocks=len(transformer.transformer_blocks), activation_quantization_fused=False)
