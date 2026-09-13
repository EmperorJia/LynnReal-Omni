"""W8A8 projections and optional post-training quantization of standard DiT."""
import torch
from torch import nn


class Int8Linear(nn.Module):
    def __init__(self, source, gemm="torch"):
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().amax(1, keepdim=True).clamp_min_(1e-8) / 127
        quantized = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer("weight_int8", quantized.contiguous().t())
        self.register_buffer("weight_scale", scale.t().contiguous())
        self.register_buffer("bias", None if source.bias is None else source.bias.detach().to(torch.bfloat16))
        self.register_buffer("_dtype_token", source.weight.new_empty(0), persistent=False)
        self.in_features, self.out_features = source.in_features, source.out_features
        self.fused_ops = False
        if gemm not in ("torch", "triton"):
            raise ValueError("unsupported INT8 GEMM backend")
        self.gemm = gemm

    @property
    def weight(self):
        return self._dtype_token

    @classmethod
    def from_packed(cls, weight, scale, bias=None, gemm="triton"):
        """Load QAT rows exactly, without a float round-trip or requantization."""
        if weight.dtype != torch.int8 or scale.shape != (weight.shape[0], 1):
            raise ValueError("expected INT8 [out,in] weights and [out,1] scales")
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.register_buffer("weight_int8", weight.t())
        layer.register_buffer("weight_scale", scale.t().contiguous())
        layer.register_buffer("bias", bias)
        layer.register_buffer("_dtype_token", torch.empty(0, device=weight.device, dtype=torch.bfloat16), persistent=False)
        layer.out_features, layer.in_features = weight.shape
        layer.gemm, layer.fused_ops = gemm, False
        return layer

    def forward_quantized(self, quantized, scale, output_shape):
        if self.gemm == "triton":
            from .int8_gemm import int8_matmul
            return int8_matmul(quantized.reshape(-1, self.in_features).contiguous(), self.weight_int8,
                scale, self.weight_scale, self.bias).view(*output_shape, self.out_features)
        flat = quantized.reshape(-1, self.in_features).contiguous()
        rows = flat.shape[0]
        pad = max(32, ((rows + 7) // 8) * 8) - rows
        if pad:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, pad))
        accumulator = torch._int_mm(flat, self.weight_int8)[:rows]
        if self.fused_ops:
            from .kernels import int8_epilogue
            return int8_epilogue(accumulator, scale, self.weight_scale, self.bias).view(*output_shape, self.out_features)
        output = accumulator.float() * scale.reshape(-1, 1).float() * self.weight_scale
        if self.bias is not None:
            output = output + self.bias.float()
        return output.to(torch.bfloat16).view(*output_shape, self.out_features)

    def forward(self, value):
        flat = value.reshape(-1, self.in_features).contiguous()
        if self.fused_ops:
            from .kernels import quantize_rows
            quantized, scale = quantize_rows(flat)
        else:
            scale = flat.float().abs().amax(-1, keepdim=True).clamp_min_(1e-8) / 127
            quantized = (flat.float() / scale).round().clamp(-127, 127).to(torch.int8)
        return self.forward_quantized(quantized, scale, value.shape[:-1])


@torch.no_grad()
def quantize_transformer(transformer, protected_edges=1, gemm="torch"):
    blocks = transformer.transformer_blocks
    if protected_edges < 0 or 2 * protected_edges >= len(blocks):
        raise ValueError("invalid number of protected edge blocks")
    converted = []
    for i, block in enumerate(blocks):
        attn = block.attn
        sources = (attn.to_q, attn.to_k, attn.to_v)
        if not all(isinstance(layer, nn.Linear) for layer in sources):
            raise ValueError("quantization requires an unconverted transformer")
        first = sources[0]
        fused = nn.Linear(first.in_features, sum(layer.out_features for layer in sources),
                          bias=any(layer.bias is not None for layer in sources),
                          device=first.weight.device, dtype=first.weight.dtype)
        fused.weight.copy_(torch.cat([layer.weight for layer in sources]))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([layer.bias if layer.bias is not None else
                layer.weight.new_zeros(layer.out_features) for layer in sources]))
        quantized = protected_edges <= i < len(blocks) - protected_edges
        attn.to_qkv = Int8Linear(fused, gemm) if quantized else fused
        attn.to_q = attn.to_k = attn.to_v = None
        attn.fused_projections = True
        if quantized:
            attn.to_out[0] = Int8Linear(attn.to_out[0], gemm)
            block.ff.net[0].proj = Int8Linear(block.ff.net[0].proj, gemm)
            block.ff.net[2] = Int8Linear(block.ff.net[2], gemm)
            converted.append(i)
    return {"format": "W8A8", "gemm": "torch._int_mm" if gemm == "torch" else "Triton INT8 with fused BF16 epilogue", "accumulator": "INT32",
            "tile_selection": "device-specific autotune with persistent cache" if gemm == "triton" else None,
            "activation_scale": "per-token", "weight_scale": "per-output-channel",
            "rounding": "nearest-even", "protected_edge_blocks": protected_edges,
            "quantized_blocks": converted, "quantized_linears": 4 * len(converted),
            "qkv_fused": True, "adaln_and_refiner_and_io_quantized": False}
