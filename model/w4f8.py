"""Load the complete packed Flash transformer and execute W4A8-FP8 projections."""
import functools
import json
from pathlib import Path
import sys
import torch
from torch import nn
from safetensors.torch import load_file
from .weights import local_weights, sha256


@functools.lru_cache(None)
def fp8_quantizer():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This Flash W4F8 artifact requires a Hopper CUDA GPU")
    sys.path.insert(0, str(Path(__file__).parent / "kernels/fbgemm_genai_120"))
    from fbgemm_gpu.experimental.gen_ai.quantize import quantize_fp8_row
    return quantize_fp8_row


class PackedLinear(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        if set(tensors) - {'weight_int4', 'weight_scale', 'channel_scale', 'bias'}:
            raise ValueError("Unexpected packed projection tensors")
        for name in ('weight_int4', 'weight_scale', 'channel_scale', 'bias'):
            self.register_buffer(name, tensors.get(name))
        if self.weight_int4.dtype not in (torch.int8, torch.uint8):
            raise ValueError("Expected nibble-packed INT4 weights")
        self.in_features = self.weight_int4.shape[1] * 2
        self.out_features = self.weight_int4.shape[0]
        self.register_buffer('_dtype', torch.empty(0, dtype=torch.bfloat16), persistent=False)

    @property
    def weight(self):
        return self._dtype

    def forward(self, x):
        q, scale = fp8_quantizer()(x.reshape(-1, x.shape[-1]).contiguous())
        y = torch.ops.fbgemm.f8i4bf16_shuffled(
            q, self.weight_int4, scale.reshape(-1), self.channel_scale, self.weight_scale)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x.shape[:-1], y.shape[-1])


def load_transformer(folder, device='cuda'):
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    folder = local_weights(folder)
    meta = json.loads((folder / 'manifest.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    arithmetic = meta['arithmetic']
    torch.set_float32_matmul_precision(arithmetic['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = arithmetic['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = arithmetic['cudnn_allow_tf32']
    fp8_quantizer()
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    remaining = set(model.state_dict())
    for name, spec in meta['layers'].items():
        path = folder / spec['file']
        expected = meta['files'][spec['file']]
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Packed tensor checksum mismatch: ' + spec['file'])
        tensors = load_file(str(path), device=device)
        if spec['kind'] == 'native_w4f8':
            layer = PackedLinear(tensors).to(device)
        elif spec['kind'] == 'protected_linear':
            layer = nn.Linear(spec['in_features'], spec['out_features'], bias=spec['bias'],
                              device='meta', dtype=tensors['weight'].dtype)
            layer.load_state_dict(tensors, strict=True, assign=True)
        else:
            raise ValueError('Unknown projection format')
        parent, _, key = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, key, layer)
        remaining.difference_update({name + '.weight', name + '.bias'})
    path = folder / 'remaining.safetensors'
    if sha256(path) != meta['files'][path.name]['sha256']:
        raise ValueError('Remaining tensor checksum mismatch')
    tensors = load_file(str(path), device=device)
    if set(tensors) != remaining:
        raise ValueError('Incomplete Flash model state')
    result = model.load_state_dict(tensors, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    return model.to(device).eval().requires_grad_(False)
