"""Boundary and real GPU arithmetic checks for the one-click safety hooks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

parser = argparse.ArgumentParser()
parser.add_argument('comfyui', type=Path)
parser.add_argument('--backend', choices=('triton', 'cuda'), default='triton')
options = parser.parse_args()
root = options.comfyui
sys.path.insert(0, str(root))
sys.argv = [sys.argv[0]]  # ComfyUI parses argv on import
spec = importlib.util.spec_from_file_location('sampling_safety', root / 'custom_nodes/ComfyUI-LynnReal/sampling_safety.py')
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)

assert safety.chunk_rows(100, 7168, 28672) == 100
assert safety.chunk_rows(83124, 7168, 28672) <= 16384
assert safety.chunk_rows(16384, 28672, 7168) <= 16384
assert safety.additional_memory({'positive': [{'cross_attn': torch.empty(1, 1000, 1, device='meta')} ]})[0] == 0
reference = {'latent': torch.empty(1, 24, 1, 128, 228, device='meta')}
refs, tokens = safety.conditioning_rows({'minimax_refs': [reference] * 3,
                                       'cross_attn': torch.empty(1, 22000, 1, device='meta')})
assert refs == 21888 and tokens == 22000

plain = {'qwen3vl_32b': [[(42, 1.0)] * 1000]}
large_image = ({'type': 'image', 'data': torch.empty(1, 2048, 3648, 3, device='meta')}, 1.0)
multimodal = {'qwen3vl_32b': [[large_image, (42, 1.0), large_image]]}
required, expanded, patches = safety.encoder_memory(multimodal)
assert (expanded, patches) == (14593, 29184)
assert required > 6 * safety.GB
assert safety.encoder_memory(plain)[0] == 0

import comfy.model_base
import comfy_kitchen as ck
ck.registry.enable(options.backend)
ck.registry.set_priority([options.backend, 'eager'])
original = ck.registry.get_implementation('int8_linear')
assert original.__module__.startswith('comfy_kitchen.backends.' + options.backend), original
safety.install()
wrapped = ck.registry.get_implementation
safety.install()
assert ck.registry.get_implementation is wrapped
from comfy.text_encoders.minimax import MiniMaxH3TEModel
encoder_estimate = MiniMaxH3TEModel.memory_estimation_function
safety.install()
assert MiniMaxH3TEModel.memory_estimation_function is encoder_estimate
# A heavy request cannot leave a reserve behind for the next plain prompt.
assert encoder_estimate(None, multimodal) == required
assert encoder_estimate(None, plain) == 0

torch.manual_seed(10)
results = []
for dtype in (torch.bfloat16, torch.float16):
    for act in (None, 'swiglu', 'gelu_tanh'):
        x = torch.randn(3, 43, 128, dtype=dtype, device='cuda')
        k = 64 if act == 'swiglu' else 128
        w = torch.randint(-127, 128, (96, k), dtype=torch.int8, device='cuda')
        for per_channel in (False, True):
            scale = torch.full((96,) if per_channel else (1,), 0.001, device='cuda')
            bias = torch.randn(96, device='cuda', dtype=dtype)
            expected = original(x, w, scale, bias=bias, out_dtype=dtype, input_act=act)
            safety.MAX_BUFFER_BYTES = 96 * 4 * 31
            actual = ck.int8_linear(x, w, scale, bias=bias, out_dtype=dtype, input_act=act)
            torch.cuda.synchronize()
            error = (actual.float() - expected.float()).abs().max().item()
            assert torch.equal(actual, expected), (dtype, act, per_channel, error)
            if act is None:
                from comfy_kitchen.tensor.int8 import _dtype_code
                via_op = torch.ops.comfy_kitchen.int8_linear(
                    x, w, scale, bias, _dtype_code(dtype), False, 256)
                assert torch.equal(via_op, expected), 'torch.ops dispatch bypassed safety'
            results.append({'dtype': str(dtype), 'activation': act,
                            'per_channel': per_channel, 'max_abs': error})
print(json.dumps({'status': 'PASS', 'backend': options.backend, 'cases': results}, indent=2))
