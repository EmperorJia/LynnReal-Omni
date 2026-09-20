"""Check estimated token expansion against the installed Qwen preprocessor."""
import importlib.util
import json
from pathlib import Path
import sys
import types

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
sys.argv = [sys.argv[0], '--cpu']
import torch
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
spec = importlib.util.spec_from_file_location('safety', root/'custom_nodes/ComfyUI-LynnReal/sampling_safety.py')
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)
checks = []
for h, w in ((17, 35), (720, 1280), (2048, 3648), (2048, 8192)):
    data = torch.zeros((1, h, w, 3))
    patches, grid = process_qwen2vl_images(data, patch_size=16)
    tokens = {'qwen3vl_32b': [[({'type': 'image', 'data': data}, 1), (42, 1)]]}
    budget, sequence, peak = safety.encoder_memory(tokens)
    assert peak == patches.shape[0] and sequence == patches.shape[0] // 4 + 1
    assert budget > 0
    checks.append({'shape': [h, w], 'patches': peak, 'expanded_sequence': sequence})
    del data, patches
plain = {'qwen3vl_32b': [[(42, 1)] * 1000]}
assert safety.encoder_memory(plain)[0] == 0
long_text = {'qwen3vl_32b': [[(42, 1)] * 10000]}
assert safety.encoder_memory(long_text)[0] > 6*safety.GB
safety.install()
from comfy.text_encoders.minimax import MiniMaxH3TEModel
estimate = MiniMaxH3TEModel.memory_estimation_function
assert estimate(None, long_text) > 0
assert estimate(None, plain) == 0
safety.install()
assert MiniMaxH3TEModel.memory_estimation_function is estimate
saved_kitchen = sys.modules['comfy_kitchen']
try:
    sys.modules['comfy_kitchen'] = types.ModuleType('comfy_kitchen')
    safety.install()
    assert MiniMaxH3TEModel.memory_estimation_function is estimate
finally:
    sys.modules['comfy_kitchen'] = saved_kitchen
print(json.dumps({'status': 'PASS', 'geometry_checks': checks, 'plain_budget': estimate(None, plain),
                  'optional_backend_absent': 'PASS'}))
