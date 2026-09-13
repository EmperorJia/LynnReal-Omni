"""Prepare common INT8 T2V shapes; no training or benchmark latency claims."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def inference_sources():
    """Follow local imports, excluding unrelated experiment and streaming scripts."""
    pending = ['model.__init__', 'model.pipeline', 'model.acceleration', 'script.precompile_profiles']
    found = {}
    while pending:
        module = pending.pop()
        path = ROOT.joinpath(*module.split('.')).with_suffix('.py')
        if module in found or not path.is_file():
            continue
        source = path.read_bytes()
        found[module] = source
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = module.split('.')[:-node.level] if node.level else []
                prefix += node.module.split('.') if node.module else []
                base = '.'.join(prefix)
                names = [base, *(base+'.'+alias.name for alias in node.names)]
            pending.extend(name for name in names if name.startswith(('model.', 'script.')))
    return found


def main():
    import torch
    if os.environ.get('LYNNREAL_PRECOMPILE','1') == '0':
        print('Common-shape preparation explicitly disabled.');return
    if not torch.cuda.is_available():
        print('WARNING: no GPU; common shapes will compile on the sampling device.');return
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['standard','flash'])
    args=p.parse_args()
    if args.variant is None:
        for variant in ('standard','flash'):
            # Release all model and compiler memory between variants.
            result=subprocess.run([sys.executable,__file__,'--variant',variant])
            if result.returncode:
                print(f'WARNING: {variant} preparation failed; sampling will prepare its shapes on demand.',flush=True)
        return
    device=torch.cuda.get_device_properties(0)
    memory=device.total_memory/2**30
    if memory<45 or (args.variant=='standard' and memory<75):
        print(f'WARNING: {args.variant} full-model preparation deferred on {memory:.1f} GiB GPU.');return
    weights=ROOT/'weight'/args.variant
    if not (weights/'inference_config.json').exists():
        print(f'WARNING: weights absent at {weights}; only standalone kernels were compiled.');return
    from model.pipeline import Pipeline,encode_conditioning,release_memory
    from model.acceleration import fastest_available
    accel=fastest_available()
    if not accel['fused']:
        print('WARNING: full-model preparation skipped without supported optional fusion.');return
    profiles=[(960,540,22)]
    if memory>=75:
        profiles += [(1344,768,n) for n in (120,240,360)]
    fingerprint=hashlib.sha256()
    sources = inference_sources()
    for name, source in sorted(sources.items()):
        fingerprint.update(name.encode()); fingerprint.update(source)
    for name in ('inference_config.json','transformer/config.json','transformer/diffusion_pytorch_model.safetensors.index.json'):
        fingerprint.update((weights/name).read_bytes())
    for name in ('config.json', 'decode_config.json', 'diffusion_pytorch_model.safetensors.index.json'):
        fingerprint.update((ROOT/'weight/light-vae'/name).read_bytes())
    import triton, diffusers
    signature=[fingerprint.hexdigest(),torch.__version__,torch.version.cuda,triton.__version__,
               diffusers.__version__,device.name,accel['attention_backend']]
    path=ROOT/'output/setup'/('profiles-'+args.variant+'-'+hashlib.sha256(repr(signature).encode()).hexdigest()[:16]+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    done=json.loads(path.read_text()) if path.exists() else {'signature':signature,'sources':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},'completed':[]}
    remaining=[x for x in profiles if list(x) not in done['completed']]
    if not remaining:
        print(f'{args.variant}: common shapes already prepared ({path}).',flush=True);return
    prompt=(ROOT/'test/t2v.txt').read_text()
    condition,_=encode_conditioning(weights,prompt,[],544,960,22,ROOT/'output/.cache')
    pipe=Pipeline(weights,int8=True,fused=True,light_vae=ROOT/'weight/light-vae',adaln_cache=True,
                  int8_gemm='triton',attention_backend=accel['attention_backend'],
                  vae_attention=accel['attention_backend'],compile_vae=True)
    pipe.pipe.vae.tile_layout='adaptive'
    if memory<64:pipe.pipe.vae.tile_batch=1
    steps=4 if args.variant=='standard' else 3
    for width,height,frames in remaining:
        canvas=(height+31)//32*32
        for iteration in range(2):
            state,timing=pipe.generate(condition,[],canvas,width,frames,7,steps)
            if len(state['videos'][0])<frames:raise RuntimeError('decoder returned too few frames')
            print(json.dumps(dict(phase='preparation',variant=args.variant,shape=[width,height,frames],
                                  iteration=iteration,**timing)),flush=True)
            del state
        done['completed'].append([width,height,frames])
        path.write_text(json.dumps(done,indent=2))
        release_memory()
    print(f'{args.variant}: compilation profiles saved to {path}',flush=True)

if __name__=='__main__':main()
