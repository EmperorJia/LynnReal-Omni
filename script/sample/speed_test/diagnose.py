"""Untimed long-clip numerical diagnostic; saves the first non-finite stage."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from model.pipeline import Pipeline,encode_conditioning
from diffusers.modular_pipelines.minimax_h3 import before_encoder
from diffusers.utils.export_utils import encode_video
from run import PROMPT
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--gemm',choices=['triton','torch'],required=True)
p.add_argument('--variant',choices=['standard','flash'],default='flash')
p.add_argument('--frames',type=int,default=360)
p.add_argument('--name')
a=p.parse_args();out=ROOT/'output/speed_test'/a.variant/(a.name or f'diagnose_768p_{a.frames}_{a.gemm}')
out.mkdir(parents=True,exist_ok=False)
before_encoder.MINIMAX_H3_MAX_DURATION=362/24
weights=ROOT/'weight'/a.variant
prompt=PROMPT.replace('0.92-second',f'{a.frames/24:.2f}-second')
condition,_=encode_conditioning(weights,prompt,[],768,1344,a.frames,ROOT/'output/.cache')
pipe=Pipeline(weights,int8=True,fused=True,light_vae=ROOT/'weight/light-vae',compile_vae=False,adaln_cache=True,int8_gemm=a.gemm,attention_backend='_flash_3',vae_attention='_flash_3')
pipe.pipe.vae.tile_layout='adaptive'
records=[]
def check(name,x):
 if not isinstance(x,torch.Tensor):return
 row=dict(stage=name,shape=list(x.shape),dtype=str(x.dtype),finite=bool(torch.isfinite(x).all()),min=float(x.min()),max=float(x.max()))
 records.append(row);(out/'stages.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
 if not row['finite']:raise RuntimeError(f'Non-finite tensor at {name}')
for i,block in enumerate(pipe.transformer.transformer_blocks):
 block.register_forward_hook(lambda m,args,result,i=i:check(f'block_{i}',result))
original=pipe.pipe.vae.decode
def decode(latents,*args,**kwargs):
 check('vae_input',latents);torch.save(latents.cpu(),out/'decoder_input.pt')
 result=original(latents,*args,**kwargs);check('vae_output',result[0] if isinstance(result,tuple) else result.sample)
 return result
pipe.pipe.vae.decode=decode
state,timing=pipe.generate(condition,[],768,1344,a.frames,77,3 if a.variant=='flash' else 4,keep_latents=True)
check('final_video_latents',state['latents'])
frames=state['videos'][0][:a.frames]
encode_video(frames,fps=24,output_path=str(out/'video.mp4'))
(out/'diagnostic.json').write_text(json.dumps(dict(untimed_diagnostic=True,gemm=a.gemm,frames=a.frames,timing_with_hooks=timing),indent=2))
