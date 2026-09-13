"""Paired cumulative execution ablation on one GPU; fixed weights, prompt and NFE."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from run import PROMPT,Tee


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['standard','flash'],required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if Path(args.name).name!=args.name or args.warmups<2 or args.repeats<3:p.error('invalid run name or repetition count')
    out=ROOT/'output/speed_test'/args.variant/args.name
    out.mkdir(parents=True,exist_ok=False)
    log=(out/'run.log').open('w',buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    import numpy as np
    import torch
    from torch import nn
    from diffusers.utils.export_utils import encode_video
    from model.pipeline import Pipeline,encode_conditioning,release_memory
    from model.provenance import snapshot_sources,changed_sources
    from model.int8 import Int8Linear
    from model.fusion import enable_fusion,cache_time_modulation
    from model.output import configure_video_output
    from model.light_vae import LightVAE
    from model.timing import measure_decoder
    from safetensors.torch import save_file
    weights=ROOT/'weight'/args.variant
    source=snapshot_sources(ROOT/'output/.cache')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem','--format=csv,noheader'],text=True).strip()
    assert torch.cuda.device_count()==1
    print(hardware,flush=True)
    prompt=PROMPT.replace('0.92-second',f'{22/24:.2f}-second')
    (out/'prompt.txt').write_text(prompt+'\n')
    condition,_=encode_conditioning(weights,prompt,[],544,960,22,ROOT/'output/.cache')
    pipe=Pipeline(weights,int8=args.variant=='flash',fused=False,int8_gemm='torch',
                  attention_backend='native',vae_attention='native')
    # Restore the upstream CPU conversion for the baseline, then add the release optimization explicitly.
    processor=pipe.pipe.video_processor
    processor.postprocess_video=type(processor).postprocess_video.__get__(processor,type(processor))
    blocks=pipe.transformer.transformer_blocks
    def qkv():
        for block in blocks:
            attn=block.attn
            parts=[attn.to_q,attn.to_k,attn.to_v]
            first=parts[0]
            layer=nn.Linear(first.in_features,sum(x.out_features for x in parts),bias=any(x.bias is not None for x in parts),device=first.weight.device,dtype=first.weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(torch.cat([x.weight for x in parts]))
                if layer.bias is not None:layer.bias.copy_(torch.cat([x.bias if x.bias is not None else x.weight.new_zeros(x.out_features) for x in parts]))
            attn.to_qkv=layer.eval().requires_grad_(False)
            attn.to_q=attn.to_k=attn.to_v=None
            attn.fused_projections=True
    def quantize():
        for block in blocks[1:-1]:
            block.attn.to_qkv=Int8Linear(block.attn.to_qkv)
            block.attn.to_out[0]=Int8Linear(block.attn.to_out[0])
            block.ff.net[0].proj=Int8Linear(block.ff.net[0].proj)
            block.ff.net[2]=Int8Linear(block.ff.net[2])
    def gemm():
        for module in pipe.transformer.modules():
            if isinstance(module,Int8Linear):module.gemm='triton'
    def attention():
        pipe.transformer.set_attention_backend('_flash_3')
        pipe.pipe.vae.set_attention_backend('_flash_3')
    def light():
        # Release full decoder residency before loading the student; the encoder is not used by T2V.
        pipe.pipe.vae.to('cpu')
        vae=LightVAE.from_pretrained(ROOT/'weight/light-vae',tile_layout='native').cuda()
        vae.core.set_attention_backend('_flash_3')
        pipe.pipe.update_components(vae=vae)
        measure_decoder(vae,pipe.decoder_events)
    def adaptive():pipe.pipe.vae.tile_layout='adaptive'
    def compile_decoder():
        decoder=pipe.pipe.vae.core.decoder
        decoder.forward=torch.compile(decoder.forward,fullgraph=True,dynamic=False)
    stages=[('baseline',lambda:None)]
    if args.variant=='standard':stages += [('qkv',qkv),('int8',quantize)]
    stages += [('fusion',lambda:enable_fusion(pipe.transformer,fused_quant=True)),('triton',gemm),
               ('fa3',attention),('modulation_cache',lambda:cache_time_modulation(pipe.transformer)),
               ('gpu_rgb',lambda:configure_video_output(processor)),('light_decoder',light),
               ('adaptive_tiles',adaptive),('compiled_decoder',compile_decoder)]
    steps=4 if args.variant=='standard' else 3
    rows=[];previous=None
    for index,(label,apply) in enumerate(stages):
        apply();release_memory()
        measurements=[]
        for i in range(args.warmups+args.repeats):
            state,timing=pipe.generate(condition,[],544,960,22,77,steps,keep_latents=i==args.warmups+args.repeats-1)
            if i>=args.warmups:measurements.append(timing)
            print(json.dumps(dict(stage=label,iteration=i,warmup=i<args.warmups,**timing)),flush=True)
        folder=out/f'{index:02d}_{label}';folder.mkdir()
        frames=[f.crop((0,2,960,542)) for f in state['videos'][0][:22]]
        rgb=np.stack([np.asarray(f) for f in frames])
        assert len(frames)==22 and rgb.max()>0,'empty or black output'
        latents={k:state[k].detach().float().cpu().contiguous() for k in ['latents','audio_latents']}
        assert all(torch.isfinite(x).all() for x in latents.values()),'non-finite latents'
        save_file(latents,str(folder/'latents.safetensors'))
        audio=state['audio'][0][...,:round(22*int(state['sampling_rate'])/24)]
        encode_video(frames,fps=24,output_path=str(folder/'video.mp4'),audio=audio,audio_sample_rate=state['sampling_rate'])
        comparison=None
        if previous is not None:
            delta=rgb.astype(np.float32)-previous['rgb'].astype(np.float32)
            old=previous['latents']['latents'];current=latents['latents']
            comparison=dict(rgb_equal=bool(np.array_equal(rgb,previous['rgb'])),rgb_rms_255=float(np.sqrt(np.mean(delta**2))),rgb_max_255=float(np.abs(delta).max()),video_latents_equal=torch.equal(current,old),video_latents_relative_l2=float((current-old).norm()/old.norm().clamp_min(1e-8)),audio_latents_equal=torch.equal(latents['audio_latents'],previous['latents']['audio_latents']))
        previous=dict(rgb=rgb,latents=latents)
        row=dict(stage=label,order=index,dit_ms=statistics.median(x['dit_ms'] for x in measurements),decoder_ms=statistics.median(x['video_decoder_ms'] for x in measurements),sum_ms=statistics.median(x['dit_ms']+x['video_decoder_ms'] for x in measurements),generate_ms=statistics.median(x['generation_and_decode_ms'] for x in measurements),peak_gib=max(x['peak_allocated_bytes'] for x in measurements)/2**30,measurements=measurements,comparison_to_previous=comparison,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),video=str(folder/'video.mp4'))
        rows.append(row)
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(dict(completed=row)),flush=True)
        del state
    (out/'ablation.json').write_text(json.dumps(dict(variant=args.variant,hardware=hardware,weights=str(weights),steps=steps,geometry=[960,540,22],native_canvas=[960,544],seed=77,prompt=prompt,warmups=args.warmups,repeats=args.repeats,rows=rows,source=source,source_changed_during_run=changed_sources(source)),indent=2))

if __name__=='__main__':main()
