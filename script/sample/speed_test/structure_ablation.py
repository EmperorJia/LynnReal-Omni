"""Same-weight token-layout and same-schedule decoder-depth controls for Flash."""
import argparse,json,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from run import PROMPT,Tee


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);a=p.parse_args()
    if Path(a.name).name!=a.name:p.error('name must be a directory basename')
    out=ROOT/'output/speed_test/flash'/a.name;out.mkdir(parents=True,exist_ok=False)
    log=(out/'run.log').open('w',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    import torch,numpy as np
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.utils.export_utils import encode_video
    from model.pipeline import Pipeline,encode_conditioning,release_memory
    from model.light_vae import LightVAE
    from model.flash import configure_flash
    from model.timing import measure_decoder
    from model.provenance import snapshot_sources,changed_sources
    from safetensors.torch import save_file
    assert torch.cuda.device_count()==1
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem','--format=csv,noheader'],text=True).strip()
    source=snapshot_sources(ROOT/'output/.cache');weights=ROOT/'weight/flash'
    prompt=PROMPT.replace('0.92-second',f'{22/24:.2f}-second')
    condition,_=encode_conditioning(weights,prompt,[],544,960,22,ROOT/'output/.cache')
    pipe=Pipeline(weights,int8=True,fused=True,int8_gemm='triton',attention_backend='_flash_3',vae_attention='_flash_3',light_vae=ROOT/'weight/light-vae',adaln_cache=True,compile_vae=True)
    pipe.pipe.vae.tile_layout='adaptive'
    def uncompressed():
        for handle in pipe.transformer._lynnreal_flash_hooks:handle.remove()
        del pipe.transformer._lynnreal_flash_hooks
    def full_batched():
        configure_flash(pipe.transformer,pipe.config)
        pipe.pipe.vae.to('cpu')
        core=AutoencoderKLMiniMaxH3.from_pretrained(ROOT/'weight/vae',torch_dtype=torch.float32,local_files_only=True)
        settings=json.loads((ROOT/'weight/light-vae/decode_config.json').read_text())
        vae=LightVAE(core,settings,tile_layout='native').eval().requires_grad_(False).cuda()
        vae.core.set_attention_backend('_flash_3');pipe.pipe.update_components(vae=vae);measure_decoder(vae,pipe.decoder_events)
    def light_batched():
        pipe.pipe.vae.to('cpu')
        vae=LightVAE.from_pretrained(ROOT/'weight/light-vae',tile_layout='native').cuda()
        vae.core.set_attention_backend('_flash_3');pipe.pipe.update_components(vae=vae);measure_decoder(vae,pipe.decoder_events)
    rows=[];states={}
    for label,apply in [('compressed',lambda:None),('uncompressed',uncompressed),('full_batched_native',full_batched),('light_batched_native',light_batched)]:
        apply();release_memory();xs=[]
        for i in range(7):
            state,timing=pipe.generate(condition,[],544,960,22,77,3,keep_latents=i==6)
            if i>=2:xs.append(timing)
            print(json.dumps(dict(stage=label,iteration=i,warmup=i<2,**timing)),flush=True)
        d=out/label;d.mkdir()
        frames=[f.crop((0,2,960,542)) for f in state['videos'][0][:22]]
        rgb=np.stack([np.asarray(x) for x in frames]);assert len(frames)==22 and rgb.max()>0
        latents={k:state[k].detach().float().cpu().contiguous() for k in ['latents','audio_latents']};assert all(torch.isfinite(x).all() for x in latents.values())
        save_file(latents,str(d/'latents.safetensors'))
        encode_video(frames,fps=24,output_path=str(d/'video.mp4'),audio=state['audio'][0][...,:round(22*state['sampling_rate']/24)],audio_sample_rate=state['sampling_rate'])
        compare=None;reference={'uncompressed':'compressed','light_batched_native':'full_batched_native'}.get(label)
        if reference:
            old=states[reference];delta=rgb.astype(np.float32)-old['rgb'].astype(np.float32)
            compare=dict(reference=reference,video_latents_equal=torch.equal(latents['latents'],old['latents']['latents']),video_latents_relative_l2=float((latents['latents']-old['latents']['latents']).norm()/old['latents']['latents'].norm().clamp_min(1e-8)),rgb_rms_255=float(np.sqrt(np.mean(delta**2))))
        states[label]=dict(rgb=rgb,latents=latents)
        rows.append(dict(stage=label,dit_ms=statistics.median(x['dit_ms'] for x in xs),decoder_ms=statistics.median(x['video_decoder_ms'] for x in xs),sum_ms=statistics.median(x['dit_ms']+x['video_decoder_ms'] for x in xs),generate_ms=statistics.median(x['generation_and_decode_ms'] for x in xs),peak_gib=max(x['peak_allocated_bytes'] for x in xs)/2**30,measurements=xs,comparison=compare,video=str(d/'video.mp4')))
        (out/'progress.json').write_text(json.dumps(rows,indent=2));del state
    (out/'structure.json').write_text(json.dumps(dict(hardware=hardware,weights=str(weights),prompt=prompt,seed=77,steps=3,geometry=[960,540,22],warmups=2,repeats=5,rows=rows,source=source,source_changed_during_run=changed_sources(source)),indent=2))

if __name__=='__main__':main()
