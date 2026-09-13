"""Native latent-overlap continuation with an optional full-codec parity check."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--conditioner-service',type=Path)
    p.add_argument('--resume-from',type=Path)
    p.add_argument('--resume-chunks',type=int,default=0)
    p.add_argument('--warmup-chunks',type=int,default=0,help='Use four-step sampling, four sinks and static text images for this many continuation chunks')
    p.add_argument('--age-horizon',type=int,default=0)
    p.add_argument('--appearance-max-scale',type=float)
    p.add_argument('--coarse-anchor',type=float,default=0.)
    p.add_argument('--joint-second-pass',action='store_true',help='Renoise audio at the native video/audio shift relationship during refinement')
    p.add_argument('--second-appearance',action=argparse.BooleanOptionalAction,default=True,help='Retain first-image moment correction during the second pass')
    p.add_argument('--rolling-history',action='store_true',help='Experimental local-history anchor; retain initial appearance moments but no permanent history latents')
    p.add_argument('--frames',type=int,default=120)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--audit-codec',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--captions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-rope',action='store_true');p.add_argument('--second-steps',type=int,choices=range(5),default=0)
    p.add_argument('--second-sigmas',type=float,nargs='+',help='Experimental sigma grid including terminal zero; no additional shift')
    p.add_argument('--mid-frames',type=int,choices=(0,2),default=0)
    p.add_argument('--compact-mid-clock',action='store_true')
    p.add_argument('--noise-correlation',type=float,default=0.,help='Correlation of second-pass Gaussian noise with original target noise')
    p.add_argument('--appearance-strength',type=float,default=0.,help='Reuse bootstrap source-image latent moments for every continuation')
    p.add_argument('--sink-frames',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--recent-text-context',action='store_true',help='Use the preceding delivered chunk endpoints as Qwen image context')
    p.add_argument('--current-text-boundary',action='store_true',help='Refresh Qwen boundary image from the last delivered frame')
    a=p.parse_args()
    if a.recent_text_context:a.current_text_boundary=True
    if a.warmup_chunks<0 or (a.warmup_chunks and (a.sink_frames>=4 or a.mid_frames or not a.recent_text_context)):
        p.error('warmup requires fewer than four final sinks, no middle latents, and recent text context')
    if a.frames < 18:
        p.error('--frames must be at least 18 for continuation')
    chunks=(a.frames-17+16)//17
    if not 0 <= a.resume_chunks < chunks or bool(a.resume_chunks)!=bool(a.resume_from):
        p.error("--resume-from and a positive --resume-chunks must be supplied together, before the final chunk")
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file,save_file
    from imageio_ffmpeg import get_ffmpeg_exe
    from model.pipeline import Pipeline
    from model.elementwise_fusion import enable_elementwise_fusion
    from model.overlap_stream import NativeOverlapStream
    from model.conditioning_service import encode_remote
    from model.provenance import snapshot_sources,changed_sources
    from model.weights import sha256
    a.output.mkdir(parents=True,exist_ok=False);source=snapshot_sources(a.output/'cache')
    deadline=time.monotonic()+600
    while not (a.bootstrap/'metadata.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Inspect the existing native bootstrap job')
        time.sleep(2)
    bm=json.loads((a.bootstrap/'metadata.json').read_text())
    if bm['changed_sources'] or sha256(a.bootstrap/'latents.safetensors')!=bm['latents_sha256']:
        raise ValueError('bootstrap is unverified or changed')
    if sha256(a.bootstrap/'rgb.npy')!=bm['rgb_sha256']:
        raise ValueError('bootstrap RGB changed')
    captions=json.loads(a.captions.read_text())
    if len(captions)<chunks or not all(isinstance(c,str) and c.strip() for c in captions):
        raise ValueError(f'Provide at least {chunks} nonempty continuation captions')
    weights=(ROOT/'weight/standard').resolve()
    original=np.load(a.bootstrap/'rgb.npy',mmap_mode='r')
    images=[a.output/'context_first.png',a.output/'context_last.png']
    for f,img in zip((original[0],original[16]),images):Image.fromarray(f).save(img)
    contexts={}
    if not a.conditioner_service:
        from model.conditioning import encode_stream_context
        from model.pipeline import release_memory
        resident={}
        static_chunks=min(chunks,a.warmup_chunks) if a.current_text_boundary else chunks
        for i in range(a.resume_chunks+1,static_chunks+1):
            contexts[i]=encode_stream_context(weights,captions[i-1],images,a.output/'cache/text',resident)
        resident.clear();release_memory()
    appearance=None
    if a.appearance_strength:
        appearance=load_file(str(a.bootstrap/'source_static_video_latents.safetensors'))['video']
    pipe=Pipeline(weights,reference=True,native_keyframes=True,vae_offload=True,attention_backend='native')
    pipe.fusion=enable_elementwise_fusion(pipe.transformer)
    stream=NativeOverlapStream(pipe,a.local_rope,0 if a.warmup_chunks else a.second_steps,
        None if a.warmup_chunks else a.second_sigmas,a.mid_frames,a.compact_mid_clock,
        0. if a.warmup_chunks else a.noise_correlation,
        appearance_reference=appearance,appearance_strength=a.appearance_strength,sink_frames=4 if a.warmup_chunks else a.sink_frames,
        age_horizon=a.age_horizon,appearance_max_scale=a.appearance_max_scale,
        coarse_anchor=0. if a.warmup_chunks else a.coarse_anchor,rolling_history=a.rolling_history,
        joint_second_pass=False if a.warmup_chunks else a.joint_second_pass,
        second_appearance=a.second_appearance)
    initial=load_file(str(a.bootstrap/'latents.safetensors'))['video']
    initial_rgb=stream.initialize_latents(initial)
    to_rgb=lambda x:(x[0].permute(1,2,3,0).cpu().numpy()*255).round().astype(np.uint8)
    first=to_rgb(initial_rgb)
    if not np.array_equal(first,original[:17]):raise RuntimeError('latent bootstrap decode differs from native I2V output')
    np.save(a.output/'initial_rgb.npy',first,allow_pickle=False)
    arrays=[first];future_latents=[];records=[dict(chunk=0,mode='i2v',first_output_frame=0,delivered_frames=17,actual_dit_forwards=4,reused_bootstrap=True)]
    if a.resume_from:
        parent=json.loads((a.resume_from/'video.json').read_text())
        if parent['changed_sources'] or parent['frames'] < (a.resume_chunks+1)*17:
            raise ValueError('resume source must contain the complete saved prefix')
        parent_boot=Path(parent['bootstrap'])
        if sha256(parent_boot/'latents.safetensors')!=sha256(a.bootstrap/'latents.safetensors'):
            raise ValueError('resume and bootstrap latents differ')
        if [r['caption'] for r in parent['chunks'][1:a.resume_chunks+1]]!=captions[:a.resume_chunks]:
            raise ValueError('resume prefix captions differ')
    delivered=17
    for i in range(1,chunks+1):
        if a.warmup_chunks and i==a.warmup_chunks+1:
            previous=future_latents[-2] if len(future_latents)>1 else initial
            stream.reduce_sinks(a.sink_frames,torch.cat((previous[:,:,-2:],future_latents[-1]),2))
            stream.second_steps,stream.second_sigmas=a.second_steps,a.second_sigmas
            stream.coarse_anchor,stream.noise_correlation=a.coarse_anchor,a.noise_correlation
            stream.joint_second_pass=a.joint_second_pass
        folder=a.output/'chunks'/f'{i:03d}';folder.mkdir(parents=True)
        if i<=a.resume_chunks:
            saved=a.resume_from/'chunks'/f'{i:03d}'
            future=load_file(str(saved/'new_latents.safetensors'))['video']
            pixels=stream.replay_chunk(future)
            rgb=to_rgb(pixels)
            if not np.array_equal(rgb,np.load(saved/'rgb.npy')):
                raise RuntimeError('saved prefix decode is not exact')
            parent_record=parent['chunks'][i]
            original_forwards=parent_record.get('original_dit_forwards',parent_record['actual_dit_forwards'])
            record=dict(chunk=i,actual_dit_forwards=0,original_dit_forwards=original_forwards,reused_saved_latents=True,
                        source=str(saved),source_sha256=sha256(saved/'new_latents.safetensors'))
            ct=dict(replayed=True)
        else:
            if i in contexts:
                conditioning,ct=contexts[i]
            elif a.conditioner_service:
                conditioning,ct=encode_remote(a.conditioner_service,weights,captions[i-1],images,768,1344,frames=22,native_keyframes=True,text_only=True,stream_context=True)
            else:
                conditioning,ct=encode_stream_context(weights,captions[i-1],images,a.output/'cache/text',resident,offload_after=True)
            pixels,future,record=stream.step(conditioning,a.seed+i,folder/'first_pass.safetensors')
            rgb=to_rgb(pixels)
        np.save(folder/'rgb.npy',rgb,allow_pickle=False)
        save_file(dict(video=future.cpu().contiguous()),str(folder/'new_latents.safetensors'))
        future_latents.append(future.cpu());count=min(17,a.frames-delivered);arrays.append(rgb[:count])
        record.update(mode='v2v',first_output_frame=delivered,delivered_frames=count,conditioning=ct,seed=a.seed+i,caption=captions[i-1])
        record['text_images']=[dict(path=str(path),sha256=sha256(path)) for path in images]
        records.append(record);delivered+=count
        if a.current_text_boundary and i>=a.warmup_chunks:
            boundary=folder/'context_boundary.png';Image.fromarray(rgb[count-1]).save(boundary)
            if a.recent_text_context:
                start=folder/"context_start.png";Image.fromarray(rgb[0]).save(start)
                images=[start,boundary]
            else:
                images=[images[0],boundary]
        (a.output/'measurements.json').write_text(json.dumps(records,indent=2)+'\n')
        (a.output/'rope_coordinates.json').write_text(json.dumps(stream.window.records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    rgb=np.concatenate(arrays);assert rgb.shape==(a.frames,768,1344,3)
    np.save(a.output/'rgb.npy',rgb,allow_pickle=False)
    # Audit only: decoding the complete latent sequence must reproduce delivered RGB.
    parity=dict(exact=None,extra_decoder_audit=False)
    if a.audit_codec:
        complete=torch.cat((initial,*future_latents),2).to(stream.device)
        with stream._vae_phase(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            decoded=stream.vae.decode(stream._latents(complete,inverse=True).contiguous(),return_dict=False)[0]
            full=to_rgb(stream._pixels(decoded.float(),inverse=True))[:a.frames]
            del decoded,complete
        parity=dict(exact=np.array_equal(full,rgb),max_abs=int(np.abs(full.astype(np.int16)-rgb.astype(np.int16)).max()),
                    extra_decoder_audit=True,scope='Full native decoder versus incremental native overlap, identical generated latents; no DiT regeneration.')
        (a.output/'codec_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
        if not parity['exact']:raise RuntimeError('incremental/full native decoder mismatch')
    command=[get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1344x768','-r','24','-i','pipe:0','-an','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p',str(a.output/'video.mp4')]
    subprocess.run(command,input=rgb.tobytes(),check=True)
    changed=changed_sources(source)
    meta=dict(frames=a.frames,fps=24,resolution=[1344,768],chunks=records,bootstrap=str(a.bootstrap),weights=str(weights),
        source_code=source,changed_sources=changed,local_rope=a.local_rope,second_pass_steps=a.second_steps,
        second_sigmas=a.second_sigmas,
        mid_frames=a.mid_frames,age_horizon=a.age_horizon,appearance_max_scale=a.appearance_max_scale,
        coarse_anchor=a.coarse_anchor,rolling_history=a.rolling_history,joint_second_pass=a.joint_second_pass,
        second_appearance=a.second_appearance,
        resume_from=str(a.resume_from) if a.resume_from else None,resume_chunks=a.resume_chunks,warmup_chunks=a.warmup_chunks,
        compact_mid_clock=a.compact_mid_clock,
        noise_correlation=a.noise_correlation,
        appearance_strength=a.appearance_strength,
        sink_frames=a.sink_frames,current_text_boundary=a.current_text_boundary,recent_text_context=a.recent_text_context,
        appearance_reference_sha256=sha256(a.bootstrap/'source_static_video_latents.safetensors') if appearance is not None else None,
        codec_parity=parity,rgb_boundary_encodes=0,history_rgb_encodes=0,initial_rgb_reencode=False,
        native_overlap_blend=True,extra_frame_blending=False,frame_removal=False,chunk_editing=False,
        temporal_interpolation=False,quality_approved=False)
    (a.output/'video.json').write_text(json.dumps(meta,indent=2)+'\n');stream.close()
    if changed:raise RuntimeError('sources changed during sampling')
    print('Completed:',a.output/'video.mp4',flush=True)


if __name__=='__main__':main()
