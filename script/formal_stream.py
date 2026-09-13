"""Generate an I2V bootstrap, then continue with the formal head-tail V2V sampler."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--video',type=Path,required=True,help='only the first RGB frame initializes I2V')
    p.add_argument('--prompt-file',type=Path,required=True)
    captions_arg=p.add_mutually_exclusive_group(required=True)
    captions_arg.add_argument('--v2v-prompt-file',type=Path,
                              help='one continuation caption describing only the next 17 frames')
    captions_arg.add_argument('--v2v-caption-plan',type=Path,
                              help='JSON list with a separate caption for each 17-frame continuation')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-service',type=Path,required=True)
    p.add_argument('--conditioner-service',type=Path,required=True)
    p.add_argument('--bootstrap-load',type=Path,help='reuse a verified I2V bootstrap from this release')
    p.add_argument('--initial-layout',choices=('reference','native'),default='reference')
    p.add_argument('--frames',type=int,default=720)
    p.add_argument('--chunk-frames',type=int,choices=(16,17,33,34,50,51),default=17,
                   help='new RGB frames per continuation; caption intervals must match')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--reference-short-edge',type=int,default=768)
    p.add_argument('--without-refs',action='store_true')
    p.add_argument('--joint-audio',action=argparse.BooleanOptionalAction,default=True,
                   help='jointly denoise audio tokens; output video remains silent')
    p.add_argument('--fixed-text-context',action=argparse.BooleanOptionalAction,default=True,
                   help='keep scene images fixed and cache by caption; disable to refresh boundary images')
    p.add_argument('--reference-text',action='store_true',help='also present fixed refs to Qwen; default keeps the formal two-history-image input')
    p.add_argument('--context-instructions',action='store_true',help='add image-role prose; default preserves the training caption')
    p.add_argument('--decoder',choices=('context','continuous','native'),default='context')
    p.add_argument('--audit-decoder',action='store_true')
    p.add_argument('--history-feedback',choices=('latent','native-tail'),default='latent')
    p.add_argument('--v2v-autocast',action=argparse.BooleanOptionalAction,default=True,
                   help='use the formal training validation BF16 autocast')
    a=p.parse_args()
    if a.output.exists() or a.frames<22 or a.reference_short_edge<32 or a.reference_short_edge%32:
        p.error('use a new output folder, >=22 frames and a reference edge divisible by 32')
    if a.chunk_frames in (33,50) and a.decoder!='native':
        p.error('33/50-frame windows require --decoder native')
    chunk_count=(a.frames-22+a.chunk_frames-1)//a.chunk_frames
    if a.v2v_caption_plan:
        captions=json.loads(a.v2v_caption_plan.read_text())
        if (not isinstance(captions,list) or len(captions)<chunk_count
                or not all(isinstance(s,str) and s.strip() for s in captions)):
            p.error(f'caption plan must contain at least {chunk_count} nonempty strings')
        captions=[s.strip() for s in captions[:chunk_count]]
    else:
        caption=a.v2v_prompt_file.read_text().strip()
        if not caption:p.error('continuation caption must not be empty')
        captions=[caption]*chunk_count
    import numpy as np
    import torch
    from PIL import Image
    from imageio_ffmpeg import get_ffmpeg_exe
    from model.pipeline import Pipeline
    from model.formal_stream import FormalStream
    from model.formal_audio_stream import AudioStream
    from model.video_window import read_video_window
    from model.conditioning_service import encode_remote
    from model.weights import sha256
    from model.provenance import snapshot_sources,changed_sources

    a.output.mkdir(parents=True)
    source=snapshot_sources(a.output/'cache')
    weights=root/'weight/standard'
    prompt=a.prompt_file.read_text().strip()
    (a.output/'prompt.txt').write_text(prompt)
    rgb,_=read_video_window(a.video,1,1344,768,'head')
    first=a.output/'reference_first.png'
    Image.fromarray(rgb[0]).save(first)
    pixels=lambda x:torch.from_numpy(np.array(x,copy=True)).permute(3,0,1,2)[None].float()/255
    identity=dict(source_frame_sha256=sha256(first),prompt=prompt,seed=a.seed,weights=str(weights),
                  initial_layout=a.initial_layout,frames=22,height=768,width=1344)
    boot=a.bootstrap_load or a.output/'bootstrap'
    if a.bootstrap_load:
        deadline=time.monotonic()+600
        while not (boot/'metadata.json').exists():
            if time.monotonic()>deadline:
                raise TimeoutError('I2V bootstrap was not completed')
            time.sleep(2)
        bm=json.loads((boot/'metadata.json').read_text())
        if bm['identity']!=identity or sha256(boot/'rgb.npy')!=bm['rgb_sha256']:
            raise ValueError('bootstrap identity or pixels changed')
        initial=np.load(boot/'rgb.npy',allow_pickle=False)
        initial_timing=dict(bm['timing'],reused_bootstrap=True,bootstrap=str(boot))
    else:
        native=a.initial_layout=='native'
        conditioning,text_time=encode_remote(a.initial_service,weights,prompt,[first],768,1344,
            frames=22,native_keyframes=native,text_only=False)
    pipe=Pipeline(weights,reference=True,native_keyframes=a.initial_layout=='native',vae_offload=True,
                  attention_backend='native')
    from model.elementwise_fusion import enable_elementwise_fusion
    pipe.fusion=enable_elementwise_fusion(pipe.transformer)
    if not a.bootstrap_load:
        state,initial_timing=pipe.generate(conditioning,[first],768,1344,22,a.seed,4)
        initial=np.stack([np.asarray(f) for f in state['videos'][0]])
        if initial.shape!=(22,768,1344,3):
            raise RuntimeError('I2V bootstrap must deliver 22 RGB frames')
        initial_timing.update(conditioning=text_time,reused_bootstrap=False)
        boot.mkdir()
        np.save(boot/'rgb.npy',initial,allow_pickle=False)
        bm=dict(identity=identity,rgb_sha256=sha256(boot/'rgb.npy'),timing=initial_timing,source_code=source)
        tmp=boot/'metadata.tmp';tmp.write_text(json.dumps(bm,indent=2));tmp.rename(boot/'metadata.json')
    second=a.output/'reference_i2v_end.png'
    early=a.output/'i2v_first.png'
    Image.fromarray(initial[-1]).save(second)
    Image.fromarray(initial[0]).save(early)
    refs=[] if a.without_refs else [first]
    stream_type=AudioStream if a.joint_audio else FormalStream
    stream=stream_type(pipe,decoder=a.decoder,audit_decoder=a.audit_decoder,history_feedback=a.history_feedback,v2v_autocast=a.v2v_autocast,chunk_frames=a.chunk_frames)
    stream.initialize(pixels(initial))
    for path in refs:
        with Image.open(path) as image:
            scale=a.reference_short_edge/min(image.size)
            size=tuple(round(v*scale/32)*32 for v in image.size)
            stream.add_reference(pixels(np.array(image.resize(size,Image.Resampling.LANCZOS))[None]))
    instructions=''
    if a.context_instructions:
        instructions=('\n<Picture 1> and <Picture 2> provide initial scene appearance and lighting context. Continue motion from the end of the ongoing video.'
                      if a.fixed_text_context else
                      '\n<Picture 1> is earlier context and <Picture 2> is the exact current video boundary. Continue motion directly after <Picture 2> in the same uninterrupted shot.')
        if refs and a.reference_text:
            instructions+=' <Picture 3> is the fixed original-frame appearance reference for scene materials, colors and exposure. It does not replace the current motion boundary.'
    captions=[caption+instructions for caption in captions]
    (a.output/'v2v_captions.json').write_text(json.dumps(captions,indent=2))
    if captions:(a.output/'v2v_prompt.txt').write_text(captions[0])
    writers=[];records=[];delivered=22;started=time.perf_counter()
    command=[get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1344x768','-r','24','-i','pipe:0',
             '-an','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p']
    with (a.output/'encode.log').open('w') as errors:
        pool=ThreadPoolExecutor(max_workers=1) if a.fixed_text_context else None
        fixed_images=[first if refs else early,second,*(refs if a.reference_text else [])]
        futures={};seen_captions=set()
        def fixed_context(caption):
            # A changed caption must never reuse another caption's embedding.
            if caption not in futures:
                futures[caption]=pool.submit(encode_remote,a.conditioner_service,weights,caption,
                    fixed_images,768,1344,frames=22,native_keyframes=True,text_only=True,stream_context=True)
                if len(futures)>2:
                    oldest=next(iter(futures))
                    del futures[oldest]
                    seen_captions.discard(oldest)
            return futures[caption]
        try:
            for name in (('video.mp4','alternate_decoder.mp4') if a.audit_decoder else ('video.mp4',)):
                writers.append(subprocess.Popen(command+[str(a.output/name)],stdin=subprocess.PIPE,stderr=errors))
            for writer in writers:writer.stdin.write(initial.tobytes())
            records.append(dict(chunk=0,mode='i2v',first_output_frame=0,delivered_frames=22,**initial_timing))
            for index in (0,10,21):Image.fromarray(initial[index]).save(a.output/f'i2v_{index:03d}.png')
            print(json.dumps(records[-1]),flush=True)
            chunk=1
            while delivered<a.frames:
                boundary=a.output/f'boundary_input_{chunk:03d}.png'
                Image.fromarray((stream.boundary[0,:,0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(boundary)
                context_prompt=captions[chunk-1]
                wait_started=time.perf_counter()
                if a.fixed_text_context:
                    context_images=fixed_images
                    conditioning,text_time=fixed_context(context_prompt).result()
                    if context_prompt in seen_captions:
                        text_time=dict(reused_scene_caption=True,seconds=0.)
                    seen_captions.add(context_prompt)
                    if chunk<len(captions):fixed_context(captions[chunk])
                else:
                    context_images=[first if refs else early,boundary,*(refs if a.reference_text else [])]
                    conditioning,text_time=encode_remote(a.conditioner_service,weights,context_prompt,context_images,
                        768,1344,frames=22,native_keyframes=True,text_only=True,stream_context=True)
                conditioning_wait_ms=(time.perf_counter()-wait_started)*1000
                output,timing=stream.step(conditioning,a.seed+chunk,4)
                array=(output[0].permute(1,2,3,0).cpu().numpy()*255).round().astype(np.uint8)
                count=min(a.chunk_frames,a.frames-delivered)
                writers[0].stdin.write(array[:count].tobytes())
                if a.audit_decoder:
                    other=stream.decoder_outputs['continuous' if a.decoder=='native' else 'native']
                    other=(other[0].permute(1,2,3,0).cpu().numpy()*255).round().astype(np.uint8)
                    writers[1].stdin.write(other[:count].tobytes())
                for index in (0,1,a.chunk_frames-1):Image.fromarray(array[index]).save(a.output/f'chunk_{chunk:03d}_{index:02d}.jpg')
                records.append(dict(chunk=chunk,mode='v2v',first_output_frame=delivered,delivered_frames=count,
                                    conditioning=text_time,qwen_images=len(context_images),
                                    qwen_image_inputs=[dict(path=str(p),sha256=sha256(p)) for p in context_images],
                                    caption=context_prompt,caption_sha256=hashlib.sha256(context_prompt.encode()).hexdigest(),
                                    prediction_time_seconds=[delivered/24,(delivered+a.chunk_frames)/24],
                                    conditioning_wait_ms=conditioning_wait_ms,
                                    seed=a.seed+chunk,**timing))
                (a.output/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                print(json.dumps(records[-1]),flush=True)
                delivered+=count;chunk+=1
        finally:
            if pool:pool.shutdown(wait=True,cancel_futures=True)
            for writer in writers:writer.stdin.close()
            codes=[writer.wait() for writer in writers]
            stream.close()
    if any(codes):raise RuntimeError(f'video encoding failed: {codes}')
    changed=changed_sources(source)
    meta=dict(frames=delivered,fps=24,resolution=[1344,768],chunks=records,source_code=source,changed_sources=changed,
        weights=str(weights),transformer=pipe.transformer_record,bootstrap=str(boot),bootstrap_identity=identity,seconds=time.perf_counter()-started,
        attention_backend=pipe.attention_backend,decoder=a.decoder,reference_short_edge=a.reference_short_edge,
        reference_text=a.reference_text,context_instructions=a.context_instructions,
        history_feedback=a.history_feedback,
        joint_audio=a.joint_audio,audio_decoded=False,fixed_text_context=a.fixed_text_context,
        caption_mode='per_chunk' if a.v2v_caption_plan else 'static',
        continuation_interval_frames=a.chunk_frames,continuation_interval_seconds=a.chunk_frames/24,
        v2v_autocast=a.v2v_autocast,
        references=[dict(path=str(p),sha256=sha256(p)) for p in refs],postprocessing=False,
        formal_branch='head_tail1_5plus17',decoder_note='context repeats two rightmost latent slots for decoder context and crops a continuous clock; native reproduces the training preview decoder')
    (a.output/'video.json').write_text(json.dumps(meta,indent=2))
    if changed:raise RuntimeError('release source changed during sampling')
    print('completed:',a.output/'video.mp4',flush=True)


if __name__=='__main__':
    main()
