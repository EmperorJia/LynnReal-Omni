"""Native first-keyframe continuation conditioned on an actual input video."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--source-window', choices=('head','tail'), default='head',
                   help='legacy ablations use head; the public launcher defaults to tail')
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=720)
    p.add_argument('--chunk-frames', type=int, default=123)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--seed-mode', choices=('increment','fixed'), default='increment')
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--width', type=int, default=1344)
    p.add_argument('--offload-blocks', type=int, help='auto: keep DiT resident when free memory and the tested canvas allow it')
    p.add_argument('--conditioner-service', type=Path)
    p.add_argument('--rolling-video-context', action='store_true')
    p.add_argument('--motion-sheet', action='store_true', help='encode current frame and labelled video history with the native image conditioner')
    p.add_argument('--appearance-reference', action='store_true', help='encode the complete original first frame beside the current boundary on every chunk')
    p.add_argument('--first-frame-reference-only', action='store_true', help='keep only the original first image as reference; carry the current boundary only in the target head')
    p.add_argument('--framepack-history', action='store_true', help='add bounded latent history beside the fixed first-image reference')
    p.add_argument('--sink-frames', type=int, choices=(1, 2), default=2)
    p.add_argument('--history-capacity', type=int, default=64)
    p.add_argument('--mid-stride', type=int, default=8)
    p.add_argument('--ref2va', action='store_true', help='diagnostic: use the full reference interface for both text and DiT conditioning')
    p.add_argument('--reference-image-short-edge', type=int, help='diagnostic DiT reference-image resolution; Ref2VA defaults to 2048')
    p.add_argument('--reference-text-short-edge', type=int, help='diagnostic Qwen reference-image resolution; Ref2VA defaults to 2048')
    p.add_argument('--history-short-edge', type=int, help='also condition both encoders on the latest 22 RGB frames at this short-edge resolution')
    p.add_argument('--chronological-references', action='store_true', help='place initial appearance, video history and current boundary in temporal order')
    p.add_argument('--appearance-latent', action='store_true', help='also retain the original image as a cached historical DiT latent')
    p.add_argument('--fixed-target-head', action='store_true', help='fix the boundary latent inside the target at every denoiser evaluation')
    p.add_argument('--head-only-boundary', action='store_true', help='remove the duplicate current boundary from reference rows')
    p.add_argument('--reuse-vae-phase', action='store_true', help='encode the next boundary before leaving the current decoder phase')
    p.add_argument('--anchor-first-frame', action='store_true', help='retain the original appearance in the motion sheet')
    p.add_argument('--elementwise-fusion', action='store_true')
    p.add_argument('--tone-anchor', action='store_true', help='experimental global RGB tone correction before clipping and next-boundary encoding')
    p.add_argument('--tone-region', type=float, nargs=4, metavar=('X0','Y0','X1','Y1'),
                   help='experimental stationary-region color fit in normalized coordinates; requires tone-anchor and a fixed camera')
    p.add_argument('--latent-boundary', action='store_true', help='experimental phase-aligned latent feedback; chunk-frames must be a multiple of 17')
    args = p.parse_args()
    if args.framepack_history and (not args.first_frame_reference_only or args.latent_boundary
                                  or args.history_capacity < 7 or args.mid_stride < 4):
        p.error('FramePack history requires first-frame-only Ref2VA, RGB target boundaries, capacity >= 7 and mid stride >= 4')
    if args.first_frame_reference_only and (not args.ref2va or not args.fixed_target_head
            or not args.appearance_reference or args.history_short_edge is not None or args.chronological_references):
        p.error('first-frame-reference-only requires fixed-head Ref2VA and excludes generated history references')
    if args.chronological_references and args.history_short_edge is None:
        p.error('chronological-references requires history-short-edge')
    if args.history_short_edge is not None and (not args.ref2va or not args.appearance_reference
            or args.history_short_edge < 32 or args.history_short_edge % 32):
        p.error('history-short-edge requires ref2va, appearance-reference and a positive multiple of 32')
    if args.reference_image_short_edge is not None and (not args.ref2va or args.reference_image_short_edge < 32 or args.reference_image_short_edge % 32):
        p.error('reference-image-short-edge requires ref2va and a positive multiple of 32')
    if args.reference_text_short_edge is not None and (not args.ref2va or args.reference_text_short_edge < 32 or args.reference_text_short_edge % 32):
        p.error('reference-text-short-edge requires ref2va and a positive multiple of 32')
    if args.ref2va and (args.appearance_latent or (args.reuse_vae_phase and not args.first_frame_reference_only)
                       or args.motion_sheet or args.rolling_video_context or (not args.conditioner_service and not args.first_frame_reference_only)):
        p.error('ref2va requires a conditioner service unless its reference is the immutable first frame')
    if args.tone_region and not args.tone_anchor:
        p.error('tone-region requires tone-anchor')
    if args.tone_anchor and not (args.reuse_vae_phase and args.appearance_reference):
        p.error('tone-anchor requires reuse-vae-phase and appearance-reference')
    if args.latent_boundary and (not args.fixed_target_head or not args.reuse_vae_phase or args.tone_anchor):
        p.error('latent-boundary requires fixed-target-head and reuse-vae-phase, without tone correction')
    if args.head_only_boundary and not args.fixed_target_head:
        p.error('head-only-boundary requires fixed-target-head')
    if args.fixed_target_head and not (args.appearance_latent or args.ref2va):
        p.error('fixed-target-head requires appearance-latent')
    if args.ref2va and args.fixed_target_head and not args.first_frame_reference_only and args.reference_image_short_edge != min(args.width,args.height):
        p.error('the fixed Ref2VA head requires reference-image-short-edge to match the output canvas')
    if args.appearance_latent and not args.appearance_reference:
        p.error('appearance-latent requires appearance-reference')
    if args.appearance_reference and ((not args.conditioner_service and not args.first_frame_reference_only) or args.motion_sheet or args.rolling_video_context):
        p.error('appearance-reference requires a conditioner service and replaces other visual history modes')
    if args.motion_sheet and (not args.conditioner_service or args.rolling_video_context):
        p.error('motion-sheet requires a conditioner service and replaces rolling-video-context')
    if args.anchor_first_frame and not args.motion_sheet:
        p.error('first-frame anchoring requires motion-sheet')
    if args.rolling_video_context and not args.conditioner_service:
        p.error('rolling video context requires a conditioner service')
    if args.output.exists() or args.frames < 1 or args.chunk_frames < 22:
        p.error('use a new output path, positive length and chunks of at least 22 frames')
    if min(args.height,args.width) < 32 or args.height % 32 or args.width % 32:
        p.error('continuation requires canvas dimensions divisible by 32')
    if args.latent_boundary and args.chunk_frames % 17:
        p.error('latent boundary chunks must contain 17*n new frames')
    if not args.latent_boundary and (args.chunk_frames-4) % 17:
        p.error('native chunks contain 17*n+4 new frames after removing their first frame')
    import numpy as np
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
    from model.pipeline import Pipeline, encode_conditioning
    from model.provenance import snapshot_sources, changed_sources
    from model.weights import sha256
    from model.video_window import read_video_window
    from diffusers.modular_pipelines.minimax_h3.packing import align_num_frames
    native_frames = align_num_frames(1+args.chunk_frames)
    if args.offload_blocks is None:
        free, _ = torch.cuda.mem_get_info()
        args.offload_blocks = 0 if free >= 78 * 1024**3 and args.width * args.height <= 1344 * 768 and args.chunk_frames <= 123 else 4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / 'cache'
    sources = snapshot_sources(cache)
    ffmpeg = get_ffmpeg_exe()
    h, w, prefix = args.height, args.width, 22
    rgb, source_window = read_video_window(args.video, prefix, w, h, args.source_window)
    appearance = None
    if args.anchor_first_frame or args.appearance_reference:
        appearance = read_video_window(args.video, 1, w, h, 'head')[0][0]
    if args.appearance_reference:
        from PIL import Image
        appearance_path = args.output.parent / 'appearance.png'
        Image.fromarray(appearance).save(appearance_path)
    pixels = torch.from_numpy(rgb).permute(3,0,1,2)[None].float()/255
    prompt = args.prompt_file.read_text().strip()
    paths = []
    if args.first_frame_reference_only:
        paths = [appearance_path]
    elif not args.conditioner_service:
        prefix_path = args.output.parent/'source_prefix.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264',
            '-crf', '16', '-pix_fmt', 'yuv420p', str(prefix_path)], input=rgb.tobytes(), check=True)
        paths = [prefix_path]
    del rgb
    if args.conditioner_service:
        condition, conditioning_time = None, {'mode':'per_chunk_service'}
    else:
        condition, conditioning_time = encode_conditioning(root/'weight/standard', prompt, paths, h, w,
                                                           native_frames, cache,
                                                           native_keyframes=False if args.first_frame_reference_only else None,
                                                           reference_image_short_edge=args.reference_text_short_edge)
    pipe = Pipeline(root/'weight/standard', reference=True, attention_backend='native',
                    native_keyframes=not args.ref2va, vae_offload=True, dit_offload_blocks=args.offload_blocks,
                    native_appearance=args.appearance_latent,
                    fixed_native_head=args.fixed_target_head,
                    head_only_boundary=args.head_only_boundary,
                    reference_image_short_edge=args.reference_image_short_edge,
                    reference_video_short_edge=args.history_short_edge,
                    fixed_head_reference_index=-1 if args.chronological_references else 0,
                    independent_target_head=args.first_frame_reference_only,
                    boundary_reuse_frames=native_frames if args.reuse_vae_phase else None)
    if args.elementwise_fusion:
        from model.elementwise_fusion import enable_elementwise_fusion
        pipe.fusion = enable_elementwise_fusion(pipe.transformer)
    hybrid = None
    if args.framepack_history:
        from model.hybrid_history import HybridHistory
        hybrid = HybridHistory(args.history_capacity, args.sink_frames, args.mid_stride)
        hybrid.initialize(pipe.pipe, pixels)
        pipe.fixed_head_denoise.history = hybrid
    if args.tone_anchor:
        from model.tone_anchor import ToneAnchor
        pipe.boundary_cache.tone_anchor = ToneAnchor(appearance, region=args.tone_region)
    command = [ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '24', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(args.output)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    records=[];delivered=0;started=time.perf_counter()
    try:
        while delivered < args.frames:
            iteration_started = time.perf_counter()
            remaining = args.frames-delivered
            # Absorb a short final tail into this chunk instead of adding four forwards.
            future = remaining if args.latent_boundary and remaining <= args.chunk_frames+17 else args.chunk_frames
            model_frames = align_num_frames(1+future)
            from PIL import Image
            boundary = args.output.parent/f'boundary_{len(records):03d}.png'
            Image.fromarray((pixels[0,:,-1].permute(1,2,0)*255).round().byte().numpy()).save(boundary)
            context_timing = None
            if args.first_frame_reference_only:
                media = [appearance_path]
                context_timing = dict(conditioning_time, reused_fixed_conditioning=bool(records))
            if args.conditioner_service:
                from model.conditioning_service import encode_remote
                media = [boundary]
                if args.appearance_reference:
                    media.append(appearance_path)
                if args.motion_sheet:
                    from model.video_context import motion_sheet
                    history = args.output.parent/f'history_{len(records):03d}.png'
                    frames = (pixels[0].permute(1,2,3,0)*255).round().byte().numpy()
                    motion_sheet(frames, appearance).save(history)
                    media.append(history)
                if args.rolling_video_context:
                    history = args.output.parent/f'history_{len(records):03d}.mp4'
                    data = (pixels[0].permute(1,2,3,0)*255).round().byte().numpy().tobytes()
                    subprocess.run([ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24',
                        '-s',f'{w}x{h}','-r','24','-i','-','-an','-c:v','libx264','-crf','16',
                        '-pix_fmt','yuv420p',str(history)],input=data,check=True)
                    media.append(history)
                if args.history_short_edge is not None:
                    history = args.output.parent/f'history_{len(records):03d}.mkv'
                    data = (pixels[0].permute(1,2,3,0)*255).round().byte().numpy().tobytes()
                    subprocess.run([ffmpeg,'-v','error','-f','rawvideo','-pix_fmt','rgb24',
                        '-s',f'{w}x{h}','-r','24','-i','-','-an','-c:v','ffv1',
                        '-level','3','-pix_fmt','bgr0',str(history)],input=data,check=True)
                    media.append(history)
                    if args.chronological_references:
                        media = [appearance_path, history, boundary]
                if args.first_frame_reference_only:
                    media = [appearance_path]
                condition, context_timing = encode_remote(args.conditioner_service,
                    root/'weight/standard',prompt,media,h,w,timeout=900,frames=model_frames,
                    native_keyframes=not (args.rolling_video_context or args.ref2va),text_only=False,
                    reference_image_short_edge=args.reference_text_short_edge,
                    reference_video_short_edge=args.history_short_edge)
            chunk_seed = args.seed + (len(records) if args.seed_mode == 'increment' else 0)
            if hybrid is not None:
                hybrid.seed = chunk_seed
            use_initial = args.appearance_latent or (args.ref2va and args.appearance_reference)
            dit_media = [boundary, appearance_path] if use_initial else [boundary]
            if args.history_short_edge is not None:
                dit_media.append(history)
            if args.chronological_references:
                dit_media = [appearance_path, history, boundary]
            if args.first_frame_reference_only:
                dit_media = [appearance_path]
            if pipe.boundary_cache is not None:
                pipe.boundary_cache.prefetch = delivered + future < args.frames
                pipe.boundary_cache.output_frames = model_frames
                if args.latent_boundary:
                    pipe.boundary_cache.boundary_frame = future
                    pipe.boundary_cache.latent_index = future // 17 * 5
            state, timing = pipe.generate(condition, dit_media, h, w, model_frames,
                                          chunk_seed, 4, boundary_image=Image.open(boundary).convert('RGB') if args.first_frame_reference_only else None)
            timing['condition_rows'] = int(state['num_condition_video_rows'])
            if args.tone_anchor:
                timing['tone_correction'] = pipe.boundary_cache.tone_anchor.audit
            if pipe.fixed_head_denoise is not None:
                timing.update(pipe.fixed_head_denoise.audit)
            timing['dit_condition_media' if args.history_short_edge is not None else 'dit_condition_images'] = [{"path": str(p), "sha256": sha256(p)} for p in dit_media]
            if pipe.boundary_cache is not None:
                timing['boundary_cache_hits'] = pipe.boundary_cache.hits
                timing['prefetched_boundaries'] = pipe.boundary_cache.encodes
                timing['latent_boundary_feedbacks'] = pipe.boundary_cache.latent_feedbacks
            if pipe.appearance_encoder is not None:
                timing['initial_appearance_encodes'] = pipe.appearance_encoder.initial_encodes
                if timing['initial_appearance_encodes'] != 1:
                    raise RuntimeError('the original appearance must remain fixed and be encoded once')
            decoded_head = args.output.parent/f'decoded_head_{len(records):03d}.png'
            state['videos'][0][0].save(decoded_head)
            timing['decoded_head'] = str(decoded_head)
            array = np.stack([np.asarray(x) for x in state['videos'][0][1:1+future]])
            if len(array) != future:
                raise RuntimeError('native sampler returned too few frames')
            timing['boundary_input'] = dict(path=str(boundary), sha256=sha256(boundary), role='target_head_only' if args.first_frame_reference_only else 'reference_and_head')
            timing.update(layout='first_reference_framepack' if hybrid is not None else 'full_reference_chain' if args.ref2va else 'native_first_keyframe', head_frames=1, future_frames=future,
                          model_frames=model_frames, latent_boundary=args.latent_boundary)
            timing['updated_conditioning'] = context_timing
            if args.motion_sheet or args.appearance_reference:
                timing['conditioning_media' if args.history_short_edge is not None else 'conditioning_images'] = [{"path": str(p), "sha256": sha256(p)} for p in media]
            count = min(future, args.frames-delivered)
            writer.stdin.write(array[:count].tobytes())
            pixels = torch.from_numpy(array[-prefix:].copy()).permute(3,0,1,2)[None].float()/255
            timing['iteration_ms'] = (time.perf_counter()-iteration_started)*1000
            record = dict(chunk=len(records), seed=chunk_seed, first_frame=delivered, delivered_frames=count, **timing)
            records.append(record);delivered += count
            with (args.output.parent/'measurements.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            print(f"DiT (4 steps): {timing['dit_ms']/1000:.3f}s | video decoder: {timing['video_decoder_ms']/1000:.3f}s | sum: {(timing['dit_ms']+timing['video_decoder_ms'])/1000:.3f}s", flush=True)
    finally:
        writer.stdin.close();code=writer.wait()
    if code:raise RuntimeError('video writer failed')
    mode = ('full-reference-chain-diagnostic' if args.ref2va else
            'native-latent-appearance-stream' if args.appearance_latent else
            'native-appearance-reference-stream' if args.appearance_reference else
            'native-motion-sheet-stream' if args.motion_sheet else
            'keyframe-chain-diagnostic' if args.conditioner_service and not args.rolling_video_context
            else 'native-keyframe-video-stream')
    args.output.with_suffix('.json').write_text(json.dumps(dict(mode=mode,
        weights=str(root/'weight/standard'), transformer=pipe.transformer_record, prompt=prompt, source_video=str(args.video.resolve()),
        source_sha256=sha256(args.video), frames=delivered, fps=24, native_canvas=[w,h],
        source_window=source_window,
        seed=args.seed, seed_mode=args.seed_mode,
        steps_per_chunk=4, chunk_frames=args.chunk_frames, retained_rgb_frames=prefix,
        first_frame_reference_only=args.first_frame_reference_only,
        framepack_history=(dict(sink_frames=args.sink_frames, sink_spatial='dense', mid_stride=args.mid_stride,
                                recent_stride=2, capacity=args.history_capacity) if hybrid is not None else None),
        text_context=('original_first_frame_only' if args.first_frame_reference_only else
                      'video_history_current_keyframe_and_initial_appearance' if args.history_short_edge is not None else
                      'current_keyframe_and_initial_appearance' if args.appearance_reference else
                      'current_keyframe_and_video_motion_sheet' if args.motion_sheet else
                      'rolling_video_and_current_keyframe' if args.rolling_video_context else
                      'current_keyframe_only' if args.conditioner_service else
                      'source_video_prefix'),
        history_feedback='generated_latents_with_rgb_target_head' if hybrid is not None else 'phase_aligned_generated_latent' if args.latent_boundary else 'rgb',
        retained_latent_frames=hybrid.history.shape[2] if hybrid is not None else int(args.latent_boundary),
        fixed_target_head=args.fixed_target_head,
        reference_layout=args.ref2va,
        dit_reference_image_short_edge=(args.reference_image_short_edge or 2048) if args.ref2va else None,
        text_reference_image_short_edge=(args.reference_text_short_edge or 2048) if args.ref2va else None,
        video_history_frames=prefix if args.history_short_edge is not None else 0,
        video_history_short_edge=args.history_short_edge,
        reference_order=('original_first_frame_only' if args.first_frame_reference_only else
                         'appearance_history_boundary' if args.chronological_references else
                         'boundary_appearance_history' if args.history_short_edge is not None else None),
        head_only_boundary=args.head_only_boundary,
        appearance_anchor=args.anchor_first_frame or args.appearance_reference,
        appearance_reference_scope=('full_reference_text_and_vae' if args.ref2va and args.appearance_reference else
                                    'native_image_conditioner_and_historical_vae_latent' if args.appearance_latent
                                    else 'native_image_conditioner_only' if appearance is not None else None),
        dit_offload_blocks=args.offload_blocks, reuse_vae_phase=args.reuse_vae_phase,
        appearance_noise_stream='independent_reference_preserving_target_rng' if args.appearance_latent else None,
        fusion=pipe.fusion,
        audio=False, postprocessing=args.tone_anchor, conditioning=conditioning_time, chunks=records,
        seconds=time.perf_counter()-started, source_code=sources,
        source_changed_during_run=changed_sources(sources), output_sha256=sha256(args.output)),indent=2))

if __name__ == '__main__':main()
