"""Four-step BF16 video continuation with an explicit source video and full log."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from run import RELEASE, frame_count, resolution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, default=RELEASE / 'test/assets/stream.mp4')
    p.add_argument('--source-window', choices=('head','tail'), default='tail',
                   help='continue from the end of the input; head reproduces older ablations')
    p.add_argument('--prompt')
    p.add_argument('--frame', type=frame_count, default='30s')
    p.add_argument('--chunk-frames', type=int, choices=(16,17), default=17)
    p.add_argument('--decode-layout', choices=('native','continuous'),
                   help='compact defaults to continuous joint decoding; native reproduces the legacy wrapper')
    p.add_argument('--elementwise-fusion', action='store_true')
    p.add_argument('--anchor-first-frame', action='store_true')
    p.add_argument('--appearance-reference', action='store_true')
    p.add_argument('--appearance-latent', action='store_true')
    p.add_argument('--fixed-target-head', action='store_true')
    p.add_argument('--head-only-boundary', action='store_true')
    p.add_argument('--resolution', type=resolution, default='768p')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--seed-mode', choices=('increment','fixed'), default='increment')
    p.add_argument('--name')
    p.add_argument('--condition-time', type=float, default=0.999)
    p.add_argument('--retain-anchor', action='store_true')
    p.add_argument('--feedback', choices=['latent','rgb'], default='latent')
    p.add_argument('--layout', choices=['reference','native','dense','compact'], default='native')
    p.add_argument('--reference-short-edge', type=int, default=2048, help='original first-frame DiT reference size in the reference layout')
    p.add_argument('--framepack-history', action='store_true', help='reference layout: add two dense sinks and bounded FramePack history (recent stride 2, middle stride 8)')
    p.add_argument('--native-chunk-frames', type=int, default=123, help='native/reference layout: new RGB frames per chunk, 17*n+4 (e.g. 55 or 123)')
    p.add_argument('--conditioner-service', type=Path)
    p.add_argument('--offload-blocks', type=int, help='native layout auto-selects residency when omitted')
    p.add_argument('--sink-frames', type=int, choices=(1, 2), default=1)
    p.add_argument('--mid-stride', type=int, choices=(4, 8, 16), default=4)
    p.add_argument('--recent-stride', type=int, choices=(2, 4), default=2)
    p.add_argument('--sink-spatial', choices=('shared', 'dense'), default='shared')
    p.add_argument('--history-pooling', choices=('point', 'mean'), default='point')
    p.add_argument('--reuse-vae-phase', action='store_true')
    p.add_argument('--boundary-rgb8', action='store_true')
    p.add_argument('--boundary-posterior', choices=('sample','mode'), default='sample')
    p.add_argument('--text-context', choices=('sink-boundary','recent-boundaries','boundary'), default='sink-boundary')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    sys.path.insert(0, str(RELEASE))
    from model.weights import standard_transformer
    transformer = standard_transformer(RELEASE / 'weight/standard')
    if args.framepack_history and args.layout != 'reference':
        p.error('--framepack-history requires --layout reference')
    if args.native_chunk_frames < 22 or (args.native_chunk_frames-4) % 17:
        p.error('--native-chunk-frames must be 17*n+4 and at least 22')
    if args.layout not in {'native', 'reference'} and args.native_chunk_frames != 123:
        p.error('--native-chunk-frames requires native/reference layout')
    if args.decode_layout is None:
        args.decode_layout = 'continuous' if args.layout == 'compact' else 'native'
    if args.layout != 'compact' and (args.sink_frames != 1 or args.mid_stride != 4 or args.boundary_rgb8
            or args.recent_stride != 2 or args.sink_spatial != 'shared' or args.history_pooling != 'point'
            or args.boundary_posterior != 'sample' or args.text_context != 'sink-boundary'
            or args.chunk_frames != 17 or args.decode_layout != 'native' or args.anchor_first_frame):
        p.error('sink, mid-stride and boundary ablations require --layout compact')
    if args.layout == 'dense' and (args.reuse_vae_phase or args.elementwise_fusion):
        p.error('VAE phase reuse and elementwise fusion require native or compact layout')
    if args.layout == 'reference' and (args.reference_short_edge < 32 or args.reference_short_edge % 32
            or args.appearance_reference or args.appearance_latent or args.fixed_target_head or args.head_only_boundary
            or args.retain_anchor or args.condition_time != 0.999 or args.feedback != 'latent'):
        p.error('reference layout uses an immutable first frame and an independent target head; native overrides are unsupported')
    if args.appearance_reference and (args.layout != 'native' or not args.conditioner_service):
        p.error('appearance-reference requires the native layout and a conditioner service')
    if args.appearance_latent and not args.appearance_reference:
        p.error('appearance-latent requires appearance-reference')
    if args.fixed_target_head and not args.appearance_latent:
        p.error('fixed-target-head requires appearance-latent')
    if args.head_only_boundary and not args.fixed_target_head:
        p.error('head-only-boundary requires fixed-target-head')
    if args.reuse_vae_phase and ((not args.conditioner_service and args.layout != 'reference') or args.feedback != 'latent'):
        p.error('--reuse-vae-phase requires a separate conditioner and latent history')
    if args.chunk_frames == 16 and args.feedback != 'latent':
        p.error('16-frame chunks require latent history')
    if args.anchor_first_frame and (args.text_context != 'sink-boundary' or args.feedback != 'latent'):
        p.error('first-frame anchoring requires sink-boundary text context and latent history')
    if args.layout == 'compact' and (args.retain_anchor or args.condition_time != 0.999):
        p.error('retain-anchor and condition-time require --layout dense')
    if args.layout not in {'native', 'reference'} and args.seed_mode != 'increment':
        p.error('--seed-mode fixed requires the native layout')
    if not args.video.is_file():p.error('--video must name an existing input video')
    name = args.name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if Path(name).name != name or name in {'.', '..'}:p.error('invalid output name')
    output = RELEASE / 'output/standard/bf16/stream' / name
    if output.exists():p.error('output already exists')
    default_prompt = ('test/stream_native.txt' if args.layout == 'native' else 'test/stream.txt')
    if args.layout == 'native' and args.conditioner_service:
        default_prompt = 'test/stream_keyframe.txt'
    if args.appearance_reference:
        default_prompt = 'test/stream_appearance.txt'
    if args.layout == 'reference':
        default_prompt = 'test/stream/urban_first_frame_appearance.txt'
    prompt = args.prompt or (RELEASE / default_prompt).read_text()
    width, height = args.resolution
    command = [sys.executable, '-u', str(RELEASE / 'script/stream.py'),
               '--weights', str(RELEASE / 'weight/standard'), '--video', str(args.video.resolve()),
               '--prompt-file', str(output / 'prompt.txt'), '--output', str(output / 'video.mp4'),
               '--cache', str(output / 'cache'), '--frames', str(args.frame), '--steps', '4',
               '--width', str(width), '--height', str(height), '--seed', str(args.seed),
               '--attention-backend', 'native', '--vae-offload']
    if args.layout in {'native', 'reference'}:
        if args.retain_anchor or args.condition_time != 0.999 or args.feedback != 'latent':
            p.error('history ablations require --layout dense')
        command = [sys.executable, '-u', str(RELEASE/'script/native_stream.py'),
                   '--video', str(args.video.resolve()), '--prompt-file', str(output/'prompt.txt'),
                   '--output', str(output/'video.mp4'), '--frames', str(args.frame),
                   '--height', str(height), '--width', str(width), '--seed', str(args.seed),
                   '--seed-mode', args.seed_mode, '--chunk-frames', str(args.native_chunk_frames)]
        if args.offload_blocks is not None:
            command += ['--offload-blocks', str(args.offload_blocks)]
        if args.conditioner_service:
            command += ['--conditioner-service', str(args.conditioner_service)]
            if args.layout == 'native' and not args.appearance_reference:command.append('--rolling-video-context')
        if args.layout == 'reference':
            command += ['--ref2va', '--first-frame-reference-only', '--appearance-reference',
                        '--fixed-target-head', '--reference-image-short-edge', str(args.reference_short_edge),
                        '--reference-text-short-edge', '768']
            if args.framepack_history:command.append('--framepack-history')
        if args.appearance_reference:command.append('--appearance-reference')
        if args.appearance_latent:command.append('--appearance-latent')
        if args.fixed_target_head:command.append('--fixed-target-head')
        if args.head_only_boundary:command.append('--head-only-boundary')
        if args.reuse_vae_phase:command.append('--reuse-vae-phase')
        if args.elementwise_fusion:command.append('--elementwise-fusion')
    elif args.layout == 'dense':
        if args.conditioner_service:p.error('the dense layout does not use a separate conditioner')
        command = [sys.executable, '-u', str(RELEASE/'script/dense_stream.py'),
                   '--video', str(args.video.resolve()), '--prompt-file', str(output/'prompt.txt'),
                   '--output', str(output/'video.mp4'), '--frames', str(args.frame),
                   '--height', str(height), '--width', str(width), '--seed', str(args.seed),
                   '--feedback', args.feedback, '--condition-time', str(args.condition_time)]
        if args.retain_anchor:command.append('--retain-anchor')
    else:
        command += ['--sink-frames', str(args.sink_frames), '--mid-stride', str(args.mid_stride),
                    '--recent-stride', str(args.recent_stride), '--sink-spatial', args.sink_spatial,
                    '--history-pooling', args.history_pooling,
                    '--boundary-posterior', args.boundary_posterior,
                    '--text-context', args.text_context,
                    '--chunk-frames', str(args.chunk_frames), '--decode-layout', args.decode_layout,
                    '--history-feedback', 'native-rgb' if args.feedback == 'rgb' else 'latent']
        if args.reuse_vae_phase:command.append('--reuse-vae-phase')
        if args.elementwise_fusion:command.append('--elementwise-fusion')
        if args.anchor_first_frame:command.append('--anchor-first-frame')
        if args.boundary_rgb8:command.append('--boundary-rgb8')
        if args.conditioner_service:command += ['--conditioner-service', str(args.conditioner_service)]
    command += ['--source-window', args.source_window]
    record = dict(command=command, transformer=transformer, prompt=prompt, frames=args.frame, fps=24, source_window=args.source_window,
                  framepack_history=args.framepack_history,
                  seed=args.seed, seed_mode=args.seed_mode if args.layout in {'native','reference'} else None,
                  precision='bf16', steps_per_chunk=4, audio=False, layout=args.layout,
                  feedback='framepack_latents_and_rgb_target_boundary' if args.framepack_history else
                           'rgb_target_boundary_only' if args.layout == 'reference' else 'rgb' if args.layout == 'native' else args.feedback)
    if args.dry_run:print(json.dumps(record, indent=2));return
    print(f'{args.layout} continuation: 4-step DiT per chunk.', flush=True)
    output.mkdir(parents=True)
    (output / 'prompt.txt').write_text(prompt.strip() + '\n')
    (output / 'request.json').write_text(json.dumps(record, indent=2))
    with (output / 'sample.log').open('w', buffering=1) as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:print(line, end='', flush=True);log.write(line)
        raise SystemExit(proc.wait())

if __name__ == '__main__':main()
