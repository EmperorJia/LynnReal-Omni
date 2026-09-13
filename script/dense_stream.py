"""Continue an input video using a rolling 22-frame prefix and four DiT forwards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(visual_context=False):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--source-window', choices=('head','tail'), default='head',
                   help='legacy ablations use head; the public launcher defaults to tail')
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=720)
    p.add_argument('--chunk-frames', type=int, default=102)
    p.add_argument('--feedback', choices=['latent','rgb'], default='latent')
    p.add_argument('--history-filter', type=float, default=0.0)
    p.add_argument('--save-prefix', action='store_true')
    p.add_argument('--retain-anchor', action='store_true')
    p.add_argument('--condition-time', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--width', type=int, default=1344)
    p.add_argument('--offload-blocks', type=int, default=0)
    args = p.parse_args()
    if args.output.exists() or args.frames < 1 or args.chunk_frames < 22:
        p.error('use a new output path, positive length and chunks of at least 22 frames')
    if min(args.height,args.width) < 32 or args.height % 32 or args.width % 32:
        p.error('dense continuation requires canvas dimensions divisible by 32')
    if args.feedback=='latent' and args.chunk_frames % 17:
        p.error('latent feedback requires chunk length divisible by 17')
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
    from model.pipeline import Pipeline, encode_conditioning
    from model.continuation import continue_video
    from model.provenance import snapshot_sources, changed_sources
    from model.weights import sha256
    from model.video_window import read_video_window
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / 'cache'
    sources = snapshot_sources(cache)
    ffmpeg = get_ffmpeg_exe()
    h, w, prefix = args.height, args.width, 22
    rgb, source_window = read_video_window(args.video, prefix, w, h, args.source_window)
    pixels = torch.from_numpy(rgb).permute(3,0,1,2)[None].float()/255
    prompt = args.prompt_file.read_text().strip()
    paths = []
    if visual_context:
        prefix_path = args.output.parent/'source_prefix.mp4'
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264',
            '-crf', '16', '-pix_fmt', 'yuv420p', str(prefix_path)], input=rgb.tobytes(), check=True)
        paths = [prefix_path]
    del rgb
    condition, conditioning_time = encode_conditioning(root/'weight/standard', prompt, paths, h, w,
                                                       prefix+args.chunk_frames, cache)
    pipe = Pipeline(root/'weight/standard', attention_backend='native')
    command = [ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
        '-r', '24', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(args.output)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    history = {} if args.feedback=='latent' else None
    records=[];delivered=0;started=time.perf_counter()
    try:
        while delivered < args.frames:
            # Each chunk is decoded jointly with its actual RGB prefix.
            rgb, _, timing = continue_video(pipe, condition, pixels, args.chunk_frames,
                                            args.seed+len(records), 4, args.offload_blocks, True, history, args.condition_time, args.retain_anchor, args.history_filter)
            if args.save_prefix and history is not None:
                torch.save(history['prefix'].cpu(), args.output.parent/f'prefix_{len(records):03d}.pt')
            rgb = (rgb*255).round().byte().cpu()
            count = min(args.chunk_frames, args.frames-delivered)
            writer.stdin.write(rgb[0,:,:count].permute(1,2,3,0).contiguous().numpy().tobytes())
            pixels = rgb[:,:,-prefix:].float()/255
            record = dict(chunk=len(records), first_frame=delivered, delivered_frames=count, **timing)
            records.append(record);delivered += count
            with (args.output.parent/'measurements.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
    finally:
        writer.stdin.close();code=writer.wait()
    if code:raise RuntimeError('video writer failed')
    args.output.with_suffix('.json').write_text(json.dumps(dict(mode='dense-prefix-stream',
        weights=str(root/'weight/standard'), transformer=pipe.transformer_record, prompt=prompt, source_video=str(args.video.resolve()),
        source_sha256=sha256(args.video), frames=delivered, fps=24, native_canvas=[w,h],
        source_window=source_window,
        steps_per_chunk=4, chunk_frames=args.chunk_frames, retained_rgb_frames=prefix,
        text_context='source_video_prefix' if visual_context else 'text_only',
        history_feedback=args.feedback, retained_latent_frames=7 if history is not None else 0,
        audio=False, postprocessing=False, conditioning=conditioning_time, chunks=records,
        seconds=time.perf_counter()-started, source_code=sources,
        source_changed_during_run=changed_sources(sources), output_sha256=sha256(args.output)),indent=2))

if __name__ == '__main__':main()
