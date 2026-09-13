"""Four-step, 17-frame continuation with two fixed appearance references."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--weights', type=Path, default=root/'weight/standard')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditioner-service', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=720)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=1344)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--reference-short-edge', type=int, default=2048)
    parser.add_argument('--raw-only', action='store_true', help='disable the training sampler exposure correction')
    args = parser.parse_args()
    if args.frames < 17 or args.height % 32 or args.width % 32:
        parser.error('at least 17 frames and dimensions divisible by 32 are required')
    if args.reference_short_edge < 32 or args.reference_short_edge % 32:
        parser.error('reference short edge must be a positive multiple of 32')
    if args.output.exists():
        parser.error('output directory already exists; use a new experiment name')
    import numpy as np
    import torch
    from PIL import Image
    from imageio_ffmpeg import get_ffmpeg_exe
    from model.pipeline import Pipeline
    from model.causal_stream import CausalStream
    from model.conditioning_service import encode_remote
    from model.video_window import read_video_window
    from model.provenance import snapshot_sources, changed_sources
    from model.weights import sha256

    args.output.mkdir(parents=True)
    cache = args.output/'cache'
    source = snapshot_sources(cache)
    prompt = args.prompt_file.read_text().strip()
    (args.output/'prompt.txt').write_text(prompt)
    pixels = lambda a: torch.from_numpy(np.array(a, copy=True)).permute(3, 0, 1, 2)[None].float()/255
    source_rgb, window = read_video_window(args.video, 22, args.width, args.height, 'tail')
    appearance, _ = read_video_window(args.video, 1, args.width, args.height, 'head')
    first = args.output/'reference_first.png'
    Image.fromarray(appearance[0]).save(first)
    refs = [first]

    def condition():
        return encode_remote(args.conditioner_service, args.weights, prompt, refs, args.height, args.width,
            frames=17, native_keyframes=False, text_only=True, reference_image_short_edge=args.reference_short_edge)

    def reference_pixels(path):
        with Image.open(path) as image:
            scale = args.reference_short_edge / min(image.size)
            size = tuple(round(v*scale/32)*32 for v in image.size)
            return pixels(np.array(image.convert('RGB').resize(size, Image.Resampling.LANCZOS))[None])

    conditioning, text_timing = condition()
    pipeline = Pipeline(args.weights, attention_backend='native')
    from model.elementwise_fusion import enable_elementwise_fusion
    pipeline.fusion = enable_elementwise_fusion(pipeline.transformer)
    stream = CausalStream(pipeline, capacity=32, stabilize_exposure=not args.raw_only)
    stream.initialize(pixels(source_rgb))
    stream.add_reference(reference_pixels(first))
    record = dict(architecture='causal_5plus17_with_fixed_image_refs', source_code=source,
        transformer=pipeline.transformer_record,
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_window=window, source_sha256=sha256(args.video),
        reference_policy='original first RGB; first generated chunk final RGB; immutable thereafter',
        conditioning='Ref2VA image presentation, fused text rows only',
        history_policy='training rolling last 32 latents, dense oldest/latest, two stride2 and eight stride4',
        attention_backend=pipeline.attention_backend, decoder='official', steps=4, fps=24)
    writers = []
    started = time.perf_counter()
    with (args.output/'encode.log').open('w') as errors:
        command = [get_ffmpeg_exe(), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{args.width}x{args.height}', '-r', '24', '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-threads', '4', '-crf', '18', '-pix_fmt', 'yuv420p']
        try:
            for name in ('video.mp4', 'raw.mp4'):
                writers.append(subprocess.Popen(command+[str(args.output/name)], stdin=subprocess.PIPE, stderr=errors))
            delivered = chunk = 0
            while delivered < args.frames:
                output, timing = stream.step(conditioning, args.seed+chunk, steps=4)
                arrays = [(v[0].permute(1, 2, 3, 0).cpu().numpy()*255).round().astype(np.uint8)
                          for v in (output, stream.raw_output)]
                count = min(17, args.frames-delivered)
                for writer, frames in zip(writers, arrays):
                    writer.stdin.write(frames[:count].tobytes())
                Image.fromarray(arrays[0][-1]).save(args.output/f'boundary_{chunk:03d}.jpg')
                timing.update(chunk=chunk, seed=args.seed+chunk, first_output_frame=delivered,
                              delivered_frames=count, conditioning=text_timing)
                with (args.output/'measurements.jsonl').open('a') as file:
                    file.write(json.dumps(timing)+'\n')
                print(json.dumps(timing), flush=True)
                delivered += count
                if chunk == 0 and delivered < args.frames:
                    second = args.output/'reference_first_chunk_end.png'
                    Image.fromarray(arrays[0][-1]).save(second)
                    refs.append(second)
                    stream.add_reference(reference_pixels(second))
                    conditioning, text_timing = condition()
                else:
                    text_timing = dict(cache_hit=True, seconds=0.)
                chunk += 1
            record.update(chunks=chunk, delivered_frames=delivered, seconds=time.perf_counter()-started,
                          reference_files=[dict(path=str(p), sha256=sha256(p)) for p in refs])
        finally:
            for writer in writers:
                writer.stdin.close()
            codes = [writer.wait() for writer in writers]
            stream.close()
        if any(codes):
            raise RuntimeError(f'video encoding failed: {codes}')
    changed = changed_sources(source)
    record['changed_sources'] = changed
    (args.output/'video.json').write_text(json.dumps(record, indent=2))
    if changed:
        raise RuntimeError('release source changed during sampling')
    print('completed:', args.output/'video.mp4', flush=True)


if __name__ == '__main__':
    main()
