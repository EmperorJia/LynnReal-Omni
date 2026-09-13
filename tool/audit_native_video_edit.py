"""Compare native four-forward video edits with explicit reference and prompt records."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--first', type=Path)
    parser.add_argument('--extra-image', type=Path, action='append', default=[])
    parser.add_argument('--image-first', action='store_true', help='Pack the first image before the source video')
    parser.add_argument('--image-edge', type=int, default=128)
    parser.add_argument('--video-edge', type=int, default=768)
    parser.add_argument('--prompt', type=Path, nargs='+', required=True)
    parser.add_argument('--conditioner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2000008)
    args = parser.parse_args()
    if args.image_first and args.first is None:
        parser.error('--image-first requires --first')
    import numpy as np
    from model.chunk_edit import write_video
    from model.conditioning_service import encode_remote
    from model.forward_budget import forward_budget
    from model.pipeline import Pipeline
    from model.provenance import snapshot_sources, changed_sources
    from model.video_window import read_video_window
    from model.weights import sha256
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'output'):
        parser.error('experiment outputs must be inside the release output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(args.output / 'cache')
    weights = root / 'weight/standard'
    refs = [args.source] + ([args.first] if args.first else []) + args.extra_image
    if args.image_first:
        refs[:2] = [args.first, args.source]
    hashes = [dict(path=str(p.resolve()), sha256=sha256(p)) for p in refs]
    source, _ = read_video_window(args.source, 22, 1344, 768, 'head')
    pipe = Pipeline(weights, reference=True, native_keyframes=False, vae_offload=True,
                    reference_image_short_edge=args.image_edge, reference_video_short_edge=args.video_edge)
    for prompt_file in args.prompt:
        output = args.output / prompt_file.stem
        output.mkdir(exist_ok=False)
        prompt = prompt_file.read_text().strip()
        context, conditioning = encode_remote(args.conditioner, weights, prompt, refs,
            768, 1344, frames=22, native_keyframes=False, text_only=False,
            reference_image_short_edge=args.image_edge, reference_video_short_edge=args.video_edge)
        with forward_budget(pipe.transformer) as budget:
            state, timing = pipe.generate(context, refs, 768, 1344, 22, args.seed, 4)
        rgb = np.stack([np.asarray(f) for f in state['videos'][0]])
        if rgb.shape != (22, 768, 1344, 3):
            raise RuntimeError('unexpected native edited video shape')
        np.save(output / 'edited_rgb.npy', rgb, allow_pickle=False)
        write_video(output / 'video.mp4', rgb[:17])
        write_video(output / 'compare.mp4', np.concatenate((source[:17], rgb[:17]), axis=2))
        changed = changed_sources(sources)
        if changed or any(sha256(p) != h['sha256'] for p, h in zip(refs, hashes)):
            raise RuntimeError('experiment sources changed')
        record = dict(prompt=prompt, seed=args.seed, actual_references=hashes,
                      reference_order='image-first' if args.image_first else 'video-first',
                      reference_image_short_edge=args.image_edge, reference_video_short_edge=args.video_edge,
                      timing=timing, forward_budget=budget, conditioning=conditioning,
                      source_code=sources, changed_sources=changed, weights=str(weights),
                      attention_backend=pipe.attention_backend, not_a_stream_rollout=True,
                      approved_for_30s=False)
        (output / 'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(output=str(output), timing=timing, forward_budget=budget)), flush=True)


if __name__ == '__main__':
    main()
