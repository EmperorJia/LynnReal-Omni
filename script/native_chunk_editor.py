"""Four-forward reference editing with a low-resolution first-image appearance cue."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--conditioner', type=Path, required=True)
    parser.add_argument('--image-edge', type=int, default=128)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    import numpy as np
    from model.chunk_edit import write_video
    from model.conditioning_service import encode_remote
    from model.forward_budget import forward_budget
    from model.pipeline import Pipeline
    from model.provenance import snapshot_sources, changed_sources
    from model.video_window import read_video_window
    from model.weights import sha256
    root = Path(__file__).resolve().parents[1]
    weights = root / 'weight/standard'
    args.queue.mkdir(parents=True, exist_ok=True)
    lock = (args.queue / 'service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_code = snapshot_sources(args.queue / 'cache')
    pipe = Pipeline(weights, reference=True, native_keyframes=False, vae_offload=True,
                    reference_image_short_edge=args.image_edge, reference_video_short_edge=768)
    started = time.monotonic()
    while time.monotonic() - started < args.max_seconds and not (args.queue / 'STOP').exists():
        pending = sorted(args.queue.glob('*.request.json'))
        if not pending:
            time.sleep(.1)
            continue
        path = pending[0]
        claimed = path.with_suffix('.claimed')
        path.rename(claimed)
        request = json.loads(claimed.read_text())
        result = dict(request=request)
        try:
            refs = [Path(r['path']) for r in request['references']]
            destination = Path(request['output']).resolve()
            if (len(refs) != 3 or not destination.is_relative_to(root / 'output')
                    or any(sha256(p) != r['sha256'] for p, r in zip(refs, request['references']))):
                raise ValueError('invalid or changed chunk inputs')
            destination.mkdir(parents=True, exist_ok=True)
            # The boundary is checked as part of the causal input record. Motion
            # comes from the source video; only the original image supplies style.
            actual_refs = [refs[1], refs[0]]
            context, text_time = encode_remote(args.conditioner, weights, request['prompt'], actual_refs,
                768, 1344, frames=22, native_keyframes=False, text_only=False,
                reference_image_short_edge=args.image_edge, reference_video_short_edge=768)
            with forward_budget(pipe.transformer) as budget:
                state, timing = pipe.generate(context, actual_refs, 768, 1344, 22, request['seed'], 4)
            rgb = np.stack([np.asarray(f) for f in state['videos'][0]])
            if rgb.shape != (22, 768, 1344, 3) or budget['actual_dit_forwards'] != 4:
                raise RuntimeError('editor output shape or forward count is wrong')
            np.save(destination / 'edited_rgb.npy', rgb, allow_pickle=False)
            write_video(destination / 'edited_full22.mp4', rgb)
            source, _ = read_video_window(refs[1], 22, 1344, 768, 'head')
            write_video(destination / 'compare.mp4', np.concatenate((source[:17], rgb[:17]), axis=2))
            changed = changed_sources(source_code)
            if changed:
                raise RuntimeError(f'editor source changed: {changed}')
            result.update(timing=timing, forward_budget=budget, conditioning=text_time,
                          fixed_head=False, actual_references=[str(p) for p in actual_refs],
                          reference_image_short_edge=args.image_edge,
                          rgb_sha256=sha256(destination / 'edited_rgb.npy'), source_code=source_code,
                          changed_sources=changed, weights=str(weights), attention_backend=pipe.attention_backend)
            (destination / 'editor_result.json').write_text(json.dumps(result, indent=2) + '\n')
        except Exception as error:
            traceback.print_exc()
            result['error'] = f'{type(error).__name__}: {error}'
        temporary = args.queue / f'{request["id"]}.result.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.rename(args.queue / f'{request["id"]}.result.json')
        print(json.dumps(dict(id=request['id'], error=result.get('error'), timing=result.get('timing'))), flush=True)
        if args.once:
            return


if __name__ == '__main__':
    main()
