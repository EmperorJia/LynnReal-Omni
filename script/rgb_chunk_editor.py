"""Native RGB chunk editing, with an optional original-first-image reference."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-only', action='store_true')
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model import conditioning_service
    from model.controlled_noise import controlled_target_noise
    from model.pipeline import Pipeline
    encode, generate = conditioning_service.encode_remote, Pipeline.generate

    def context(queue, weights, prompt, paths, *a, **kw):
        if len(paths) != 2 or Path(paths[0]).suffix != '.mp4':
            raise ValueError('expected current RGB video and original first image')
        if args.video_only:
            del paths[1:]
        return encode(queue, weights, prompt, paths, *a, **kw)

    def sample(self, conditioning, paths, height, width, frames, seed, steps, **kw):
        if len(paths) != (1 if args.video_only else 2) or steps != 4:
            raise ValueError('unexpected editor reference count or forward budget')
        with controlled_target_noise(seed) as noise:
            state, timing = generate(self, conditioning, paths, height, width, frames, seed, steps, **kw)
        timing.update(controlled_target_noise=noise,
                      editor_reference_mode='video-only' if args.video_only else 'video-and-first-image',
                      input_preprocessing=False)
        return state, timing

    conditioning_service.encode_remote = context
    Pipeline.generate = sample
    entry = root / 'script/native_chunk_editor.py'
    try:
        sys.argv = [str(entry), *rest]
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        conditioning_service.encode_remote = encode
        Pipeline.generate = generate


if __name__ == '__main__':
    main()
