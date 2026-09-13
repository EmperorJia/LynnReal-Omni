"""Run native four-step editing comparisons with identical target video and audio noise."""
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model.controlled_noise import controlled_target_noise
    from model.pipeline import Pipeline
    original = Pipeline.generate

    def generate(self, conditioning, paths, height, width, frames, seed, steps, **kw):
        with controlled_target_noise(seed) as noise:
            state, timing = original(self, conditioning, paths, height, width, frames, seed, steps, **kw)
        timing['controlled_target_noise'] = noise
        return state, timing

    Pipeline.generate = generate
    entry = root / 'tool/audit_native_video_edit.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
