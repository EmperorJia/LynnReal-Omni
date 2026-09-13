"""Use a coarse original image for appearance and full-resolution current video."""
import argparse
from pathlib import Path
import runpy
import sys


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-edge', type=int, choices=(128, 256), required=True)
    a, rest = p.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model.pipeline import Pipeline
    from model import conditioning_service
    initialize, generate = Pipeline.__init__, Pipeline.generate
    encode = conditioning_service.encode_remote

    def init(self, *args, **kwargs):
        kwargs['reference_image_short_edge'] = a.image_edge
        initialize(self, *args, **kwargs)

    def context(*args, **kwargs):
        kwargs['reference_image_short_edge'] = a.image_edge
        return encode(*args, **kwargs)

    def sample(self, *args, **kwargs):
        state, timing = generate(self, *args, **kwargs)
        timing['reference_image_short_edge'] = a.image_edge
        self.pipe._blocks.sub_blocks['denoise'].audit['reference_image_short_edge'] = a.image_edge
        return state, timing

    Pipeline.__init__, Pipeline.generate = init, sample
    conditioning_service.encode_remote = context
    entry = root / 'script/flow_chunk_editor.py'
    sys.argv = [str(entry), *rest]
    runpy.run_path(str(entry), run_name='__main__')
