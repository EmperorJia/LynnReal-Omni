"""Use a verified crop of the immutable first image for four-step structure editing."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appearance-crop', type=Path, required=True)
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model import conditioning_service
    from model.weights import sha256
    crop = args.appearance_crop.resolve()
    record = json.loads(crop.with_suffix('.json').read_text())
    if record['resized'] or sha256(crop) != record['output_sha256']:
        raise ValueError('appearance crop or its provenance changed')
    original = conditioning_service.encode_remote

    def encode(queue, weights, prompt, paths, *a, **kw):
        if (len(paths) != 2 or sha256(paths[1]) != record['source_sha256']
                or sha256(crop) != record['output_sha256']):
            raise ValueError('crop must derive from this exact immutable first image')
        paths[1] = crop
        context, timing = original(queue, weights, prompt, paths, *a, **kw)
        timing['appearance_crop'] = record
        return context, timing

    conditioning_service.encode_remote = encode
    entry = root/'script/structure_chunk_editor.py'
    sys.argv = [str(entry), *rest]
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        conditioning_service.encode_remote = original


if __name__ == '__main__':
    main()
