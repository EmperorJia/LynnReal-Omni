"""Native four-forward chunk editing from structure and the original first image."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structure', choices=('gray', 'edges'), required=True)
    parser.add_argument('--chroma-retention', type=float, default=0)
    parser.add_argument('--preprocess-python', required=True)
    parser.add_argument('--preserve-rgb-head', action='store_true')
    parser.add_argument('--boundary-edge', type=int, choices=(0,128,768), default=0)
    args, rest = parser.parse_known_args()
    if not 0 <= args.chroma_retention <= 1 or (args.structure != 'gray' and args.chroma_retention):
        parser.error('chroma retention requires gray mode and a value in [0,1]')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from model import conditioning_service
    from model.controlled_noise import controlled_target_noise
    from model.pipeline import Pipeline
    encode = conditioning_service.encode_remote
    generate = Pipeline.generate
    prepared = {}
    boundary_geometry = {}
    if args.boundary_edge:
        from model.boundary_reference import configure_boundary_reference
        boundary_geometry = configure_boundary_reference(args.boundary_edge)

    def structure_context(queue, weights, prompt, paths, *a, **kw):
        if len(paths) != 2 or Path(paths[0]).suffix != '.mp4':
            raise ValueError('expected current video and original first image')
        destination = Path(paths[0]).parent / 'structure_input'
        boundary = Path(paths[0]).parent / 'boundary.png'
        started = time.perf_counter()
        subprocess.run([args.preprocess_python, str(root / 'tool/audit_structure_video_edit.py'),
                        '--prepare-only', '--structure', args.structure,
                        '--chroma-retention', str(args.chroma_retention),
                        '--source', str(paths[0]), '--output', str(destination)]
                       + (['--preserve-rgb-head'] if args.preserve_rgb_head else [])
                       + (['--boundary', str(boundary)] if args.boundary_edge else []), check=True)
        paths[0] = destination / 'structure_lossless.mp4'
        prepared.clear()
        prepared.update(json.loads((destination / 'structure.json').read_text()))
        prepared.update(preprocessing_ms=(time.perf_counter()-started)*1000, prompt=prompt)
        if args.boundary_edge:
            paths.append(boundary)
        return encode(queue, weights, prompt, paths, *a, **kw)

    def structured_generate(self, conditioning, paths, height, width, frames, seed, steps, **kw):
        with controlled_target_noise(seed) as noise:
            state, timing = generate(self, conditioning, paths, height, width, frames, seed, steps, **kw)
        timing.update(controlled_target_noise=noise, structure_preprocessing=prepared.copy())
        if args.boundary_edge:
            expected = 7056 + 1008 + boundary_geometry['boundary_image_rows']
            if int(state['num_condition_video_rows']) != expected:
                raise RuntimeError('boundary image was not encoded at its requested token count')
            timing['boundary_reference'] = dict(boundary_geometry, actual_condition_rows=expected,
                source='previous delivered RGB frame', fixed_target_head=False)
        return state, timing

    conditioning_service.encode_remote = structure_context
    Pipeline.generate = structured_generate
    entry = root / 'script/native_chunk_editor.py'
    sys.argv = [str(entry), *rest]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
