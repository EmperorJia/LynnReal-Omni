"""Use a deterministic structure rendition of the current chunk for native video editing."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structure', choices=('gray', 'edges'), required=True)
    parser.add_argument('--chroma-retention', type=float, default=0,
                        help='For gray input, retain this fraction of source RGB chroma')
    parser.add_argument('--inference-python', help='Optional Python environment for DiT inference')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preserve-rgb-head', action='store_true')
    parser.add_argument('--boundary', type=Path)
    args, rest = parser.parse_known_args()
    if not 0 <= args.chroma_retention <= 1 or (args.structure != 'gray' and args.chroma_retention):
        parser.error('chroma retention requires gray mode and a value in [0,1]')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import cv2
    import numpy as np
    from model.chunk_edit import write_video
    from model.video_window import read_video_window
    from model.weights import sha256
    cv2.setNumThreads(2)
    source = Path(rest[rest.index('--source') + 1])
    output = Path(rest[rest.index('--output') + 1])
    if not output.resolve().is_relative_to(root / 'output'):
        parser.error('outputs must stay inside release/output')
    output.mkdir(parents=True, exist_ok=False)
    rgb, _ = read_video_window(source, 22, 1344, 768, 'head')
    if args.boundary:
        from PIL import Image
        if not np.array_equal(np.asarray(Image.open(args.boundary).convert('RGB')), rgb[0]):
            raise RuntimeError('boundary reference is not the previous delivered source head')
    control = []
    for frame in rgb:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if args.structure == 'edges':
            gray = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.), 64, 128, L2gradient=True)
        control.append(np.repeat(gray[..., None], 3, axis=-1))
    control = np.stack(control)
    if args.chroma_retention:
        control = np.rint(control.astype(np.float32) + args.chroma_retention *
                          (rgb.astype(np.float32) - control)).clip(0,255).astype(np.uint8)
    if args.preserve_rgb_head:
        control[0] = rgb[0]
    path = output / 'structure_lossless.mp4'
    write_video(path, control, lossless=True)
    decoded, _ = read_video_window(path, 22, 1344, 768, 'head')
    if not np.array_equal(decoded, control):
        raise RuntimeError('structure input changed during lossless interchange')
    record = dict(mode='desaturated' if args.chroma_retention else args.structure,
                  chroma_retention=args.chroma_retention,
                  original_source=dict(path=str(source.resolve()), sha256=sha256(source)),
                  editing_source=dict(path=str(path.resolve()), sha256=sha256(path)),
                  fps=24, frame_count=22, frame_resampling=False, optical_flow_warping=False,
                  grayscale='OpenCV RGB2GRAY', edges='Gaussian 5x5 sigma1; Canny thresholds64,128 L2gradient',
                  output_postprocessing=False, editing_forward_limit=4,
                  preserved_source_rgb_frames=[0] if args.preserve_rgb_head else [])
    if args.boundary:
        record['causal_boundary'] = dict(path=str(args.boundary.resolve()), sha256=sha256(args.boundary),
                                        exact_source_head=True)
    (output / 'structure.json').write_text(json.dumps(record, indent=2) + '\n')
    if args.prepare_only:
        return
    rest[rest.index('--source') + 1] = str(path)
    rest[rest.index('--output') + 1] = str(output / 'edited')
    entry = root / 'tool/audit_matched_edit_noise.py'
    if args.inference_python:
        subprocess.run([args.inference_python, '-u', str(entry), *rest], check=True)
    else:
        sys.argv = [str(entry), *rest]
        runpy.run_path(str(entry), run_name='__main__')
    for p in (output / 'edited').glob('*/edited_rgb.npy'):
        edited = np.load(p, allow_pickle=False)
        write_video(p.parent / 'compare_original.mp4', np.concatenate((rgb[:17], edited[:17]), axis=2))


if __name__ == '__main__':
    main()
