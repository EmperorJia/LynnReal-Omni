"""Experimental eight-forward source-preserving chunk repair service."""
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
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    import numpy as np
    from model.chunk_edit import write_video
    from model.conditioning_service import encode_remote
    from model.pipeline import Pipeline
    from model.provenance import snapshot_sources, changed_sources
    from model.weights import sha256
    root = Path(__file__).resolve().parents[1]
    weights = root/'weight/standard'
    args.queue.mkdir(parents=True, exist_ok=True)
    lock = (args.queue/'service.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_code = snapshot_sources(args.queue/'cache')
    from model import offload
    from model.background_last_flow_edit import configure_background_flow as configure_visual_flow
    from model.video_window import read_video_window
    from PIL import Image
    original = offload.configure_vae_offload
    def configure(blocks, *a, **kw):
        original(blocks, *a, **kw)
        configure_visual_flow(blocks)
    offload.configure_vae_offload = configure
    source_prompt = (root/'test/stream/urban_repair_source.txt').read_text().strip()
    pipe = Pipeline(weights, reference=True, native_keyframes=False, vae_offload=True,
                    reference_image_short_edge=384, reference_video_short_edge=768)
    started = time.monotonic()
    while time.monotonic()-started < args.max_seconds and not (args.queue/'STOP').exists():
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
            destination = Path(request['output'])
            if (len(refs) != 3 or not destination.is_relative_to(root/'output')
                    or any(sha256(p) != r['sha256'] for p, r in zip(refs, request['references']))):
                raise ValueError('invalid or changed chunk inputs')
            source_rgb, _ = read_video_window(refs[1], 22, 1344, 768, 'head')
            source_image = destination/'source_appearance.png'
            Image.fromarray(source_rgb[16,:round(source_rgb.shape[1]*.45)]).save(source_image)
            original_crop = destination/'original_facade.png'
            with Image.open(refs[0]) as image:
                image.crop((0,0,image.width,round(image.height*.45))).save(original_crop)
            target_refs = [refs[1], original_crop]
            source_context, source_text_time = encode_remote(args.conditioner, weights, source_prompt,
                [refs[1], source_image], 768, 1344, frames=22, native_keyframes=False, text_only=False,
                reference_image_short_edge=384, reference_video_short_edge=768)
            context, text_time = encode_remote(args.conditioner, weights, request['prompt'], target_refs,
                768, 1344, frames=22, native_keyframes=False, text_only=False,
                reference_image_short_edge=384, reference_video_short_edge=768)
            denoise = pipe.pipe._blocks.sub_blocks['denoise']
            denoise.source_conditioning = source_context
            denoise.mode = 'difference'
            denoise.seed = request['seed']
            state, timing = pipe.generate(context, target_refs, 768, 1344, 22, request['seed'], 8)
            rgb = np.stack([np.asarray(f) for f in state['videos'][0]])
            if rgb.shape != (22, 768, 1344, 3) or timing['actual_dit_forwards'] != 8:
                raise RuntimeError('editor output shape or forward count is wrong')
            np.save(destination/'edited_rgb.npy', rgb, allow_pickle=False)
            write_video(destination/'edited_full22.mp4', rgb)
            if changed_sources(source_code):
                raise RuntimeError('editor source changed during execution')
            result.update(timing=timing, conditioning=text_time, flow_edit=denoise.audit, fixed_head=False, appearance_crop_fraction=.45, source_appearance_frame=16,
                          source_conditioning=source_text_time,
                          actual_references=[str(p) for p in target_refs],
                          source_appearance=dict(path=str(source_image),sha256=sha256(source_image)),
                          rgb_sha256=sha256(destination/'edited_rgb.npy'), source_code=source_code,
                          weights=str(weights), attention_backend=pipe.attention_backend)
        except Exception as error:
            traceback.print_exc()
            result['error'] = f'{type(error).__name__}: {error}'
        temporary = args.queue / f'{request["id"]}.result.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.rename(args.queue / f'{request["id"]}.result.json')
        print(json.dumps(dict(id=request['id'], error=result.get('error'))), flush=True)


if __name__ == '__main__':
    main()
