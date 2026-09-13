"""Generate a 768p, 22-frame Standard BF16 bootstrap with four DiT forwards."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sample(args):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from model.elementwise_fusion import enable_elementwise_fusion
    from model.first_appearance import constrain_moments, encode_static_reference
    from model.output import encode_video
    from model.pipeline import Pipeline, encode_conditioning
    from model.provenance import changed_sources, snapshot_sources
    from model.refinement import target_latents
    from model.weights import sha256

    image = Image.open(args.image).convert('RGB')
    if image.size != (1344, 768):
        raise ValueError('Supply a 1344 x 768 first image; no implicit crop is applied.')
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text().strip()
    if not prompt.strip():
        raise ValueError('The first-chunk prompt must not be empty.')
    weights = ROOT / 'weight/standard'
    sources = snapshot_sources(args.output / 'cache')
    (args.output / 'prompt.txt').write_text(prompt + '\n')
    image.save(args.output / 'reference_first.png')
    context, text_timing = encode_conditioning(
        weights, prompt, [args.image], 768, 1344, 22, args.output / 'cache',
        native_keyframes=False, text_only=False, reference_image_short_edge=768)
    pipeline = Pipeline(
        weights, reference=True, native_keyframes=False, vae_offload=True,
        attention_backend='native', fixed_native_head=True, reference_image_short_edge=768)
    pipeline.fusion = enable_elementwise_fusion(pipeline.transformer)
    reference = encode_static_reference(pipeline, image).cuda()
    original_step = pipeline.pipe.scheduler.step
    constraint = constrain_moments(pipeline.pipe.scheduler, reference, args.appearance_strength)
    expected = context['prompt_embeds'].cuda()
    forwards = []

    def inspect(module, positional, keyword):
        if not torch.equal(keyword['encoder_hidden_states'], expected):
            raise RuntimeError('Image-aware text conditioning changed during denoising.')
        forwards.append(dict(embedding_exact=True, shape=list(expected.shape)))

    handle = pipeline.transformer.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        with torch.inference_mode():
            state, timing = pipeline.generate(
                context, [args.image], 768, 1344, 22, args.seed, 4, keep_latents=True)
    finally:
        handle.remove()
        pipeline.pipe.scheduler.step = original_step
    video, audio = target_latents(state, pipeline.pipe)
    rgb = np.stack([np.asarray(frame) for frame in state['videos'][0]])
    if len(forwards) != 4 or rgb.shape != (22, 768, 1344, 3):
        raise RuntimeError('Unexpected bootstrap geometry or DiT forward count.')
    np.save(args.output / 'rgb.npy', rgb, allow_pickle=False)
    save_file(dict(video=video.contiguous(), audio=audio.contiguous()), str(args.output / 'latents.safetensors'))
    save_file(dict(video=reference.cpu()), str(args.output / 'source_static_video_latents.safetensors'))
    encode_video([Image.fromarray(frame) for frame in rgb], 24, args.output / 'video.mp4')
    encode_video([Image.fromarray(np.concatenate((np.asarray(image), frame), axis=1))
                  for frame in rgb], 24, args.output / 'compare.mp4')
    changed = changed_sources(sources)
    if changed:
        raise RuntimeError(f'Sampling source changed: {changed}')
    metadata = dict(
        weights=str(weights), transformer=pipeline.transformer_record, image=str(args.image),
        image_sha256=sha256(args.image), prompt=prompt, seed=args.seed, frames=22, fps=24,
        precision='bf16', conditioning_abi='unified_ref_native', native_keyframes=False,
        appearance_strength=args.appearance_strength, appearance_constraint=constraint,
        fixed_head=pipeline.fixed_head_denoise.audit, forwards=forwards,
        timing=timing, conditioning=text_timing, source_code=sources, changed_sources=changed,
        rgb_sha256=sha256(args.output / 'rgb.npy'), latents_sha256=sha256(args.output / 'latents.safetensors'))
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), timing=timing, appearance_constraint=constraint), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixtures = ROOT / 'test/stream/first_chunk'
    parser.add_argument('--image', type=Path, default=fixtures / 'street.png')
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path, default=fixtures / 'street.txt')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--appearance-strength', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 <= args.appearance_strength <= 1:
        parser.error('--appearance-strength must be in [0, 1]')
    args.output = args.output or ROOT / 'output/standard/bf16/first_chunk' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    args.output.mkdir(parents=True, exist_ok=False)

    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, value):
            self.terminal.write(value)
            self.log.write(value)
            self.flush()
            return len(value)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    with (args.output / 'sample.log').open('w') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            sample(args)


if __name__ == '__main__':
    main()
