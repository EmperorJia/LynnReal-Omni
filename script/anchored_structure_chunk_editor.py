"""Keep the corrected causal boundary fixed during four-forward structure editing."""
from pathlib import Path
import runpy
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
    from model import offload
    from model.native_head import InitialReferenceEncoder, FixedReferenceHeadDenoise
    from model.offload import StagedReferenceEncoder
    from model.pipeline import Pipeline
    from model.weights import sha256
    configure = offload.configure_vae_offload
    generate = Pipeline.generate

    class BoundaryEncoder(InitialReferenceEncoder):
        __call__ = StagedReferenceEncoder.__call__

        def encode_references(self, components, references, device=None):
            video, audio = StagedReferenceEncoder.encode_references(components, references, device)
            # The source's structure rendition has no appearance information.
            # Encode the previously delivered RGB boundary, not its edge map.
            with Image.open(self.boundary_path) as image:
                self.boundary_image = image.convert('RGB')
            self.current_rows = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(
                components, [self.boundary_image], device)
            self.initial_encodes += 1
            return video, audio

    def configure_boundary(blocks, *a, **kw):
        configure(blocks, *a, **kw)
        encoder = BoundaryEncoder()
        offload_blocks = blocks.sub_blocks['denoise'].offload_blocks
        blocks.sub_blocks['reference_encoder'] = encoder
        blocks.sub_blocks['denoise'] = FixedReferenceHeadDenoise(encoder, offload_blocks)

    def anchored_generate(self, conditioning, paths, *a, **kw):
        boundary = Path(paths[0]).parent.parent / 'boundary.png'
        if Path(paths[0]).parent.name != 'structure_input' or not boundary.is_file():
            raise ValueError('missing corrected causal RGB boundary')
        blocks = self.pipe._blocks.sub_blocks
        blocks['reference_encoder'].boundary_path = boundary
        state, timing = generate(self, conditioning, paths, *a, **kw)
        audit = blocks['denoise'].audit
        if audit['fixed_head_forwards'] != 4 or not audit['endpoint_restored']:
            raise RuntimeError('boundary was not fixed across the four editing forwards')
        timing.update(source_head=audit, boundary_added_to_reference_prefix=False,
                      head_source=dict(path=str(boundary), sha256=sha256(boundary)))
        return state, timing

    offload.configure_vae_offload = configure_boundary
    Pipeline.generate = anchored_generate
    entry = root / 'script/structure_chunk_editor.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
