"""Anchor the edited target head to the current source without adding a boundary reference."""
import numpy as np
from PIL import Image
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3KeyframeVaeEncoderStep
from .native_head import InitialReferenceEncoder, FixedReferenceHeadDenoise
from .offload import StagedReferenceEncoder


class SourceBoundaryEncoder(InitialReferenceEncoder):
    __call__ = StagedReferenceEncoder.__call__

    def encode_references(self, components, references, device=None):
        video, audio = StagedReferenceEncoder.encode_references(components, references, device)
        sources = [r for r in references if r.kind == 'video']
        if len(sources) != 1:
            raise ValueError('source boundary editing requires exactly one current video')
        self.boundary_image = Image.fromarray(np.array(sources[0].frames[0], copy=True))
        self.current_rows = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes(
            components, [self.boundary_image], device)
        self.initial_encodes += 1
        return video, audio


def configure_source_boundary(blocks):
    encoder = SourceBoundaryEncoder()
    offload_blocks = blocks.sub_blocks['denoise'].offload_blocks
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = FixedReferenceHeadDenoise(encoder, offload_blocks)
