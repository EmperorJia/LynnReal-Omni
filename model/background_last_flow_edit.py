"""Use only a first-frame facade crop as the immutable appearance reference."""
import copy
import numpy as np
from PIL import Image
import torch
from .flow_edit import SourceVideoEncoder
from .offload import StagedReferenceEncoder
from .visual_flow_edit import VisualFlowEdit


class BackgroundEncoder(SourceVideoEncoder):
    def encode_references(self, components, references, device=None):
        video, audio = super().encode_references(components, references, device)
        clips = [r for r in references if r.kind == 'video']
        images = [r for r in references if r.kind == 'image']
        if len(clips) != 1 or len(images) != 1:
            raise ValueError('one source video and one appearance crop are required')
        frame = np.array(clips[0].frames[16], copy=True)
        image = copy.copy(images[0])
        image.image = Image.fromarray(frame[:round(len(frame)*.45)]).resize(
            images[0].image.size, Image.Resampling.LANCZOS)
        source, _ = StagedReferenceEncoder.encode_references(components, [image], device)
        self.condition_delta = torch.zeros_like(video)
        cursor = 0
        for reference in references:
            count = reference.num_video_rows if reference.kind != 'audio' else 0
            if reference.kind == 'image':
                if source.shape != video[cursor:cursor+count].shape:
                    raise RuntimeError('source and target crop grids differ')
                self.condition_delta[cursor:cursor+count] = .999*(source-video[cursor:cursor+count])
            cursor += count
        return video, audio


def configure_background_flow(blocks):
    encoder = BackgroundEncoder()
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = VisualFlowEdit(encoder)
