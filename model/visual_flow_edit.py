"""FlowEdit diagnostic with distinct source and target appearance images."""
import copy
import numpy as np
from PIL import Image
import torch
from .flow_edit import SourceVideoEncoder, FlowEditDenoise
from .offload import StagedReferenceEncoder


class AppearanceEncoder(SourceVideoEncoder):
    def encode_references(self, components, references, device=None):
        video, audio = super().encode_references(components, references, device)
        clips = [r for r in references if r.kind == 'video']
        images = [r for r in references if r.kind == 'image']
        if len(images) != 1 or len(clips) != 1:
            raise ValueError('expected one current video and one original appearance image')
        alternative = copy.copy(images[0])
        alternative.image = Image.fromarray(np.array(clips[0].frames[0], copy=True)).resize(
            images[0].image.size, Image.Resampling.LANCZOS)
        source, _ = StagedReferenceEncoder.encode_references(components, [alternative], device)
        self.condition_delta = torch.zeros_like(video)
        cursor = 0
        for reference in references:
            count = reference.num_video_rows if reference.kind != 'audio' else 0
            if reference.kind == 'image':
                if source.shape != video[cursor:cursor+count].shape:
                    raise RuntimeError('source and target appearance grids must match')
                self.condition_delta[cursor:cursor+count] = .999 * (source-video[cursor:cursor+count])
            cursor += count
        return video, audio


class VisualFlowEdit(FlowEditDenoise):
    @torch.no_grad()
    def __call__(self, components, state):
        calls = 0
        delta = self.encoder.condition_delta.to(components._execution_device)

        def source_reference(module, args, kwargs):
            nonlocal calls
            # The parent evaluates target then source. A hybrid final step
            # has only the target call. Both branches share conditioning noise.
            is_source = calls % 2 == 1
            calls += 1
            if is_source and self.mode != 'identity':
                hidden = kwargs['hidden_states'].clone()
                hidden[:, :len(delta)] += delta
                kwargs = dict(kwargs, hidden_states=hidden)
            return args, kwargs

        handle = components.transformer_ref.register_forward_pre_hook(source_reference, with_kwargs=True)
        try:
            result = super().__call__(components, state)
        finally:
            handle.remove()
        if calls != self.audit['forwards']:
            raise RuntimeError('unexpected source/target query ordering')
        self.audit.update(visual_condition_difference=True,
                          source_appearance='first frame of current source clip',
                          target_appearance='original scene image',
                          shared_visual_condition_noise=True)
        return result


def configure_visual_flow(blocks):
    encoder = AppearanceEncoder()
    blocks.sub_blocks['reference_encoder'] = encoder
    blocks.sub_blocks['denoise'] = VisualFlowEdit(encoder)
