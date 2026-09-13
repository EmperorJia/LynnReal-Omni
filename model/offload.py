"""Stage unused DiT blocks during whole VAE phases on memory-limited GPUs."""
from contextlib import contextmanager
import sys
import torch
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3KeyframeVaeEncoderStep, MiniMaxH3Ref2VAReferenceEncoderStep,
)
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep


@contextmanager
def temporarily_on_cpu(module):
    device = next(module.parameters()).device
    module.to("cpu")
    torch.cuda.empty_cache()
    try:
        yield
    finally:
        # Do not hide an inference exception behind a failed CUDA restoration.
        if sys.exc_info()[0] is None:
            module.to(device)
        torch.cuda.empty_cache()


@contextmanager
def staged_transformer(components, count):
    transformer = getattr(components, "transformer_ref", None)
    if transformer is None:
        transformer = components.transformer
    if not 0 <= count <= len(transformer.transformer_blocks):
        raise ValueError("DiT offload count exceeds the number of transformer blocks")
    blocks = transformer.transformer_blocks[:count]
    if not count:
        yield
        return
    device = next(transformer.parameters()).device
    handles = []

    def load(module, args):
        module.to(device)

    def unload(module, args, output):
        module.to("cpu")

    with temporarily_on_cpu(blocks):
        try:
            for block in blocks:
                handles.append(block.register_forward_pre_hook(load))
                handles.append(block.register_forward_hook(unload, always_call=True))
            yield
        finally:
            for handle in handles:
                handle.remove()


@contextmanager
def vae_phase(components):
    transformer = getattr(components, "transformer_ref", None)
    if transformer is None:
        transformer = components.transformer
    blocks = transformer.transformer_blocks[:8]
    # Input projections remain resident, so native pipeline device discovery
    # still identifies CUDA. No precision, tiling or latent arithmetic changes.
    with temporarily_on_cpu(blocks):
        yield


class StagedReferenceEncoder(MiniMaxH3Ref2VAReferenceEncoderStep):
    def __call__(self, components, state):
        with vae_phase(components):
            return super().__call__(components, state)


class StagedKeyframeEncoder(MiniMaxH3KeyframeVaeEncoderStep):
    def __call__(self, components, state):
        with vae_phase(components):
            return super().__call__(components, state)


class StagedVideoDecoder(MiniMaxH3VideoDecodeStep):
    def __call__(self, components, state):
        with vae_phase(components):
            return super().__call__(components, state)


class DenoiseMemory:
    offload_blocks = 0

    def __call__(self, components, state):
        # The VAE encoder remains on CUDA for native device discovery.
        with temporarily_on_cpu(components.vae.decoder), staged_transformer(components, self.offload_blocks):
            return super().__call__(components, state)


class StagedDenoise(DenoiseMemory, MiniMaxH3DenoiseStep):
    pass


class StagedReferenceDenoise(DenoiseMemory, MiniMaxH3Ref2VADenoiseStep):
    pass


def configure_vae_offload(blocks, dit_offload_blocks=0):
    if "reference_encoder" in blocks.sub_blocks:
        blocks.sub_blocks["reference_encoder"] = StagedReferenceEncoder()
        blocks.sub_blocks["denoise"] = StagedReferenceDenoise()
    else:
        blocks.sub_blocks["denoise"] = StagedDenoise()
    blocks.sub_blocks["denoise"].offload_blocks = dit_offload_blocks
    if "vae_encoder" in blocks.sub_blocks:
        blocks.sub_blocks["vae_encoder"] = StagedKeyframeEncoder()
    blocks.sub_blocks["decode"].sub_blocks["video"] = StagedVideoDecoder()
