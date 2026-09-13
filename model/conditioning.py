"""Retain only the Qwen layer consumed by H3, with the native multimodal forward."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from diffusers.modular_pipelines.minimax_h3.encoders import MINIMAX_H3_TEXT_ENCODER_LAYER


@contextmanager
def conditioning_layer(model):
    """Scope the optimization to this encoder call; restore the complete model."""
    decoder = model.language_model
    layers = decoder.layers
    depth = MINIMAX_H3_TEXT_ENCODER_LAYER
    if len(layers) < depth:
        raise ValueError("Qwen checkpoint does not contain H3's conditioning layer")
    original = model.forward
    local_forward = model.__dict__.get("forward")
    captured = []

    def capture(module, inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    def forward(*args, **kwargs):
        captured.clear()
        kwargs.update(output_hidden_states=False, logits_to_keep=1)
        original(*args, **kwargs)
        if len(captured) != 1:
            raise RuntimeError("Qwen conditioning layer must execute exactly once")
        return SimpleNamespace(hidden_states=(None,) * depth + (captured[0],))

    handle = layers[depth - 1].register_forward_hook(capture)
    decoder.layers = type(layers)(list(layers[:depth]))
    model.forward = forward
    try:
        yield
    finally:
        decoder.layers = layers
        if local_forward is None:
            del model.forward
        else:
            model.forward = local_forward
        handle.remove()


def encode_stream_context(weights, prompt, paths, cache, resident, offload_after=False):
    """Formal V2V image-fused text: early frame, current boundary, optional appearance refs."""
    import hashlib
    import json
    from pathlib import Path
    import time
    import torch
    from PIL import Image
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
    from .weights import local_weights, sha256
    from .cache import save_cache
    weights = local_weights(weights)
    if len(paths) not in (2, 3, 4):
        raise ValueError('stream context requires two boundary images and optionally one or two fixed refs')
    identity = dict(abi='formal_head_tail_native_text_v1', weights=str(weights), prompt=prompt,
                    inputs=[sha256(p) for p in paths])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/(key+'.pt')
    if path.exists():
        value = torch.load(path, map_location='cpu', weights_only=True)
        if value['identity'] != identity:
            raise ValueError('stream text cache mismatch')
        return value['tensors'], dict(cache_hit=True, seconds=0., key=key)
    if not resident:
        manager = ComponentsManager()
        pipe = MiniMaxH3TextEncoderStep().init_pipeline(str(weights), components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(weights), local_files_only=True)
        decoder = pipe.text_encoder.model.language_model
        decoder.layers = type(decoder.layers)(list(decoder.layers[:MINIMAX_H3_TEXT_ENCODER_LAYER]))
        if not offload_after:
            pipe.to('cuda')
        resident.update(pipe=pipe, manager=manager, weights=str(weights))
    if resident['weights'] != str(weights):
        raise ValueError('stream conditioner cannot change weights')
    images = []
    for p in paths:
        with Image.open(p) as image:
            images.append(image.convert('RGB'))
    pipe = resident['pipe']
    torch.cuda.synchronize()
    started = time.perf_counter()
    device_context=conditioning_device(pipe) if offload_after else nullcontext()
    with torch.inference_mode(), device_context, conditioning_layer(pipe.text_encoder.model):
        embeds, tags = MiniMaxH3TextEncoderStep.encode_prompt(pipe, prompt, images,
            device=torch.device('cuda'), dtype=torch.bfloat16)
    keep = tags == MINIMAX_H3_TEXT_TAG
    tensors = dict(prompt_embeds=embeds[:, keep].contiguous().cpu(), text_token_tags=tags[keep].cpu())
    torch.cuda.synchronize()
    timing = dict(cache_hit=False, seconds=time.perf_counter()-started, key=key,
                  image_count=len(images), vision_rows_filtered=int((~keep).sum()))
    if offload_after:
        pipe.to('cpu')
        torch.cuda.empty_cache()
    save_cache(dict(identity=identity, tensors=tensors), path)
    return tensors, timing


@contextmanager
def conditioning_device(pipe, offload_blocks=None):
    """Stage text layers without changing their weights, order or arithmetic."""
    import torch
    decoder = pipe.text_encoder.model.language_model
    layers = decoder.layers
    if offload_blocks is None:
        needed = sum(p.numel()*p.element_size() for p in pipe.text_encoder.parameters())
        budget = max(0, torch.cuda.mem_get_info()[0] - 8*2**30)
        sizes = [sum(p.numel()*p.element_size() for p in block.parameters()) for block in layers]
        offload_blocks = 0
        while offload_blocks < len(layers) and needed > budget:
            needed -= sizes[offload_blocks]
            offload_blocks += 1
    if not 0 <= offload_blocks <= len(layers):
        raise ValueError('invalid conditioning offload count')
    handles = []
    # Exclude staged layers before .to(cuda), avoiding an initial allocation peak.
    decoder.layers = type(layers)(list(layers[offload_blocks:]))
    try:
        pipe.to('cuda')
    finally:
        decoder.layers = layers
    if offload_blocks:
        print(f'Conditioning: staging {offload_blocks}/{len(layers)} text layers on CPU; precision unchanged.', flush=True)
    def load(module, args):
        module.to('cuda')
    def unload(module, args, output):
        module.to('cpu')
    try:
        for block in layers[:offload_blocks]:
            block.to('cpu')
            handles.append(block.register_forward_pre_hook(load))
            handles.append(block.register_forward_hook(unload, always_call=True))
        yield offload_blocks
    finally:
        for handle in handles:
            handle.remove()
