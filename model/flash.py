"""Flash's trained spatial selection and residual restoration on upstream H3."""
import inspect
import torch


def spatial_layout(positions, tags, stride=2):
    """Keep both grid boundaries; map dropped video rows to same-time anchors."""
    if positions.ndim != 2 or tags.ndim != 1 or positions.shape != (tags.numel(), 3):
        raise ValueError("Flash requires a single shared packed layout")
    video = tags == 0
    coordinates = positions[video]
    if not coordinates.numel():
        raise ValueError("Flash requires video tokens")
    spatial = torch.ones_like(video)
    for axis in (1, 2):
        alphabet = torch.unique(coordinates[:, axis], sorted=True)
        retained = alphabet[::stride]
        if retained[-1] != alphabet[-1]:
            retained = torch.cat((retained, alphabet[-1:]))
        spatial &= torch.isin(positions[:, axis], retained)
    mask = ~video | spatial
    keep = mask.nonzero().flatten()
    inverse = torch.empty_like(tags, dtype=torch.long)
    inverse[keep] = torch.arange(keep.numel(), device=tags.device)
    dropped = (video & ~mask).nonzero().flatten()
    anchors = (video & mask).nonzero().flatten()
    target = positions[anchors].float()
    for rows in dropped.split(1024):
        source = positions[rows].float()
        distance = torch.cdist(source[:, 1:], target[:, 1:])
        distance.masked_fill_(source[:, None, 0] != target[None, :, 0], float("inf"))
        inverse[rows] = inverse[anchors[distance.argmin(dim=1)]]
    return keep, inverse


def configure_flash(transformer, config):
    """Install inference hooks; weights and upstream transformer code stay intact."""
    blocks = transformer.transformer_blocks
    compression = config["token_compression"]
    if len(blocks) != config["num_layers"]:
        raise ValueError("Flash depth and exported weights disagree")
    if (compression["reduction"] != "select" or compression["spatial_stride"] != 2
            or not compression["preserve_text_tokens"] or not compression["preserve_audio_tokens"]
            or compression.get("full_refresh_blocks")
            or compression.get("text_stride", 1) != 1 or compression.get("audio_stride", 1) != 1):
        raise ValueError("unsupported Flash token layout")
    start, end = compression["full_prefix_blocks"], len(blocks) - compression["full_suffix_blocks"]
    if not 0 < start < end < len(blocks):
        raise ValueError("Flash requires nonempty full-resolution prefix and suffix")
    if hasattr(transformer, "_lynnreal_flash_hooks"):
        raise ValueError("Flash is already configured")
    # Disable the optional experimental branch in older local Diffusers forks.
    transformer.h3_token_compression = None
    signature = inspect.signature(transformer.forward)
    state = {}
    layout_cache = {}

    def begin(module, args, kwargs):
        if module.training or torch.is_grad_enabled():
            raise ValueError("Flash hooks require frozen inference")
        if state:
            raise RuntimeError("Flash transformer does not support concurrent forwards")
        inputs = signature.bind(*args, **kwargs).arguments
        positions, tags = inputs["position_ids"], inputs["token_tags"]
        if (not layout_cache or not torch.equal(positions, layout_cache["positions"])
                or not torch.equal(tags, layout_cache["tags"])):
            keep, inverse = spatial_layout(positions, tags)
            layout_cache.update(positions=positions.clone(), tags=tags.clone(), keep=keep, inverse=inverse)
        state["keep"], state["inverse"] = layout_cache["keep"], layout_cache["inverse"]

    def compress(module, args):
        hidden, temb, indices, rotary, mask = args
        keep = state["keep"]
        if "full" not in state:
            state["full"] = hidden
            hidden = hidden.index_select(1, keep)
            state["input"] = hidden
        return (hidden, temb, indices.index_select(0, keep),
                tuple(value.index_select(-2, keep) for value in rotary),
                None if mask is None else mask.index_select(-1, keep))

    def restore(module, args):
        delta = (args[0] - state.pop("input")) * float(compression.get("residual_gain", 1.0))
        hidden = state.pop("full") + delta.index_select(1, state["inverse"])
        return (hidden, *args[1:])

    def finish(*_):
        state.clear()

    handles = [transformer.register_forward_pre_hook(begin, with_kwargs=True)]
    handles += [block.register_forward_pre_hook(compress) for block in blocks[start:end]]
    handles += [blocks[end].register_forward_pre_hook(restore),
                transformer.register_forward_hook(finish, always_call=True)]
    transformer._lynnreal_flash_hooks = handles
