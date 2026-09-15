"""Flash token compression for MiniMax-H3, as a ComfyUI model patch.

Port of `model/flash.py` from the LynnReal release. The Flash DiT's middle blocks run on a
stride-2 spatial subset of the packed video rows; the change those blocks make is scattered
back onto the full sequence before the suffix blocks run at full resolution, so a dropped row
receives its nearest same-timestep anchor's update. Text and audio rows are never dropped.

The release installs this with forward pre-hooks on the diffusers transformer blocks. ComfyUI
exposes the same seam as a per-block replacement -- `set_model_patch_replace(..., "dit",
"double_block", index)` -- whose arguments carry the packed `layout`, which is all
`spatial_layout` needs.

ComfyUI's H3 blocks accumulate their residual in place, so the compressed stream is a copy and
the pre-block value is kept for the delta; the release's blocks return a fresh tensor, and both
forms give the same arithmetic.
"""

import bisect

import torch
from typing_extensions import override

import comfy.model_prefetch
from comfy_api.latest import ComfyExtension, io

# Packed rows the Flash export tags as video: keyframe rows, reference images and the video
# streams themselves all take the video token tag upstream. Audio rows keep the audio tag and
# text rows the text tag, so neither is ever dropped.
VIDEO_KINDS = ("cond", "ref_img", "video")


def video_tags(layout, device):
    tags = torch.ones(layout.seq_len, dtype=torch.long, device=device)
    for start, stop, kind in layout.segments:
        if kind in VIDEO_KINDS:
            tags[start:stop] = 0
    return tags


def spatial_layout(positions, tags, stride):
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


def repack_segments(segments, keep):
    """Rewrite `(start, stop, mod_row)` spans onto the compressed row numbering.

    A span's compressed start and stop are the ranks of its own bounds, so the spans stay
    contiguous and ordered. A per-row modulation table is selected the same way the rows were:
    the kept rows inside a span are exactly `keep[first:last]`.
    """
    kept = keep.tolist()
    packed = []
    for start, stop, row in segments:
        first = bisect.bisect_left(kept, start)
        last = bisect.bisect_left(kept, stop)
        if last <= first:
            continue
        if isinstance(row, torch.Tensor):
            row = row[keep[first:last] - start]
        packed.append((first, last, row))
    return packed


class FlashBlockPatch:
    """Replaces one transformer block, sharing the compression state with its neighbours."""

    def __init__(self, shared, index, start, end, stride, gain):
        self.shared = shared
        self.index = index
        self.start = start
        self.end = end
        self.stride = stride
        self.gain = gain

    def __call__(self, args, extra_args):
        original = extra_args["original_block"]
        state = self.shared["state"]
        if self.index == self.end:
            # `original_block` already answers with the `{"img": ...}` mapping the model reads.
            return self._restore(args, original, state)
        if self.index == self.start:
            with comfy.model_prefetch.pause_malloc_graph():
                self._compress(args, state)
        inner = dict(args)
        inner["rope_freqs"] = state["rope_freqs"]
        inner["mod_segments"] = state["mod_segments"]
        if self.index == self.start:
            inner["img"] = state["compressed"]
        return original(inner)

    def _compress(self, args, state):
        layout = args["layout"]
        full = args["img"]
        # `PackedLayout` keeps its position table on the host; the row selection has to land on
        # the device the hidden state lives on.
        positions = layout.position_ids.to(full.device)
        tags = video_tags(layout, full.device)
        cache = self.shared["cache"]
        if ("keep" not in cache or not torch.equal(positions, cache["positions"])
                or not torch.equal(tags, cache["tags"])):
            keep, inverse = spatial_layout(positions, tags, self.stride)
            cache.update(positions=positions.clone(), tags=tags.clone(), keep=keep, inverse=inverse)
        keep, inverse = cache["keep"], cache["inverse"]

        compressed = full.index_select(0, keep).contiguous()
        state.clear()
        state.update(
            full=full,
            keep=keep,
            inverse=inverse,
            input=compressed.clone(),
            compressed=compressed,
            rope_freqs=args["rope_freqs"].index_select(1, keep).contiguous(),
            mod_segments=repack_segments(args["mod_segments"], keep),
        )

    def _restore(self, args, original, state):
        delta = (args["img"] - state["input"]) * self.gain
        full = state["full"]
        with comfy.model_prefetch.pause_malloc_graph():
            full.add_(delta.index_select(0, state["inverse"]))
            state.clear()
        inner = dict(args)
        inner["img"] = full
        return original(inner)

    def to(self, device_or_dtype):
        return self

    def cleanup(self):
        self.shared["state"].clear()

    def models(self):
        return []


class LynnRealFlashTokenCompression(io.ComfyNode):
    """Run the Flash DiT's middle blocks on the stride-2 spatial subset."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LynnRealFlashTokenCompression",
            display_name="LynnReal Flash token compression (MiniMax H3)",
            description=(
                "Spatial stride-2 token selection with full-resolution residual restoration, the "
                "structure the LynnReal Flash DiT was trained with. Blocks [start, end) run on the "
                "compressed rows and block `end` restores the full sequence before it runs. Insert "
                "between the Flash UNET loader and the guider."
            ),
            category="model/conditioning/minimax",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("start_block", default=2, min=0, max=256,
                             tooltip="First block that runs compressed (Flash exports 2)."),
                io.Int.Input("end_block", default=28, min=1, max=256,
                             tooltip="First block at full resolution; Flash exports 42 - 14 = 28."),
                io.Int.Input("spatial_stride", default=2, min=2, max=8,
                             tooltip="Grid stride of the retained rows; Flash is trained at 2."),
                io.Float.Input("residual_gain", default=1.0, min=0.0, max=4.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, start_block: int, end_block: int, spatial_stride: int,
                residual_gain: float) -> io.NodeOutput:
        diffusion = getattr(model, "model", model).diffusion_model
        blocks = diffusion.blocks
        if not 0 < start_block < end_block < len(blocks):
            raise ValueError(
                "Flash token compression needs 0 < start_block < end_block < {} blocks".format(len(blocks)))

        patched = model.clone()
        shared = {"cache": {}, "state": {}}
        for index in range(start_block, end_block + 1):
            patched.set_model_patch_replace(
                FlashBlockPatch(shared, index, start_block, end_block, spatial_stride, residual_gain),
                "dit", "double_block", index)
        return io.NodeOutput(patched)


class LynnRealFlashExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LynnRealFlashTokenCompression]


async def comfy_entrypoint() -> LynnRealFlashExtension:
    return LynnRealFlashExtension()
