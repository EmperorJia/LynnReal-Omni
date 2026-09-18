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

The same node also hands every block its attention callable, which is where the release's
FlashAttention 3 path is injected (ComfyUI's own selection stops at FA2); see
``attention_fa3.py``. ``LYNNREAL_FA3=0`` keeps the stock attention.
"""

import bisect
import logging
import os

import torch
from typing_extensions import override

import comfy.model_prefetch
from comfy_api.latest import ComfyExtension, io

from . import adaln_exact, attention_fa3, bench, fast_blocks

# Packed rows the Flash export tags as video: keyframe rows, reference images and the video
# streams themselves all take the video token tag upstream. Audio rows keep the audio tag and
# text rows the text tag, so neither is ever dropped.
VIDEO_KINDS = ("cond", "ref_img", "video")

# A 15 s clip packs ~109k rows; that is where ComfyUI's INT8 MLP overflows 32-bit indexing, and
# the pack now splits those GEMMs (see fast_blocks), which is what makes 15 s run at all (44.5 s
# warm against the release's 38.8 s). Anything longer is untested and would fault the context,
# so refuse it with a clear message instead of crashing.
LONG_CLIP_ROWS = 130000


def _check_clip_length(rows: int) -> None:
    if rows > LONG_CLIP_ROWS and os.environ.get("LYNNREAL_ALLOW_LONG_CLIPS") != "1":
        raise ValueError(
            "This clip packs {} rows; the accelerated Flash path is only validated up to {} "
            "(10 s at 1344x768, ~62k rows). Longer clips crash inside comfy-kitchen's INT8 "
            "kernels on this build (an illegal access that can take the GPU with it), so this "
            "request is refused rather than run. Set LYNNREAL_ALLOW_LONG_CLIPS=1 to try anyway."
            .format(rows, LONG_CLIP_ROWS))


def _layout_report(layout, positions, tags, groups, keep, stride) -> None:
    """``LYNNREAL_LAYOUT_DEBUG=1``: what the packed layout is made of, and what compression kept.

    A reference image that does not land on the latent grid shows up here first: the counts per
    segment kind say how many rows the conditioning added, and the per-axis alphabet says
    whether the video rows still share a regular grid for `spatial_layout` to halve.
    """
    if os.environ.get("LYNNREAL_LAYOUT_DEBUG") != "1":
        return
    dump = os.environ.get("LYNNREAL_LAYOUT_DUMP")
    if dump:
        # Hand the exact packed coordinates to a script, so the release's own `spatial_layout` can be run on this
        # layout and compared with ours (see tools/probe_release_layout.py).
        torch.save({"positions": positions.cpu(), "tags": tags.cpu(),
                    "segments": [(int(a), int(b), k) for a, b, k in layout.segments],
                    "signature": layout.signature}, dump)
    kinds = {}
    spans = []
    for start, stop, kind in layout.segments:
        kinds[kind] = kinds.get(kind, 0) + (stop - start)
        spans.append("%s:%d-%d" % (kind, start, stop))

    def _kept(rows):
        coordinates = positions[rows]
        hits = torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
        sizes = []
        for axis in (1, 2):
            alphabet = torch.unique(coordinates[:, axis], sorted=True)
            retained = alphabet[::stride]
            if retained[-1] != alphabet[-1]:
                retained = torch.cat((retained, alphabet[-1:]))
            sizes.append((alphabet.numel(), retained.numel()))
            hits &= torch.isin(coordinates[:, axis], retained)
        return int(hits.sum()), sizes

    detail = []
    for index, rows in enumerate(groups):
        kept, sizes = _kept(rows)
        detail.append("group%d rows=%d kept=%d(%d%%) alphabets=%s"
                      % (index, rows.numel(), kept, round(100 * kept / max(rows.numel(), 1)),
                         sizes))
    logging.info("LYNNREAL_LAYOUT rows=%d kept=%d video_rows=%d groups=%d kept_video=%d "
                 "kinds=%s | %s | %s", tags.numel(), keep.numel(), int((tags == 0).sum()),
                 len(groups), int((tags[keep] == 0).sum()), kinds, " ".join(spans),
                 " ".join(detail))


def video_tags(layout, device):
    """Rows the model reads as video modality: keyframes, references and the generated stream."""
    tags = torch.ones(layout.seq_len, dtype=torch.long, device=device)
    for start, stop, kind in layout.segments:
        if kind in VIDEO_KINDS:
            tags[start:stop] = 0
    return tags


def compression_groups(layout, device):
    """The row spans the stride is applied to, each on its own spatial grid.

    A `ref2va` reference is prepared at its own resolution -- the release gives images a 2048 pixel short edge -- and
    carries its *own* aspect-normalised spatial grid, so two references and the generated stream are three unrelated
    coordinate alphabets. `spatial_layout` halves one alphabet; handing it the union of three is what makes it drop
    the generated rows (see `spatial_layout`). Each reference is therefore its own group, strided on its own grid:
    every group keeps about a quarter of its rows, which is the density the stride was trained at.

    `LYNNREAL_STRIDE_REFS=0` instead leaves references (and any other conditioning span that is not the generated
    stream) at full resolution, which costs rows in the middle blocks but keeps every reference pixel.
    """
    groups = []
    stride_refs = os.environ.get("LYNNREAL_STRIDE_REFS", "1") != "0"
    for start, stop, kind in layout.segments:
        if kind == "video" or (stride_refs and kind in VIDEO_KINDS):
            groups.append(torch.arange(start, stop, device=device))
    return groups


def _layout_key(layout, device):
    """Identify a packed layout without comparing its 100k-row position table.

    The signature covers the text/latent/audio sizes and the row count and segment table cover
    the conditioning blocks (keyframes, references), which is everything `spatial_layout` reads.
    """
    return (layout.signature, layout.seq_len, len(layout.segments), device.type, device.index)


def spatial_layout(positions, tags, stride, groups=None):
    """Keep both grid boundaries; map dropped video rows to same-time anchors.

    ``groups`` is the list of row spans that each carry their own spatial grid: the generated stream, and -- when
    references are strided too -- one per reference. The stride is applied inside each group, to that group's own
    coordinate alphabet. Sampling the union of two unrelated grids instead is what drops the generated video: the
    two alphabets interleave, "every other sorted coordinate" lands on one of them, and the rows that need the
    other lose every column they have. Rows outside every group are left alone.
    """
    if positions.ndim != 2 or tags.ndim != 1 or positions.shape != (tags.numel(), 3):
        raise ValueError("Flash requires a single shared packed layout")
    video = tags == 0
    if not bool(video.any()):
        raise ValueError("Flash requires video tokens")
    if groups is None:
        groups = [video.nonzero().flatten()]

    spatial = torch.zeros_like(video)
    compressible = torch.zeros_like(video)
    for rows in groups:
        if not rows.numel():
            continue
        coordinates = positions[rows]
        hits = torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
        for axis in (1, 2):
            alphabet = torch.unique(coordinates[:, axis], sorted=True)
            retained = alphabet[::stride]
            if retained[-1] != alphabet[-1]:
                retained = torch.cat((retained, alphabet[-1:]))
            hits &= torch.isin(coordinates[:, axis], retained)
        spatial[rows] = hits
        compressible[rows] = True
    compressible &= video
    compressible = _guard_degenerate(compressible, spatial)
    if not bool(compressible.any()):
        # Nothing is sampled this request: keep every row and let the middle blocks run at full resolution.
        spatial = torch.ones_like(video)
    mask = ~compressible | spatial
    keep = mask.nonzero().flatten()
    inverse = torch.empty_like(tags, dtype=torch.long)
    inverse[keep] = torch.arange(keep.numel(), device=tags.device)
    dropped = (~mask).nonzero().flatten()
    anchors = (compressible & mask).nonzero().flatten()
    target = positions[anchors].float()
    for rows in dropped.split(1024):
        source = positions[rows].float()
        distance = torch.cdist(source[:, 1:], target[:, 1:])
        distance.masked_fill_(source[:, None, 0] != target[None, :, 0], float("inf"))
        inverse[rows] = inverse[anchors[distance.argmin(dim=1)]]
    return keep, inverse


# The stride keeps every other coordinate on each of the two spatial axes, so a healthy layout retains about a
# quarter of the on-grid video rows. Anything far below that means the retained coordinates and the rows they were
# meant to keep do not agree -- the failure mode that turns a render into stripe noise -- and running the middle
# blocks on a sequence that small is worse than not compressing at all.
MINIMUM_KEPT_FRACTION = 0.15


def _guard_degenerate(compressible, spatial):
    """Give up on sampling this request when the stride selects almost nothing.

    The model then runs its middle blocks at full resolution -- slower for that request, but the render is still the
    clip the prompt asked for, and the reason is logged with the numbers that produced it.
    """
    on_grid = int(compressible.sum())
    kept = int((compressible & spatial).sum())
    if not on_grid or kept >= MINIMUM_KEPT_FRACTION * on_grid:
        return compressible
    logging.warning(
        "LynnReal: the Flash token stride kept only %d of %d on-grid video rows (expected about %d); running "
        "this request at full resolution instead of compressing. The packed layout's spatial grids disagree -- "
        "a reference whose aspect-normalised grid differs from the generated video's, most likely.",
        kept, on_grid, on_grid // 4)
    return torch.zeros_like(compressible)


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

    def __init__(self, shared, index, start, end, stride, gain, attention=None, compress=True,
                 block=None):
        self.shared = shared
        self.index = index
        self.start = start
        self.end = end
        self.stride = stride
        self.gain = gain
        self.attention = attention
        self.compress = compress
        self.block = block

    def _run(self, inner, original):
        """Call the block, through the fused kernels when they are verified on this GPU."""
        if self.block is not None and fast_blocks.usable():
            try:
                if os.environ.get("LYNNREAL_FAST_DEBUG") == "1":
                    fast_blocks.debug_compare(
                        self.block, inner["img"], inner["t_emb"], inner["mod_segments"],
                        inner["rope_freqs"], inner["transformer_options"],
                        inner.get("attention"), original(inner)["img"])
                return {"img": fast_blocks.block_forward(
                    self.block, inner["img"], inner["t_emb"], inner["mod_segments"],
                    inner["rope_freqs"], inner["transformer_options"], inner.get("attention"))}
            except Exception as error:
                fast_blocks._disable(f"{type(error).__name__}: {error}")
        if self.block is not None:
            rows = inner["img"].shape[0]
            if fast_blocks.needs_chunking(self.block, rows):
                # long clip without the fused kernels (e.g. DynamicVRAM disabled them): use the
                # stock math with only the MLP split, which ComfyUI's INT8 kernels can address
                return {"img": fast_blocks.stock_block_forward(
                    self.block, inner["img"], inner["t_emb"], inner["mod_segments"],
                    inner["rope_freqs"], inner["transformer_options"], inner.get("attention"))}
            fast_blocks.check_stock_mlp(self.block, rows)
        return original(inner)

    def __call__(self, args, extra_args):
        original = extra_args["original_block"]
        if not self.compress:
            inner = dict(args)
            if self.attention is not None:
                inner["attention"] = self.attention
            return self._run(inner, original)
        state = self.shared["state"]
        if self.index == self.end:
            # `original_block` already answers with the `{"img": ...}` mapping the model reads.
            return self._restore(args, original, state)
        if self.index == self.start:
            _check_clip_length(args["img"].shape[0])
            with comfy.model_prefetch.pause_malloc_graph():
                self._compress(args, state)
        inner = dict(args)
        if state:  # inside [start, end): the compressed stream is live
            inner["rope_freqs"] = state["rope_freqs"]
            inner["mod_segments"] = state["mod_segments"]
            if self.index == self.start:
                inner["img"] = state["compressed"]
        if self.attention is not None:
            inner["attention"] = self.attention
        return self._run(inner, original)

    def _compress(self, args, state):
        layout = args["layout"]
        full = args["img"]
        cache = self.shared["cache"]
        key = _layout_key(layout, full.device)
        entry = cache.get(key)
        if entry is None:
            # `PackedLayout` keeps its position table on the host; the row selection has to land
            # on the device the hidden state lives on. The map only depends on the packed
            # layout's signature, so it is built once per shape instead of once per block.
            positions = layout.position_ids.to(full.device)
            tags = video_tags(layout, full.device)
            groups = compression_groups(layout, full.device)
            keep, inverse = spatial_layout(positions, tags, self.stride, groups)
            _layout_report(layout, positions, tags, groups, keep, self.stride)
            cache.clear()
            entry = cache[key] = {"keep": keep, "inverse": inverse}
        keep, inverse = entry["keep"], entry["inverse"]
        bench.note_shapes(full.shape[0], keep.numel(), layout.signature)

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
        if self.attention is not None:
            inner["attention"] = self.attention
        return self._run(inner, original)

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
        bench.install()
        attention_fa3.probe_once()
        # A selector-style adaLN table (the locked Flash build) needs the projection to run in the
        # model dtype; detection is structural, so a plain curve checkpoint is left alone.
        adaln_exact.install(patched)
        use_fa3 = attention_fa3.usable()
        # diagnostics only: LYNNREAL_NO_COMPRESSION=1 changes the arithmetic (the Flash DiT is
        # trained with compression) and exists to price the patch itself
        compress = os.environ.get("LYNNREAL_NO_COMPRESSION") != "1"
        fast_blocks.probe(blocks[0])
        # Every block is replaced: the compressed band also swaps the fused stream, and all
        # blocks take the attention callable (FA3 when available).
        for index in range(len(blocks)):
            attention = attention_fa3.make_dit_attention(blocks[index].attn) if use_fa3 else None
            patched.set_model_patch_replace(
                FlashBlockPatch(shared, index, start_block, end_block, spatial_stride, residual_gain,
                                attention, compress, blocks[index]),
                "dit", "double_block", index)
        if use_fa3:
            logging.info("LynnReal: Flash blocks 0-%d run with FlashAttention 3 (compression on "
                         "%d-%d).", len(blocks) - 1, start_block, end_block)
        return io.NodeOutput(patched)


class LynnRealFlashExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LynnRealFlashTokenCompression]


async def comfy_entrypoint() -> LynnRealFlashExtension:
    return LynnRealFlashExtension()
