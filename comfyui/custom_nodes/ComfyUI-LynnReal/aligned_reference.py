"""LynnReal frame-aligned reference control for MiniMax-H3 Ref2VA.

Port of `model/reference.py::AlignedReferenceSetup` from the LynnReal release:
the block `model/pipeline.py::selected_blocks` swaps into the reference pipeline
whenever a case marks a video reference in `aligned_indices` -- which is every
`test/demo/cases/{pose,hand}/*.json` case.

That block is a diffusers ModularPipeline step. It reads `block.num_frames`,
`block.height`, `block.width` and rewrites the prepared reference in place, so
it cannot be called from here: ComfyUI never builds a `PipelineState`, and its
H3 implementation (`comfy/ldm/minimax/model.py`) is a separate port whose layout
is `PackedLayout`, not diffusers' `MiniMaxH3PackedSequence`.

The pixels it produces are a pure function of (frames, fps, target canvas,
target frame count), so those three passes run here on the decoded frames:

1. `resample_reference_frames` -- onto MiniMax-H3's own 24 fps grid, dropping
   and repeating whole frames exactly as `ffmpeg`'s `fps` filter did;
2. truncate to the target frame count;
3. rescale onto the *target* canvas and hold the last frame to reach the count.

Step 3 is where this differs from ComfyUI's own reference video input, which
leaves a clip on the canvas its own aspect ratio resolves to (`adapt_canvas`).
It is also what makes `AlignedReferenceLayout`'s invariant hold: the control
block's latent geometry then equals the target's, so its rows share the target
spatial grid and sit a constant positive lead ahead of it on the rotary clock.
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from typing_extensions import override

import av

from comfy_api.latest import ComfyExtension, io

# MiniMax-H3's own clock and video-VAE frame grid; identical to
# `MINIMAX_H3_FPS`, `MINIMAX_H3_FRAMES_PER_CHUNK` and `MINIMAX_H3_LATENTS_PER_CHUNK`.
H3_FPS = 24.0
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5


def align_frame_count(n: int) -> int:
    """Snap a frame count up to the next `17 * n + 5` the video VAE can encode."""
    n = max(H3_LATENTS_PER_CHUNK, int(n))
    while n % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        n += 1
    return n


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Put decoded reference frames on MiniMax-H3's 24 fps grid.

    A verbatim port of `diffusers.modular_pipelines.minimax_h3.packing_ref2va.resample_reference_frames`:
    every source frame lands on the output slot its timestamp rounds to and a slot
    holds the last frame that landed on it, so a frame whose successor shares its
    slot is dropped and one whose successor skips slots is repeated. The stream end
    rounds onto the grid the same way, which is what fixes the output length at
    `round(num_frames * 24 / fps)`. Frames already at 24 fps flow through untouched.
    """
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    if fps == H3_FPS:
        return frames

    scale = H3_FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    return np.repeat(frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)


def decode_reference_video(video):
    """Decode a file-backed clip exactly as the release did, or return `None`.

    The release's `decode_reference_video` reads every frame with PyAV's
    `to_ndarray(format="rgb24")` and undoes the display-matrix rotation. ComfyUI's
    own loader instead decodes through a planar float format, which lands within a
    level or two of it almost everywhere but differs on about 2% of pixels, so
    taking the release's own path is what makes this node reproduce the demo's
    control frames exactly.

    Only a plain, untrimmed, uncropped file-backed clip takes this path; anything
    else (a `Video Slice`/`VideoCrop` upstream, or an in-memory video) falls back
    to ComfyUI's `get_components()` so those edits are never silently dropped.
    """
    get_source = getattr(video, "get_stream_source", None)
    if get_source is None:
        return None
    try:
        source = get_source()
        start_time, duration = video.get_active_trim_window()
    except Exception:
        return None
    if not isinstance(source, (str, os.PathLike)) or start_time or duration:
        return None

    try:
        with av.open(os.fspath(source)) as container:
            stream = container.streams.video[0]
            decoded, rotation = [], 0.0
            for frame in container.decode(stream):
                # The display matrix rotation rides on every frame of its stream.
                rotation = frame.rotation
                decoded.append(frame.to_ndarray(format="rgb24"))
            fps = float(stream.average_rate or stream.guessed_rate)
    except Exception:
        return None
    if not decoded:
        return None

    frames = np.stack(decoded)
    turns = round(rotation / 90.0) % 4
    if turns:
        frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
    # An upstream crop would not survive the raw file decode; hand it back instead.
    try:
        width, height = video.get_dimensions()
    except Exception:
        return None
    if (width, height) != (frames.shape[2], frames.shape[1]):
        return None
    return frames, fps


class LynnRealAlignedReference(io.ComfyNode):
    """Frame-aligned control clip: 24 fps, exact frame count, target canvas.

    The control clip reaches the reference node as pixels on the *target* canvas
    and on the target's frame count, so `ref_videos.ref_video_0` reproduces the
    demo's aligned reference instead of a clip on its own canvas.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LynnRealAlignedReference",
            display_name="LynnReal Aligned Reference (pose/hand control clip)",
            description=(
                "Frame-aligned control clip for MiniMax H3 Ref2VA: resamples to the model's 24 fps "
                "(dropping and repeating whole frames, as the reference decode did), truncates to the "
                "generation's frame count, rescales onto the target canvas and holds the last frame to "
                "reach that count. Pass the same width / height / length as the reference node."
            ),
            category="conditioning/minimax",
            inputs=[
                io.Video.Input("video", tooltip="Control clip at any frame rate; it is resampled to 24 fps."),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32,
                             tooltip="Target canvas width, the same value the reference node generates at."),
                io.Int.Input("height", default=768, min=32, max=16384, step=32,
                             tooltip="Target canvas height, the same value the reference node generates at."),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Generation frame count at 24 fps; snapped up to the model's 17k+5 grid."),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, video, width: int, height: int, length: int) -> io.NodeOutput:
        decoded = decode_reference_video(video)
        if decoded is not None:
            frames, fps = decoded
        else:
            components = video.get_components()
            fps = float(components.frame_rate)
            # ComfyUI decodes to float; uint8 over 255 round-trips exactly.
            frames = (components.images * 255.0).round().clamp_(0, 255).to(torch.uint8).numpy()

        num_frames = align_frame_count(length)
        frames = resample_reference_frames(frames, fps)[:num_frames]
        if not len(frames):
            raise ValueError("empty frame control: the control clip decoded to no frames")
        if frames.shape[1:3] != (height, width):
            tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear",
                                   align_corners=False, antialias=True)
            frames = tensor.round().clamp_(0, 255).byte().permute(0, 2, 3, 1).numpy()
        if len(frames) < num_frames:
            frames = np.concatenate(
                (frames, np.repeat(frames[-1:], num_frames - len(frames), axis=0))
            )

        images = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0
        return io.NodeOutput(images)


class LynnRealAlignedReferenceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LynnRealAlignedReference]


async def comfy_entrypoint() -> LynnRealAlignedReferenceExtension:
    return LynnRealAlignedReferenceExtension()
