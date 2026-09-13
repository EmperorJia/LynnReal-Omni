"""Depth-distilled H3 VAE with trained tile geometry and unchanged temporal chunking."""
import json
import math
from pathlib import Path
import torch
from diffusers import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.vae import DecoderOutput

class LightVAE(torch.nn.Module):
    def __init__(self, core, settings, tile_batch=0, tile_layout=None):
        super().__init__()
        if settings["temporal_decode_protocol"] != "released-segment-prepadding-v3":
            raise ValueError("unsupported temporal decoding protocol")
        if tile_batch < 0:
            raise ValueError("tile batch must be nonnegative; zero batches all tiles")
        self.core, self.settings, self.tile_batch = core, settings, tile_batch
        self.tile_layout = tile_layout or settings.get("tile_layout", "adaptive")
        if self.tile_layout not in {"native", "adaptive"}:
            raise ValueError("tile layout must be native or adaptive")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("core"), name)

    @classmethod
    def from_pretrained(cls, path, tile_batch=0, *, tile_layout=None):
        from .weights import local_weights
        path = local_weights(path)
        settings = json.loads((path / "decode_config.json").read_text())
        core = AutoencoderKLMiniMaxH3.from_pretrained(path, torch_dtype=torch.float32, local_files_only=True)
        return cls(core, settings, tile_batch, tile_layout).eval().requires_grad_(False)

    def _geometry(self, frames, height, width):
        if self.tile_layout == "native":
            return (self.core.tile_sample_min_height, self.core.tile_sample_min_width,
                    self.core.tile_sample_min_overlap_height, self.core.tile_sample_min_overlap_width)
        s = self.settings
        if frames == 1:
            return s["image_tile_size"], s["image_long_axis_tile"], s["student_tile_overlap"], s["student_tile_overlap"]
        short, long = s["student_tile_size"], s["student_long_axis_tile"]
        overlap_short, overlap_long = s["student_short_axis_overlap"], s["student_long_axis_overlap"]
        if height < width:
            return short, long, overlap_short, overlap_long
        if height > width:
            return long, short, overlap_long, overlap_short
        overlap = max(overlap_short, overlap_long)
        return short, short, overlap, overlap

    def _decode_clip(self, latents):
        ratio = self.core.spatial_compression_ratio
        height, width = latents.shape[-2] * ratio, latents.shape[-1] * ratio
        tile_h, tile_w, overlap_h, overlap_w = self._geometry(latents.shape[2], height, width)
        ys, heights, y_overlaps = self._split_tiles(height, tile_h, overlap_h)
        xs, widths, x_overlaps = self._split_tiles(width, tile_w, overlap_w)
        tiles = [latents[..., y // ratio:(y + h) // ratio, x // ratio:(x + w) // ratio]
                 for y, h in zip(ys, heights) for x, w in zip(xs, widths)]
        # The native splitter produces equal tiles. Guard this before batching.
        if any(tile.shape != tiles[0].shape for tile in tiles):
            raise RuntimeError("native spatial tiles have inconsistent geometry")
        batch = self.tile_batch or len(tiles)
        decoded = []
        for i in range(0, len(tiles), batch):
            packed = torch.cat(tiles[i:i + batch], dim=0)
            output = self.core.decoder(self.core.post_quant_conv(packed))
            decoded.extend(output.split(latents.shape[0]))
        rows = [decoded[i:i + len(xs)] for i in range(0, len(decoded), len(xs))]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _split_tiles(self, length, size, overlap):
        if self.tile_layout == "native":
            return self.core._split_tiles(length, size, overlap)
        # The trained adaptive path expands a tile by at most one overlap,
        # avoiding near-identical crops at an axis just above a tile boundary.
        if length <= size + overlap:
            return [0], [length], []
        ratio = self.core.spatial_compression_ratio
        count = math.ceil(length / size)
        required = math.ceil((length + overlap * (count - 1)) / count / ratio) * ratio
        if required > size + overlap:
            raise ValueError("adaptive tile expansion exceeds one overlap")
        size = max(size, required)
        while size * count - overlap * (count - 1) < length:
            count += 1
        overlaps = [overlap] * (count - 1)
        remaining = size * count - sum(overlaps) - length
        for i in range(remaining // ratio):
            overlaps[i % (count - 1)] += ratio
        starts = [0]
        for value in overlaps:
            starts.append(starts[-1] + size - value)
        return starts, [size] * count, overlaps

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps):
        # The distilled geometry permits zero overlap; native H3's :-0 crop
        # would discard a whole tile. Match the trained LightX2V stitch order.
        rows = []
        for i, row in enumerate(tiles):
            columns = []
            for j, tile in enumerate(row):
                if i and height_overlaps[i - 1] > 0:
                    tile = self.core._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j and width_overlaps[j - 1] > 0:
                    tile = self.core._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1 and height_overlaps[i] > 0:
                    tile = tile[..., :-height_overlaps[i], :]
                if j < len(row) - 1 and width_overlaps[j] > 0:
                    tile = tile[..., :, :-width_overlaps[j]]
                columns.append(tile)
            rows.append(torch.cat(columns, dim=-1))
        return torch.cat(rows, dim=-2)

    @torch.inference_mode()
    def decode(self, latents, return_dict=True):
        single = latents.shape[2] == 1
        latents = latents.to(dtype=torch.float16)
        if single:
            latents = latents.expand(-1, -1, self.settings["image_context_tokens"], -1, -1).contiguous()
        with torch.autocast(latents.device.type, dtype=torch.float16):
            if latents.shape[2] < self.core.tokens_chunk_size + self.core.token_overlap:
                output = self._decode_clip(latents)
                start = self.settings["image_context_output_phase"] if single else self.core.frame_pre_padding
                output = output[:, :, start:start + 1] if single else output[:, :, start:]
            else:
                output = AutoencoderKLMiniMaxH3._decode(self, latents)
        output = output.float()
        return DecoderOutput(sample=output) if return_dict else (output,)
