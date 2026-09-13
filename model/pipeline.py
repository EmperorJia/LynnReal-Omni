"""Native H3 compression with the LynnReal unified denoiser and exact NFE accounting."""
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image, ImageOps
from diffusers import ComponentsManager
from diffusers.modular_pipelines.minimax_h3 import before_encoder
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3TextEncoderStep
from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks
from diffusers.utils import is_flash_attn_3_available
from .weights import local_weights, sha256, standard_transformer
from .flash import configure_flash
from .reference import AlignedReferenceSetup, AlignedReferenceLayout
from .conditioning import conditioning_layer, conditioning_device
from .cache import save_cache

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# LynnReal supports short clips. Keep H3's geometry, alignment and reference checks.
before_encoder.MINIMAX_H3_MIN_DURATION = 0.0
# Fifteen delivered seconds require 362 native padded frames at 24 fps.
before_encoder.MINIMAX_H3_MAX_DURATION = 362 / 24

def selected_blocks(reference, names, aligned_indices=()):
    blocks = MiniMaxH3Ref2VABlocks() if reference else MiniMaxH3Blocks()
    if aligned_indices:
        if not reference:
            raise ValueError("frame-aligned controls require the reference pipeline")
        blocks.sub_blocks["setup"] = AlignedReferenceSetup(aligned_indices)
        blocks.sub_blocks["prepare_layout"] = AlignedReferenceLayout()
    for name in list(blocks.sub_blocks):
        if name not in names:
            del blocks.sub_blocks[name]
    return blocks

def references(paths):
    result = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        result.append(MiniMaxH3Reference(**{"image" if path.suffix.lower() in IMAGE_SUFFIXES else "video": str(path)}))
    return result

def release_memory():
    gc.collect()
    torch.cuda.empty_cache()

def model_config(weights):
    return json.loads((local_weights(weights) / "inference_config.json").read_text())

def input_media(paths, native_keyframes=False):
    if not paths:
        return {}
    if native_keyframes:
        if len(paths) not in (1, 2) or any(Path(p).suffix.lower() not in IMAGE_SUFFIXES for p in paths):
            raise ValueError("native keyframes require one or two images")
        result = {}
        for key, path in zip(("image", "last_image"), paths):
            with Image.open(path) as image:
                result[key] = ImageOps.exif_transpose(image).convert("RGB")
        return result
    return {"references": references(paths)}

def encode_conditioning(weights, prompt, paths, height, width, frames, cache, *, native_keyframes=None, text_only=False, aligned_indices=(), resident=None, resident_device="cpu", reference_image_short_edge=None, reference_video_short_edge=None):
    """Conditioning cache includes text, input bytes, geometry and encoder config."""
    if resident_device not in {"cpu", "cuda"} or (resident_device == "cuda" and resident is None):
        raise ValueError("CUDA conditioning residency requires an explicit resident holder")
    native = model_config(weights).get("conditioning_abi") == "native_fl2va" if native_keyframes is None else native_keyframes
    if reference_image_short_edge is not None and (native or not paths or aligned_indices
            or reference_image_short_edge < 32 or reference_image_short_edge % 32):
        raise ValueError("reference image sizing requires unaligned Ref2VA inputs and a positive multiple of 32")
    if reference_video_short_edge is not None and (native or not paths or aligned_indices
            or reference_video_short_edge < 32 or reference_video_short_edge % 32):
        raise ValueError("reference video sizing requires unaligned Ref2VA inputs and a positive multiple of 32")
    encoder = (Path(weights) / "text_encoder").resolve()
    identity = {"prompt": prompt, "inputs": [sha256(p) for p in paths], "height": height, "width": width,
                "frames": frames, "mode": "reference" if paths else "t2v",
                "native_keyframes": native, "text_only": text_only, "aligned_indices": list(aligned_indices), "encoder_folder": str(encoder),
                "encoder_files": [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                                  for p in sorted(encoder.glob("*.safetensors"))],
                "encoder_config": sha256(Path(weights) / "text_encoder/config.json"),
                "processor_config": sha256(Path(weights) / "processor/preprocessor_config.json")}
    if reference_image_short_edge is not None:
        identity["reference_image_short_edge"] = reference_image_short_edge
    if reference_video_short_edge is not None:
        identity["reference_video_short_edge"] = reference_video_short_edge
    legacy_identity = dict(identity)
    if not paths:
        for field in ('height', 'width', 'frames'):
            identity.pop(field)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / f"{key}.pt"
    if not paths and not path.exists():
        legacy_key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
        legacy_path = Path(cache) / f'{legacy_key}.pt'
        if legacy_path.exists():
            saved = torch.load(legacy_path, weights_only=True, map_location='cpu')
            if saved['identity'] != legacy_identity:
                raise ValueError('legacy conditioning cache identity mismatch')
            save_cache({'identity': identity, 'tensors': saved['tensors']}, path)
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["identity"] != identity:
            raise ValueError("conditioning cache identity mismatch")
        return saved["tensors"], {"cache_hit": True, "seconds": 0.0, "key": key}
    contract = (str(Path(weights).resolve()), bool(paths), native, tuple(aligned_indices), resident_device, reference_image_short_edge, reference_video_short_edge)
    if resident and resident["contract"] != contract:
        raise ValueError("resident conditioner cannot change its model or task layout")
    if resident:
        pipe, manager = resident["pipe"], resident["manager"]
    else:
        manager = ComponentsManager()
        block = selected_blocks(not native, {"setup", "text_encoder"}, aligned_indices) if paths else MiniMaxH3TextEncoderStep()
        if reference_image_short_edge is not None or reference_video_short_edge is not None:
            from .reference import SizedReferenceSetup
            block.sub_blocks["setup"] = SizedReferenceSetup(reference_image_short_edge or 2048, reference_video_short_edge)
        pipe = block.init_pipeline(str(weights), components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(weights), local_files_only=True)
        # The native H3 feature extractor consumes layer 50 and already skips
        # later layers during its forward. They need not occupy GPU memory.
        from diffusers.modular_pipelines.minimax_h3.encoders import MINIMAX_H3_TEXT_ENCODER_LAYER
        decoder = pipe.text_encoder.model.language_model
        decoder.layers = type(decoder.layers)(list(decoder.layers[:MINIMAX_H3_TEXT_ENCODER_LAYER]))
    kwargs = {**input_media(paths, native), "height": height, "width": width, "num_frames": frames} if paths else {"keyframes": []}
    with conditioning_device(pipe) as staged:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with conditioning_layer(pipe.text_encoder.model):
            state = pipe(prompt=prompt, output=["prompt_embeds", "text_token_tags"], **kwargs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tensors = {k: v.cpu() for k, v in state.items()}
    if text_only:
        keep = tensors["text_token_tags"] == MINIMAX_H3_TEXT_TAG
        tensors["prompt_embeds"] = tensors["prompt_embeds"][:, keep].contiguous()
        tensors["text_token_tags"] = tensors["text_token_tags"][keep].contiguous()
    save_cache({"identity": identity, "tensors": tensors}, path)
    if resident is not None:
        pipe.to(resident_device)
        resident.update(pipe=pipe, manager=manager, contract=contract)
    del state, pipe, manager
    release_memory()
    return tensors, {"cache_hit": False, "seconds": seconds, "key": key, "staged_text_layers": staged}

class Pipeline:
    def __init__(self, weights, reference=False, aligned_indices=(), int8=False, fused=False, light_vae=None, adaln_cache=False, int8_gemm="torch", vae_attention=None, refinement=None, vae_offload=False, attention_backend="auto", compile_vae=False, dit_offload_blocks=0, native_keyframes=None, native_appearance=False, boundary_reuse_frames=None, fixed_native_head=False, head_only_boundary=False, reference_image_short_edge=None, reference_video_short_edge=None, fixed_head_reference_index=0, independent_target_head=False):
        self.weights = Path(weights)
        self.config = model_config(weights)
        self.transformer_record = (standard_transformer(weights)
                                   if self.config.get('variant') == 'standard' else None)
        if dit_offload_blocks < 0 or (dit_offload_blocks and not vae_offload):
            raise ValueError("DiT offload requires VAE offload and a nonnegative block count")
        if compile_vae and (light_vae is None or vae_offload):
            raise ValueError("decoder compilation requires a resident light VAE")
        if attention_backend == "auto":
            from .attention import select_attention
            attention_backend = select_attention()[0]
        if attention_backend not in {"native", "_flash_3", "flash", "_native_flash", "_native_cudnn"}:
            raise ValueError("unsupported attention backend")
        if attention_backend == "_flash_3" and not is_flash_attn_3_available():
            raise ValueError("_flash_3 requires an installed FlashAttention 3 package")
        self.attention_backend = attention_backend
        self.native_keyframes = self.config.get("conditioning_abi") == "native_fl2va" if native_keyframes is None else native_keyframes
        self.conditioned = reference
        self.reference = reference and not self.native_keyframes
        keep = {"setup", "prepare_layout", "prepare_latents", "set_timesteps", "denoise", "decode"}
        if self.reference:
            keep.add("reference_encoder")
        elif reference:
            keep.add("vae_encoder")
        self.manager = ComponentsManager()
        blocks = selected_blocks(self.reference, keep, aligned_indices)
        if reference_image_short_edge is not None or reference_video_short_edge is not None:
            if not self.reference or aligned_indices:
                raise ValueError('reference resolution ablation requires unaligned Ref2VA images')
            from .reference import SizedReferenceSetup
            blocks.sub_blocks['setup'] = SizedReferenceSetup(reference_image_short_edge or 2048, reference_video_short_edge)
        self.boundary_cache = self.appearance_encoder = None
        if (native_appearance or boundary_reuse_frames is not None) and (
                not reference or not (self.native_keyframes or independent_target_head) or not vae_offload):
            raise ValueError("native continuation options require keyframes and VAE phase offload")
        if vae_offload:
            if int8:
                raise ValueError("VAE phase offload requires BF16 DiT")
            from .offload import configure_vae_offload
            configure_vae_offload(blocks, dit_offload_blocks)
        if head_only_boundary and not fixed_native_head:
            raise ValueError('head-only boundary requires a fixed target head')
        if native_appearance:
            from .native_appearance import configure_initial_appearance
            self.appearance_encoder = configure_initial_appearance(blocks, head_only_boundary)
        if independent_target_head and not (fixed_native_head and self.reference):
            raise ValueError("an independent target head requires fixed-head Ref2VA")
        self.independent_target_head = independent_target_head
        if fixed_native_head:
            if not native_appearance and not (self.reference and vae_offload):
                raise ValueError("fixed native head requires initial appearance conditioning")
            from .native_head import configure_fixed_head
            configure_fixed_head(blocks, fixed_head_reference_index, independent_target_head)
        if boundary_reuse_frames is not None:
            from .native_boundary import configure_boundary_reuse
            self.boundary_cache = configure_boundary_reuse(blocks, boundary_reuse_frames)
        if refinement is not None:
            from .refinement import configure_refinement
            configure_refinement(blocks, **refinement)
        self.pipe = blocks.init_pipeline(str(weights), components_manager=self.manager)
        if int8:
            self.pipe.set_progress_bar_config(disable=True)
        if self.transformer_record is not None:
            name = 'transformer_ref' if self.reference else 'transformer'
            dit_specs = {k: v for k, v in self.pipe._component_specs.items()
                         if k in {'transformer', 'transformer_ref'}}
            if list(dit_specs) != [name] or dit_specs[name].subfolder != 'transformer':
                raise RuntimeError('Standard must load exactly one DiT from transformer/')
            self.transformer_record = dict(self.transformer_record, component=name)
        # init_pipeline deep-copies blocks; use the executing cache, not its template.
        runtime = self.pipe._blocks.sub_blocks
        self.fixed_head_denoise = runtime['denoise'] if fixed_native_head else None
        if fixed_native_head and self.fixed_head_denoise.encoder is not runtime['reference_encoder' if self.reference else 'vae_encoder']:
            raise RuntimeError('fixed head must use the executing image encoder')
        if native_appearance:
            self.appearance_encoder = runtime["vae_encoder"]
        if boundary_reuse_frames is not None:
            self.boundary_cache = runtime["reference_encoder" if self.reference else "vae_encoder"].cache
            if self.boundary_cache is not runtime["decode"].sub_blocks["video"].cache:
                raise RuntimeError("boundary encoder and decoder must share their runtime cache")
        if self.config.get("variant") == "flash":
            if self.config.get("quantization") != "int8" or not int8:
                raise ValueError("Flash requires its trained INT8 weights and INT8 inference")
            from .flash_int8 import load_transformer
            self.pipe.update_components(transformer=load_transformer(self.weights / "transformer", gemm=int8_gemm))
        elif self.config.get("quantization") == "w4f8":
            if int8 or fused or dit_offload_blocks:
                raise ValueError("Packed Flash uses its own W4F8 kernels; INT8/fusion/block-offload are unsupported")
            from .w4f8 import load_transformer
            self.pipe.update_components(transformer=load_transformer(self.weights / "transformer"))
        self.pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16},
                                  pretrained_model_name_or_path=str(weights), local_files_only=True)
        from .output import configure_video_output
        configure_video_output(self.pipe.video_processor)
        self.vae_record = None
        if light_vae is not None:
            from .light_vae import LightVAE
            vae = LightVAE.from_pretrained(local_weights(light_vae))
            self.pipe.update_components(vae=vae)
            self.vae_record = {"folder": str(Path(light_vae).resolve()), "decode_config": vae.settings,
                "config_sha256": sha256(Path(light_vae) / "config.json")}
        self.pipe.to("cuda")
        if vae_attention is not None:
            vae = getattr(self.pipe.vae, "core", self.pipe.vae)
            vae.set_attention_backend(vae_attention)
        if compile_vae:
            decoder = self.pipe.vae.core.decoder
            from .acceleration import compile_decoder
            self.vae_record["compilation"] = compile_decoder(decoder)
        self.transformer = self.pipe.transformer_ref if self.reference else self.pipe.transformer
        self.transformer.eval().requires_grad_(False)
        self.transformer.set_attention_backend(attention_backend)
        if self.config.get("variant") == "flash":
            configure_flash(self.transformer, self.config)
        self.quantization = (json.loads((self.weights / "transformer/quantization_config.json").read_text())
                             if self.config.get("quantization") in {"w4f8", "int8"} else None)
        if int8 and self.config.get("quantization") != "int8":
            from .int8 import quantize_transformer
            self.quantization = quantize_transformer(self.transformer, gemm=int8_gemm)
        self.fusion = None
        if fused:
            if self.config.get("variant") not in {"standard", "flash"}:
                raise ValueError("operator fusion requires a standard or Flash model")
            from .fusion import enable_fusion
            self.fusion = enable_fusion(self.transformer, fused_quant=int8)
        self.events = []
        from .timing import measure_decoder
        self.decoder_events = []
        measure_decoder(self.pipe.vae, self.decoder_events)
        self.transformer.register_forward_pre_hook(self._start_forward)
        self.transformer.register_forward_hook(self._end_forward)
        self.modulation_cache = None
        if adaln_cache:
            from .fusion import cache_time_modulation
            self.modulation_cache = cache_time_modulation(self.transformer)

    def _start_forward(self, *_):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.events.append((start, end))

    def _end_forward(self, *_):
        self.events[-1][1].record()

    @torch.inference_mode()
    def generate(self, conditioning, paths, height, width, frames, seed, steps, keep_latents=False, boundary_image=None):
        if frames < 1 or steps < 1:
            raise ValueError("frames and denoiser forwards must be positive")
        if bool(paths) != self.conditioned:
            raise ValueError("request and pipeline conditioning modes differ")
        if self.independent_target_head != (boundary_image is not None):
            raise ValueError("independent target head and boundary image must be supplied together")
        if boundary_image is not None:
            if boundary_image.size != (width, height):
                raise ValueError("target boundary must match the output canvas")
            self.fixed_head_denoise.encoder.boundary_image = boundary_image
        self.events.clear()
        self.decoder_events.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        kwargs = input_media(paths, self.native_keyframes)
        outputs = ["videos", "audio", "sampling_rate"]
        if keep_latents:
            outputs += ["latents", "audio_latents", "num_latent_frames", "latent_height", "latent_width"]
        if self.conditioned:
            outputs += ["num_condition_video_rows", "num_condition_audio_rows"]
        state = self.pipe(prompt_embeds=conditioning["prompt_embeds"].to("cuda"),
            text_token_tags=conditioning["text_token_tags"], height=height, width=width, num_frames=frames,
            num_inference_steps=steps + 1, generator=torch.Generator().manual_seed(seed),
            output_type="pil", output=outputs, **kwargs)
        torch.cuda.synchronize()
        timing = {"generation_and_decode_ms": (time.perf_counter() - started) * 1000,
                  "dit_forward_ms": [a.elapsed_time(b) for a, b in self.events],
                  "actual_dit_forwards": len(self.events), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        if len(self.events) != steps:
            raise RuntimeError(f"expected {steps} actual DiT forwards, observed {len(self.events)}")
        if self.conditioned and int(state["num_condition_video_rows"]) <= 0:
            raise RuntimeError("reference input produced no conditioning rows")
        timing["dit_ms"] = sum(timing["dit_forward_ms"])
        timing["video_decoder_ms"] = sum(a.elapsed_time(b) for a, b in self.decoder_events)
        return state, timing
