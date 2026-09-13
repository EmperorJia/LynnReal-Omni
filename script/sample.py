"""Generate through the release pipeline; preserve prompts, references and measured NFE."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=["t2v", "ti2v", "reference", "image-edit", "video-edit"], required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--aligned-reference", type=int, action="append", default=[],
                        help="zero-based video reference index aligned to the target timeline")
    parser.add_argument("--reference-image-short-edge", type=int,
                        help="image conditioning resolution, a positive multiple of 32 (unaligned references only)")
    parser.add_argument("--reference-video-short-edge", type=int,
                        help="video conditioning resolution, preserving reference frame order (unaligned references only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--native-keyframes", action="store_true", default=None,
                        help="use native first/last-frame conditioning with a unified checkpoint")
    parser.add_argument("--fast", action="store_true", help="probe and select supported FA3 and fused CUDA operators")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--int8", action="store_true", help="experimental dense W8A8; record quality and latency separately")
    parser.add_argument("--int8-gemm", choices=["torch", "triton"], default="torch")
    parser.add_argument("--fused", action="store_true", help="experimental dense operator fusion")
    parser.add_argument("--adaln-cache", action="store_true", help="cache exact time-only modulation tables for immutable weights")
    parser.add_argument("--light-vae", type=Path, help="optional distilled VAE bundle for paired decoder comparisons")
    parser.add_argument("--vae-tiles", choices=("native", "adaptive"), help="explicit lightweight decoder tile layout")
    parser.add_argument("--refine-strength", type=float, help="Flash T2V second-pass starting noise fraction")
    parser.add_argument("--refine-steps", type=int, default=2)
    parser.add_argument("--refine-first-height", type=int, choices=(544, 640, 768), default=544,
                        help="first-pass canvas height cap for spatial refinement")
    parser.add_argument("--refine-schedule", choices=("trained-tail", "linear"), default="trained-tail")
    parser.add_argument("--compile-vae", action="store_true", help="experimental light-decoder compilation; warm per shape, RGB may differ slightly")
    parser.add_argument("--vae-attention", choices=["auto", "native", "_flash_3", "flash", "_native_flash", "_native_cudnn"], help="explicit VAE attention backend ablation")
    parser.add_argument("--vae-tile-batch", type=int, default=0, help="light decoder tile batch; 0 batches all tiles")
    parser.add_argument("--vae-offload", action="store_true", help="stage unused BF16 DiT blocks on CPU during VAE phases")
    parser.add_argument("--dit-offload-blocks", type=int, default=0,
                        help="with --vae-offload, load this many DiT blocks only for their forward calls")
    parser.add_argument("--image-frame", type=int, default=0,
                        help="frame selected for image-edit PNG; the full generated clip is also retained")
    parser.add_argument("--profile", type=Path, help="record the last iteration's CPU/CUDA trace; its timing includes profiler overhead")
    parser.add_argument("--attention-backend", choices=("auto", "native", "_flash_3", "flash", "_native_flash", "_native_cudnn"), default="auto",
                        help="auto probes FA3, FA2 and native Flash SDPA; the selected backend is recorded")
    args = parser.parse_args()
    reference_sizes = (args.reference_image_short_edge, args.reference_video_short_edge)
    if any(size is not None for size in reference_sizes):
        if any(size is not None and (size < 32 or size % 32) for size in reference_sizes):
            parser.error("reference short edges must be positive multiples of 32")
        if not args.reference or args.aligned_reference or args.native_keyframes or args.mode == "ti2v":
            parser.error("reference sizing requires unaligned references without native keyframes")
    if args.vae_tile_batch < 0 or (args.vae_tile_batch and args.light_vae is None):
        parser.error("--vae-tile-batch requires a light VAE and a nonnegative value")
    if args.dit_offload_blocks < 0 or (args.dit_offload_blocks and not args.vae_offload):
        parser.error("--dit-offload-blocks requires --vae-offload and a nonnegative count")
    if args.adaln_cache and not args.fused:
        parser.error("--adaln-cache requires --fused")
    if args.compile_vae and (args.light_vae is None or args.vae_offload):
        parser.error("--compile-vae requires --light-vae without --vae-offload")
    if args.int8_gemm != "torch" and not args.int8:
        parser.error("--int8-gemm requires --int8")
    if (args.mode == "t2v") == bool(args.reference):
        parser.error("t2v takes no references; conditioned modes require references")
    if args.mode == "ti2v" and (len(args.reference) not in [1, 2] or any(p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"] for p in args.reference)):
        parser.error("ti2v expects a first-frame picture and an optional last-frame picture")
    if min(args.height, args.width, args.frames, args.steps, args.repeats) < 1 or args.warmups < 0:
        parser.error("dimensions, frames, steps and repeats must be positive")
    if args.refine_strength is not None and (args.mode != "t2v" or args.repeats != 1
            or not 0 < args.refine_strength < 1 or not 1 <= args.refine_steps <= 4):
        parser.error("refinement requires one T2V call, strength in (0,1), and 1..4 extra steps")
    if (args.refine_strength is not None and args.refine_schedule == "trained-tail"
            and args.refine_steps > 1 and args.refine_strength <= 12/14):
        parser.error("multi-step trained-tail refinement requires strength > 12/14")
    if args.output.exists():
        parser.error("output exists; select a new experiment path")
    if args.mode == "image-edit":
        if args.output.suffix.lower() != ".png" or not 0 <= args.image_frame < args.frames:
            parser.error("image-edit requires a PNG output and an image frame within the generated clip")
        if args.output.with_suffix(".mp4").exists():
            parser.error("image-edit companion clip already exists")
    elif args.output.suffix.lower() != ".mp4":
        parser.error("video outputs must use .mp4")
    if len(set(args.aligned_reference)) != len(args.aligned_reference) or any(
            i < 0 or i >= len(args.reference) for i in args.aligned_reference):
        parser.error("aligned reference indices must be unique and within the reference list")
    import torch
    acceleration = None
    if args.fast:
        from model.acceleration import fastest_available
        acceleration = fastest_available(None if args.attention_backend == "auto" else args.attention_backend)
        if args.attention_backend == "auto":
            args.attention_backend = acceleration["attention_backend"]
        acceleration["attention_backend"] = args.attention_backend
        args.fused = acceleration["fused"]
        variant = json.loads((args.weights / "inference_config.json").read_text())["variant"]
        args.adaln_cache = args.fused and (args.adaln_cache or variant in {"standard", "flash"})
        if args.int8 and not args.fused:
            args.int8_gemm = "torch"
        if args.compile_vae and not args.fused:
            args.compile_vae = False
            acceleration["fallbacks"].append("VAE compilation disabled without supported Triton kernels")
        print(json.dumps({"acceleration": acceleration}), flush=True)
    if args.int8:
        from model.output import encode_video
    else:
        from diffusers.utils.export_utils import encode_video
    if args.vae_attention == 'auto':
        from model.attention import select_attention
        args.vae_attention = select_attention()[0]
    from model.pipeline import Pipeline, encode_conditioning
    from model.weights import sha256
    from model.provenance import snapshot_sources, changed_sources
    source_record = snapshot_sources(args.cache)
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        parser.error("prompt must not be empty")
    height, width = ((args.height + 31) // 32) * 32, ((args.width + 31) // 32) * 32
    started = time.perf_counter()
    condition, encode_timing = encode_conditioning(args.weights, prompt, args.reference, height, width, args.frames, args.cache,
                                                 aligned_indices=args.aligned_reference, native_keyframes=args.native_keyframes,
                                                 reference_image_short_edge=args.reference_image_short_edge,
                                                 reference_video_short_edge=args.reference_video_short_edge)
    pipe = Pipeline(args.weights, reference=bool(args.reference),
                    aligned_indices=args.aligned_reference,
                    int8=args.int8, fused=args.fused, light_vae=args.light_vae, adaln_cache=args.adaln_cache,
                    compile_vae=args.compile_vae,
                    int8_gemm=args.int8_gemm, vae_attention=args.vae_attention, vae_offload=args.vae_offload,
                    attention_backend=args.attention_backend, dit_offload_blocks=args.dit_offload_blocks,
                    native_keyframes=args.native_keyframes,
                    reference_image_short_edge=args.reference_image_short_edge,
                    reference_video_short_edge=args.reference_video_short_edge)
    if args.light_vae is not None:
        pipe.pipe.vae.tile_batch = args.vae_tile_batch
        pipe.vae_record["tile_batch"] = args.vae_tile_batch
    if args.vae_tiles:
        if args.light_vae is None:
            parser.error("--vae-tiles requires --light-vae")
        pipe.pipe.vae.tile_layout = args.vae_tiles
        pipe.vae_record["tile_layout"] = args.vae_tiles
    loaded_seconds = time.perf_counter() - started
    first_height = min(height, args.refine_first_height) if args.refine_strength is not None else height
    first_width = ((round(width * first_height / height) + 31) // 32) * 32
    refinement_record = None
    timings = []
    preparation_timings = []
    for i in range(args.warmups + args.repeats):
        profiled = args.profile is not None and i == args.warmups + args.repeats - 1
        context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) if profiled else nullcontext()
        with context as profiler:
            state, timing = pipe.generate(condition, args.reference, first_height, first_width, args.frames, args.seed, args.steps,
                                          keep_latents=args.refine_strength is not None)
        if profiled:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(args.profile))
            args.profile.with_suffix(".txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))
            timing["profiler_overhead_included"] = True
        if i >= args.warmups:
            timings.append(timing)
        else:
            preparation_timings.append(timing)
        print(json.dumps({"iteration": i, "warmup": i < args.warmups, **timing}), flush=True)
        phase = "Preparation (includes compilation/tuning)" if i < args.warmups else "Measured inference"
        print(f"{phase} | DiT ({timing['actual_dit_forwards']} steps): {timing['dit_ms']/1000:.3f}s | "
              f"video decoder: {timing['video_decoder_ms']/1000:.3f}s | "
              f"sum: {(timing['dit_ms'] + timing['video_decoder_ms'])/1000:.3f}s | "
              f"generation incl. decode/RGB/staging: {timing['generation_and_decode_ms']/1000:.3f}s", flush=True)
    if args.refine_strength is not None:
        from model.refinement import target_latents, configure_refinement, refinement_sigmas
        from model.pipeline import release_memory
        if pipe.config.get("variant") != "flash":
            parser.error("the refined entry uses weight/flash")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        first_video = args.output.with_name("first_pass.mp4")
        encode_video(state["videos"][0][:args.frames], fps=24, output_path=str(first_video),
                     audio=state["audio"][0][..., :round(args.frames/24*state["sampling_rate"])],
                     audio_sample_rate=state["sampling_rate"])
        video_latent, audio_latent = target_latents(state, pipe.pipe)
        from safetensors.torch import save_file
        first_latents = args.output.with_name("first_latents.safetensors")
        save_file({"video": video_latent, "audio": audio_latent}, str(first_latents))
        sigmas = refinement_sigmas(args.refine_strength, args.refine_steps, args.refine_schedule)
        # Replace only the preparer and scheduler update; keep loaded INT8 weights.
        configure_refinement(pipe.pipe._blocks, video_latent, audio_latent, sigmas)
        pipe.pipe.set_progress_bar_config(disable=True)
        del state
        release_memory()
        refined_preparation = []
        for j in range(args.warmups + 1):
            state, refined_timing = pipe.generate(condition, [], height, width, args.frames,
                                                 args.seed + 1, args.refine_steps, keep_latents=True)
            if j < args.warmups:
                refined_preparation.append(refined_timing)
                print(json.dumps({'refinement_preparation': refined_timing}), flush=True)
                del state
        refined_video, refined_audio = target_latents(state, pipe.pipe)
        final_latents = args.output.with_name("final_latents.safetensors")
        save_file({"video": refined_video, "audio": refined_audio}, str(final_latents))
        print(json.dumps({"refinement": refined_timing, "sigmas": sigmas}), flush=True)
        refinement_record = {"first_pass": str(first_video), "first_latents": str(first_latents),
            "final_latents": str(final_latents),
            "first_canvas": [first_width, first_height],
            "first_timing": timings[0], "refinement_timing": refined_timing,
            "preparation_timings": refined_preparation,
            "refinement_sigmas": sigmas, "audio": "first-pass clean latent, frozen",
            "schedule": args.refine_schedule,
            "total_dit_forwards": args.steps + args.refine_steps,
            "method": "spatial latent upscale, flow re-noising, video-only refinement"}
    frames = state["videos"][0]
    if len(frames) < args.frames:
        raise RuntimeError("decoder returned fewer frames than requested")
    left, top = (width - args.width) // 2, (height - args.height) // 2
    frames = [frame.crop((left, top, left + args.width, top + args.height)) for frame in frames[:args.frames]]
    audio = state["audio"][0][..., :round(args.frames * int(state["sampling_rate"]) / 24)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_path = args.output.with_suffix(".mp4")
    encode_video(frames, fps=24, output_path=str(video_path), audio=audio, audio_sample_rate=state["sampling_rate"])
    if args.mode == "image-edit":
        frames[args.image_frame].save(args.output)
    record = {"mode": args.mode, "weights": str(args.weights.resolve()), "prompt": prompt,
              "transformer": pipe.transformer_record,
              "references": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in args.reference],
              "aligned_reference_indices": args.aligned_reference,
              "reference_image_short_edge": args.reference_image_short_edge,
              "reference_video_short_edge": args.reference_video_short_edge,
              "output": str(args.output.resolve()), "sha256": sha256(args.output), "seed": args.seed,
              "requested_geometry": [args.width, args.height, args.frames], "native_canvas": [width, height],
              "fps": 24, "steps": args.steps, "sigma_grid_points": args.steps + 1,
              "conditioning": encode_timing, "load_and_condition_seconds": loaded_seconds,

              "quantization": pipe.quantization,
              "fusion": pipe.fusion, "attention_backend": pipe.attention_backend,
              "video_encoding": "lossless_rgb_h264" if args.int8 else "upstream_h264",
              "acceleration": acceleration, "native_keyframes": pipe.native_keyframes,
              "modulation_cache": pipe.modulation_cache,
              "light_vae": pipe.vae_record,
              "vae_attention_backend": args.vae_attention or "upstream_default",
              "vae_phase_offload": args.vae_offload,
              "vae_offloaded_dit_blocks": 8 if args.vae_offload else 0,
              "denoise_offloaded_vae_decoder": args.vae_offload,
              "denoise_offloaded_dit_blocks": args.dit_offload_blocks,
              "image_frame": args.image_frame if args.mode == "image-edit" else None,
              "image_edit_route": "selected frame from camera-locked video request" if args.mode == "image-edit" else None,
              "generated_clip": {"path": str(video_path), "sha256": sha256(video_path)},
              "source": source_record, "source_changed_during_run": changed_sources(source_record),
              "timings": timings, "preparation_timings": preparation_timings, "warmups": args.warmups, "refinement": refinement_record,
              "device": torch.cuda.get_device_name(), "torch": torch.__version__}
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
    args.output.with_suffix(".txt").write_text(prompt + "\n")

if __name__ == "__main__":
    main()
