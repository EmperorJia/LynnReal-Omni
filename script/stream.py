"""Continue a source video in 16- or 17-frame chunks with bounded latent history."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-window", choices=("head", "tail"), default="head",
                        help="legacy ablations use head; the public launcher explicitly defaults to tail")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=720)
    parser.add_argument("--steps", type=int, default=4, help="actual DiT forwards per chunk")
    parser.add_argument("--chunk-frames", type=int, choices=(16,17), default=17)
    parser.add_argument("--decode-layout", choices=("native","continuous"), default="continuous")
    parser.add_argument("--elementwise-fusion", action="store_true")
    parser.add_argument("--anchor-first-frame", action="store_true",
                        help="encode the source's first RGB frame once into the existing first sink slot")
    parser.add_argument("--condition-frames", type=int, default=22)
    parser.add_argument("--history-capacity", type=int, default=64)
    parser.add_argument("--sink-frames", type=int, choices=(1, 2), default=1,
                        help="retained initial latent frames; --sink-spatial chooses their spatial budget")
    parser.add_argument("--mid-stride", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--recent-stride", type=int, choices=(2, 4), default=2)
    parser.add_argument("--sink-spatial", choices=("shared", "dense"), default="shared")
    parser.add_argument("--history-pooling", choices=("point", "mean"), default="point")
    parser.add_argument("--reuse-vae-phase", action="store_true",
                        help="keep DiT blocks offloaded between decode and the next boundary encode")
    parser.add_argument("--boundary-rgb8", action="store_true",
                        help="feed the same rounded RGB boundary to the VAE and text conditioner")
    parser.add_argument("--boundary-posterior", choices=("sample", "mode"), default="sample")
    parser.add_argument("--text-context", choices=("sink-boundary", "recent-boundaries", "boundary"),
                        default="sink-boundary", help="visual inputs to Qwen; latent sinks are unchanged")
    parser.add_argument("--history-feedback", choices=("latent", "native-rgb"), default="latent",
                        help="experimental native RGB re-encoding of bounded history; adds VAE compute")
    parser.add_argument("--history-posterior", choices=("sample", "mode"), default="sample",
                        help="native fixed-seed sample; mode reproduces earlier development runs")
    parser.add_argument("--light-vae", type=Path,
                        help="experimental decoder replacement; also changes later RGB boundary feedback")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae-offload", action="store_true", help="stage unused DiT blocks during VAE encoding and decoding")
    parser.add_argument("--reload-conditioner", action="store_true",
                        help="reload Qwen each chunk to reduce CPU memory use; default retains it on CPU")
    parser.add_argument("--resident-conditioner", action="store_true",
                        help="retain Qwen's 50 used layers on the DiT GPU; requires sufficient combined VRAM")
    parser.add_argument("--conditioner-service", type=Path,
                        help="shared queue served by script/conditioner.py on a separate GPU")
    parser.add_argument("--stabilize-exposure", action="store_true",
                        help="apply original V2V causal correction; also save uncorrected decoder RGB")
    parser.add_argument("--attention-backend", choices=("auto", "native", "_flash_3"), default="auto",
                        help="auto selects FA3 for standard DiT when installed, otherwise native")
    args = parser.parse_args()
    if args.reuse_vae_phase and (not args.vae_offload or not args.conditioner_service or args.history_feedback != "latent"):
        parser.error("--reuse-vae-phase requires --vae-offload, --conditioner-service and latent history")
    if sum((args.resident_conditioner, args.reload_conditioner, bool(args.conditioner_service))) > 1:
        parser.error("resident, reloaded and remote conditioning modes are mutually exclusive")
    if min(args.frames, args.height, args.width, args.steps) < 1 or args.height % 2 or args.width % 2:
        parser.error("positive dimensions and even output width/height are required")
    if args.condition_frames < 5 or (args.condition_frames - 5) % 17:
        parser.error("--condition-frames must equal 17*n+5")
    if args.output.exists():
        parser.error("output exists; select a new experiment path")
    raw_path = args.output.with_name(args.output.stem + ".raw.mp4") if args.stabilize_exposure else None
    if raw_path and raw_path.exists():
        parser.error("uncorrected output exists; select a new experiment path")
    import numpy as np
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
    from PIL import Image
    from model.pipeline import Pipeline, encode_conditioning, release_memory
    from model.stream import Stream
    from model.video_window import read_video_window
    from model.weights import sha256
    from model.provenance import snapshot_sources, changed_sources
    source_record = snapshot_sources(args.cache)
    initialization_started = time.perf_counter()
    resident = None if args.reload_conditioner else {}
    resident_device = "cuda" if args.resident_conditioner else "cpu"
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        parser.error("prompt must not be empty")
    height, width = (args.height + 31) // 32 * 32, (args.width + 31) // 32 * 32

    def condition_pair(first, boundary):
        pictures = [boundary] if args.text_context == "boundary" else [first, boundary]
        if args.conditioner_service:
            from model.conditioning_service import encode_remote
            return encode_remote(args.conditioner_service, args.weights, prompt, pictures, height, width)
        return encode_conditioning(args.weights, prompt, pictures, height, width, 22,
            args.cache, native_keyframes=True, text_only=True, resident=resident, resident_device=resident_device)

    folder = args.output.parent / (args.output.stem + "_chunks")
    folder.mkdir(parents=True, exist_ok=False)
    ffmpeg = get_ffmpeg_exe()
    rgb, source_window = read_video_window(args.video, args.condition_frames, width, height, args.source_window)
    first = folder / "sink.png"
    boundary = folder / "boundary_00000.png"
    appearance, appearance_window = None, None
    if args.anchor_first_frame:
        if args.text_context != "sink-boundary" or args.history_feedback != "latent":
            parser.error("first-frame anchoring requires sink-boundary text context and latent history")
        appearance, appearance_window = read_video_window(args.video, 1, width, height, "head")
    Image.fromarray(appearance[0] if appearance is not None else rgb[0]).save(first)
    Image.fromarray(rgb[-1]).save(boundary)
    condition, text_timing = condition_pair(first, boundary)
    pipeline = Pipeline(args.weights,
                        light_vae=args.light_vae, attention_backend=args.attention_backend)
    if args.elementwise_fusion:
        from model.elementwise_fusion import enable_elementwise_fusion
        pipeline.fusion = enable_elementwise_fusion(pipeline.transformer)
    stream = Stream(pipeline, args.history_capacity, stabilize_exposure=args.stabilize_exposure,
                    history_posterior=args.history_posterior, history_feedback=args.history_feedback,
                    vae_offload=args.vae_offload, sink_frames=args.sink_frames,
                    mid_stride=args.mid_stride, boundary_rgb8=args.boundary_rgb8,
                    recent_stride=args.recent_stride, sink_spatial=args.sink_spatial,
                    history_pooling=args.history_pooling, reuse_vae_phase=args.reuse_vae_phase,
                    boundary_posterior=args.boundary_posterior, chunk_frames=args.chunk_frames,
                    decode_layout=args.decode_layout)
    appearance_pixels = None if appearance is None else torch.from_numpy(appearance).permute(3, 0, 1, 2)[None].float() / 255
    stream.initialize(torch.from_numpy(rgb).permute(3, 0, 1, 2)[None].float() / 255, appearance_pixels)
    del appearance, appearance_pixels
    del rgb
    initialization_seconds = time.perf_counter() - initialization_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_command = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", "24", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(args.output)]
    total_started = time.perf_counter()
    delivered, chunk = 0, 0
    with (folder / "encode.log").open("w") as errors:
        process = subprocess.Popen(output_command, stdin=subprocess.PIPE, stderr=errors)
        writers = [process]
        try:
            if raw_path:
                raw_process = subprocess.Popen(output_command[:-1] + [str(raw_path)],
                                               stdin=subprocess.PIPE, stderr=errors)
                writers.append(raw_process)
            while delivered < args.frames:
                started = time.perf_counter()
                if chunk:
                    # Qwen and DiT are staged on one GPU to keep the exact two-image
                    # text context. Record transfer + encoding in end-to-end latency.
                    if not args.resident_conditioner and not args.conditioner_service:
                        pipeline.pipe.to("cpu")
                        release_memory()
                    condition, text_timing = condition_pair(first, boundary)
                    if not args.resident_conditioner and not args.conditioner_service:
                        pipeline.pipe.to("cuda")
                output, timing = stream.step(condition, seed=args.seed + chunk, steps=args.steps)
                frames = (output[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
                if args.text_context == "recent-boundaries":
                    first = boundary
                boundary = folder / f"boundary_{chunk + 1:05d}.png"
                Image.fromarray(frames[-1]).save(boundary)
                count = min(args.chunk_frames, args.frames - delivered)
                top, left = (height - args.height) // 2, (width - args.width) // 2
                process.stdin.write(frames[:count, top:top + args.height, left:left + args.width].tobytes())
                raw_boundary = None
                if raw_path:
                    raw_frames = (stream.raw_output[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    raw_process.stdin.write(raw_frames[:count, top:top + args.height, left:left + args.width].tobytes())
                    raw_boundary = folder / f"raw_boundary_{chunk + 1:05d}.png"
                    Image.fromarray(raw_frames[-1]).save(raw_boundary)
                record = {"chunk": chunk, "seed": args.seed + chunk, "first_output_frame": delivered,
                    "delivered_frames": count, "conditioning": text_timing, **timing,
                    "iteration_with_text_and_transfer_ms": (time.perf_counter() - started) * 1000,
                    "boundary": str(boundary), "boundary_sha256": sha256(boundary),
                    "uncorrected_boundary_sha256": sha256(raw_boundary) if raw_boundary else None}
                with (folder / "measurements.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                print(f"Chunk {chunk}: {args.steps}-step DiT {timing['dit_ms']:.3f} ms + "
                      f"decoder {timing['video_decoder_ms']:.3f} ms", flush=True)
                delivered += count
                chunk += 1
        finally:
            for writer in writers:
                writer.stdin.close()
            return_codes = [writer.wait() for writer in writers]
            stream.close(*sys.exc_info())
        if any(return_codes):
            raise RuntimeError("video writer failed; see encode.log")
    args.output.with_suffix(".json").write_text(json.dumps({"mode": "v2v_stream", "prompt": prompt, "transformer": pipeline.transformer_record,
        "source": str(args.video.resolve()), "source_sha256": sha256(args.video),
        "source_window": source_window,
        "output_sha256": sha256(args.output), "frames": delivered, "fps": 24, "chunks": chunk,
        "native_canvas": [width, height], "output_geometry": [args.width, args.height],
        "history_capacity": args.history_capacity, "sink_retained": True,
        "sink_frames": args.sink_frames, "mid_stride": args.mid_stride,
        "recent_stride": args.recent_stride, "sink_spatial": args.sink_spatial,
        "history_pooling": args.history_pooling, "reuse_vae_phase": args.reuse_vae_phase,
        "appearance_reference": {"path": str(first), "sha256": sha256(first), "source_window": appearance_window,
            "compression": "image-encoded existing first sink slot; no added rows"} if args.anchor_first_frame else None,
        "boundary_rgb8": args.boundary_rgb8,
        "boundary_posterior": args.boundary_posterior,
        "steps_per_chunk": args.steps,
        "chunk_frames": args.chunk_frames, "decode_layout": args.decode_layout,
        "fusion": pipeline.fusion,
        "history_posterior": args.history_posterior,
        "history_feedback": args.history_feedback,
        "light_vae": pipeline.vae_record, "attention_backend": pipeline.attention_backend,
        "text_context": args.text_context,
        "text_rows": "image-fused text rows only",
        "text_residency": "Qwen resident on a separate GPU via shared filesystem" if args.conditioner_service else (
            "50 used Qwen layers resident on DiT GPU" if args.resident_conditioner else "staged on same GPU; " + ("reload per chunk" if args.reload_conditioner else "CPU resident weights")),
        "conditioner_service": str(args.conditioner_service.resolve()) if args.conditioner_service else None,
        "initialization_seconds": initialization_seconds, "audio": "omitted by trained continuation contract",
        "exposure_stabilization": args.stabilize_exposure,
        "uncorrected_decode": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path),
            "scope": "raw decoder pixels from this run; later chunks already use corrected RGB boundary feedback"} if raw_path else None,
        "seconds_excluding_initialization": time.perf_counter() - total_started,

        "source_code": source_record, "source_changed_during_run": changed_sources(source_record),
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "measurements": str(folder / "measurements.jsonl")}, indent=2))

if __name__ == "__main__":
    main()
