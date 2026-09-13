"""Keep Qwen on a second GPU and answer native continuation conditioning requests."""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=21600)
    parser.add_argument("--idle-seconds", type=float, default=1200)
    parser.add_argument('--stream-context', action='store_true', help='serve formal V2V boundary images plus optional fixed refs')
    args = parser.parse_args()
    if min(args.max_seconds, args.idle_seconds) <= 0:
        parser.error("service time limits must be positive")
    import torch
    from model.pipeline import encode_conditioning
    from model.weights import sha256
    from model.provenance import snapshot_sources, changed_sources
    args.queue.mkdir(parents=True, exist_ok=True)
    lock = (args.queue / "service.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = snapshot_sources(args.cache)
    resident = {}
    started = last_request = time.monotonic()
    while time.monotonic() - started < args.max_seconds:
        if (args.queue / "STOP").exists() or time.monotonic() - last_request > args.idle_seconds:
            break
        requests = sorted(args.queue.glob("*.request.json"))
        if not requests:
            time.sleep(0.1)
            continue
        path = requests[0]
        claimed = path.with_suffix(".claimed")
        path.rename(claimed)
        request = json.loads(claimed.read_text())
        identity = path.name.removesuffix(".request.json")
        metadata = {"request": request, "device": torch.cuda.get_device_name(),
                    "host": socket.gethostname(), "source": source}
        try:
            if (request["id"] != identity or request["weights"] != str(args.weights.resolve())
                    or request["contract"] != "multimodal-conditioning-v1" or request["frames"] < 1):
                raise ValueError("conditioning request does not match this service")
            pictures = [Path(p["path"]) for p in request["pictures"]]
            if not pictures or any(sha256(p) != r["sha256"] for p, r in zip(pictures, request["pictures"])):
                raise ValueError("conditioning pictures changed or are incomplete")
            if bool(request.get('stream_context')) != args.stream_context:
                raise ValueError('request does not match the conditioner context mode')
            if args.stream_context:
                from model.conditioning import encode_stream_context
                state, timing = encode_stream_context(args.weights, request['prompt'], pictures, args.cache, resident)
            else:
                state, timing = encode_conditioning(args.weights, request["prompt"], pictures,
                    request["height"], request["width"], request["frames"], args.cache,
                    native_keyframes=request['native_keyframes'],
                    text_only=request['text_only'], resident=resident, resident_device="cuda",
                    reference_image_short_edge=request.get('reference_image_short_edge'),
                    reference_video_short_edge=request.get('reference_video_short_edge'))
            if changed_sources(source):
                raise RuntimeError("conditioning service source changed during execution")
            output = args.queue / (identity + ".pt.tmp")
            torch.save({k: state[k].detach().cpu() for k in ("prompt_embeds", "text_token_tags")}, output)
            output.rename(args.queue / (identity + ".pt"))
            metadata["timing"] = timing
        except Exception as error:
            metadata["error"] = f"{type(error).__name__}: {error}"
        output = args.queue / (identity + ".result.tmp")
        output.write_text(json.dumps(metadata, indent=2))
        output.rename(args.queue / (identity + ".result.json"))
        last_request = time.monotonic()
        print(json.dumps({"request": identity, "error": metadata.get("error")}), flush=True)


if __name__ == "__main__":
    main()
