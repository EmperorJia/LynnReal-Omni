"""File-queue conditioning RPC for two GPU workers sharing a filesystem."""
import json
from pathlib import Path
import time
import uuid


def encode_remote(directory, weights, prompt, pictures, height, width, timeout=300,
                  *, frames=22, native_keyframes=True, text_only=True, reference_image_short_edge=None, reference_video_short_edge=None,
                  stream_context=False):
    import torch
    from .weights import sha256
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    request = {"id": identity, "weights": str(Path(weights).resolve()), "prompt": prompt,
               "pictures": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in pictures],
               "height": height, "width": width, "frames": frames,
               "native_keyframes": native_keyframes, "text_only": text_only,
               "contract": "multimodal-conditioning-v1"}
    if stream_context:
        request['stream_context'] = True
    if reference_image_short_edge is not None:
        request["reference_image_short_edge"] = reference_image_short_edge
    if reference_video_short_edge is not None:
        request["reference_video_short_edge"] = reference_video_short_edge
    pending = directory / (identity + ".tmp")
    pending.write_text(json.dumps(request))
    pending.rename(directory / (identity + ".request.json"))
    result = directory / (identity + ".result.json")
    started = time.monotonic()
    while not result.exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"conditioning service did not answer {identity} in {timeout}s")
        time.sleep(0.05)
    metadata = json.loads(result.read_text())
    if metadata.get("request") != request or metadata.get("error"):
        raise RuntimeError(f"conditioning service failed: {metadata.get('error', 'request mismatch')}")
    state = torch.load(directory / (identity + ".pt"), map_location="cpu", weights_only=True)
    if set(state) != {"prompt_embeds", "text_token_tags"} or not state["prompt_embeds"].isfinite().all():
        raise RuntimeError("invalid conditioning service tensors")
    timing = dict(metadata["timing"], service_wall_seconds=time.monotonic() - started,
                  service_record=str(result), service_device=metadata["device"], service_host=metadata["host"])
    return state, timing
