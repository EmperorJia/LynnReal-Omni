# ComfyUI-LynnReal

Everything the LynnReal release needs from ComfyUI that is not core: the Flash token
compression, the Light VAE loader, the INT8 backend helper and the aligned-reference node
for the pose/hand workflows. **No file under `comfy/` is patched.**

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-LynnReal     # or copy the folder in
```

Restart ComfyUI. No launcher flag is required: `LynnRealInt8Backend` enables the
comfy-kitchen Triton backend when the CUDA backend is unavailable (torch < cu130), which is
what makes INT8 checkpoints fast -- about 3x versus the eager fallback
(`--disable-triton-backend` or `LYNNREAL_NO_TRITON=1` opts out).

## VRAM policy (`runtime.py`)

* **Default: no extra reserve.** With ComfyUI's own setting the 61.7 GiB DiT and the 15 GiB text
  encoder stay resident and a warm 4-step 1344x768 t2v takes ~50 s. Reserving VRAM makes
  ComfyUI evict the text encoder between runs, which cost ~38 s per run in our measurements.
* **DynamicVRAM on a cu13x torch.** ComfyUI enables it itself when the torch build is >= 2.8 and
  CUDA >= 13; if it did not, the pack retries the same initialisation and logs the outcome
  (`LYNNREAL_NO_DYNAMIC_VRAM=1` opts out).
* **Pose control workflows need `--reserve-vram 10`** (or `LYNNREAL_RESERVE_VRAM=10`): a
  frame-aligned control clip packs reference and target into one sequence and OOMs at the first
  sampling step next to a fully resident DiT. The flag always wins over anything the pack does.

## Nodes

| Node | What it is for |
|---|---|
| `LynnReal Flash token compression (MiniMax H3)` | The Flash DiT's trained token compression. Insert between the Flash model loader and the guider (`start_block=2`, `end_block=28`, `stride=2`). |
| `Load LynnReal H3 VAE (Light VAE aware)` | Loads any MiniMax-H3 video VAE with the decoder depth taken from the checkpoint. Use it for `lynnreal_omni_light_vae_fp16.safetensors`; the official VAE loads identically to the stock loader. |
| `LynnReal INT8 backend info` | Reports which comfy-kitchen backend serves the quantized ops. |
| `LynnReal Aligned Reference` | Frame-aligned reference setup for the pose/hand control workflows. |

### Why a VAE loader instead of the stock one

The release's Light VAE is a 26-block distilled decoder; the official H3 video VAE has 36.
ComfyUI builds the H3 VAE with the depth hardcoded to 36 and loads state dicts with
`strict=False`, so the stock `VAELoader` accepts the Light VAE with nothing but a
`Missing VAE keys [...]` warning and decodes with ten randomly initialized blocks -- nothing
but garbage frames. This loader reads the depth from the checkpoint, applies the Light VAE's own tile geometry
(`272/16`, from its `decode_config.json`) and compiles the decoder the way the release's
`--compile-vae` does. Set `LYNNREAL_NO_COMPILE_VAE=1` to decode eagerly.

## Requirements

* ComfyUI with MiniMax-H3 support (`comfy/ldm/minimax/`, `ResolutionSelector`,
  `ComfyMathExpression`, `ComfySwitchNode`) and comfy-kitchen INT8
  (`int8_tensorwise`). Tested against ComfyUI 0.35.0.
* An 80 GB-class GPU for the 1344x768 / 124-frame workflows; `--reserve-vram 5` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` are recommended.
* `flash-attn` (FA2) for the fast attention path, `triton` for the INT8 GEMM.
