# LynnReal-Omni · ComfyUI 🧩

ComfyUI workflows for the **Standard four-step** checkpoints, plus the small node pack they
need and every file they load. The folder mirrors a ComfyUI install, so copying three
directories into place is the whole installation.

> [!NOTE]
> This port is **experimental and still being built out**. It runs the same checkpoints and the
> same schedules as `script/sample/`, but the pipeline around them is ComfyUI's: numbers and
> timings should be taken from the **original scripts**, and the same seed does **not** produce
> the same sample in both engines (different noise source and decoder path). We keep improving
> it — issues and pull requests are very welcome.

## Layout

```
comfyui/
├── workflows/                       -> ComfyUI/user/default/workflows/
│   ├── t2v_lynnreal_4step.json          text → video + audio
│   ├── i2v_lynnreal_4step.json          first frame → video
│   ├── r2v_lynnreal_4step.json          reference images/videos → video
│   ├── pose2v_lynnreal_4step.json       pose control clip → video
│   └── v2v_lynnreal_4step.json          video continuation
├── custom_nodes/ComfyUI-LynnReal/   -> ComfyUI/custom_nodes/
├── models/                          -> ComfyUI/models/            (see the tables below)
├── input/                           -> ComfyUI/input/             (demo assets the workflows load)
├── tools/quantize_h3_standard_int8.py   how the INT8 checkpoint was built
└── VERIFICATION.md                      Flash / Light-VAE end-to-end record
```

All model files are on Hugging Face:
[🤗 stdstu123/LynnReal-Onmi-beta-0.1 · comfyui/models](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1/tree/main/comfyui/models)

## What each task needs

Every task needs the six items below; they differ only in the workflow, the extra input, and
whether the node pack is required.

| Task | Workflow | Diffusion model (`models/diffusion_models/`) | Text encoder (`models/text_encoders/`) | Video VAE (`models/vae/`) | Audio VAE (`models/vae/`) | Embedding (`models/embeddings/`) | Node pack | Extra input (`input/`) |
|---|---|---|---|---|---|---|---|---|
| Text → video | `t2v_lynnreal_4step.json` | `lynnreal_omni_standard_bf16.safetensors` (or `lynnreal_omni_standard_int8.safetensors` via the switch) | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `minimax_h3_video_vae_fp16.safetensors` | `minimax_h3_audio_vae_fp32.safetensors` | `minimaxh3_art_is_explosion.safetensors` | — | — |
| First frame → video | `i2v_lynnreal_4step.json` | same | same | same | same | same | — | `transparent_rgb_gaming_mouse.png` |
| References → video | `r2v_lynnreal_4step.json` | same | same | same | same | same | — | `red_superboy_on_city_roof.png`, `mecha_dragon_lightning.png` |
| Pose control | `pose2v_lynnreal_4step.json` | same (INT8 switch **on** by default) | same | same | same | same | `ComfyUI-LynnReal` | `pose_boxing_first.png`, `pose_boxing_control.mp4` |
| Video continuation | `v2v_lynnreal_4step.json` | same | same | same | same | same | — | `snowboard.mp4` |

The embedding is optional: the demo prompts reference it as
`embedding:minimaxh3_art_is_explosion`. Drop it and remove that token to run without it.

### Files and sizes

| File | Size | Destination | Source |
|---|---|---|---|
| `lynnreal_omni_standard_bf16.safetensors` | 61.7 GiB | `models/diffusion_models/` | this release |
| `lynnreal_omni_standard_int8.safetensors` | 41.4 GiB | `models/diffusion_models/` | this release (optional) |
| `minimax_h3_video_vae_fp16.safetensors` | 4.9 GiB | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_audio_vae_fp32.safetensors` | 577 MiB | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GiB | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimaxh3_art_is_explosion.safetensors` | 500 KiB | `models/embeddings/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

💨 **The Flash three-step checkpoint is coming — we are pushing it out as fast as we can**, as a
separate release together with its Light VAE. On an H100 80 GB a 1344×768, five-second t2v takes
about **7–8 s of GPU kernel time** (three denoiser steps plus video decoding — the ≈8 s figure we
quote for Flash) with the official scripts. The ComfyUI port is not there yet: the same request currently needs
quote for Flash) with the official scripts. The ComfyUI port is not there yet: the same request
currently needs ~12.6 s of GPU time and ~17 s warm wall clock including file encoding, and
closing that gap is part of the packaging work.

## The INT8 switch

All five workflows carry a **`Use INT8 model?`** boolean (default **off** except `pose2v`). It
swaps `UNETLoader` to `lynnreal_omni_standard_int8.safetensors` — the same W8A8 contract as the
release's `--precision int8` path (per-output-channel weight scales, per-token activation
scales, INT32 accumulate; block 0, the last block, the token refiner, adaLN and the IO
projections stay BF16).

## Install

1. ComfyUI recent enough to have MiniMax-H3 (`comfy/ldm/minimax/`), `ResolutionSelector`,
   `ComfyMathExpression`, `ComfySwitchNode` and comfy-kitchen INT8 (`int8_tensorwise`).
   Tested with 0.35.0.
2. Copy `custom_nodes/ComfyUI-LynnReal` into `ComfyUI/custom_nodes/`.
3. Copy `models/*` and `input/*` into the matching ComfyUI folders.
4. Start ComfyUI and open a workflow. An 80 GB-class GPU is required at 1344×768;
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended.

   **VRAM:** the node pack keeps ComfyUI's own reserve, which keeps the 61.7 GiB DiT and the
   15 GiB text encoder resident — a warm 4-step 1344×768 t2v runs in ~50 s. Reserving VRAM
   instead makes ComfyUI evict the text encoder between runs (measured 46 s → 85 s).
   `pose2v` defaults to INT8, so it fits without any flag; if you flip it back to bf16, run it
   with `--reserve-vram 10` (or `LYNNREAL_RESERVE_VRAM=10`).

   On a cu13x torch (≥ 2.8) ComfyUI uses DynamicVRAM instead and manages the split itself; the
   pack reports which path is active at startup.

## Notes

* `minimax_h3_video_vae_fp16.safetensors` is loaded by ComfyUI's stock `VAELoader`.
* Same-seed output is not comparable across engines: the official scripts and ComfyUI draw
  their noise differently, and ComfyUI's decoder adds run-to-run spread at the same magnitude
  as ~44 dB PSNR (measured, see `VERIFICATION.md`).
