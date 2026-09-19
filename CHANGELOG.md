# Changelog · LynnReal-Omni

User-visible changes to the released bundle — checkpoints, the ComfyUI workflows, the
`ComfyUI-LynnReal` node pack and the documentation. Newest first; dates are UTC+8.

**Source of truth.** Edit this file first (and add the new line to the "Latest updates" list in
the Hugging Face model card). The Hugging Face copies of `CHANGELOG.md` and `README.md` in
[`stdstu123/LynnReal-Onmi-beta-0.1`](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1)
mirror this one.

**How to update.** Everything the ComfyUI port needs lives under
[`comfyui/`](comfyui): copy `comfyui/custom_nodes/ComfyUI-LynnReal` into
`ComfyUI/custom_nodes/`, the workflows into `ComfyUI/user/default/workflows/`, and the files
under `comfyui/models/` into the matching `ComfyUI/models/` folders. The same tree ships in the
[Hugging Face bundle](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1/tree/main/comfyui).

## 2026-09-19 — Standard Lite: exact four-step BF16 and INT8 checkpoints

**What.** Added two optional Standard checkpoints:
`lynnreal_omni_standard_bf16_lite.safetensors` is **37.6 GiB instead of 61.7 GiB**, and
`lynnreal_omni_standard_int8_lite.safetensors` is **20.4 GiB instead of 44.5 GiB**. Their loaded
DiT footprints are 38.6 GB and 20.8 GB respectively, down from 63.2 GB and 45.5 GB.

**Same sampler trajectory.** At seed `970000`, every sampler state and denoiser output from all
four steps matched the corresponding original checkpoint exactly for t2v, i2v, r2v, pose2v and
v2v (`max |Δ| = 0`). The checks used cold model loads, required four non-empty comparison dumps,
and verified the default INT8 branch used by `pose2v`.

**How.** The checkpoints store the original model's adaLN modulation vectors for every timestep
and call shape visited by the shipped schedules, with a curve fallback for other timesteps. The
node pack now installs this selector for Standard models as well as Flash models.

**How to use it.** Install the current `ComfyUI-LynnReal` node pack, put the Lite checkpoints in
`models/diffusion_models/`, and open one of the five new `*_4step_lite.json` workflows. Their
BF16/INT8 switch selects the matching Lite pair. The original checkpoints and workflows are
unchanged and remain available.

## 2026-09-19 — Flash Lite: the same three-step model in 16.7 GiB instead of 37.0 GiB

**What.** [`lynnreal_omni_flash_int8_lite.safetensors`](https://huggingface.co/stdstu123/LynnReal-Onmi-beta-0.1/tree/main/comfyui/models/diffusion_models)
is an extra, optional Flash checkpoint. Its adaLN step-embedding table is stored as the modulation
vectors the three-step schedule actually visits — taken verbatim from the original checkpoint — so
it needs **16.7 GiB on disk instead of 37.0 GiB** and **17.0 GB of VRAM instead of 37.9 GB**, at
the same speed.

**Same frames.** At the same seed the sampler trajectory is identical to the original checkpoint:
compared step by step on an H100, t2v / ti2v / ref2v at 5 s and 10 s, max |Δ| = 0.

**How to use it.** Drop the file in `models/diffusion_models/`, keep the node pack up to date and
open one of the three new `*_lite.json` workflows — no launcher flags, no configuration. The
original Flash checkpoint and its workflows are unchanged and stay available.

**Caveat.** The exact table is pinned to the shipped schedule (`euler` + `simple`, three steps,
stock shifts); another step count or sampler falls back to the table's curve columns instead of
failing.

## 2026-09-18 — ComfyUI: Flash workflows no longer crash on DynamicVRAM machines

**Symptom.** On a machine where comfy-aimdo's DynamicVRAM is on (ComfyUI enables it by itself on
a cu13x torch, e.g. 2.13.0+cu312 on an RTX 5090), the Flash three-step workflows stopped on the
first sampling step:

```text
[INFO] LynnReal: fused DiT blocks running (first block has 64 rows, ...)
[INFO] LynnReal: fused DiT blocks verified on this GPU (relative error 0.0000, max abs 0.0000).
[INFO] Comfy model compiler graph breaks: 0, rogues: 12
[ERROR] !!! Exception during processing !!! aimdo memory compile error
```

**Cause.** `fast_blocks.probe()` is the guard that stands the fused DiT-block path down while
comfy-aimdo's malloc graph is recording — that planner rejects the allocation pattern of the
pack's two Triton kernels. The guard was handed `device=None`, and
`comfy.model_prefetch.malloc_graph_enabled()` starts with `is_device_cuda(device)`, which is
`False` for `None`; the fused blocks therefore stayed enabled, and the planner's failure surfaced
on the first block it tried to compile. (Bypassing the `LynnRealFlashTokenCompression` node
appeared to help only because that node is what installs the fused blocks.)

**Fix.** Resolve the compute device before the guard (`fast_blocks.py`), so the pack disables the
fused path on DynamicVRAM machines exactly as designed and the Flash graph runs on ComfyUI's own
block math. `tools/verify_node_pack.py` gained a fourth, GPU-less check that fails if the guard is
ever handed `device=None` again ([1a731cd](https://github.com/LynnReal-AI/LynnReal-Omni/commit/1a731cd)).

**What to do.** Update `comfyui/custom_nodes/ComfyUI-LynnReal` and re-run the workflow; no launcher
flags are needed. Standard four-step workflows, and Flash on machines without aimdo (e.g. a cu126
build), are unaffected.

## 2026-09-17 — Flash three-step is live in the ComfyUI bundle

- Added the three Flash workflows — `t2v_lynnreal_flash_3_step.json`,
  `ti2v_lynnreal_flash_3_step.json`, `ref2v_lynnreal_flash_3_step.json` — with their demo inputs.
- Added the `LynnRealH3VAELoader` node (Light VAE, 26 decoder blocks, 272/16 tiles) and the
  `LynnRealInt8Backend` node that reports/enables the INT8 fast path.
- Measured on one H100 80 GB at 1344×768, warm: 5 s clip ≈8.4 s end to end (three Flash steps),
  against ≈50 s for the Standard four-step model; 15 s clips run with the same recipes.
- Flash clips that pack more than 130k rows are refused with a clear message instead of crashing
  in the INT8 kernels.

## 2026-09-16 — Standard four-step ComfyUI port

- Added the five Standard workflows (`t2v`, `i2v`, `r2v`, `pose2v`, `v2v`) plus their demo inputs
  under `comfyui/input/`.
- Added a per-task weight guide (`comfyui/README.md`) covering the INT8 switch and the VRAM
  behaviour, and the end-to-end verification record (`comfyui/VERIFICATION.md`).

## 2026-09-15 — Initial beta

- Diffusers-format Standard weights at the repository root, the ComfyUI bundle under `comfyui/`,
  the paper and the demo videos.
