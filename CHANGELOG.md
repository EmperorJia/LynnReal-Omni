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
