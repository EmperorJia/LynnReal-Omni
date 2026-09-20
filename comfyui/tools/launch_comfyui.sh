#!/usr/bin/env bash
# One launcher for all shipped LynnReal workflows. The node pack budgets each
# sampling request and bounds large INT8 launches; no workflow-specific flags.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/main.py" ]]; then
  DEFAULT_DIR="$SCRIPT_DIR"
else
  DEFAULT_DIR="$PWD"
fi
COMFY_DIR="${COMFYUI_DIR:-$DEFAULT_DIR}"
if [[ ! -f "$COMFY_DIR/main.py" ]]; then
  echo 'Set COMFYUI_DIR to the ComfyUI directory (containing main.py).' >&2
  exit 2
fi
SHARED_ENV="/inspire/qb-ilm/project/3d-display/public/conda/envs/${COMFYUI_ENV:-lynnreal-comfyui}"
if [[ -x "$SHARED_ENV/bin/python" ]]; then
  DEFAULT_PYTHON="$SHARED_ENV/bin/python"
else
  DEFAULT_PYTHON=python
fi
COMFY_PYTHON="${COMFYUI_PYTHON:-$DEFAULT_PYTHON}"
OPTIONS=(--listen "${COMFYUI_HOST:-0.0.0.0}" --port "${COMFYUI_PORT:-8188}")
if [[ -n "${COMFYUI_RESERVE_VRAM:-}" ]]; then
  OPTIONS+=(--reserve-vram "$COMFYUI_RESERVE_VRAM")
fi
if [[ "${COMFYUI_TRITON:-auto}" == 0 ]]; then
  OPTIONS+=(--disable-triton-backend)
elif [[ "${COMFYUI_TRITON:-auto}" == 1 ]]; then
  OPTIONS+=(--enable-triton-backend)
fi
cd "$COMFY_DIR"
exec "$COMFY_PYTHON" main.py "${OPTIONS[@]}" "$@"
