#!/usr/bin/env bash
set -euo pipefail
SAMPLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${LYNNREAL_PYTHON:-python}" -u "$SAMPLE_ROOT/stream_native.py" "$@"
