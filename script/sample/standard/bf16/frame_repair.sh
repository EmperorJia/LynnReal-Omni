#!/usr/bin/env bash
set -euo pipefail
STANDARD_SAMPLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Every source frame uses four DiT forwards from weight/standard/transformer/.
"${LYNNREAL_PYTHON:-python}" "$STANDARD_SAMPLE_ROOT/standard/check_weights.py" "$@"
exec "${LYNNREAL_PYTHON:-python}" -u "$STANDARD_SAMPLE_ROOT/../repair_frames.py" "$@"
