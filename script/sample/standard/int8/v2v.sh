#!/usr/bin/env bash
set -euo pipefail
STANDARD_SAMPLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Every Standard task loads its DiT from weight/standard/transformer/.
"${LYNNREAL_PYTHON:-python}" "$STANDARD_SAMPLE_ROOT/standard/check_weights.py" "$@"
exec "${LYNNREAL_PYTHON:-python}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/run.py" --variant standard --precision int8 --task v2v "$@"
