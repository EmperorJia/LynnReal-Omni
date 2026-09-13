#!/usr/bin/env bash
set -euo pipefail
exec "${LYNNREAL_PYTHON:-python}" "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/run.py" --variant flash --precision int8 --task t2v --refine "$@"
