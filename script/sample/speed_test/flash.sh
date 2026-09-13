#!/usr/bin/env bash
set -euo pipefail
exec "${LYNNREAL_PYTHON:-python}" "$(dirname -- "${BASH_SOURCE[0]}")/run.py" --variant flash "$@"
