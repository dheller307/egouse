#!/usr/bin/env bash
# Run a command in the 100DOH venv (venv_doh).
# Usage: scripts/run_doh.sh src/smoke_detector.py --backend 100doh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/venv_doh/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${ROOT}/venv_doh/bin/python" "$@"
