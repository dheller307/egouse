#!/usr/bin/env bash
# Batch raw hybrid student features (dual-venv orchestration via Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQ_LIST="${1:-${ROOT}/config/seq_lists/train.txt}"
shift || true
export PYTHONUNBUFFERED=1
exec "${ROOT}/venv/bin/python" "${ROOT}/src/run_student_feature_batch.py" \
  --seq-list "${SEQ_LIST}" \
  --continue-on-error \
  --skip-existing \
  "$@"
