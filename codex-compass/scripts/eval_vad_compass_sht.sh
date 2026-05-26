#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mllm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/codex/codex-compass}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/eval_vad_compass_internvl2_sht.py" \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${CHECKPOINT:-outputs/vad_compass_internvl2_sht/vad_compass_final.pt}" \
  --output-dir "${OUTPUT_DIR:-outputs/vad_compass_eval}" \
  --split "${SPLIT:-testing}" \
  "$@"
