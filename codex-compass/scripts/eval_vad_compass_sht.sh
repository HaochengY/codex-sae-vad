#!/usr/bin/env bash
set -euo pipefail
】
PROJECT_ROOT="${PROJECT_ROOT:-/root/codex/codex-compass}"
python "${PROJECT_ROOT}/scripts/eval_vad_compass_internvl2_sht.py" \
  --project-root "${PROJECT_ROOT}" \
  --checkpoint "${CHECKPOINT:-outputs/vad_compass_internvl2_sht/vad_compass_final.pt}" \
  --output-dir "${OUTPUT_DIR:-outputs/vad_compass_eval}" \
  --split "${SPLIT:-testing}" \
  "$@"
