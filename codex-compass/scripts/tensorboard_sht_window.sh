#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-outputs/tensorboard/sht_window_smoke}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6007}"

"${PYTHON_BIN}" -m tensorboard.main \
  --logdir "${TENSORBOARD_LOGDIR}" \
  --host "${TENSORBOARD_HOST}" \
  --port "${TENSORBOARD_PORT}"
