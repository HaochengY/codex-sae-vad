#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
METRICS_JSONL="${METRICS_JSONL:-outputs/sht_window_smoke/metrics.jsonl}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-7860}"

"${PYTHON_BIN}" "scripts/live_metrics_dashboard.py" \
  --metrics "${METRICS_JSONL}" \
  --host "${DASHBOARD_HOST}" \
  --port "${DASHBOARD_PORT}"
