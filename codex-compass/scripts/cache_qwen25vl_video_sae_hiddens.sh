#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mllm/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

DATA_PATHS="${DATA_PATHS:-}"
DATASET_ROOTS="${DATASET_ROOTS:-}"
SPLITS="${SPLITS:-}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/get_hf/Qwen2.5-VL-3B-Instruct}"
HIDDEN_SAVE_DIR="${HIDDEN_SAVE_DIR:-outputs/sae_hiddens/qwen25vl_video_l12}"
SAE_LAYER_K="${SAE_LAYER_K:-12}"
NUM_FRAMES="${NUM_FRAMES:-8}"
VIDEO_FPS="${VIDEO_FPS:-0}"
MIN_PIXELS="${MIN_PIXELS:-0}"
MAX_PIXELS="${MAX_PIXELS:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1400}"
BATCH_WORKERS="${BATCH_WORKERS:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SUPPORT_BF16="${SUPPORT_BF16:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
USER_PROMPT="${USER_PROMPT:-{text}}"
CSV_TEXT_FIELDS="${CSV_TEXT_FIELDS:-name}"
VIDEO_TMP_DIR="${VIDEO_TMP_DIR:-/tmp/qwen25vl_video_sae_cache}"

if [ -z "${DATA_PATHS}" ]; then
  echo "ERROR: set DATA_PATHS to one or more JSON/JSONL video index files." >&2
  exit 1
fi

launcher=("${PYTHON_BIN}" -m sae_video_harness.cache_qwen25vl_video_hiddens)
if [ "${NPROC_PER_NODE}" != "1" ]; then
  launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"
    --module sae_video_harness.cache_qwen25vl_video_hiddens
  )
fi

cmd=(
  "${launcher[@]}"
  --data-paths ${DATA_PATHS}
  --model-path "${MODEL_PATH}"
  --hidden-save-dir "${HIDDEN_SAVE_DIR}"
  --sae-layer-k "${SAE_LAYER_K}"
  --num-frames "${NUM_FRAMES}"
  --video-fps "${VIDEO_FPS}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --max-prompt-length "${MAX_PROMPT_LENGTH}"
  --num-workers "${BATCH_WORKERS}"
  --max-samples "${MAX_SAMPLES}"
  --support-bf16 "${SUPPORT_BF16}"
  --system-prompt "${SYSTEM_PROMPT}"
  --user-prompt "${USER_PROMPT}"
  --csv-text-fields "${CSV_TEXT_FIELDS}"
  --video-tmp-dir "${VIDEO_TMP_DIR}"
)

if [ -n "${DATASET_ROOTS}" ]; then
  cmd+=(--dataset-roots ${DATASET_ROOTS})
fi
if [ -n "${SPLITS}" ]; then
  cmd+=(--splits ${SPLITS})
fi

"${cmd[@]}"
