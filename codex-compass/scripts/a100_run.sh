#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/petrelfs/wangxiaoyang/yhc/dataset/sht/window3s_step1s_dataset}"
CLIPS_SUBDIRS="${CLIPS_SUBDIRS:-clips/train,clips/test}"
LABELS_SUBDIRS="${LABELS_SUBDIRS:-frame_labels/train,frame_labels/test}"

MODEL_PATH="${MODEL_PATH:-/mnt/petrelfs/wangxiaoyang/yhc/get_data/InternVL2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/a100_run}"
TRAIN_JSON="${TRAIN_JSON:-outputs/a100_run/train_index.json}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-outputs/tensorboard/a100_run}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"

NUM_FRAMES="${NUM_FRAMES:-12}"
INPUT_SIZE="${INPUT_SIZE:-448}"
MAX_PATCHES_PER_FRAME="${MAX_PATCHES_PER_FRAME:-1}"
HOOK_LAYER="${HOOK_LAYER:-12}"
SAE_PATH="${SAE_PATH:-/mnt/petrelfs/wangxiaoyang/yhc/codex-sae-vad/outputs/sae_l${HOOK_LAYER}/sae_final.pt}"
DTYPE="${DTYPE:-bfloat16}"

K_SLOTS="${K_SLOTS:-4}"
POS_TOKEN="${POS_TOKEN:-<RULE>}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-16}"
NUM_LATENTS="${NUM_LATENTS:-0}"
SAE_TOPK="${SAE_TOPK:-0}"
SLOT_DIM="${SLOT_DIM:-512}"
SLOT_HEADS="${SLOT_HEADS:-8}"
SLOT_LAYERS="${SLOT_LAYERS:-2}"

ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

LR="${LR:-1.6e-6}"
LR_QB_RATE="${LR_QB_RATE:-30}"
LR_HEAD_RATE="${LR_HEAD_RATE:-25}"
LR_PE_RATE="${LR_PE_RATE:-10}"
LR_DECODER_RATE="${LR_DECODER_RATE:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

LAMBDA_GRPO="${LAMBDA_GRPO:-1.0}"
LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.05}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.3}"
TASK_WEIGHT="${TASK_WEIGHT:-0.7}"
CLIPRANGE="${CLIPRANGE:-0.2}"

TRAIN_SAE="${TRAIN_SAE:-0}"
TRAIN_RULE_EMBEDDING="${TRAIN_RULE_EMBEDDING:-0}"
FREEZE_INTERNVL="${FREEZE_INTERNVL:-0}"
DEBUG_ROLLOUTS="${DEBUG_ROLLOUTS:-0}"

mkdir -p "$(dirname "${TRAIN_JSON}")" "${OUTPUT_DIR}" "${TENSORBOARD_LOGDIR}"

"${PYTHON_BIN}" "scripts/prepare_sht_window_smoke_json.py" \
  --dataset-root "${DATASET_ROOT}" \
  --clips-subdirs "${CLIPS_SUBDIRS}" \
  --labels-subdirs "${LABELS_SUBDIRS}" \
  --output-json "${TRAIN_JSON}" \
  --max-samples "${MAX_SAMPLES}" \
  --seed "${SEED}"

if [ ! -s "${TRAIN_JSON}" ]; then
  echo "ERROR: failed to create non-empty training index: ${TRAIN_JSON}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [ "${TRAIN_RULE_EMBEDDING}" = "1" ]; then
  EXTRA_ARGS+=(--train-rule-embedding)
fi
if [ "${TRAIN_SAE}" = "1" ]; then
  EXTRA_ARGS+=(--train-sae)
fi
if [ "${FREEZE_INTERNVL}" = "1" ]; then
  EXTRA_ARGS+=(--freeze-internvl)
fi
if [ "${DEBUG_ROLLOUTS}" = "1" ]; then
  EXTRA_ARGS+=(--debug-rollouts)
fi

"${PYTHON_BIN}" "scripts/train_vad_compass_internvl2_sht.py" \
  --project-root "." \
  --data "${TRAIN_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-split training \
  --epochs "${EPOCHS}" \
  --max-samples 0 \
  --max-steps "${MAX_STEPS}" \
  --sae-path "${SAE_PATH}" \
  --num-frames "${NUM_FRAMES}" \
  --input-size "${INPUT_SIZE}" \
  --max-patches-per-frame "${MAX_PATCHES_PER_FRAME}" \
  --hook-layer "${HOOK_LAYER}" \
  --k-slots "${K_SLOTS}" \
  --pos-token "${POS_TOKEN}" \
  --expansion-factor "${EXPANSION_FACTOR}" \
  --num-latents "${NUM_LATENTS}" \
  --sae-topk "${SAE_TOPK}" \
  --slot-dim "${SLOT_DIM}" \
  --slot-heads "${SLOT_HEADS}" \
  --slot-layers "${SLOT_LAYERS}" \
  --rollout-n "${ROLLOUT_N}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --lr "${LR}" \
  --lr-qb-rate "${LR_QB_RATE}" \
  --lr-head-rate "${LR_HEAD_RATE}" \
  --lr-pe-rate "${LR_PE_RATE}" \
  --lr-decoder-rate "${LR_DECODER_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --lambda-grpo "${LAMBDA_GRPO}" \
  --lambda-bce "${LAMBDA_BCE}" \
  --lambda-recon "${LAMBDA_RECON}" \
  --format-weight "${FORMAT_WEIGHT}" \
  --task-weight "${TASK_WEIGHT}" \
  --cliprange "${CLIPRANGE}" \
  --dtype "${DTYPE}" \
  --log-every "${LOG_EVERY}" \
  --save-every "${SAVE_EVERY}" \
  --tensorboard-logdir "${TENSORBOARD_LOGDIR}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
