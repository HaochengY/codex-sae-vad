#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Dataset layout expected under DATASET_ROOT:
#   clips/train/*.mp4
#   clips/test/*.mp4
#   frame_labels/train/*.npy
#   frame_labels/test/*.npy
DATASET_ROOT="${DATASET_ROOT:-data/window3s_step1s_dataset}"
CLIPS_SUBDIRS="${CLIPS_SUBDIRS:-clips/train,clips/test}"
LABELS_SUBDIRS="${LABELS_SUBDIRS:-frame_labels/train,frame_labels/test}"

# Model directory. Keep this relative or pass MODEL_PATH from the shell.
MODEL_PATH="${MODEL_PATH:-models/InternVL2}"

# Smoke data and outputs.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sht_window_smoke}"
SMOKE_JSON="${SMOKE_JSON:-outputs/sht_window_smoke/smoke_train.json}"
TENSORBOARD_LOGDIR="${TENSORBOARD_LOGDIR:-outputs/tensorboard/sht_window_smoke}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-4}"
SEED="${SEED:-42}"

# Training controls. Increase EPOCHS for repeated passes over the smoke set.
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-2}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"

# Video/model controls.
NUM_FRAMES="${NUM_FRAMES:-4}"
INPUT_SIZE="${INPUT_SIZE:-448}"
MAX_PATCHES_PER_FRAME="${MAX_PATCHES_PER_FRAME:-1}"
HOOK_LAYER="${HOOK_LAYER:-12}"
DTYPE="${DTYPE:-bfloat16}"

# VAD-Compass controls.
K_SLOTS="${K_SLOTS:-4}"
POS_TOKEN="${POS_TOKEN:-<RULE>}"
SAE_ROOT="${SAE_ROOT:-/mnt/petrelfs/wangxiaoyang/yhc/codex-sae-vad/outputs}"
SAE_PATH="${SAE_PATH:-${SAE_ROOT}/sae_l${HOOK_LAYER}/sae_final.pt}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-16}"
NUM_LATENTS="${NUM_LATENTS:-1024}"
SAE_TOPK="${SAE_TOPK:-32}"
SLOT_DIM="${SLOT_DIM:-128}"
SLOT_HEADS="${SLOT_HEADS:-4}"
SLOT_LAYERS="${SLOT_LAYERS:-1}"

# Optimization and reward controls.
LR="${LR:-1.6e-6}"
LR_QB_RATE="${LR_QB_RATE:-30}"
LR_HEAD_RATE="${LR_HEAD_RATE:-25}"
LR_PE_RATE="${LR_PE_RATE:-10}"
LR_DECODER_RATE="${LR_DECODER_RATE:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
ROLLOUT_N="${ROLLOUT_N:-4}"
LAMBDA_GRPO="${LAMBDA_GRPO:-1.0}"
LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.05}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.3}"
TASK_WEIGHT="${TASK_WEIGHT:-0.7}"
CLIPRANGE="${CLIPRANGE:-0.2}"

"${PYTHON_BIN}" "scripts/prepare_sht_window_smoke_json.py" \
  --dataset-root "${DATASET_ROOT}" \
  --clips-subdirs "${CLIPS_SUBDIRS}" \
  --labels-subdirs "${LABELS_SUBDIRS}" \
  --output-json "${SMOKE_JSON}" \
  --max-samples "${SMOKE_SAMPLES}" \
  --seed "${SEED}"

"${PYTHON_BIN}" "scripts/train_vad_compass_internvl2_sht.py" \
  --project-root "." \
  --data "${SMOKE_JSON}" \
  --dataset-root "${DATASET_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-split training \
  --epochs "${EPOCHS}" \
  --max-samples "${SMOKE_SAMPLES}" \
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
  "$@"
