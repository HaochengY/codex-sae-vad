#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mllm/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/codex/codex-compass}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_vad_compass_internvl2_sht.py" \
  --project-root "${PROJECT_ROOT}" \
  --data "${DATA_JSON:-../data/sht_clip_32_160_conversations_filtered.json}" \
  --dataset-root "${DATASET_ROOT:-../../autodl-tmp/sht_clip_32_160}" \
  --model-path "${MODEL_PATH:-../../autodl-tmp/get_hf/InternVL2}" \
  --output-dir "${OUTPUT_DIR:-outputs/vad_compass_internvl2_sht}" \
  --train-split "${TRAIN_SPLIT:-all}" \
  --epochs "${EPOCHS:-1}" \
  --num-frames "${NUM_FRAMES:-8}" \
  --max-patches-per-frame "${MAX_PATCHES_PER_FRAME:-1}" \
  --hook-layer "${HOOK_LAYER:-12}" \
  --k-slots "${K_SLOTS:-4}" \
  --expansion-factor "${EXPANSION_FACTOR:-16}" \
  --sae-topk "${SAE_TOPK:-256}" \
  --slot-dim "${SLOT_DIM:-512}" \
  --slot-heads "${SLOT_HEADS:-8}" \
  --slot-layers "${SLOT_LAYERS:-2}" \
  --rollout-n "${ROLLOUT_N:-4}" \
  --lambda-grpo "${LAMBDA_GRPO:-1.0}" \
  --lambda-bce "${LAMBDA_BCE:-1.0}" \
  --lambda-recon "${LAMBDA_RECON:-0.05}" \
  --dtype "${DTYPE:-bfloat16}" \
  "$@"
