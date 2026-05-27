#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mllm/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

CONFIG="${CONFIG:-sae_video_harness/train_scripts/qwen25vl_video_initial.yaml}"
HIDDEN_SAVE_DIR="${HIDDEN_SAVE_DIR:-outputs/sae_hiddens/qwen25vl_video_l12}"
SAE_SAVE_DIR="${SAE_SAVE_DIR:-outputs/sae_checkpoints/qwen25vl_video_l12/default}"
D_IN="${D_IN:-2048}"
D_SAE="${D_SAE:-65536}"
HOOK_LAYER="${HOOK_LAYER:-12}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"

launcher=("${PYTHON_BIN}" -m sae_video_harness.sae_trainer)
if [ "${NPROC_PER_NODE}" != "1" ]; then
  launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}"
    --module sae_video_harness.sae_trainer
  )
fi

"${launcher[@]}" \
  --config "${CONFIG}" \
  data.cached_dir="${HIDDEN_SAVE_DIR}" \
  data.batch_size="${BATCH_SIZE}" \
  ckpt.save_dir="${SAE_SAVE_DIR}" \
  ckpt.load_path=null \
  train.max_epochs="${MAX_EPOCHS}" \
  sae_model.d_in="${D_IN}" \
  sae_model.d_sae="${D_SAE}" \
  sae_model.hook_layer="${HOOK_LAYER}" \
  sae_model.hook_name="model.language_model.layers.${HOOK_LAYER}" \
  "$@"
