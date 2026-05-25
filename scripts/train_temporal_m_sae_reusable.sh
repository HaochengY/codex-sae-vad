#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Reusable Temporal Matryoshka SAE training command.

Defaults reproduce the latest known run when paths exist, but every input can be
overridden through flags or environment variables.

Usage:
  train_temporal_m_sae_reusable.sh [options] [-- extra trainer args]

Common options:
  --data PATH                  Dataset JSON/JSONL. Env: SAE_DATA
  --model-path PATH            InternVL-compatible model weights. Env: SAE_MODEL_PATH
  --output-dir PATH            Output directory. Env: SAE_OUTPUT_DIR
  --python PATH                Python executable. Env: SAE_PYTHON
  --codex-dir PATH             Project root containing trainer. Env: CODEX_DIR
  --trainer PATH               Training script. Env: SAE_TRAINER
  --gpu IDS                    CUDA_VISIBLE_DEVICES value. Env: CUDA_VISIBLE_DEVICES

Training options:
  --hook-layer N               Default: 12
  --num-frames N               Default: 8
  --max-patches-per-frame N    Default: 1
  --epochs N                   Default: 1
  --max-samples N              Default: 0
  --num-latents N              Default: 0, uses hidden_size * expansion_factor
  --expansion-factor N         Default: 16
  --k N                        Default: 256
  --lr FLOAT                   Default: 2e-4
  --lambda-temp FLOAT          Default: 0.1
  --tau FLOAT                  Default: 0.1
  --alpha-mat FLOAT            Default: 0.1
  --high-frac FLOAT            Default: 0.2
  --log-every N                Default: 20
  --save-every N               Default: 1000
  --seed N                     Default: 0
  --dtype bfloat16|float16|float32
                               Default: bfloat16

Utility:
  --dry-run                    Print resolved command without running.
  --print-config               Print resolved config and exit.
  -h, --help                   Show this help.

Examples:
  # Reproduce the latest clip10k filtered training defaults.
  /root/codex/scripts/train_temporal_m_sae_reusable.sh

  # New production environment with custom model/data/output.
  /root/codex/scripts/train_temporal_m_sae_reusable.sh \
    --model-path /models/InternVL2 \
    --data /data/my_dataset.jsonl \
    --output-dir /outputs/my_sae_run \
    --epochs 3 --k 256 --gpu 0

  # Pass through new trainer args without changing this wrapper.
  /root/codex/scripts/train_temporal_m_sae_reusable.sh -- --some-new-arg value
EOF
}

first_existing_file() {
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

first_existing_dir() {
  for path in "$@"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

CODEX_DIR="${CODEX_DIR:-/root/codex}"
PYTHON_BIN="${SAE_PYTHON:-}"
TRAINER="${SAE_TRAINER:-}"
DATA="${SAE_DATA:-}"
MODEL_PATH="${SAE_MODEL_PATH:-}"
OUTPUT_DIR="${SAE_OUTPUT_DIR:-}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

HOOK_LAYER="${SAE_HOOK_LAYER:-12}"
NUM_FRAMES="${SAE_NUM_FRAMES:-8}"
MAX_PATCHES_PER_FRAME="${SAE_MAX_PATCHES_PER_FRAME:-1}"
EPOCHS="${SAE_EPOCHS:-1}"
MAX_SAMPLES="${SAE_MAX_SAMPLES:-0}"
NUM_LATENTS="${SAE_NUM_LATENTS:-0}"
EXPANSION_FACTOR="${SAE_EXPANSION_FACTOR:-16}"
K="${SAE_K:-256}"
LR="${SAE_LR:-0.0002}"
LAMBDA_TEMP="${SAE_LAMBDA_TEMP:-0.1}"
TAU="${SAE_TAU:-0.1}"
ALPHA_MAT="${SAE_ALPHA_MAT:-0.1}"
HIGH_FRAC="${SAE_HIGH_FRAC:-0.2}"
LOG_EVERY="${SAE_LOG_EVERY:-20}"
SAVE_EVERY="${SAE_SAVE_EVERY:-1000}"
SEED="${SAE_SEED:-0}"
DTYPE="${SAE_DTYPE:-bfloat16}"

DRY_RUN=0
PRINT_CONFIG=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --model-path|--model) MODEL_PATH="$2"; shift 2 ;;
    --output-dir|--out) OUTPUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --codex-dir) CODEX_DIR="$2"; shift 2 ;;
    --trainer) TRAINER="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --hook-layer) HOOK_LAYER="$2"; shift 2 ;;
    --num-frames) NUM_FRAMES="$2"; shift 2 ;;
    --max-patches-per-frame) MAX_PATCHES_PER_FRAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --num-latents) NUM_LATENTS="$2"; shift 2 ;;
    --expansion-factor) EXPANSION_FACTOR="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --lambda-temp) LAMBDA_TEMP="$2"; shift 2 ;;
    --tau) TAU="$2"; shift 2 ;;
    --alpha-mat) ALPHA_MAT="$2"; shift 2 ;;
    --high-frac) HIGH_FRAC="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(first_existing_file \
    /root/miniconda3/envs/mllm/bin/python \
    /opt/conda/envs/mllm/bin/python \
    /root/miniconda3/bin/python \
    /opt/conda/bin/python \
    /usr/bin/python3 \
    /usr/local/bin/python3 || true)"
fi

if [[ -z "$TRAINER" ]]; then
  TRAINER="$CODEX_DIR/train_temporal_matryoshka_sae_internvl_sht.py"
fi

if [[ -z "$DATA" ]]; then
  DATA="$(first_existing_file \
    "$CODEX_DIR/data/sht_clip_32_160_conversations_filtered.json" \
    "$CODEX_DIR/data/sht_clip_32_160_conversations_filtered_normstr.json" \
    "$CODEX_DIR/data/sht_clip_32_160_conversations_filtered.jsonl" \
    "$CODEX_DIR/data/sht_clip_32_160_conversations_filtered_normstr.jsonl" \
    "$CODEX_DIR/data/sht_video_concept_conversations_clean.json" \
    "$CODEX_DIR/data/sht_video_concept_conversations_clean.jsonl" || true)"
fi

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="$(first_existing_dir \
    /root/autodl-tmp/get_hf/InternVL2 \
    /root/autodl-tmp/InternVL2 \
    /models/InternVL2 \
    /mnt/models/InternVL2 || true)"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$CODEX_DIR/outputs/temporal_m_sae_$(basename "${MODEL_PATH:-model}")_$(basename "${DATA:-data}" | sed 's/\.[^.]*$//')_$(timestamp)"
fi

missing=0
for pair in \
  "python:$PYTHON_BIN" \
  "trainer:$TRAINER" \
  "data:$DATA" \
  "model_path:$MODEL_PATH"; do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ -z "$path" ]]; then
    echo "ERROR: could not resolve $name. Pass --$name or set the corresponding env var." >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$TRAINER" ]]; then
  echo "ERROR: trainer not found: $TRAINER" >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: data not found: $DATA" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model path not found: $MODEL_PATH" >&2
  exit 2
fi

CMD=(
  "$PYTHON_BIN" "$TRAINER"
  --data "$DATA"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --hook-layer "$HOOK_LAYER"
  --num-frames "$NUM_FRAMES"
  --max-patches-per-frame "$MAX_PATCHES_PER_FRAME"
  --epochs "$EPOCHS"
  --max-samples "$MAX_SAMPLES"
  --num-latents "$NUM_LATENTS"
  --expansion-factor "$EXPANSION_FACTOR"
  --k "$K"
  --lr "$LR"
  --lambda-temp "$LAMBDA_TEMP"
  --tau "$TAU"
  --alpha-mat "$ALPHA_MAT"
  --high-frac "$HIGH_FRAC"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --seed "$SEED"
  --dtype "$DTYPE"
)

if [[ "${#PASSTHROUGH[@]}" -gt 0 ]]; then
  CMD+=("${PASSTHROUGH[@]}")
fi

print_config() {
  cat <<EOF
Resolved Temporal Matryoshka SAE training config:
  CODEX_DIR=$CODEX_DIR
  PYTHON=$PYTHON_BIN
  TRAINER=$TRAINER
  CUDA_VISIBLE_DEVICES=$GPU
  DATA=$DATA
  MODEL_PATH=$MODEL_PATH
  OUTPUT_DIR=$OUTPUT_DIR
  HOOK_LAYER=$HOOK_LAYER
  NUM_FRAMES=$NUM_FRAMES
  MAX_PATCHES_PER_FRAME=$MAX_PATCHES_PER_FRAME
  EPOCHS=$EPOCHS
  MAX_SAMPLES=$MAX_SAMPLES
  NUM_LATENTS=$NUM_LATENTS
  EXPANSION_FACTOR=$EXPANSION_FACTOR
  K=$K
  LR=$LR
  LAMBDA_TEMP=$LAMBDA_TEMP
  TAU=$TAU
  ALPHA_MAT=$ALPHA_MAT
  HIGH_FRAC=$HIGH_FRAC
  LOG_EVERY=$LOG_EVERY
  SAVE_EVERY=$SAVE_EVERY
  SEED=$SEED
  DTYPE=$DTYPE
EOF
}

print_config
printf 'Command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=$GPU" "${CMD[@]}"
printf '\n'

if [[ "$PRINT_CONFIG" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec env "CUDA_VISIBLE_DEVICES=$GPU" "${CMD[@]}"
