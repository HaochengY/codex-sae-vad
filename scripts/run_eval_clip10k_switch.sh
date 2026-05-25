#!/usr/bin/env bash
set -euo pipefail
cd /root/codex
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/mllm/bin/python scripts/eval_temporal_m_clip_firing_switch.py 2>&1 | tee /root/codex/outputs/temporal_m_sae_clip10k_filtered_switch_eval/eval.log
