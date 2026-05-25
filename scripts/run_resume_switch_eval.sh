#!/usr/bin/env bash
set -euo pipefail
cd /root/codex
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/mllm/bin/python scripts/resume_switch_eval_fill_failed.py 2>&1 | tee /root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval/eval.log
