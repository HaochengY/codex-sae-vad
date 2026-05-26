#!/usr/bin/env python
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad_compass.internvl_utils import build_internvl, extract_compass_hidden, resolve_hook_module
from vad_compass.modeling import VadCompassHead
from vad_compass.path_utils import load_json_rows, portable_rows, resolve_path
from vad_compass.reward import synthetic_response


def label_to_int(row):
    label = str(row.get("label", "")).strip().lower()
    if label == "abnormal":
        return 1
    if label == "normal":
        return 0
    raise ValueError(f"unknown label for {row.get('id')}: {row.get('label')}")


def auc_score(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for group in np.where(counts > 1)[0]:
        mask = inv == group
        ranks[mask] = ranks[mask].mean()
    rpos = ranks[y == 1].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def binary_metrics(y, score, threshold=0.5):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    return {
        "accuracy": float((pred == y).mean()) if len(y) else float("nan"),
        "balanced_acc": float((tpr + tnr) / 2),
        "tpr_abnormal": float(tpr),
        "tnr_normal": float(tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_rows(rows, split="testing", max_samples=0, seed=42):
    if split != "all":
        rows = [r for r in rows if str(r.get("split", "")).lower() == split]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples:
        by_label = {0: [], 1: []}
        for row in rows:
            try:
                by_label[label_to_int(row)].append(row)
            except ValueError:
                continue
        if all(by_label.values()) and max_samples >= 2:
            half = max_samples // 2
            rows = by_label[0][:half] + by_label[1][: max_samples - half]
            rng.shuffle(rows)
        else:
            rows = rows[:max_samples]
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint", default="outputs/vad_compass_internvl2_sht/vad_compass_final.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="outputs/vad_compass_eval")
    parser.add_argument("--split", choices=["all", "training", "testing"], default="testing")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--response-mode",
        choices=["predicted", "neutral"],
        default="predicted",
        help="predicted: build POS-token response from the model's own first-pass prediction; neutral: fixed non-label answer placeholder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--max-patches-per-frame", type=int, default=None)
    parser.add_argument("--hook-layer", type=int, default=None)
    return parser


def make_eval_response(action, prob, pos_token, k_slots, mode="predicted"):
    if mode == "neutral":
        pos_tokens = " ".join([pos_token] * int(k_slots))
        return (
            "<think> Evaluate the surveillance clip for abnormal activity using the visual evidence. </think>\n"
            f"Here are the {int(k_slots)} anomaly concentration tokens:\n"
            f"{pos_tokens}\n"
            "Answer: normal"
        )
    return synthetic_response(
        label=int(action),
        action=int(action),
        abnormal_prob=float(prob),
        pos_token=pos_token,
        k_slots=k_slots,
    )


@torch.no_grad()
def evaluate(args):
    project_root = Path(args.project_root).resolve()
    ckpt_path = resolve_path(args.checkpoint, project_root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("args", {})

    def cfg_value(name, default=None):
        return getattr(args, name.replace("-", "_"), None) or cfg.get(name.replace("-", "_"), default)

    data = args.data or cfg.get("data", "data/sht_clip_32_160_conversations_filtered.json")
    dataset_root = args.dataset_root or cfg.get("dataset_root", ".")
    model_path = args.model_path or cfg.get("model_path", "models/InternVL2")
    dtype_name = args.dtype or cfg.get("dtype", "bfloat16")

    data_path = resolve_path(data, project_root)
    dataset_root = resolve_path(dataset_root, project_root)
    model_path = resolve_path(model_path, project_root)
    out_dir = resolve_path(args.output_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = portable_rows(load_json_rows(data_path), dataset_root)
    rows = select_rows(rows, split=args.split, max_samples=args.max_samples, seed=args.seed)

    k_slots = int(cfg.get("k_slots", 4))
    pos_token = cfg.get("pos_token", "<ANOM_POS>")
    question_template = cfg.get("question")
    if not question_template:
        raise ValueError("checkpoint config does not contain the training question template")
    question = question_template.format(
        K=k_slots,
        POS_TOKEN=pos_token,
        POS_TOKENS=" ".join([pos_token] * k_slots),
    )

    model, tokenizer, torch_dtype, special_token_ids = build_internvl(
        model_path,
        dtype=dtype_name,
        special_tokens=[pos_token],
        train_special_tokens=False,
    )
    pos_token_id = special_token_ids[pos_token]
    hook_layer = args.hook_layer if args.hook_layer is not None else int(cfg.get("hook_layer", 12))
    _, hook_module = resolve_hook_module(model, hook_layer)
    _, final_hook_module = resolve_hook_module(model, model.config.llm_config.num_hidden_layers - 1)

    d_in = int(ckpt.get("d_in", cfg.get("d_in", model.config.llm_config.hidden_size)))
    num_latents = int(ckpt.get("num_latents", cfg.get("num_latents")))
    vad = VadCompassHead(
        d_in=d_in,
        num_latents=num_latents,
        topk=int(cfg.get("sae_topk", 256)),
        k_slots=k_slots,
        slot_dim=int(cfg.get("slot_dim", 512)),
        nhead=int(cfg.get("slot_heads", 8)),
        num_layers=int(cfg.get("slot_layers", 2)),
    ).cuda().eval()
    vad.load_state_dict(ckpt["model"], strict=True)

    run_args = argparse.Namespace(
        num_frames=args.num_frames if args.num_frames is not None else int(cfg.get("num_frames", 8)),
        input_size=args.input_size if args.input_size is not None else int(cfg.get("input_size", 448)),
        max_patches_per_frame=(
            args.max_patches_per_frame
            if args.max_patches_per_frame is not None
            else int(cfg.get("max_patches_per_frame", 1))
        ),
        k_slots=k_slots,
    )

    records = []
    errors = []
    start = time.time()
    for row in tqdm(rows, desc="Evaluating VAD-Compass"):
        try:
            y = label_to_int(row)

            # First pass: use a fixed non-label response only to obtain the POS-token
            # anchors. This produces a model-side preliminary VAD decision without
            # reading the ground-truth label.
            neutral_response = make_eval_response(
                action=0,
                prob=0.5,
                pos_token=pos_token,
                k_slots=k_slots,
                mode="neutral",
            )
            neutral_compass = extract_compass_hidden(
                row,
                model,
                tokenizer,
                torch_dtype,
                hook_module,
                final_hook_module,
                question,
                neutral_response,
                pos_token_id,
                run_args,
            )
            neutral_out = vad(neutral_compass["sae_tokens"], neutral_compass["pos_hidden"])
            prelim_prob = float(neutral_out["video_prob"].detach().cpu().item())
            predicted_action = int(prelim_prob >= args.threshold)

            # Second pass: mirror SegCompass eval more closely by feeding a response
            # that contains the model-side predicted answer plus POS tokens. GT is
            # used only after this point to compute metrics.
            response = make_eval_response(
                action=predicted_action,
                prob=prelim_prob,
                pos_token=pos_token,
                k_slots=k_slots,
                mode=args.response_mode,
            )
            compass = extract_compass_hidden(
                row,
                model,
                tokenizer,
                torch_dtype,
                hook_module,
                final_hook_module,
                question,
                response,
                pos_token_id,
                run_args,
            )
            out = vad(compass["sae_tokens"], compass["pos_hidden"])
            prob = float(out["video_prob"].detach().cpu().item())
            pred = int(prob >= args.threshold)
            records.append(
                {
                    "id": row.get("id"),
                    "video_rel": row.get("video_rel"),
                    "label": y,
                    "prelim_score_abnormal": prelim_prob,
                    "response_mode": args.response_mode,
                    "score_abnormal": prob,
                    "pred": pred,
                    "correct": int(pred == y),
                    "slot_probs": [float(x) for x in out["slot_probs"][0].detach().cpu().tolist()],
                }
            )
        except Exception as exc:
            errors.append({"id": row.get("id"), "video": row.get("video"), "error": str(exc)})

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if errors:
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for rec in errors:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "id",
            "video_rel",
            "label",
            "prelim_score_abnormal",
            "response_mode",
            "score_abnormal",
            "pred",
            "correct",
            "slot_probs",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    y = np.array([r["label"] for r in records], dtype=int)
    s = np.array([r["score_abnormal"] for r in records], dtype=float)
    summary = {
        "checkpoint": str(ckpt_path),
        "data": str(data_path),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "max_samples": args.max_samples,
        "evaluated": len(records),
        "failed": len(errors),
        "label_counts": dict(Counter(["abnormal" if int(v) else "normal" for v in y])),
        "auc": auc_score(y, s) if len(records) else float("nan"),
        **binary_metrics(y, s, threshold=args.threshold),
        "threshold": args.threshold,
        "elapsed_sec": round(time.time() - start, 2),
        "outputs": {
            "predictions_jsonl": str(out_dir / "predictions.jsonl"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
