#!/usr/bin/env python
import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/root/codex")
import train_topk_sae_internvl_sht as topk_train


CONCEPTS = {
    "bicycle": {
        "zh": ["自行车"],
        "prompt": "Does this video clip contain a bicycle, bike, or cyclist?",
    },
    "skateboard": {
        "zh": ["滑板"],
        "prompt": "Does this video clip contain a skateboard or skateboarding?",
    },
    "running": {
        "zh": ["追逐奔跑", "奔跑", "跑步", "抢了别人包在奔跑", "小孩子在奔跑"],
        "prompt": "Does this video clip contain running, chasing, or jogging?",
    },
    "vehicle": {
        "zh": ["摩托车", "汽车", "三轮车", "滑板车", "骑平衡车"],
        "prompt": "Does this video clip contain a vehicle such as a car, motorcycle, tricycle, scooter, or balance bike?",
    },
}


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def stratified_split(y, test_frac=0.2, seed=42):
    rng = random.Random(seed)
    train, test = [], []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0].tolist()
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.array(train, dtype=int), np.array(test, dtype=int)


def build_model(args):
    model_path = Path(args.model_path).resolve()
    (model_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(model_path.parent))
    module = importlib.import_module(f"{model_path.name}.modeling_internvl_chat")
    cls = module.InternVLChatModel
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, dtype


def make_prompt(model, tokenizer, question, num_patches_list):
    template_mod = importlib.import_module(f"{Path(model.config.name_or_path).name}.conversation")
    template = template_mod.get_conv_template(model.template)
    template.system_message = model.system_message
    image_prefix = "\n".join(["<image>"] * len(num_patches_list))
    template.append_message(template.roles[0], f"{image_prefix}\nAnswer with exactly one word: yes or no. {question}")
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for n in num_patches_list:
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token * n + "</img>", 1)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return query


@torch.no_grad()
def score_questions(row, questions, model, tokenizer, dtype, args):
    pixel_values, num_patches_list = topk_train.load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        max_num=args.max_patches_per_frame,
    )
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    vit_embeds = model.extract_feature(pixel_values)
    yes_ids = tokenizer(" yes", add_special_tokens=False).input_ids + tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" no", add_special_tokens=False).input_ids + tokenizer("no", add_special_tokens=False).input_ids
    names = list(questions.keys())
    prompts = [make_prompt(model, tokenizer, questions[name], num_patches_list) for name in names]
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        tok = tokenizer(prompts, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_padding_side
    input_ids = tok["input_ids"].cuda()
    attention_mask = tok["attention_mask"].cuda()
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1])
    for b in range(input_ids.shape[0]):
        selected = input_ids[b] == model.img_context_token_id
        input_embeds[b, selected] = flat_vit.to(input_embeds.dtype)
    out = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=False)
    last_idx = attention_mask.sum(dim=1) - 1
    logits = out.logits[torch.arange(input_ids.shape[0], device=input_ids.device), last_idx].float()
    yes_logit = torch.logsumexp(logits[:, yes_ids], dim=1)
    no_logit = torch.logsumexp(logits[:, no_ids], dim=1)
    margins = (yes_logit - no_logit).detach().cpu().tolist()
    return {name: float(score) for name, score in zip(names, margins)}


def load_baseline_scores(path):
    if not path.exists():
        return {}
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                scores[rec["id"]] = rec
    return scores


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def eval_feature_method(rows, labels, train_idx, test_idx, args):
    data = np.load(args.feature_cache)
    firing = data["firing"].astype(bool)
    metas = [json.loads(line) for line in open(args.feature_meta, encoding="utf-8")]
    ok = np.array([m.get("ok", False) for m in metas], dtype=bool)
    records = []
    for task_name, task in make_tasks(rows).items():
        if task_name == "overall_abnormal":
            pos_mask = labels == 1
            neg_mask = labels == 0
        else:
            pos_zh = set(task["zh"])
            cats = np.array([r.get("category_zh") for r in rows])
            pos_mask = np.isin(cats, list(pos_zh))
            neg_mask = labels == 0
        tr = np.array([i for i in train_idx if ok[i] and (pos_mask[i] or neg_mask[i])], dtype=int)
        te = np.array([i for i in test_idx if ok[i] and (pos_mask[i] or neg_mask[i])], dtype=int)
        y_train = pos_mask[tr].astype(int)
        y_test = pos_mask[te].astype(int)
        h_train = firing[tr]
        h_test = firing[te]
        delta = h_train[y_train == 1].mean(axis=0) - h_train[y_train == 0].mean(axis=0)
        feats = np.argsort(-delta)[: args.feature_topk]
        count_score = h_test[:, feats].sum(axis=1).astype(float)
        switch_score = h_test[:, feats].any(axis=1).astype(float)
        records.append({
            "task": task_name,
            "method": f"feature_count_top{args.feature_topk}",
            "auc": auc_score(y_test, count_score),
            "switch_auc": auc_score(y_test, switch_score),
            "n_test": int(len(te)),
            "pos_test": int(y_test.sum()),
            "neg_test": int((1 - y_test).sum()),
            "features": json.dumps([int(x) for x in feats[:10]]),
        })
    return records


def make_tasks(rows):
    tasks = {k: dict(v) for k, v in CONCEPTS.items()}
    tasks["overall_abnormal"] = {
        "zh": [],
        "prompt": "Does this video clip contain abnormal, suspicious, dangerous, or rule-breaking behavior?",
    }
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.json")
    p.add_argument("--model-path", default="/root/autodl-tmp/get_hf/InternVL2")
    p.add_argument("--feature-cache", default="/root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval/firing_presence_uint8.npz")
    p.add_argument("--feature-meta", default="/root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval/firing_meta.jsonl")
    p.add_argument("--output-dir", default="/root/codex/outputs/internvl2_baseline_vs_feature_auc_20260524")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-patches-per-frame", type=int, default=1)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--feature-topk", type=int, default=10)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_path = out / "baseline_scores.jsonl"
    rows = load_rows(args.data)
    labels = np.array([1 if r.get("label") == "abnormal" else 0 for r in rows], dtype=int)
    train_idx, test_idx = stratified_split(labels, args.test_frac, args.seed)
    if args.limit_test:
        test_idx = test_idx[: args.limit_test]

    tasks = make_tasks(rows)
    feature_records = eval_feature_method(rows, labels, train_idx, test_idx, args)
    cached = load_baseline_scores(baseline_path)
    todo = [int(i) for i in test_idx if rows[int(i)]["id"] not in cached]
    if todo:
        model, tokenizer, dtype = build_model(args)
        start = time.time()
        questions = {k: v["prompt"] for k, v in tasks.items()}
        for n, i in enumerate(todo, 1):
            row = rows[i]
            rec = {"id": row["id"], "index": int(i), "label": row.get("label"), "category_zh": row.get("category_zh")}
            try:
                rec["scores"] = score_questions(row, questions, model, tokenizer, dtype, args)
            except Exception as exc:
                rec["error"] = str(exc)
            append_jsonl(baseline_path, rec)
            if n % args.progress_every == 0:
                print(f"baseline scored {n}/{len(todo)} elapsed={time.time()-start:.1f}s", flush=True)

    cached = load_baseline_scores(baseline_path)
    baseline_records = []
    cats = np.array([r.get("category_zh") for r in rows])
    for task_name, task in tasks.items():
        if task_name == "overall_abnormal":
            pos_mask = labels == 1
            neg_mask = labels == 0
        else:
            pos_mask = np.isin(cats, list(task["zh"]))
            neg_mask = labels == 0
        idx = [int(i) for i in test_idx if (pos_mask[int(i)] or neg_mask[int(i)]) and rows[int(i)]["id"] in cached and not cached[rows[int(i)]["id"]].get("error")]
        y = np.array([int(pos_mask[i]) for i in idx], dtype=int)
        s = np.array([cached[rows[i]["id"]]["scores"][task_name] for i in idx], dtype=float)
        baseline_records.append({
            "task": task_name,
            "method": "internvl2_yes_no_logit",
            "auc": auc_score(y, s),
            "switch_auc": "",
            "n_test": int(len(idx)),
            "pos_test": int(y.sum()),
            "neg_test": int((1 - y).sum()),
            "features": "",
        })

    all_records = baseline_records + feature_records
    with open(out / "auc_comparison.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["task", "method", "auc", "switch_auc", "n_test", "pos_test", "neg_test", "features"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_records)
    by_task = {}
    for task in tasks:
        b = next(r for r in baseline_records if r["task"] == task)
        m = next(r for r in feature_records if r["task"] == task)
        by_task[task] = {
            "baseline_auc": b["auc"],
            "feature_count_auc": m["auc"],
            "feature_minus_baseline": m["auc"] - b["auc"] if not math.isnan(b["auc"]) else float("nan"),
            "feature_switch_auc": m["switch_auc"],
            "n_test": b["n_test"],
            "pos_test": b["pos_test"],
            "neg_test": b["neg_test"],
            "top_features": json.loads(m["features"]),
        }
    summary = {
        "data": args.data,
        "split": f"stratified mixed split seed={args.seed}, test_frac={args.test_frac}",
        "baseline": "InternVL2 yes/no next-token logit margin: logit(yes)-logit(no)",
        "feature_method": f"SAE feature firing count over train-selected top {args.feature_topk} positive-vs-normal features",
        "limit_test": args.limit_test,
        "results": by_task,
        "outputs": {
            "baseline_scores": str(baseline_path),
            "comparison_csv": str(out / "auc_comparison.csv"),
            "summary": str(out / "summary.json"),
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
