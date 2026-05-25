#!/usr/bin/env python
import argparse
import csv
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/root/codex")
import train_temporal_matryoshka_sae_internvl_sht as temp_train
import train_topk_sae_internvl_sht as topk_train


QUESTION = "Does this video clip contain abnormal, suspicious, dangerous, or rule-breaking behavior?"


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


def load_sae(args):
    ckpt = torch.load(args.sae_path, map_location="cpu")
    sae_args = ckpt["args"]
    sae = temp_train.TopKMatryoshkaSae(
        ckpt["d_in"],
        ckpt["num_latents"],
        sae_args["k"],
        sae_args.get("high_frac", 0.2),
    ).cuda().eval()
    sae.load_state_dict(ckpt["sae"])
    return sae


def make_prompt(model, tokenizer, num_patches_list):
    template_mod = importlib.import_module(f"{Path(model.config.name_or_path).name}.conversation")
    template = template_mod.get_conv_template(model.template)
    template.system_message = model.system_message
    image_prefix = "\n".join(["<image>"] * len(num_patches_list))
    template.append_message(template.roles[0], f"{image_prefix}\nAnswer with exactly one word: yes or no. {QUESTION}")
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for n in num_patches_list:
        query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token * n + "</img>", 1)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return query


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["id"], float(rec["alpha"])))
    return done


@torch.no_grad()
def score_row_alphas(row, active_features, alphas, model, tokenizer, dtype, hook_module, sae, args):
    pixel_values, num_patches_list = topk_train.load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        max_num=args.max_patches_per_frame,
    )
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    vit_embeds = model.extract_feature(pixel_values)
    prompt = make_prompt(model, tokenizer, num_patches_list)
    tok = tokenizer(prompt, return_tensors="pt")
    batch = len(alphas)
    input_ids = tok["input_ids"].cuda().repeat(batch, 1)
    attention_mask = tok["attention_mask"].cuda().repeat(batch, 1)
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    flat_vit = vit_embeds.reshape(-1, vit_embeds.shape[-1])
    for b in range(batch):
        selected = input_ids[b] == model.img_context_token_id
        input_embeds[b, selected] = flat_vit.to(input_embeds.dtype)
    img_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    yes_ids = tokenizer(" yes", add_special_tokens=False).input_ids + tokenizer("yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" no", add_special_tokens=False).input_ids + tokenizer("no", add_special_tokens=False).input_ids
    handle = None
    if active_features and any(float(a) != 0 for a in alphas):
        steer_vec = sae.decoder.weight[:, active_features].sum(dim=1).detach().float()
        alpha_tensor = torch.tensor([float(a) for a in alphas], dtype=torch.float32, device="cuda").view(batch, 1, 1)

        def hook(_, __, output):
            tuple_out = isinstance(output, tuple)
            h = output[0] if tuple_out else output
            mask = (input_ids.to(h.device) == img_token_id).view(batch, -1, 1)
            h = h + mask.to(h.dtype) * alpha_tensor.to(h.dtype) * steer_vec.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
            if tuple_out:
                return (h,) + output[1:]
            return h

        handle = hook_module.register_forward_hook(hook)
    try:
        out = model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = out.logits[:, -1].float()
        yes_logit = torch.logsumexp(logits[:, yes_ids], dim=1)
        no_logit = torch.logsumexp(logits[:, no_ids], dim=1)
        scores = (yes_logit - no_logit).detach().cpu().tolist()
        return {float(alpha): float(score) for alpha, score in zip(alphas, scores)}
    finally:
        if handle is not None:
            handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.json")
    p.add_argument("--model-path", default="/root/autodl-tmp/get_hf/InternVL2")
    p.add_argument("--sae-path", default="/root/codex/outputs/temporal_m_sae_internvl_sht_clips10k_filtered_8f_16x_k256_1ep/sae_final.pt")
    p.add_argument("--feature-cache", default="/root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval/firing_presence_uint8.npz")
    p.add_argument("--feature-meta", default="/root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval/firing_meta.jsonl")
    p.add_argument("--output-dir", default="/root/codex/outputs/overall_abnormal_steered_auc_20260525")
    p.add_argument("--features", default="18392,10730,8385,9459,9567,11176,21398,16573,28158,25253")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--hook-layer", type=int, default=12)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-patches-per-frame", type=int, default=1)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--steer-all-selected", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / "steered_scores.jsonl"
    rows = json.load(open(args.data, encoding="utf-8"))
    y = np.array([1 if r.get("label") == "abnormal" else 0 for r in rows], dtype=int)
    _, test_idx = stratified_split(y, args.test_frac, args.seed)
    if args.limit:
        test_idx = test_idx[: args.limit]

    features = [int(x) for x in args.features.split(",") if x.strip()]
    firing = np.load(args.feature_cache)["firing"].astype(bool)
    metas = [json.loads(line) for line in open(args.feature_meta, encoding="utf-8")]
    ok = np.array([m.get("ok", False) for m in metas], dtype=bool)
    test_idx = np.array([int(i) for i in test_idx if ok[int(i)]], dtype=int)

    model, tokenizer, dtype = build_model(args)
    _, hook_module = topk_train.resolve_hook_module(model, args.hook_layer)
    sae = load_sae(args)
    done = read_done(score_path)
    start = time.time()
    total = len(test_idx) * len(args.alpha)
    n = 0
    for i in test_idx:
        row = rows[int(i)]
        active = features if args.steer_all_selected else [f for f in features if firing[int(i), f]]
        pending_alphas = [float(alpha) for alpha in args.alpha if (row["id"], float(alpha)) not in done]
        if not pending_alphas:
            continue
        row_scores = None
        try:
            row_scores = score_row_alphas(row, active, pending_alphas, model, tokenizer, dtype, hook_module, sae, args)
        except Exception as exc:
            row_error = str(exc)
        for alpha in pending_alphas:
            key = (row["id"], float(alpha))
            rec = {
                "id": row["id"],
                "index": int(i),
                "label": row.get("label"),
                "category_zh": row.get("category_zh"),
                "alpha": float(alpha),
                "active_features": active,
            }
            if row_scores is None:
                rec["error"] = row_error
            else:
                rec["score"] = row_scores[float(alpha)]
            append_jsonl(score_path, rec)
            n += 1
            if n % args.progress_every == 0:
                print(f"scored {n}/{total} elapsed={time.time()-start:.1f}s", flush=True)

    records = []
    all_scores = [json.loads(line) for line in open(score_path, encoding="utf-8") if line.strip()]
    for alpha in args.alpha:
        part = [r for r in all_scores if float(r.get("alpha", -999)) == float(alpha) and not r.get("error")]
        by_id = {r["id"]: r["score"] for r in part}
        idx = [int(i) for i in test_idx if rows[int(i)]["id"] in by_id]
        yy = np.array([y[int(i)] for i in idx], dtype=int)
        ss = np.array([by_id[rows[int(i)]["id"]] for i in idx], dtype=float)
        records.append({
            "alpha": float(alpha),
            "auc": auc_score(yy, ss),
            "n_test": int(len(idx)),
            "normal": int((yy == 0).sum()),
            "abnormal": int((yy == 1).sum()),
        })
    with open(out_dir / "auc_by_alpha.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "auc", "n_test", "normal", "abnormal"])
        w.writeheader()
        w.writerows(records)
    summary = {
        "question": QUESTION,
        "features": features,
        "method": (
            "For each clip, add alpha * sum(SAE decoder vectors for all selected top10 features) "
            "to layer-12 image-token hidden states, then score yes/no logit margin."
            if args.steer_all_selected
            else "For each clip, add alpha * sum(SAE decoder vectors for selected top10 features that fired in that clip) to layer-12 image-token hidden states, then score yes/no logit margin."
        ),
        "steer_all_selected": bool(args.steer_all_selected),
        "records": records,
        "outputs": {
            "scores": str(score_path),
            "auc_by_alpha": str(out_dir / "auc_by_alpha.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
