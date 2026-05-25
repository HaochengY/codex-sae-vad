#!/usr/bin/env python
import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/root/codex")
import train_temporal_matryoshka_sae_internvl_sht as temp_train
import train_topk_sae_internvl_sht as topk_train


BICYCLE_RE = re.compile(r"\b(bicycle|bike|cycling|cyclist|biking)\b", re.I)


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def build_model(args):
    model_path = Path(args.model_path).resolve()
    (model_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(model_path.parent))
    module = importlib.import_module(f"{model_path.name}.modeling_internvl_chat")
    InternVLChatModel = module.InternVLChatModel
    if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
        InternVLChatModel.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = InternVLChatModel.from_pretrained(
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


def make_question(num_images=1):
    image_prefix = "\n".join(["<image>"] * num_images)
    return (
        f"{image_prefix}\n"
        "Answer with exactly one word: bicycle or no_bicycle. "
        "Does this video clip contain a bicycle, bike, or cyclist?"
    )


def is_bicycle_answer(text):
    text = text.strip().lower()
    if "no_bicycle" in text or "no bicycle" in text or "not" in text:
        return False
    if re.search(r"\byes\b", text):
        return True
    return bool(BICYCLE_RE.search(text))


def load_video(row, dtype, args):
    pixel_values, num_patches_list = topk_train.load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        max_num=args.max_patches_per_frame,
    )
    return pixel_values.to(dtype=dtype, device="cuda"), num_patches_list


@torch.no_grad()
def answer_clip(row, model, tokenizer, dtype, args, hook_module=None, steer_vec=None, alpha=0.0):
    pixel_values, num_patches_list = load_video(row, dtype, args)
    handle = None
    if hook_module is not None and steer_vec is not None and alpha != 0:
        img_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        state = {"used": False}

        def hook(_, __, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
                tuple_out = True
            else:
                h = output
                rest = ()
                tuple_out = False
            # Only steer the full prefill pass. Later generation steps have seq_len=1.
            if not state["used"] and h.ndim == 3 and h.shape[0] == 1:
                ids = getattr(model, "_steer_input_ids", None)
                if ids is not None and ids.shape[-1] == h.shape[1]:
                    mask = (ids[0].to(h.device) == img_token_id).view(1, -1, 1)
                    h = h + mask.to(h.dtype) * (alpha * steer_vec.to(device=h.device, dtype=h.dtype)).view(1, 1, -1)
                    state["used"] = True
            if tuple_out:
                return (h,) + rest
            return h

        handle = hook_module.register_forward_hook(hook)
    try:
        # model.chat constructs input_ids internally; reproduce enough state for the hook by
        # monkey-patching generate's input_ids argument just before the language model runs.
        orig_generate = model.generate

        def wrapped_generate(*g_args, **g_kwargs):
            model._steer_input_ids = g_kwargs.get("input_ids")
            return orig_generate(*g_args, **g_kwargs)

        model.generate = wrapped_generate
        response = model.chat(
            tokenizer,
            pixel_values,
            make_question(len(num_patches_list)),
            dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1),
            num_patches_list=num_patches_list,
        )
        return response
    finally:
        model.generate = orig_generate
        if hasattr(model, "_steer_input_ids"):
            delattr(model, "_steer_input_ids")
        if handle is not None:
            handle.remove()


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_done_ids(path):
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.json")
    p.add_argument("--model-path", default="/root/autodl-tmp/get_hf/InternVL2")
    p.add_argument("--sae-path", default="/root/codex/outputs/temporal_m_sae_internvl_sht_clips10k_filtered_8f_16x_k256_1ep/sae_final.pt")
    p.add_argument("--output-dir", default="/root/codex/outputs/feature28158_bicycle_steer_failcase")
    p.add_argument("--feature", type=int, default=28158)
    p.add_argument("--alpha", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    p.add_argument("--hook-layer", type=int, default=12)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-patches-per-frame", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--steer-existing-fails-only", action="store_true")
    p.add_argument("--target", choices=["baseline_fail", "baseline_success"], default="baseline_fail")
    p.add_argument("--baseline-source", default="")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "baseline_bicycle_answers.jsonl"
    steer_path = out_dir / ("steered_successcase_answers.jsonl" if args.target == "baseline_success" else "steered_failcase_answers.jsonl")

    rows = [r for r in load_rows(args.data) if r.get("category_zh") == "自行车"]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer, dtype = build_model(args)
    hook_name, hook_module = topk_train.resolve_hook_module(model, args.hook_layer)
    sae = load_sae(args)
    steer_vec = sae.decoder.weight[:, args.feature].detach().float()

    baseline_read_path = Path(args.baseline_source) if args.baseline_source else baseline_path

    if args.target == "baseline_fail" and not args.steer_existing_fails_only:
        done = read_done_ids(baseline_path)
        start = time.time()
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            try:
                response = answer_clip(row, model, tokenizer, dtype, args)
                rec = {
                    "id": row["id"],
                    "video": row["video"],
                    "category_zh": row.get("category_zh"),
                    "baseline_response": response,
                    "baseline_pred_bicycle": is_bicycle_answer(response),
                }
            except Exception as exc:
                rec = {"id": row["id"], "video": row.get("video"), "error": str(exc)}
            append_jsonl(baseline_path, rec)
            if i % 25 == 0:
                print(f"baseline {i}/{len(rows)} elapsed={time.time()-start:.1f}s", flush=True)

    baseline = []
    with open(baseline_read_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                baseline.append(json.loads(line))
    baseline_valid = [r for r in baseline if not r.get("error")]
    baseline_errors = [r for r in baseline if r.get("error")]
    fail_ids = {r["id"] for r in baseline_valid if not r.get("baseline_pred_bicycle", False)}
    success_ids = {r["id"] for r in baseline_valid if r.get("baseline_pred_bicycle", False)}
    id_to_row = {r["id"]: r for r in rows}
    target_ids = success_ids if args.target == "baseline_success" else fail_ids
    target_rows = [id_to_row[i] for i in sorted(target_ids) if i in id_to_row]
    if args.limit:
        target_rows = target_rows[: args.limit]

    done_steer = set()
    if steer_path.exists():
        with open(steer_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_steer.add((rec["id"], float(rec["alpha"])))
    start = time.time()
    for i, row in enumerate(target_rows, 1):
        for alpha in args.alpha:
            key = (row["id"], float(alpha))
            if key in done_steer:
                continue
            try:
                response = answer_clip(
                    row,
                    model,
                    tokenizer,
                    dtype,
                    args,
                    hook_module=hook_module,
                    steer_vec=steer_vec,
                    alpha=float(alpha),
                )
                rec = {
                    "id": row["id"],
                    "video": row["video"],
                    "feature": args.feature,
                    "hook": hook_name,
                    "alpha": float(alpha),
                    "steered_response": response,
                    "steered_pred_bicycle": is_bicycle_answer(response),
                    "target": args.target,
                }
            except Exception as exc:
                rec = {"id": row["id"], "video": row.get("video"), "feature": args.feature, "alpha": float(alpha), "error": str(exc)}
            append_jsonl(steer_path, rec)
        if i % 10 == 0:
            print(f"steer {args.target} rows {i}/{len(target_rows)} elapsed={time.time()-start:.1f}s", flush=True)

    steered = []
    if steer_path.exists():
        with open(steer_path, encoding="utf-8") as f:
            steered = [json.loads(line) for line in f if line.strip()]
    by_alpha = {}
    for alpha in args.alpha:
        part = [r for r in steered if float(r.get("alpha", -1)) == float(alpha) and not r.get("error")]
        by_alpha[str(alpha)] = {
            "n": len(part),
            "rescued": sum(1 for r in part if r.get("steered_pred_bicycle")),
            "rescue_rate": (sum(1 for r in part if r.get("steered_pred_bicycle")) / len(part)) if part else None,
            "suppressed": sum(1 for r in part if not r.get("steered_pred_bicycle")),
            "suppression_rate": (sum(1 for r in part if not r.get("steered_pred_bicycle")) / len(part)) if part else None,
        }
    summary = {
        "data": args.data,
        "sae_path": args.sae_path,
        "feature": args.feature,
        "hook": hook_name,
        "question": make_question(args.num_frames),
        "n_bicycle_rows": len(rows),
        "baseline_done": len(baseline),
        "baseline_valid": len(baseline_valid),
        "baseline_errors": len(baseline_errors),
        "baseline_success": sum(1 for r in baseline_valid if r.get("baseline_pred_bicycle")),
        "baseline_fail": len(fail_ids),
        "target": args.target,
        "target_count": len(target_rows),
        "alphas": args.alpha,
        "steer_by_alpha": by_alpha,
        "outputs": {
            "baseline": str(baseline_path),
            "steered": str(steer_path),
            "summary": str(out_dir / "summary.json"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
