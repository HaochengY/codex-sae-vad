#!/usr/bin/env python
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad_compass.internvl_utils import (
    build_internvl,
    extract_compass_hidden,
    load_video_as_pixel_values,
    resolve_hook_module,
)
from vad_compass.modeling import VadCompassHead
from vad_compass.path_utils import load_json_rows, portable_rows, resolve_path
from vad_compass.reward import (
    format_score,
    grpo_advantages,
    video_bce_loss,
)


DEFAULT_QUESTION = (
    "You are a video anomaly detection assistant.\n\n"
    "Known anomaly rule list:\n"
    "1. Skateboarding is anomalous.\n"
    "2. Riding a bicycle is anomalous.\n"
    "3. Running is anomalous.\n"
    "4. Fighting is anomalous.\n"
    "5. Driving or riding a vehicle is anomalous.\n"
    "6. Falling is anomalous.\n"
    "7. Jumping is anomalous.\n\n"
    "Determine whether this video clip contains any behavior that violates the rules above.\n\n"
    "Requirements:\n"
    "1. First, briefly explain the reasoning inside <think> </think>.\n"
    "2. Then output `Violated rules:` followed by exactly {K} rule-focus tokens. "
    "Each token must be written as {POS_TOKEN}.\n"
    "3. Do not output Answer or any other extra text.\n\n"
    "Format:\n"
    "<think> your reasoning here </think>\n"
    "Violated rules:{POS_TOKENS}"
)


def label_to_int(row):
    label = str(row.get("label", "")).strip().lower()
    if label == "abnormal":
        return 1
    if label == "normal":
        return 0
    raise ValueError(f"unknown label for {row.get('id')}: {row.get('label')}")


def split_rows(rows, train_split="all", max_samples=0, seed=0):
    if train_split == "all":
        train = rows[:]
    else:
        train = [r for r in rows if str(r.get("split", "")).lower() == train_split]
    if not train:
        train = rows[:]
    rng = random.Random(seed)
    rng.shuffle(train)
    if max_samples:
        by_label = {0: [], 1: []}
        for row in train:
            try:
                by_label[label_to_int(row)].append(row)
            except ValueError:
                continue
        if all(by_label.values()) and max_samples >= 2:
            half = max_samples // 2
            sampled = by_label[0][:half] + by_label[1][: max_samples - half]
            rng.shuffle(sampled)
            train = sampled
        else:
            train = train[:max_samples]
    return train


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reference_response(label, pos_token="<RULE>", k_slots=4):
    evidence = (
        "The clip contains behavior that appears to violate the listed rules."
        if int(label) == 1
        else "The clip does not show clear behavior that violates the listed rules."
    )
    pos_tokens = " ".join([pos_token] * int(k_slots))
    return f"<think> {evidence} </think>\nViolated rules:{pos_tokens}"


def response_for_forward(response, pos_token="<RULE>", k_slots=4):
    missing = int(k_slots) - response.count(pos_token)
    if missing <= 0:
        return response
    separator = " " if response and not response.endswith((" ", "\n")) else ""
    return response + separator + " ".join([pos_token] * missing)


def vad_response_reward(label, response, video_prob, format_weight=0.3, task_weight=0.7, pos_token="<RULE>", k_slots=4):
    fs = format_score(response, pos_token=pos_token, k_slots=k_slots)
    prob = float(video_prob)
    task_score = prob if int(label) == 1 else 1.0 - prob
    return format_weight * fs + task_weight * task_score, fs, task_score


@torch.no_grad()
def generate_rollout_responses(row, model, tokenizer, dtype, question, args):
    pixel_values, num_patches_list = load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        input_size=args.input_size,
        max_num=args.max_patches_per_frame,
    )
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    image_prefix = "\n".join(["<image>"] * len(num_patches_list))
    rollout_question = f"{image_prefix}\n{question.strip()}"
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    was_training = model.training
    model.eval()
    responses = []
    for _ in range(args.rollout_n):
        response = model.chat(
            tokenizer,
            pixel_values,
            rollout_question,
            generation_config.copy(),
            num_patches_list=num_patches_list,
        )
        responses.append(response.strip())
    if was_training:
        model.train()
    return responses


def response_log_prob(compass, tokenizer, response):
    output = compass["model_output"]
    if not hasattr(output, "logits"):
        raise ValueError("model output does not expose logits for GRPO policy loss")
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not response_ids:
        return output.logits.sum() * 0.0
    logits = output.logits[0, :-1].float()
    targets = compass["input_ids"][1:].to(logits.device)
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    n = min(len(response_ids), max(1, log_probs.numel() - 1))
    if log_probs.numel() > n:
        return log_probs[-(n + 1) : -1].sum()
    return log_probs[-n:].sum()


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data", default="data/sht_clip_32_160_conversations_filtered.json")
    parser.add_argument("--dataset-root", default="../../autodl-tmp/sht_clip_32_160")
    parser.add_argument("--model-path", default="../../autodl-tmp/get_hf/InternVL2")
    parser.add_argument("--output-dir", default="outputs/vad_compass_internvl2_sht")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--pos-token", default="<RULE>")
    parser.add_argument("--hook-layer", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-patches-per-frame", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-split", choices=["all", "training", "testing"], default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--sae-path", default="")
    parser.add_argument("--num-latents", type=int, default=0)
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--sae-topk", type=int, default=256)
    parser.add_argument("--k-slots", type=int, default=4)
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--slot-heads", type=int, default=8)
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--freeze-internvl", action="store_true")
    parser.add_argument("--lr", type=float, default=1.6e-6)
    parser.add_argument("--lr-qb-rate", type=float, default=30.0)
    parser.add_argument("--lr-head-rate", type=float, default=25.0)
    parser.add_argument("--lr-pe-rate", type=float, default=10.0)
    parser.add_argument("--lr-decoder-rate", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-sae", action="store_true")
    parser.add_argument("--train-rule-embedding", action="store_true")
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-grpo", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.05)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--format-weight", type=float, default=0.3)
    parser.add_argument("--task-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--tensorboard-logdir", default="")
    parser.add_argument("--debug-rollouts", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def train(args):
    project_root = Path(args.project_root).resolve()
    data_path = resolve_path(args.data, project_root)
    dataset_root = resolve_path(args.dataset_root, project_root)
    model_path = resolve_path(args.model_path, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    sae_path = resolve_path(args.sae_path, project_root) if args.sae_path else None
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = portable_rows(load_json_rows(data_path), dataset_root)
    rows = split_rows(
        rows,
        train_split=args.train_split,
        max_samples=args.max_samples or (2 if args.smoke else 0),
        seed=args.seed,
    )
    label_counts = Counter(str(r.get("label", "")).lower() for r in rows)
    if args.smoke:
        args.epochs = 1
        args.max_steps = min(args.max_steps or 2, 2)

    model, tokenizer, dtype, special_token_ids = build_internvl(
        model_path,
        dtype=args.dtype,
        special_tokens=[args.pos_token],
        train_special_tokens=args.train_rule_embedding,
    )
    if args.freeze_internvl:
        model.eval()
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
    pos_token_id = special_token_ids[args.pos_token]
    hook_name, hook_module = resolve_hook_module(model, args.hook_layer)
    _, final_hook_module = resolve_hook_module(model, model.config.llm_config.num_hidden_layers - 1)
    d_in = model.config.llm_config.hidden_size
    sae_checkpoint = None
    if sae_path:
        layer_match = re.search(r"sae_l(\d+)", str(sae_path))
        if layer_match and int(layer_match.group(1)) != int(args.hook_layer):
            raise ValueError(
                f"SAE path layer sae_l{layer_match.group(1)} does not match --hook-layer {args.hook_layer}: {sae_path}"
            )
        sae_checkpoint = torch.load(sae_path, map_location="cpu")
        ckpt_d_in = int(sae_checkpoint.get("d_in", d_in)) if isinstance(sae_checkpoint, dict) else d_in
        if ckpt_d_in != int(d_in):
            raise ValueError(f"SAE d_in {ckpt_d_in} does not match model hidden size {d_in}: {sae_path}")
        if not args.num_latents and isinstance(sae_checkpoint, dict) and sae_checkpoint.get("num_latents"):
            args.num_latents = int(sae_checkpoint["num_latents"])
        ckpt_args = sae_checkpoint.get("args", {}) if isinstance(sae_checkpoint, dict) else {}
        if args.sae_topk == 0 and ckpt_args.get("k"):
            args.sae_topk = int(ckpt_args["k"])
    num_latents = args.num_latents or d_in * args.expansion_factor
    vad = VadCompassHead(
        d_in=d_in,
        num_latents=num_latents,
        topk=args.sae_topk,
        k_slots=args.k_slots,
        slot_dim=args.slot_dim,
        nhead=args.slot_heads,
        num_layers=args.slot_layers,
    ).cuda()
    if sae_path:
        checkpoint = sae_checkpoint
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        state = checkpoint.get("sae", state)
        sae_state = {}
        for key, value in state.items():
            if key.startswith("sae."):
                sae_state[key[len("sae.") :]] = value
            elif key.startswith("module.sae."):
                sae_state[key[len("module.sae.") :]] = value
            elif key in vad.sae.state_dict():
                sae_state[key] = value
        missing, unexpected = vad.sae.load_state_dict(sae_state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected SAE keys from {sae_path}: {unexpected}")
        print(f"loaded SAE from {sae_path} missing={missing}")
    if not args.train_sae:
        for p in vad.sae.parameters():
            p.requires_grad_(False)
    param_groups = []

    def add_group(name, params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            param_groups.append({"name": name, "params": params, "lr": lr})

    add_group("query_book", vad.query_book.parameters(), args.lr * args.lr_qb_rate)
    add_group("head", list(vad.query_fuse.parameters()) + list(vad.conf_head.parameters()), args.lr * args.lr_head_rate)
    add_group("pe", vad.pos_proj.parameters(), args.lr * args.lr_pe_rate)
    if args.train_sae:
        add_group("sae", vad.sae.parameters(), args.lr * args.lr_decoder_rate)
    add_group("internvl", model.parameters(), args.lr)
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = output_dir / "metrics.jsonl"
    errors_path = output_dir / "errors.jsonl"
    tb_writer = None
    if args.tensorboard_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_logdir = resolve_path(args.tensorboard_logdir, project_root)
            tb_logdir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_logdir))
        except ModuleNotFoundError as exc:
            print(f"TensorBoard logging disabled: {exc}")

    config = vars(args).copy()
    config.update(
        {
            "project_root": str(project_root),
            "data_resolved": str(data_path),
            "dataset_root_resolved": str(dataset_root),
            "model_path_resolved": str(model_path),
            "hook": hook_name,
            "d_in": d_in,
            "num_latents": num_latents,
            "sae_path_resolved": str(sae_path) if sae_path else "",
            "optimizer_groups": [
                {"name": group.get("name", ""), "lr": group["lr"], "num_params": sum(p.numel() for p in group["params"])}
                for group in param_groups
            ],
            "pos_token": args.pos_token,
            "pos_token_id": pos_token_id,
            "internvl_trainable": not args.freeze_internvl,
            "label_counts": dict(label_counts),
            "loss": "lambda_grpo*GRPO(generated response reward) + lambda_bce*BCE(video_prob,label); SAE_MSE is added only when --train-sae is set",
            "tensorboard_logdir": str(resolve_path(args.tensorboard_logdir, project_root)) if args.tensorboard_logdir else "",
            "path_policy": "All CLI paths are resolved relative to --project-root unless absolute. JSON media paths are converted to paths relative to --dataset-root.",
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    global_step = 0
    start = time.time()
    for epoch in range(args.epochs):
        random.shuffle(rows)
        pbar = tqdm(rows, desc=f"VAD-Compass epoch {epoch + 1}/{args.epochs}")
        for row_idx, row in enumerate(pbar):
            if args.max_steps and global_step >= args.max_steps:
                break
            try:
                label = torch.tensor([label_to_int(row)], device="cuda", dtype=torch.float32)
                question = args.question.format(
                    K=args.k_slots,
                    POS_TOKEN=args.pos_token,
                    POS_TOKENS=" ".join([args.pos_token] * args.k_slots),
                )
                gold_response = reference_response(int(label.item()), pos_token=args.pos_token, k_slots=args.k_slots)
                compass = extract_compass_hidden(
                    row,
                    model,
                    tokenizer,
                    dtype,
                    hook_module,
                    final_hook_module,
                    question,
                    gold_response,
                    pos_token_id,
                    args,
                )
                out = vad(compass["sae_tokens"], compass["pos_hidden"])
                video_prob = out["video_prob"]
                prob_value = float(video_prob.detach().item())
                bce_loss = video_bce_loss(video_prob, label)
                recon_loss = out["recon_loss"]

                rewards, format_scores, task_scores = [], [], []
                candidate_log_probs = []
                old_log_probs = []
                rollout_debug = []
                rollout_responses = generate_rollout_responses(row, model, tokenizer, dtype, question, args)
                for rollout_idx, response in enumerate(rollout_responses):
                    forward_response = response_for_forward(response, pos_token=args.pos_token, k_slots=args.k_slots)
                    cand = extract_compass_hidden(
                        row,
                        model,
                        tokenizer,
                        dtype,
                        hook_module,
                        final_hook_module,
                        question,
                        forward_response,
                        pos_token_id,
                        args,
                    )
                    cand_out = vad(cand["sae_tokens"], cand["pos_hidden"])
                    lp = response_log_prob(cand, tokenizer, response)
                    candidate_log_probs.append(lp)
                    old_log_probs.append(lp.detach())
                    reward, fs, ts = vad_response_reward(
                        int(label.item()),
                        response,
                        float(cand_out["video_prob"].detach().item()),
                        format_weight=args.format_weight,
                        task_weight=args.task_weight,
                        pos_token=args.pos_token,
                        k_slots=args.k_slots,
                    )
                    rewards.append(reward)
                    format_scores.append(fs)
                    task_scores.append(ts)
                    rollout_debug.append(
                        {
                            "idx": rollout_idx,
                            "response": response,
                            "format_score": float(fs),
                            "task_score": float(ts),
                            "reward": float(reward),
                            "video_prob": float(cand_out["video_prob"].detach().item()),
                            "log_prob": float(lp.detach().cpu()),
                        }
                    )

                rewards_t = torch.tensor(rewards, device="cuda", dtype=torch.float32)
                advantages = grpo_advantages(rewards_t, args.rollout_n)
                log_probs = torch.stack(candidate_log_probs)
                old_log_probs = torch.stack(old_log_probs)
                ratio = torch.exp(log_probs - old_log_probs)
                pg_losses = -advantages * ratio
                pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                grpo_loss = torch.max(pg_losses, pg_losses2).mean()
                approx_kl = (old_log_probs - log_probs).mean()
                clipfrac = (pg_losses2 > pg_losses).float().mean()

                loss = (
                    args.lambda_grpo * grpo_loss
                    + args.lambda_bce * bce_loss
                    + (args.lambda_recon * recon_loss if args.train_sae else 0.0)
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.train_sae:
                    vad.sae.remove_decoder_parallel_grad()
                torch.nn.utils.clip_grad_norm_(vad.parameters(), args.max_grad_norm)
                opt.step()
                if args.train_sae:
                    vad.sae.normalize_decoder()

                pred = int(prob_value >= 0.5)
                rec = {
                    "step": global_step,
                    "epoch": epoch,
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "video_rel": row.get("video_rel"),
                    "label": int(label.item()),
                    "pred": pred,
                    "video_prob": prob_value,
                    "loss": float(loss.detach().cpu()),
                    "bce_loss": float(bce_loss.detach().cpu()),
                    "grpo_loss": float(grpo_loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "reward_mean": float(np.mean(rewards)),
                    "format_score_mean": float(np.mean(format_scores)),
                    "task_score_mean": float(np.mean(task_scores)),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "clipfrac": float(clipfrac.detach().cpu()),
                    "slot_probs": [float(x) for x in out["slot_probs"][0].detach().cpu().tolist()],
                    "rollout_n": len(rollout_responses),
                    "sample_responses": rollout_responses[:2],
                    "elapsed_sec": round(time.time() - start, 2),
                }
                if rollout_debug:
                    best_rollout = max(rollout_debug, key=lambda item: item["reward"])
                    rec["best_rollout_idx"] = best_rollout["idx"]
                    rec["best_rollout_reward"] = best_rollout["reward"]
                    rec["best_rollout_response"] = best_rollout["response"]
                    if args.debug_rollouts:
                        rec["rollouts"] = rollout_debug
                if global_step % args.log_every == 0:
                    append_jsonl(metrics_path, rec)
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss/total", rec["loss"], global_step)
                        tb_writer.add_scalar("loss/bce", rec["bce_loss"], global_step)
                        tb_writer.add_scalar("loss/grpo", rec["grpo_loss"], global_step)
                        tb_writer.add_scalar("loss/recon", rec["recon_loss"], global_step)
                        tb_writer.add_scalar("reward/mean", rec["reward_mean"], global_step)
                        tb_writer.add_scalar("reward/format_score", rec["format_score_mean"], global_step)
                        tb_writer.add_scalar("reward/task_score", rec["task_score_mean"], global_step)
                        tb_writer.add_scalar("prediction/video_prob", rec["video_prob"], global_step)
                        tb_writer.add_scalar("prediction/label", rec["label"], global_step)
                        tb_writer.add_scalar("optimization/approx_kl", rec["approx_kl"], global_step)
                        tb_writer.add_scalar("optimization/clipfrac", rec["clipfrac"], global_step)
                        for slot_idx, slot_prob in enumerate(rec["slot_probs"]):
                            tb_writer.add_scalar(f"slot_conf/slot_{slot_idx}", slot_prob, global_step)
                        tb_writer.flush()
                    pbar.set_postfix(
                        p=round(rec["video_prob"], 3),
                        y=rec["label"],
                        loss=round(rec["loss"], 4),
                        reward=round(rec["reward_mean"], 3),
                    )
                if global_step > 0 and global_step % args.save_every == 0:
                    torch.save(
                        {
                            "model": vad.state_dict(),
                            "args": config,
                            "step": global_step,
                            "d_in": d_in,
                            "num_latents": num_latents,
                        },
                        output_dir / f"vad_compass_step_{global_step}.pt",
                    )
                global_step += 1
            except Exception as exc:
                append_jsonl(
                    errors_path,
                    {"step": global_step, "id": row.get("id"), "video": row.get("video"), "error": str(exc)},
                )
                global_step += 1
                continue
        if args.max_steps and global_step >= args.max_steps:
            break

    final_path = output_dir / "vad_compass_final.pt"
    torch.save(
        {
            "model": vad.state_dict(),
            "args": config,
            "step": global_step,
            "d_in": d_in,
            "num_latents": num_latents,
        },
        final_path,
    )
    internvl_final = ""
    if not args.freeze_internvl:
        internvl_dir = output_dir / "internvl_final"
        internvl_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(internvl_dir))
        tokenizer.save_pretrained(str(internvl_dir))
        internvl_final = str(internvl_dir)
    print(json.dumps({"final": str(final_path), "internvl_final": internvl_final, "steps": global_step}, ensure_ascii=False, indent=2))
    if tb_writer is not None:
        tb_writer.close()
    return final_path


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
