#!/usr/bin/env python
import argparse
import importlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

import train_topk_sae_internvl_sht as base


class TopKMatryoshkaSae(nn.Module):
    def __init__(self, d_in, num_latents, k, high_frac=0.2):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k
        self.high_latents = max(1, int(num_latents * high_frac))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, num_latents)
        self.decoder = nn.Linear(num_latents, d_in, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def remove_decoder_parallel_grad(self):
        if self.decoder.weight.grad is None:
            return
        grad = self.decoder.weight.grad
        weight = self.decoder.weight.data
        parallel = (grad * weight).sum(dim=0, keepdim=True) * weight
        grad.sub_(parallel)

    def pre_acts(self, x):
        return F.relu(self.encoder(x - self.b_dec))

    def encode_topk(self, x):
        acts = self.pre_acts(x)
        values, indices = torch.topk(acts, k=min(self.k, acts.shape[-1]), dim=-1, sorted=False)
        return values, indices

    def decode_topk(self, values, indices, high_only=False):
        if high_only:
            mask = indices < self.high_latents
            values = values * mask.to(values.dtype)
        # decoder.weight.T is [num_latents, d_in], so embedding gathers decoder vectors.
        dec_vecs = F.embedding(indices, self.decoder.weight.T)
        return (dec_vecs * values.unsqueeze(-1)).sum(dim=1) + self.b_dec

    def dense_sparse(self, values, indices):
        z = torch.zeros(values.shape[0], self.num_latents, device=values.device, dtype=values.dtype)
        z.scatter_(1, indices, values)
        return z

    def forward(self, x, need_dense=False):
        values, indices = self.encode_topk(x)
        x_hat = self.decode_topk(values, indices)
        x_hat_high = self.decode_topk(values, indices, high_only=True)
        z = self.dense_sparse(values, indices) if need_dense else None
        return x_hat, x_hat_high, values, indices, z


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def build_model(args):
    model_path = Path(args.model_path).resolve()
    (model_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(model_path.parent))
    InternVLChatModel = importlib.import_module(
        f"{model_path.name}.modeling_internvl_chat"
    ).InternVLChatModel
    if not hasattr(InternVLChatModel, "all_tied_weights_keys"):
        InternVLChatModel.all_tied_weights_keys = {}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = InternVLChatModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer, dtype


def extract_image_hidden(row, model, tokenizer, dtype, hook_module, args):
    pixel_values, num_patches_list = base.load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        max_num=args.max_patches_per_frame,
    )
    prompt = base.build_full_prompt(model, tokenizer, row["conversations"], num_patches_list)
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].cuda()
    attention_mask = tok["attention_mask"].cuda()
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device="cuda")
    buffer = {}

    def hook(_, __, output):
        if isinstance(output, tuple):
            output = output[0]
        buffer["hidden"] = output.detach()

    handle = hook_module.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
        )
    handle.remove()

    valid = attention_mask[0].bool()
    h = buffer["hidden"][0][valid].float()
    ids = input_ids[0][valid]
    h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    image_positions = (ids == model.img_context_token_id).nonzero(as_tuple=False).flatten()
    h_img = h[image_positions]
    expected = sum(num_patches_list) * model.num_image_token
    if h_img.shape[0] != expected:
        raise ValueError(
            f"image token mismatch for {row.get('id')}: got {h_img.shape[0]}, expected {expected}"
        )
    # This script is designed for one dynamic image patch per frame; otherwise P varies.
    if len(set(num_patches_list)) != 1:
        raise ValueError(f"variable patches per frame are unsupported: {num_patches_list}")
    t = len(num_patches_list)
    p = num_patches_list[0] * model.num_image_token
    return h_img.reshape(t, p, -1)


def temporal_contrastive_loss(z_grid, tau=0.1):
    # z_grid: [T, P, H], dense sparse activations after TopK.
    if z_grid.shape[0] < 2:
        return z_grid.new_tensor(0.0), 0.0
    losses = []
    with torch.no_grad():
        # Positive feature reuse rate is a useful diagnostic for temporal coherence.
        prev_fire = z_grid[:-1] > 0
        next_fire = z_grid[1:] > 0
        union = (prev_fire | next_fire).float().sum(dim=-1).clamp_min(1.0)
        overlap = ((prev_fire & next_fire).float().sum(dim=-1) / union).mean().item()
    for t in range(z_grid.shape[0] - 1):
        a = F.normalize(z_grid[t], dim=-1)
        b = F.normalize(z_grid[t + 1], dim=-1)
        logits = a @ b.T / tau
        labels = torch.arange(logits.shape[0], device=logits.device)
        losses.append(F.cross_entropy(logits, labels))
    return torch.stack(losses).mean(), overlap


def _resolve_path(path, root):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root).resolve() / path


def _video_path_relative_to_dataset(value, dataset_root):
    """Map JSON video paths onto the caller-provided dataset_root.

    Old dataset JSONs may contain machine-specific absolute paths such as
    /root/autodl-tmp/sht_clip_32_160/train_normal/foo.mp4. For portable runs,
    only the path below the dataset directory should be used.
    """
    path = Path(value)
    if not path.is_absolute():
        return path

    parts = path.parts
    dataset_name = Path(dataset_root).name
    anchors = [dataset_name, "sht_clip_32_160"]
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx + 1 :])
    return Path(path.name)


def _resolve_row_video_paths(rows, dataset_root):
    dataset_root = Path(dataset_root).resolve()
    out = []
    for row in rows:
        row = dict(row)
        for key in ("video",):
            value = row.get(key)
            if value:
                row[key] = str(dataset_root / _video_path_relative_to_dataset(value, dataset_root))
        out.append(row)
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--data", default="data/sht_video_concept_conversations_clean.json")
    parser.add_argument("--model-path", default="models/InternVL2")
    parser.add_argument("--output-dir", default="outputs/temporal_matryoshka_sae_internvl_sht")
    parser.add_argument("--hook-layer", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-patches-per-frame", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-latents", type=int, default=0)
    parser.add_argument("--expansion-factor", type=int, default=16)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-temp", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alpha-mat", type=float, default=0.1)
    parser.add_argument("--high-frac", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser


def train_temporal_matryoshka_sae(
    *,
    data="data/sht_video_concept_conversations_clean.json",
    dataset_root=".",
    model_path="models/InternVL2",
    output_dir="outputs/temporal_matryoshka_sae_internvl_sht",
    project_root=".",
    hook_layer=12,
    num_frames=8,
    max_patches_per_frame=1,
    epochs=10,
    max_samples=0,
    num_latents=0,
    expansion_factor=16,
    k=256,
    lr=2e-4,
    lambda_temp=0.1,
    tau=0.1,
    alpha_mat=0.1,
    high_frac=0.2,
    log_every=5,
    save_every=100,
    seed=0,
    dtype="bfloat16",
):
    args = argparse.Namespace(
        data=str(_resolve_path(data, project_root)),
        dataset_root=str(_resolve_path(dataset_root, project_root)),
        model_path=str(_resolve_path(model_path, project_root)),
        output_dir=str(_resolve_path(output_dir, project_root)),
        project_root=str(Path(project_root).resolve()),
        hook_layer=hook_layer,
        num_frames=num_frames,
        max_patches_per_frame=max_patches_per_frame,
        epochs=epochs,
        max_samples=max_samples,
        num_latents=num_latents,
        expansion_factor=expansion_factor,
        k=k,
        lr=lr,
        lambda_temp=lambda_temp,
        tau=tau,
        alpha_mat=alpha_mat,
        high_frac=high_frac,
        log_every=log_every,
        save_every=save_every,
        seed=seed,
        dtype=dtype,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, dtype = build_model(args)
    hook_name, hook_module = base.resolve_hook_module(model, args.hook_layer)
    d_in = model.config.llm_config.hidden_size
    num_latents = args.num_latents or d_in * args.expansion_factor
    sae = TopKMatryoshkaSae(
        d_in=d_in,
        num_latents=num_latents,
        k=args.k,
        high_frac=args.high_frac,
    ).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    rows = _resolve_row_video_paths(load_rows(args.data), args.dataset_root)
    if args.max_samples:
        rows = rows[:]
        random.shuffle(rows)
        rows = rows[: args.max_samples]

    metrics_path = out_dir / "metrics.jsonl"
    errors_path = out_dir / "errors.jsonl"
    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(rows)
        pbar = tqdm(rows, desc=f"Temporal+M SAE epoch {epoch + 1}/{args.epochs}")
        for row_idx, row in enumerate(pbar):
            try:
                h_grid = extract_image_hidden(row, model, tokenizer, dtype, hook_module, args)
                t, p, d = h_grid.shape
                x = h_grid.reshape(t * p, d)
                x_hat, x_hat_high, values, indices, z = sae(x, need_dense=True)
                recon_loss = F.mse_loss(x_hat, x)
                mat_loss = F.mse_loss(x_hat_high, x)
                z_grid = z.reshape(t, p, -1)
                temp_loss, temporal_overlap = temporal_contrastive_loss(z_grid, tau=args.tau)
                loss = recon_loss + args.alpha_mat * mat_loss + args.lambda_temp * temp_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                sae.remove_decoder_parallel_grad()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                sae.normalize_decoder()

                denom = x.var(dim=0).sum().clamp_min(1e-6)
                fvu = ((x_hat - x).pow(2).sum(dim=-1).mean() / denom).detach()
                high_fvu = ((x_hat_high - x).pow(2).sum(dim=-1).mean() / denom).detach()
                active = float(values.shape[1])
                if global_step % args.log_every == 0:
                    rec = {
                        "step": global_step,
                        "epoch": epoch,
                        "row_idx": row_idx,
                        "id": row.get("id"),
                        "hook": hook_name,
                        "image_tokens": int(x.shape[0]),
                        "loss": float(loss.detach().cpu()),
                        "recon_loss": float(recon_loss.detach().cpu()),
                        "mat_loss": float(mat_loss.detach().cpu()),
                        "temp_loss": float(temp_loss.detach().cpu()),
                        "fvu": float(fvu.cpu()),
                        "high_fvu": float(high_fvu.cpu()),
                        "active_latents": active,
                        "temporal_overlap": float(temporal_overlap),
                    }
                    with open(metrics_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pbar.set_postfix(
                        loss=rec["loss"],
                        fvu=rec["fvu"],
                        temp=rec["temp_loss"],
                        overlap=rec["temporal_overlap"],
                    )

                if global_step > 0 and global_step % args.save_every == 0:
                    torch.save(
                        {
                            "sae": sae.state_dict(),
                            "args": vars(args),
                            "step": global_step,
                            "epoch": epoch,
                            "hook": hook_name,
                            "d_in": d_in,
                            "num_latents": num_latents,
                            "high_latents": sae.high_latents,
                        },
                        out_dir / f"sae_step_{global_step}.pt",
                    )
                global_step += 1
            except Exception as exc:
                with open(errors_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"epoch": epoch, "id": row.get("id"), "error": str(exc)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

    torch.save(
        {
            "sae": sae.state_dict(),
            "args": vars(args),
            "step": global_step,
            "epoch": args.epochs,
            "hook": hook_name,
            "d_in": d_in,
            "num_latents": num_latents,
            "high_latents": sae.high_latents,
        },
        out_dir / "sae_final.pt",
    )
    final_path = out_dir / "sae_final.pt"
    print(f"saved final Temporal+M SAE to {final_path}")
    return final_path


def main():
    args = build_arg_parser().parse_args()
    return train_temporal_matryoshka_sae(
        data=args.data,
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        project_root=args.project_root,
        hook_layer=args.hook_layer,
        num_frames=args.num_frames,
        max_patches_per_frame=args.max_patches_per_frame,
        epochs=args.epochs,
        max_samples=args.max_samples,
        num_latents=args.num_latents,
        expansion_factor=args.expansion_factor,
        k=args.k,
        lr=args.lr,
        lambda_temp=args.lambda_temp,
        tau=args.tau,
        alpha_mat=args.alpha_mat,
        high_frac=args.high_frac,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
