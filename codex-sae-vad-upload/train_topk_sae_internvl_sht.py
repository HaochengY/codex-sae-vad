#!/usr/bin/env python
import argparse
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def sample_video_frames(video_path, num_frames):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) == 0:
        raise ValueError(f"empty video: {video_path}")
    indices = np.linspace(0, len(vr) - 1, num=min(num_frames, len(vr)), dtype=int)
    return [Image.fromarray(frame) for frame in vr.get_batch(indices).asnumpy()]


def load_video_as_pixel_values(video_path, num_frames=8, input_size=448, max_num=1):
    transform = build_transform(input_size)
    frames = sample_video_frames(video_path, num_frames)
    pixel_values = []
    num_patches_list = []
    for frame in frames:
        images = dynamic_preprocess(
            frame, image_size=input_size, use_thumbnail=False, max_num=max_num
        )
        patches = torch.stack([transform(img) for img in images])
        pixel_values.append(patches)
        num_patches_list.append(patches.shape[0])
    return torch.cat(pixel_values, dim=0), num_patches_list


class SHTConversationDataset(Dataset):
    def __init__(self, json_path, max_samples=0, seed=0):
        with open(json_path, "r", encoding="utf-8") as f:
            if str(json_path).endswith(".jsonl"):
                self.rows = [json.loads(line) for line in f if line.strip()]
            else:
                self.rows = json.load(f)
        if max_samples:
            rng = random.Random(seed)
            self.rows = self.rows[:]
            rng.shuffle(self.rows)
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class TopKSae(nn.Module):
    def __init__(self, d_in, num_latents, k):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k
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

    def encode(self, x):
        acts = F.relu(self.encoder(x - self.b_dec))
        k = min(self.k, acts.shape[-1])
        values, indices = torch.topk(acts, k=k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indices, values)
        return sparse

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z) + self.b_dec
        return x_hat, z


def build_full_prompt(model, tokenizer, conversations, num_patches_list):
    model_path = Path(model.config.name_or_path).resolve()
    init_file = model_path / "__init__.py"
    init_file.touch(exist_ok=True)
    sys.path.insert(0, str(model_path.parent))
    get_conv_template = importlib.import_module(
        f"{model_path.name}.conversation"
    ).get_conv_template

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    roles = template.roles
    image_prefix = "\n".join(["<image>"] * len(num_patches_list))
    first_human = True
    for msg in conversations:
        role = roles[0] if msg["from"] == "human" else roles[1]
        value = msg["value"]
        if msg["from"] == "human" and first_human:
            value = value.replace("<video>", image_prefix).replace("<image>", image_prefix)
            first_human = False
        else:
            value = value.replace("<video>", "").replace("<image>", "")
        template.append_message(role, value.strip())
    query = template.get_prompt()
    image_tokens = []
    for num_patches in num_patches_list:
        image_tokens.append(
            "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        )
    for token_str in image_tokens:
        query = query.replace("<image>", token_str, 1)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return query


def resolve_hook_module(model, hook_layer):
    name = f"language_model.model.layers.{hook_layer}"
    return name, model.get_submodule(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/codex/data/sht_video_concept_conversations_clean.json")
    parser.add_argument("--model-path", default="/root/autodl-tmp/get_hf/InternVL2")
    parser.add_argument("--output-dir", default="/root/codex/outputs/topk_sae_internvl_sht")
    parser.add_argument("--hook-layer", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-patches-per-frame", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--num-latents", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    hook_name, hook_module = resolve_hook_module(model, args.hook_layer)
    d_in = model.config.llm_config.hidden_size
    sae = TopKSae(d_in=d_in, num_latents=args.num_latents, k=args.k).cuda()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    dataset = SHTConversationDataset(args.data, max_samples=args.max_samples, seed=args.seed)

    metrics_path = out_dir / "metrics.jsonl"
    global_step = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(dataset, desc="training TopK SAE")
    for row_idx, row in enumerate(pbar):
        if global_step >= args.max_steps:
            break
        try:
            pixel_values, num_patches_list = load_video_as_pixel_values(
                row["video"],
                num_frames=args.num_frames,
                max_num=args.max_patches_per_frame,
            )
            prompt = build_full_prompt(model, tokenizer, row["conversations"], num_patches_list)
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

            h = buffer["hidden"][0]
            valid = attention_mask[0].bool()
            h = h[valid].float()
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            if h.numel() == 0:
                continue

            x_hat, z = sae(h)
            mse = F.mse_loss(x_hat, h)
            denom = h.var(dim=0).sum().clamp_min(1e-6)
            fvu = ((x_hat - h).pow(2).sum(dim=-1).mean() / denom).detach()
            loss = mse / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sae.normalize_decoder()

            active = (z > 0).float().sum(dim=-1).mean().item()
            if global_step % args.log_every == 0:
                rec = {
                    "step": global_step,
                    "row_idx": row_idx,
                    "id": row.get("id"),
                    "hook": hook_name,
                    "tokens": int(h.shape[0]),
                    "loss": float(mse.detach().cpu()),
                    "fvu": float(fvu.cpu()),
                    "active_latents": active,
                }
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.set_postfix(loss=rec["loss"], fvu=rec["fvu"], active=active)

            if global_step > 0 and global_step % args.save_every == 0:
                ckpt = {
                    "sae": sae.state_dict(),
                    "args": vars(args),
                    "step": global_step,
                    "hook": hook_name,
                    "d_in": d_in,
                }
                torch.save(ckpt, out_dir / f"sae_step_{global_step}.pt")
            global_step += 1
        except Exception as exc:
            with open(out_dir / "errors.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": row.get("id"), "error": str(exc)}, ensure_ascii=False) + "\n")
            continue

    torch.save(
        {
            "sae": sae.state_dict(),
            "args": vars(args),
            "step": global_step,
            "hook": hook_name,
            "d_in": d_in,
        },
        out_dir / "sae_final.pt",
    )
    print(f"saved final SAE to {out_dir / 'sae_final.pt'}")


if __name__ == "__main__":
    main()
