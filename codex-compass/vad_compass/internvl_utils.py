import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
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
        if min_num <= i * j <= max_num
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


def build_internvl(model_path, dtype="bfloat16", special_tokens=None, train_special_tokens=False):
    model_path = Path(model_path).resolve()
    (model_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(model_path.parent))
    module = importlib.import_module(f"{model_path.name}.modeling_internvl_chat")
    cls = module.InternVLChatModel
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, use_fast=False
    )
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = cls.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    ).eval().cuda()
    special_token_ids = {}
    if special_tokens:
        added = tokenizer.add_tokens(list(special_tokens), special_tokens=True)
        if added > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
        for token in special_tokens:
            special_token_ids[token] = int(tokenizer.convert_tokens_to_ids(token))
    for p in model.parameters():
        p.requires_grad_(False)
    if train_special_tokens and special_tokens:
        emb = model.language_model.get_input_embeddings()
        emb.weight.requires_grad_(True)
        emb.weight.register_hook(
            lambda grad: _mask_embedding_grad(grad, list(special_token_ids.values()))
        )
    return model, tokenizer, torch_dtype, special_token_ids


def _mask_embedding_grad(grad, token_ids):
    mask = torch.zeros_like(grad)
    mask[token_ids] = 1
    return grad * mask


def build_prompt(model, tokenizer, question, num_patches_list, assistant_response=None):
    model_path = Path(model.config.name_or_path).resolve()
    sys.path.insert(0, str(model_path.parent))
    get_conv_template = importlib.import_module(
        f"{model_path.name}.conversation"
    ).get_conv_template
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    image_prefix = "\n".join(["<image>"] * len(num_patches_list))
    template.append_message(template.roles[0], f"{image_prefix}\n{question.strip()}")
    template.append_message(template.roles[1], assistant_response)
    query = template.get_prompt()
    for num_patches in num_patches_list:
        token_str = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", token_str, 1)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return query


def resolve_hook_module(model, hook_layer):
    name = f"language_model.model.layers.{hook_layer}"
    return name, model.get_submodule(name)


def extract_compass_hidden(row, model, tokenizer, dtype, sae_hook_module, final_hook_module, question, response, pos_token_id, args):
    pixel_values, num_patches_list = load_video_as_pixel_values(
        row["video"],
        num_frames=args.num_frames,
        input_size=args.input_size,
        max_num=args.max_patches_per_frame,
    )
    prompt = build_prompt(model, tokenizer, question, num_patches_list, assistant_response=response)
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].cuda()
    attention_mask = tok["attention_mask"].cuda()
    pixel_values = pixel_values.to(dtype=dtype, device="cuda")
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device="cuda")
    buffers = {}

    def sae_hook(_, __, output):
        if isinstance(output, tuple):
            output = output[0]
        buffers["sae_hidden"] = output

    def final_hook(_, __, output):
        if isinstance(output, tuple):
            output = output[0]
        buffers["final_hidden"] = output

    sae_handle = sae_hook_module.register_forward_hook(sae_hook)
    final_handle = final_hook_module.register_forward_hook(final_hook)
    try:
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
        )
    finally:
        sae_handle.remove()
        final_handle.remove()

    valid = attention_mask[0].bool()
    sae_h = buffers["sae_hidden"][0][valid].float()
    final_h = buffers["final_hidden"][0][valid].float()
    ids = input_ids[0][valid]
    sae_h = torch.nan_to_num(sae_h, nan=0.0, posinf=0.0, neginf=0.0)
    final_h = torch.nan_to_num(final_h, nan=0.0, posinf=0.0, neginf=0.0)
    image_positions = (ids == model.img_context_token_id).nonzero(as_tuple=False).flatten()
    h_img = sae_h[image_positions]
    expected = sum(num_patches_list) * model.num_image_token
    if h_img.shape[0] != expected:
        raise ValueError(f"image token mismatch: got {h_img.shape[0]}, expected {expected}")
    if len(set(num_patches_list)) != 1:
        raise ValueError(f"variable patches per frame are unsupported: {num_patches_list}")
    t = len(num_patches_list)
    p = num_patches_list[0] * model.num_image_token
    pos = (ids == int(pos_token_id)).nonzero(as_tuple=False).flatten()
    if pos.numel() < args.k_slots:
        raise ValueError(f"expected at least {args.k_slots} POS tokens, got {int(pos.numel())}")
    pos = pos[-args.k_slots :]
    return {
        "sae_tokens": sae_h,
        "image_grid": h_img.reshape(t, p, -1),
        "pos_hidden": final_h[pos],
        "input_ids": ids,
        "prompt": prompt,
        "model_output": out,
    }
