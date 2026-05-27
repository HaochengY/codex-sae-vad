import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from sae_video_harness.video_dataset import build_video_text_dataset


def init_dist():
    if "RANK" not in os.environ:
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def set_seeds_and_env(seed=0, support_bf16=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not support_bf16:
        os.environ["VLLM_ATTENTION_BACKEND"] = "SDPA"
        os.environ["VLLM_USE_TRITON"] = "0"
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def video_suffix(url_or_path):
    parsed = urlparse(str(url_or_path))
    suffix = Path(parsed.path).suffix or Path(str(url_or_path)).suffix
    return suffix if suffix else ".mp4"


def download_video(url, tmp_dir):
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix(url), dir=tmp_dir)
    tmp_path = handle.name
    handle.close()
    try:
        if shutil.which("wget"):
            subprocess.run(["wget", "-q", "-O", tmp_path, url], check=True)
        else:
            from urllib.request import urlretrieve

            urlretrieve(url, tmp_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return tmp_path


def download_file(url, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "-O", str(output_path), url], check=True)
    else:
        urlretrieve(url, output_path)
    return str(output_path)


def format_csv_url(template, index):
    try:
        return template.format(index=index, i=index)
    except (IndexError, KeyError):
        return template.format(index)


def build_remote_csv_urls(args):
    urls = []
    if args.remote_csv_urls:
        urls.extend(args.remote_csv_urls)
    if args.webvid_first_n_csvs > 0:
        urls.extend(format_csv_url(args.webvid_csv_url_template, i) for i in range(args.webvid_first_n_csvs))
    return urls


def materialize_data_paths(args, rank):
    data_paths = []
    data_path_urls = []
    for data_path in args.data_paths or []:
        if str(data_path).startswith(("http://", "https://")):
            data_path_urls.append(data_path)
        else:
            data_paths.append(data_path)
    remote_csv_urls = build_remote_csv_urls(args)
    remote_csv_urls = data_path_urls + remote_csv_urls
    if not data_paths and not remote_csv_urls:
        raise ValueError("provide --data-paths, --remote-csv-urls, or --webvid-first-n-csvs")

    if not remote_csv_urls:
        return data_paths, None

    if args.remote_csv_cache_dir:
        csv_dir = Path(args.remote_csv_cache_dir) / f"rank{rank}"
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"qwen25vl_csv_rank{rank}_")
        csv_dir = Path(temp_dir.name)

    for idx, url in enumerate(remote_csv_urls):
        parsed = urlparse(url)
        name = Path(parsed.path).name or f"remote_{idx:04d}.csv"
        if not name.lower().endswith(".csv"):
            name = f"{Path(name).stem or f'remote_{idx:04d}'}.csv"
        output_path = csv_dir / f"{idx:04d}_{name}"
        if not output_path.exists():
            download_file(url, output_path)
        data_paths.append(str(output_path))
    return data_paths, temp_dir


def materialize_video(sample, tmp_dir):
    video = str(sample["video"])
    if video.startswith(("http://", "https://")):
        return download_video(video, tmp_dir), True
    return video, False


def make_messages(video_path: str, text: str, args):
    video_item = {"type": "video", "video": video_path}
    if args.video_fps > 0:
        video_item["fps"] = args.video_fps
    if args.num_frames > 0:
        video_item["nframes"] = args.num_frames
    if args.min_pixels > 0:
        video_item["min_pixels"] = args.min_pixels
    if args.max_pixels > 0:
        video_item["max_pixels"] = args.max_pixels
    content = [video_item]
    content.append({"type": "text", "text": args.user_prompt.format(text=text)})
    return [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": content},
    ]


def collate(batch):
    return batch


def create_hex_buckets(hidden_save_dir):
    base = Path(hidden_save_dir)
    base.mkdir(parents=True, exist_ok=True)
    for i in range(256):
        (base / f"b{i:02x}").mkdir(exist_ok=True)
    (base / "others").mkdir(exist_ok=True)


def move_to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def init_qwen(args, device):
    dtype = torch.bfloat16 if args.support_bf16 else torch.float16
    try:
        import flash_attn  # noqa: F401

        has_flash_attn = True
    except ModuleNotFoundError:
        has_flash_attn = False
    attn_impl = "flash_attention_2" if args.support_bf16 and has_flash_attn else "sdpa"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    return model, processor


@torch.no_grad()
def forward_and_collect_hidden(args, model, processor, sample: Dict, device):
    local_video, should_delete = materialize_video(sample, args.video_tmp_dir)
    try:
        messages = make_messages(local_video, sample["text"], args)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=False,
            max_length=args.max_prompt_length,
            return_tensors="pt",
        )
        inputs = move_to_device(inputs, device)
        outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[int(args.sae_layer_k)][0]
        attention_mask = inputs["attention_mask"][0]
        return hidden, attention_mask
    finally:
        if should_delete:
            Path(local_video).unlink(missing_ok=True)


def maybe_write_zero_sample(hidden_save_dir, hidden, attention_mask):
    zero_path = Path(hidden_save_dir) / "zero_sample.npz"
    if zero_path.exists():
        return
    hs_np = torch.zeros_like(hidden).detach().cpu().to(torch.float16).numpy()
    am_np = torch.zeros_like(attention_mask).detach().cpu().numpy()
    np.savez(zero_path, hidden_states=hs_np, attention_mask=am_np)


def main(args):
    set_seeds_and_env(args.seed, args.support_bf16)
    rank, world_size, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    create_hex_buckets(args.hidden_save_dir)
    model, processor = init_qwen(args, device)

    data_paths, csv_temp_dir = materialize_data_paths(args, rank)
    csv_text_fields = [item.strip() for item in args.csv_text_fields.split(",") if item.strip()]
    try:
        dataset = build_video_text_dataset(
            data_paths,
            args.dataset_roots,
            args.splits,
            csv_text_fields=csv_text_fields,
        )
        if args.max_samples > 0:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=False,
        )
        if rank == 0:
            print(f"#data_paths={len(data_paths)} #samples={len(dataset)} #batches={len(dataloader)}")

        iterator = dataloader if rank != 0 else tqdm(dataloader, desc="caching qwen video activations")
        for batch in iterator:
            sample = batch[0]
            save_path = Path(args.hidden_save_dir) / sample["hidden_path"]
            if save_path.exists():
                continue
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                hidden, attention_mask = forward_and_collect_hidden(args, model, processor, sample, device)
                maybe_write_zero_sample(args.hidden_save_dir, hidden, attention_mask)
                hs_np = hidden.detach().cpu().to(torch.float16).numpy()
                am_np = attention_mask.detach().cpu().numpy()
                np.savez(save_path, hidden_states=hs_np, attention_mask=am_np)
            except Exception as exc:
                if rank == 0:
                    print(f"[warning] failed sample id={sample.get('id')} video={sample.get('video')}: {exc}")
    finally:
        if csv_temp_dir is not None:
            csv_temp_dir.cleanup()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-paths", nargs="+", default=None)
    parser.add_argument("--remote-csv-urls", nargs="+", default=None)
    parser.add_argument("--webvid-first-n-csvs", type=int, default=0)
    parser.add_argument(
        "--webvid-csv-url-template",
        default="https://huggingface.co/datasets/TempoFunk/webvid-10M/resolve/main/data/train/partitions/{index:04d}.csv?download=true",
    )
    parser.add_argument("--remote-csv-cache-dir", default=None)
    parser.add_argument("--dataset-roots", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hidden-save-dir", required=True)
    parser.add_argument("--sae-layer-k", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--video-fps", type=float, default=0.0)
    parser.add_argument("--min-pixels", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1400)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--user-prompt", default="{text}")
    parser.add_argument("--csv-text-fields", default="name")
    parser.add_argument("--video-tmp-dir", default="/tmp/qwen25vl_video_sae_cache")
    parser.add_argument("--support-bf16", type=lambda s: False if str(s).lower() == "false" else True, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(build_parser())
