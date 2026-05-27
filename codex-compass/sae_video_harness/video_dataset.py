import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset, ConcatDataset


def load_json_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_media_path(value, dataset_root=None):
    path = Path(value)
    if path.is_absolute() or dataset_root is None:
        return path
    return Path(dataset_root).resolve() / path


def infer_text(row):
    for key in ("text", "prompt", "question", "caption", "description"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = row.get("conversations") or row.get("conversation")
    if isinstance(conversations, list):
        for item in conversations:
            value = item.get("value") if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                return value.replace("<video>", "").replace("<image>", "").strip()
    label = str(row.get("label", "")).strip()
    if label:
        return f"Describe this video. The dataset label is {label}."
    return "Describe this video."


def infer_csv_text(row, text_fields=None):
    fields = text_fields or ["name"]
    parts = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if parts:
        return " ".join(parts)
    return infer_text(row)


def stable_hidden_name(row, video_path):
    raw_id = str(row.get("id") or row.get("videoid") or Path(str(video_path)).stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._") or Path(video_path).stem
    last2 = stem[-2:].lower() if len(stem) >= 2 else ""
    bucket = f"b{last2}" if re.match(r"^[0-9a-f]{2}$", last2) else "others"
    return str(Path(bucket) / f"{stem}.npz")


class JsonVideoTextDataset(Dataset):
    """Default video-text dataset adapter.

    Replace or extend this class when the final video dataset format is fixed.
    Required row field: video. Optional text fields: text/prompt/question/caption/description.
    """

    def __init__(self, data_path, dataset_root=None, split: Optional[str] = None):
        self.data_path = Path(data_path)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        rows = load_json_rows(self.data_path)
        if split:
            rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = dict(self.rows[idx])
        if not row.get("video"):
            raise KeyError("video dataset rows must contain a 'video' field")
        video_path = resolve_media_path(row["video"], self.dataset_root)
        text = infer_text(row)
        return {
            "id": str(row.get("id") or video_path.stem),
            "video": str(video_path),
            "source_url": str(row.get("source_url") or row.get("contentUrl") or ""),
            "text": text,
            "hidden_path": stable_hidden_name(row, video_path),
            "row": row,
        }


class CsvVideoTextDataset(Dataset):
    """CSV adapter for rows like: videoid,contentUrl,duration,page_dir,name.

    Videos are not downloaded here. The cache script downloads each URL into a
    temporary file, forwards it through Qwen2.5-VL, then deletes it.
    """

    def __init__(self, data_path, dataset_root=None, split: Optional[str] = None, text_fields=None):
        self.data_path = Path(data_path)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None
        self.text_fields = text_fields or ["name"]
        with self.data_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if split and "split" in (rows[0] if rows else {}):
            rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict:
        row = dict(self.rows[idx])
        url = (row.get("contentUrl") or row.get("url") or row.get("video") or "").strip()
        if not url:
            raise KeyError("CSV video rows must contain contentUrl, url, or video")
        video_id = str(row.get("videoid") or row.get("id") or Path(url).stem)
        local_video = ""
        if not url.startswith(("http://", "https://")):
            local_video = str(resolve_media_path(url, self.dataset_root))
        text = infer_csv_text(row, self.text_fields)
        return {
            "id": video_id,
            "video": local_video or url,
            "source_url": url if url.startswith(("http://", "https://")) else "",
            "text": text,
            "hidden_path": stable_hidden_name({"id": video_id}, video_id),
            "row": row,
        }


def build_video_text_dataset(
    data_paths: List[str],
    dataset_roots: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    csv_text_fields: Optional[List[str]] = None,
) -> Dataset:
    """Dataset import interface for SAE video hidden caching.

    The current implementation supports JSON/JSONL rows with a video path and text-like
    fields. Keep this function as the extension point for future SHT/VideoChat/Obelics
    video formats.
    """

    dataset_roots = [None] * len(data_paths) if dataset_roots is None else dataset_roots
    splits = [None] * len(data_paths) if splits is None else splits
    if not (len(data_paths) == len(dataset_roots) == len(splits)):
        raise ValueError("--data-paths, --dataset-roots, and --splits must have the same length")
    datasets = [
        (
            CsvVideoTextDataset(
                data_path,
                dataset_root=dataset_root,
                split=split,
                text_fields=csv_text_fields,
            )
            if str(data_path).lower().endswith(".csv")
            else JsonVideoTextDataset(data_path, dataset_root=dataset_root, split=split)
        )
        for data_path, dataset_root, split in zip(data_paths, dataset_roots, splits)
    ]
    return ConcatDataset(datasets)
