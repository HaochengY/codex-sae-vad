import json
from pathlib import Path


def resolve_path(path, root="."):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root).resolve() / path


def relative_under_anchor(value, anchors=("sht_clip_32_160", "sht")):
    """Convert machine-specific absolute dataset paths to portable relatives."""
    path = Path(value)
    if not path.is_absolute():
        return path
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            return Path(*parts[idx + 1 :])
    return Path(path.name)


def resolve_media_path(value, dataset_root):
    return Path(dataset_root).resolve() / relative_under_anchor(value)


def load_json_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def portable_rows(rows, dataset_root):
    out = []
    for row in rows:
        row = dict(row)
        if row.get("video"):
            row["video_rel"] = str(relative_under_anchor(row["video"]))
            row["video"] = str(resolve_media_path(row["video"], dataset_root))
        if row.get("source_video"):
            row["source_video_rel"] = str(relative_under_anchor(row["source_video"], anchors=("sht",)))
        out.append(row)
    return out


def save_portable_json(input_json, output_json, dataset_root):
    rows = load_json_rows(input_json)
    output = []
    for row in rows:
        row = dict(row)
        if row.get("video"):
            row["video"] = str(relative_under_anchor(row["video"]))
        if row.get("source_video"):
            row["source_video"] = str(relative_under_anchor(row["source_video"], anchors=("sht",)))
        output.append(row)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output_json
