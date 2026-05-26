import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/window3s_step1s_dataset")
    parser.add_argument("--clips-subdirs", default="clips/train,clips/test")
    parser.add_argument("--labels-subdirs", default="frame_labels/train,frame_labels/test")
    parser.add_argument("--output-json", default="outputs/smoke/sht_window_train_smoke.json")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_csv(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def clip_to_label_rel(clip_rel, labels_subdir):
    clip_rel = Path(clip_rel)
    return (Path(labels_subdir) / f"{clip_rel.stem}.npy").as_posix()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    clips_subdirs = split_csv(args.clips_subdirs)
    labels_subdirs = split_csv(args.labels_subdirs)
    if len(clips_subdirs) != len(labels_subdirs):
        raise ValueError("--clips-subdirs and --labels-subdirs must have the same number of entries")

    groups = []
    for clips_subdir, labels_subdir in zip(clips_subdirs, labels_subdirs):
        clips_dir = dataset_root / clips_subdir
        clips = sorted(clips_dir.glob("*.mp4"))
        if not clips:
            raise FileNotFoundError(f"no mp4 clips found under {clips_dir}")
        groups.append([(clip_path, Path(labels_subdir)) for clip_path in clips])
    if not groups:
        raise FileNotFoundError(f"no mp4 clips found under {dataset_root}")

    rng = random.Random(args.seed)
    for group in groups:
        rng.shuffle(group)
    items = []
    max_len = max(len(group) for group in groups)
    for i in range(max_len):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for group_idx in order:
            if i < len(groups[group_idx]):
                items.append(groups[group_idx][i])
    if args.max_samples > 0:
        items = items[: args.max_samples]

    rows = []
    for idx, (clip_path, labels_subdir) in enumerate(items):
        clip_rel = clip_path.relative_to(dataset_root).as_posix()
        label_rel = clip_to_label_rel(clip_rel, labels_subdir)
        label_path = dataset_root / label_rel
        label_clip = 0
        if label_path.exists():
            labels = np.load(label_path)
            label_clip = int(np.asarray(labels).max() > 0)
        rows.append(
            {
                "id": f"sht_window_smoke_{idx:04d}",
                "split": "training",
                "video": clip_rel,
                "video_rel": clip_rel,
                "frame_label": label_rel,
                "label": "abnormal" if label_clip else "normal",
                "label_clip": label_clip,
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "output_json": output_json.as_posix(),
                "samples": len(rows),
                "clips_subdirs": clips_subdirs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
