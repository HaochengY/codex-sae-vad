import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/window3s_step1s_dataset")
    parser.add_argument("--clips-subdir", default="clips/train")
    parser.add_argument("--labels-subdir", default="frame_labels/train")
    parser.add_argument("--output-json", default="outputs/smoke/sht_window_train_smoke.json")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clip_to_label_rel(clip_rel, labels_subdir):
    clip_rel = Path(clip_rel)
    return (Path(labels_subdir) / f"{clip_rel.stem}.npy").as_posix()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    clips_dir = dataset_root / args.clips_subdir
    labels_subdir = Path(args.labels_subdir)
    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"no mp4 clips found under {clips_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(clips)
    if args.max_samples > 0:
        clips = clips[: args.max_samples]

    rows = []
    for idx, clip_path in enumerate(clips):
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
    print(json.dumps({"output_json": output_json.as_posix(), "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
