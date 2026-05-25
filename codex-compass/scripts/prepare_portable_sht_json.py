#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad_compass.path_utils import save_portable_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", default="data/sht_clip_32_160_conversations_filtered.json")
    parser.add_argument("--output-json", default="data/sht_clip_32_160_conversations_filtered_portable.json")
    parser.add_argument("--dataset-root", default="data/sht_clip_32_160")
    args = parser.parse_args()
    out = save_portable_json(args.input_json, args.output_json, args.dataset_root)
    print(f"wrote portable json to {Path(out)}")


if __name__ == "__main__":
    main()
