#!/usr/bin/env python
import argparse
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

MIN_LEN = 32
MAX_LEN = 160
GAP_MERGE = 8
EVENT_CONTEXT = 16

ANOMALY_EN = {
    '自行车': 'bicycle or cycling',
    '滑板': 'skateboarding',
    '打架': 'fighting',
    '追逐奔跑': 'chasing or running',
    '奔跑': 'running',
    '汽车': 'car or vehicle',
    '摩托车': 'motorcycle',
    '跳跃': 'jumping',
    '跑步': 'running',
    '摔倒': 'falling',
    '不按规则乱转圈': 'irregular circling movement',
    '把包扔到天上接住': 'throwing and catching a bag',
    '三轮车': 'tricycle',
    '抢了别人包在奔跑': 'snatching a bag and running',
    '手持一个东西乱甩': 'waving an object irregularly',
    '翻越栏杆': 'climbing over a railing',
    '滑板，跑步': 'skateboarding and running',
    '滑板车': 'scooter',
    '小孩子在奔跑': 'child running',
    '骑平衡车': 'riding a balance vehicle',
    '弯腰捡东西': 'bending down to pick something up',
}

REMOVE_TERMS = [
    'bicycle', 'cycling', 'cyclist', 'bike', 'skateboard', 'skateboarding',
    'fight', 'fighting', 'chasing', 'chase', 'running', 'runner', 'jumping',
    'motorcycle', 'tricycle', 'scooter', 'falling', 'abnormal', 'unusual',
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def video_info(path):
    vr = VideoReader(str(path), ctx=cpu(0))
    return len(vr), float(vr.get_avg_fps())


def ensure_bounds(start, length, nframes):
    length = int(max(MIN_LEN, min(MAX_LEN, length, nframes)))
    if nframes <= length:
        return 0, nframes
    start = int(max(0, min(start, nframes - length)))
    return start, length


def sample_lengths(rng, nframes, n=3):
    if nframes <= MIN_LEN:
        return [nframes]
    bands = [(MIN_LEN, 64), (65, 112), (113, MAX_LEN)]
    out = []
    for lo, hi in bands[:n]:
        lo = min(lo, nframes)
        hi = min(hi, nframes)
        if lo > hi:
            lo = min(MIN_LEN, nframes)
        out.append(rng.randint(lo, hi))
    return out


def normal_clips_for_video(nframes, rng, n=3):
    clips = []
    for length in sample_lengths(rng, nframes, n=n):
        if nframes <= length:
            start = 0
        else:
            start = rng.randint(0, nframes - length)
        start, length = ensure_bounds(start, length, nframes)
        clips.append((start, start + length - 1, length))
    return dedupe_clips(clips)


def normal_clips_for_segment(seg_start, seg_end, rng):
    seg_len = seg_end - seg_start + 1
    if seg_len < MIN_LEN:
        return []
    n = 1 if seg_len < 96 else 2
    clips = []
    for length in sample_lengths(rng, seg_len, n=n):
        max_start = seg_end - length + 1
        start = seg_start if max_start <= seg_start else rng.randint(seg_start, max_start)
        clips.append((start, start + length - 1, length))
    return dedupe_clips(clips)


def contiguous_regions(mask_value):
    idx = np.flatnonzero(mask_value)
    if len(idx) == 0:
        return []
    regions = []
    start = prev = int(idx[0])
    for x in idx[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            regions.append((start, prev))
            start = prev = x
    regions.append((start, prev))
    return regions


def merge_regions(regions, max_gap=GAP_MERGE):
    if not regions:
        return []
    merged = [regions[0]]
    for s, e in regions[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def abnormal_clips_for_region(region, nframes, rng):
    s, e = region
    event_len = e - s + 1
    clips = []
    if event_len <= MAX_LEN:
        base_len = min(MAX_LEN, max(MIN_LEN, event_len + 2 * EVENT_CONTEXT))
        lengths = sorted(set([base_len, min(MAX_LEN, max(MIN_LEN, event_len + EVENT_CONTEXT)), min(MAX_LEN, max(MIN_LEN, event_len))]))
        # Prefer longer first to preserve context, then shorter variants.
        for length in sorted(lengths, reverse=True)[:3]:
            center = (s + e) // 2
            jitter = rng.randint(-max(1, length // 10), max(1, length // 10)) if nframes > length else 0
            start = center - length // 2 + jitter
            start, length = ensure_bounds(start, length, nframes)
            if start <= e and start + length - 1 >= s:
                clips.append((start, start + length - 1, length, s, e))
    else:
        stride = MAX_LEN // 2
        starts = list(range(s, e + 1, stride))
        for start in starts:
            if start > e:
                continue
            start = min(start, max(0, e - MAX_LEN + 1))
            start, length = ensure_bounds(start, MAX_LEN, nframes)
            clips.append((start, start + length - 1, length, s, e))
            if start + length - 1 >= e:
                break
    return dedupe_clips(clips)


def dedupe_clips(clips):
    seen = set(); out = []
    for c in clips:
        key = (c[0], c[1])
        if key not in seen:
            seen.add(key); out.append(c)
    return out


def ffmpeg_clip(src, dst, start, end, fps, dry_run=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    if dry_run:
        return
    start_t = start / fps
    frames = end - start + 1
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-ss', f'{start_t:.6f}', '-i', str(src),
        '-frames:v', str(frames), '-an',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-pix_fmt', 'yuv420p', str(dst),
    ]
    subprocess.run(cmd, check=True)


def clean_text(text):
    text = re.sub(r'Ground-truth abnormal event:[^.;]*[.;]?', '', text, flags=re.I)
    for term in REMOVE_TERMS:
        text = re.sub(r'\b' + re.escape(term) + r'\b', 'normal movement', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_summary(row):
    raw = row.get('source_summary') or ''
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def base_scene(row):
    s = parse_summary(row)
    scene = s.get('scene') if isinstance(s, dict) else None
    return scene or 'An outdoor surveillance scene.'


def gpt_value_for_existing(row):
    vals = [m.get('value', '') for m in row.get('conversations', []) if m.get('from') == 'gpt']
    return clean_text(' '.join(vals)) if vals else 'People and background objects are visible in the scene.'


def make_conversations(row, label, clip_kind, anomaly_text=None, start=None, end=None):
    scene = base_scene(row)
    existing = gpt_value_for_existing(row)
    if label == 'normal':
        event = 'This clip contains normal/background activity only; no ground-truth abnormal frames are included.'
        actors = existing or 'People and ordinary scene objects are visible.'
    else:
        event = f'This clip contains the abnormal event: {anomaly_text or row.get("category_zh") or "abnormal event"}.'
        actors = existing or 'People and relevant objects involved in the event are visible.'
    return [
        {'from': 'human', 'value': '<video>\nDescribe the scene in this clip.'},
        {'from': 'gpt', 'value': scene},
        {'from': 'human', 'value': 'What actors and objects are visible?'},
        {'from': 'gpt', 'value': actors},
        {'from': 'human', 'value': 'What movement or event is happening in this clip?'},
        {'from': 'gpt', 'value': event},
    ]


def make_record(row, clip_path, clip_id, split, label, clip_kind, start, end, nframes, fps, source_video, extra=None):
    extra = extra or {}
    rec = {
        'id': clip_id,
        'video': str(clip_path),
        'video_name': clip_path.name,
        'split': split,
        'label': label,
        'clip_kind': clip_kind,
        'source_video': str(source_video),
        'source_video_name': Path(source_video).name,
        'source_id': row.get('id'),
        'category_zh': row.get('category_zh', 'normal' if label == 'normal' else 'unknown'),
        'start_frame': int(start),
        'end_frame': int(end),
        'num_frames': int(nframes),
        'fps': fps,
        'duration_sec': float(nframes / fps),
        **extra,
    }
    anomaly_text = ANOMALY_EN.get(rec['category_zh'], rec['category_zh']) if label == 'abnormal' else None
    rec['conversations'] = make_conversations(row, label, clip_kind, anomaly_text, start, end)
    return rec


def synthetic_missing_row(video_path):
    name = Path(video_path).name
    stem = Path(video_path).stem
    return {
        'id': f'sht_abnormal_{stem}',
        'video': str(video_path),
        'video_name': name,
        'split': 'testing',
        'label': 'abnormal',
        'category_zh': 'unknown',
        'source_summary': '',
        'conversations': [
            {'from': 'human', 'value': '<video>\nDescribe the scene in this video.'},
            {'from': 'gpt', 'value': 'A surveillance video containing an annotated abnormal event.'},
        ],
    }



def random_clip_in_span(span_start, span_end, rng, min_len=MIN_LEN, max_len=MAX_LEN):
    span_len = span_end - span_start + 1
    if span_len < min_len:
        return None
    max_len = min(max_len, span_len)
    # Length distribution: mostly medium clips, with some short and long clips.
    r = rng.random()
    if r < 0.35:
        lo, hi = min_len, min(64, max_len)
    elif r < 0.75:
        lo, hi = min(65, max_len), min(112, max_len)
    else:
        lo, hi = min(113, max_len), max_len
    if lo > hi:
        lo, hi = min_len, max_len
    length = rng.randint(lo, hi)
    start_max = span_end - length + 1
    start = span_start if start_max <= span_start else rng.randint(span_start, start_max)
    return start, start + length - 1, length


def random_clip_covering_region(region, nframes, rng):
    s, e = region
    event_len = e - s + 1
    if event_len >= MAX_LEN:
        start_min = s
        start_max = max(s, e - MAX_LEN + 1)
        start = rng.randint(start_min, start_max) if start_max > start_min else start_min
        start, length = ensure_bounds(start, MAX_LEN, nframes)
        return start, start + length - 1, length, s, e
    min_len = max(MIN_LEN, min(MAX_LEN, event_len))
    max_len = min(MAX_LEN, nframes)
    length = rng.randint(min_len, max_len)
    # Choose any start whose clip intersects/covers the event as much as possible.
    lo = max(0, e - length + 1)
    hi = min(s, nframes - length)
    if lo > hi:
        center = (s + e) // 2
        start = center - length // 2
    else:
        start = rng.randint(lo, hi)
    start, length = ensure_bounds(start, length, nframes)
    return start, start + length - 1, length, s, e


def allocation_counts(total, items):
    if not items:
        return []
    base = total // len(items)
    rem = total % len(items)
    return [base + (1 if i < rem else 0) for i in range(len(items))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/codex/data/sht_video_concept_conversations.json')
    ap.add_argument('--train-dir', default='/root/autodl-tmp/sht/train')
    ap.add_argument('--test-dir', default='/root/autodl-tmp/sht/test')
    ap.add_argument('--mask-dir', default='/root/autodl-tmp/sht/test_frame_mask')
    ap.add_argument('--clip-root', default='/root/autodl-tmp/sht_clip_32_160')
    ap.add_argument('--output-json', default='/root/codex/data/sht_clip_32_160_conversations.json')
    ap.add_argument('--output-jsonl', default='/root/codex/data/sht_clip_32_160_conversations.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--target-records', type=int, default=10000)
    ap.add_argument('--train-normal-frac', type=float, default=0.60)
    ap.add_argument('--test-abnormal-frac', type=float, default=0.25)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_json(args.data)
    train_rows = [r for r in rows if r.get('label') == 'normal']
    test_rows_by_name = {r.get('video_name'): r for r in rows if r.get('label') == 'abnormal'}
    clip_root = Path(args.clip_root)
    records = []
    stats = {'train_normal_clips': 0, 'test_abnormal_clips': 0, 'test_normal_clips': 0, 'missing_test_metadata': []}

    target_train = int(args.target_records * args.train_normal_frac)
    target_test_abn = int(args.target_records * args.test_abnormal_frac)
    target_test_norm = args.target_records - target_train - target_test_abn

    train_items = []
    for row in train_rows:
        src = Path(row['video'])
        nframes, fps = video_info(src)
        if nframes >= MIN_LEN:
            train_items.append((row, src, nframes, fps))
    for item_idx, ((row, src, nframes, fps), count) in enumerate(zip(train_items, allocation_counts(target_train, train_items))):
        used = set()
        tries = 0
        made = 0
        while made < count and tries < count * 20:
            tries += 1
            c = random_clip_in_span(0, nframes - 1, rng)
            if c is None:
                break
            start, end, length = c
            key = (start, end)
            if key in used:
                continue
            used.add(key)
            clip_id = f'{row["id"]}_clip{made:03d}_{start:05d}_{end:05d}'
            dst = clip_root / 'train_normal' / f'{clip_id}.mp4'
            ffmpeg_clip(src, dst, start, end, fps, dry_run=args.dry_run)
            records.append(make_record(row, dst, clip_id, 'training', 'normal', 'train_normal', start, end, length, fps, src))
            stats['train_normal_clips'] += 1
            made += 1
        if (item_idx + 1) % 50 == 0:
            print(f'processed train {item_idx+1}/{len(train_items)} records={len(records)}', flush=True)

    test_abn_items = []
    test_norm_items = []
    for vp in sorted(Path(args.test_dir).glob('*.mp4')):
        name = vp.name
        row = test_rows_by_name.get(name)
        if row is None:
            row = synthetic_missing_row(vp)
            stats['missing_test_metadata'].append(name)
        nframes, fps = video_info(vp)
        mask_path = Path(args.mask_dir) / f'{vp.stem}.npy'
        mask = np.load(mask_path).astype(bool)
        if len(mask) != nframes:
            m = np.zeros(nframes, dtype=bool)
            m[: min(nframes, len(mask))] = mask[: min(nframes, len(mask))]
            mask = m
        abnormal_regions = merge_regions(contiguous_regions(mask), GAP_MERGE)
        for ri, region in enumerate(abnormal_regions):
            test_abn_items.append((row, vp, nframes, fps, ri, region))
        for ri, (seg_s, seg_e) in enumerate(contiguous_regions(~mask)):
            if seg_e - seg_s + 1 >= MIN_LEN:
                test_norm_items.append((row, vp, nframes, fps, ri, (seg_s, seg_e)))

    for item_idx, ((row, vp, nframes, fps, ri, region), count) in enumerate(zip(test_abn_items, allocation_counts(target_test_abn, test_abn_items))):
        used = set(); made = 0; tries = 0
        while made < count and tries < count * 30:
            tries += 1
            start, end, length, ev_s, ev_e = random_clip_covering_region(region, nframes, rng)
            key = (start, end)
            if key in used:
                continue
            used.add(key)
            clip_id = f'{row["id"]}_abn{ri:02d}_clip{made:03d}_{start:05d}_{end:05d}'
            dst = clip_root / 'test_abnormal' / f'{clip_id}.mp4'
            ffmpeg_clip(vp, dst, start, end, fps, dry_run=args.dry_run)
            extra = {'event_start_frame': int(ev_s), 'event_end_frame': int(ev_e), 'event_start_in_clip': int(max(0, ev_s - start)), 'event_end_in_clip': int(min(end, ev_e) - start)}
            records.append(make_record(row, dst, clip_id, 'testing', 'abnormal', 'test_abnormal_event', start, end, length, fps, vp, extra))
            stats['test_abnormal_clips'] += 1
            made += 1
        if (item_idx + 1) % 50 == 0:
            print(f'processed abnormal items {item_idx+1}/{len(test_abn_items)} records={len(records)}', flush=True)

    for item_idx, ((row, vp, nframes, fps, ri, seg), count) in enumerate(zip(test_norm_items, allocation_counts(target_test_norm, test_norm_items))):
        seg_s, seg_e = seg
        used = set(); made = 0; tries = 0
        while made < count and tries < count * 30:
            tries += 1
            c = random_clip_in_span(seg_s, seg_e, rng)
            if c is None:
                break
            start, end, length = c
            key = (start, end)
            if key in used:
                continue
            used.add(key)
            clip_id = f'{row["id"]}_norm{ri:02d}_clip{made:03d}_{start:05d}_{end:05d}'
            dst = clip_root / 'test_normal' / f'{clip_id}.mp4'
            ffmpeg_clip(vp, dst, start, end, fps, dry_run=args.dry_run)
            records.append(make_record(row, dst, clip_id, 'testing', 'normal', 'test_non_abnormal', start, end, length, fps, vp, {'source_was_abnormal_video': True}))
            stats['test_normal_clips'] += 1
            made += 1
        if (item_idx + 1) % 50 == 0:
            print(f'processed test-normal items {item_idx+1}/{len(test_norm_items)} records={len(records)}', flush=True)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    stats.update({'total_records': len(records), 'output_json': str(out_json), 'output_jsonl': args.output_jsonl, 'clip_root': str(clip_root)})
    with open(out_json.with_suffix('.summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
