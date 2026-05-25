#!/usr/bin/env python
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_KEYWORDS = [
    # motion/action terms requested by user
    'riding', 'ride', 'rides', 'ridden', 'running', 'run', 'runs', 'runner', 'jumping', 'jump', 'jumps',
    # vehicle terms
    'vehicle', 'vehicles', 'car', 'cars', 'truck', 'trucks', 'bus', 'buses', 'van', 'vans', 'motorcycle',
    'motorbike', 'bike', 'bicycle', 'cycling', 'cyclist', 'scooter', 'skateboard', 'skateboarding',
    'tricycle', 'traffic',
    # Chinese counterparts in case category/source metadata remains
    '骑', '自行车', '单车', '骑车', '滑板', '滑板车', '摩托', '摩托车', '汽车', '车辆', '车', '三轮车',
    '平衡车', '奔跑', '跑步', '在跑', '跳跃', '跳', '追逐',
]

# Avoid filtering harmless phrases like "no abnormal frames". We only search fields that describe visible content.
TEXT_FIELDS = ['category_zh', 'source_summary']
META_FIELDS = ['clip_kind', 'source_video_name', 'video_name']

def load_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def row_text(row):
    parts = []
    for k in TEXT_FIELDS:
        v = row.get(k)
        if isinstance(v, str):
            parts.append(v)
    for msg in row.get('conversations', []):
        # include gpt answers and human prompts because some human prompts can contain <video> only, harmless
        parts.append(str(msg.get('value', '')))
    return '\n'.join(parts)

def compile_patterns(keywords):
    pats = []
    for kw in keywords:
        if re.fullmatch(r'[A-Za-z0-9_ -]+', kw):
            pats.append((kw, re.compile(r'(?<![A-Za-z])' + re.escape(kw.lower()) + r'(?![A-Za-z])')))
        else:
            pats.append((kw, re.compile(re.escape(kw))))
    return pats

def find_hits(text, patterns):
    low = text.lower()
    hits = []
    for kw, pat in patterns:
        target = low if re.fullmatch(r'[A-Za-z0-9_ -]+', kw) else text
        if pat.search(target):
            hits.append(kw)
    return sorted(set(hits))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/root/codex/data/sht_clip_32_160_conversations.json')
    ap.add_argument('--output', default='/root/codex/data/sht_clip_32_160_conversations_filtered.json')
    ap.add_argument('--jsonl-output', default='/root/codex/data/sht_clip_32_160_conversations_filtered.jsonl')
    ap.add_argument('--report', default='/root/codex/data/sht_clip_32_160_normal_keyword_hits.csv')
    ap.add_argument('--summary', default='/root/codex/data/sht_clip_32_160_filter_summary.json')
    ap.add_argument('--keywords', nargs='*', default=DEFAULT_KEYWORDS)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rows = load_rows(args.input)
    patterns = compile_patterns(args.keywords)
    kept = []
    removed = []
    hit_counts = Counter()
    by_kind = Counter()
    for row in rows:
        if row.get('label') != 'normal':
            kept.append(row)
            continue
        hits = find_hits(row_text(row), patterns)
        if hits:
            removed.append({'id': row.get('id'), 'video': row.get('video'), 'clip_kind': row.get('clip_kind'), 'source_video_name': row.get('source_video_name'), 'hits': hits, 'text_preview': row_text(row).replace('\n', ' ')[:500]})
            hit_counts.update(hits)
            by_kind[row.get('clip_kind')] += 1
        else:
            kept.append(row)

    summary = {
        'input': args.input,
        'output': args.output,
        'total_rows': len(rows),
        'kept_rows': len(kept),
        'removed_normal_rows': len(removed),
        'labels_before': dict(Counter(r.get('label') for r in rows)),
        'labels_after': dict(Counter(r.get('label') for r in kept)),
        'clip_kind_before': dict(Counter(r.get('clip_kind') for r in rows)),
        'clip_kind_after': dict(Counter(r.get('clip_kind') for r in kept)),
        'removed_by_clip_kind': dict(by_kind),
        'top_hit_keywords': hit_counts.most_common(50),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['id', 'video', 'clip_kind', 'source_video_name', 'hits', 'text_preview']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in removed:
            rr = dict(r)
            rr['hits'] = ';'.join(rr['hits'])
            w.writerow(rr)

    with open(args.summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.dry_run:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        with open(args.jsonl_output, 'w', encoding='utf-8') as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
