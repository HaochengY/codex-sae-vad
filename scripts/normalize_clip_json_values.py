#!/usr/bin/env python
import json
from pathlib import Path

def stringify(v):
    if isinstance(v, str):
        return v
    if v is None:
        return ''
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

inp=Path('/root/codex/data/sht_clip_32_160_conversations_filtered.json')
out=Path('/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.json')
outl=Path('/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.jsonl')
rows=json.load(open(inp,encoding='utf-8'))
changed=0
for r in rows:
    for m in r.get('conversations',[]):
        if not isinstance(m.get('value'), str):
            m['value']=stringify(m.get('value'))
            changed += 1
with open(out,'w',encoding='utf-8') as f:
    json.dump(rows,f,ensure_ascii=False,indent=2)
with open(outl,'w',encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'input':str(inp),'output':str(out),'rows':len(rows),'changed_values':changed},ensure_ascii=False,indent=2))
