#!/usr/bin/env python
import csv, json, sys
sys.path.insert(0, '/root/codex')
from pathlib import Path
from collections import Counter
import numpy as np
import torch

import scripts.eval_temporal_m_clip_firing_switch as ev

DATA='/root/codex/data/sht_clip_32_160_conversations_filtered_normstr.json'
OLD=Path('/root/codex/outputs/temporal_m_sae_clip10k_filtered_switch_eval')
OUT=Path('/root/codex/outputs/temporal_m_sae_clip10k_filtered_normstr_switch_eval')
OUT.mkdir(parents=True,exist_ok=True)

rows=json.load(open(DATA,encoding='utf-8'))
old_firing=np.load(OLD/'firing_presence_uint8.npz')['firing']
old_metas=[json.loads(l) for l in open(OLD/'firing_meta.jsonl',encoding='utf-8')]
failed=[i for i,m in enumerate(old_metas) if not m.get('ok')]
print(f'old ok={sum(m.get("ok") for m in old_metas)} failed={len(failed)} total={len(old_metas)}', flush=True)

firing=old_firing.copy()
metas=list(old_metas)
if failed:
    class Args: pass
    args=Args()
    args.model_path='/root/autodl-tmp/get_hf/InternVL2'
    args.dtype='bfloat16'
    args.hook_layer=12
    args.sae_path='/root/codex/outputs/temporal_m_sae_internvl_sht_clips10k_filtered_8f_16x_k256_1ep/sae_final.pt'
    args.num_frames=8
    args.max_patches_per_frame=1
    args.progress_every=100
    subrows=[rows[i] for i in failed]
    sub_firing, sub_metas=ev.extract_firing(subrows,args)
    for pos,idx in enumerate(failed):
        firing[idx]=sub_firing[pos]
        metas[idx]=sub_metas[pos]
np.savez_compressed(OUT/'firing_presence_uint8.npz',firing=firing)
with open(OUT/'firing_meta.jsonl','w',encoding='utf-8') as f:
    for m in metas:
        f.write(json.dumps(m,ensure_ascii=False)+'\n')

labels=np.array([1 if r.get('label')=='abnormal' else 0 for r in rows],dtype=int)
ok=np.array([m.get('ok',False) for m in metas],dtype=bool)
valid_idx=np.where(ok)[0]
y=labels[valid_idx]
h=firing[valid_idx].astype(bool)
train_rel,test_rel=ev.stratified_split(y,0.2,42)
h_train,h_test=h[train_rel],h[test_rel]
y_train,y_test=y[train_rel],y[test_rel]
p_abn=h_train[y_train==1].mean(axis=0)
p_norm=h_train[y_train==0].mean(axis=0)
delta=p_abn-p_norm
order=np.argsort(-delta)

top_rows=[]
for rank,feat in enumerate(order[:1000],1):
    top_rows.append({'rank':rank,'feature':int(feat),'delta_train':float(delta[feat]),'p_abnormal_train':float(p_abn[feat]),'p_normal_train':float(p_norm[feat]),'abnormal_train_count':int(h_train[y_train==1,feat].sum()),'normal_train_count':int(h_train[y_train==0,feat].sum())})
with open(OUT/'train_delta_abnormal_top_features.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(top_rows[0].keys()))
    w.writeheader(); w.writerows(top_rows)
results=[]
for k in [1,3,10]:
    feats=order[:k]
    switch_score=h_test[:,feats].any(axis=1).astype(int)
    count_score=h_test[:,feats].sum(axis=1).astype(float)
    rec={'k':k,'features':[int(x) for x in feats],'train_deltas':[float(delta[x]) for x in feats],'switch_rule':'abnormal if any selected feature fires in the clip','switch_auc':ev.auc_score(y_test,switch_score),'count_auc':ev.auc_score(y_test,count_score),**ev.binary_metrics(y_test,switch_score),'test_positive_rate_pred_abnormal':float(switch_score.mean())}
    results.append(rec)
summary={'data':DATA,'sae_path':'/root/codex/outputs/temporal_m_sae_internvl_sht_clips10k_filtered_8f_16x_k256_1ep/sae_final.pt','split':'stratified mixed train/test, test_frac=0.2, seed=42','valid_rows':int(len(valid_idx)),'failed_rows':int((~ok).sum()),'label_counts_all':dict(Counter(['abnormal' if v else 'normal' for v in y])),'label_counts_train':dict(Counter(['abnormal' if v else 'normal' for v in y_train])),'label_counts_test':dict(Counter(['abnormal' if v else 'normal' for v in y_test])),'feature_selection':'Delta_f on train only: Pr(feature fires | abnormal) - Pr(feature fires | normal), h=1[max_t activation>0]','results':results,'top10_train_delta':top_rows[:10],'outputs':{'dir':str(OUT),'firing_cache':str(OUT/'firing_presence_uint8.npz'),'top_features_csv':str(OUT/'train_delta_abnormal_top_features.csv')}}
with open(OUT/'switch_eval_summary.json','w',encoding='utf-8') as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
with open(OUT/'switch_eval_results.csv','w',newline='',encoding='utf-8') as f:
    fields=['k','features','train_deltas','switch_rule','switch_auc','count_auc','balanced_acc','accuracy','tpr_abnormal','tnr_normal','tp','tn','fp','fn','test_positive_rate_pred_abnormal']
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader(); w.writerows(results)
print(json.dumps(summary,ensure_ascii=False,indent=2), flush=True)
