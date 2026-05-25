#!/usr/bin/env python
import sys
sys.path.insert(0, '/root/codex')
import argparse, csv, json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import train_temporal_matryoshka_sae_internvl_sht as temp_train
import train_topk_sae_internvl_sht as topk_train

def auc_score(y, s):
    y=np.asarray(y).astype(int); s=np.asarray(s).astype(float)
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    order=np.argsort(s); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(s)+1)
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    for g in np.where(cnt>1)[0]:
        idx=inv==g; ranks[idx]=ranks[idx].mean()
    rpos=ranks[y==1].sum()
    return float((rpos-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def binary_metrics(y,pred):
    y=np.asarray(y).astype(int); pred=np.asarray(pred).astype(int)
    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
    fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    tpr=tp/max((y==1).sum(),1); tnr=tn/max((y==0).sum(),1)
    return {'balanced_acc':float((tpr+tnr)/2),'accuracy':float((pred==y).mean()),'tpr_abnormal':float(tpr),'tnr_normal':float(tnr),'tp':tp,'tn':tn,'fp':fp,'fn':fn}

def stratified_split(y,test_frac=0.2,seed=42):
    rng=random.Random(seed); train=[]; test=[]
    for cls in [0,1]:
        idx=np.where(y==cls)[0].tolist(); rng.shuffle(idx)
        n_test=max(1,round(len(idx)*test_frac))
        test.extend(idx[:n_test]); train.extend(idx[n_test:])
    rng.shuffle(train); rng.shuffle(test)
    return np.array(train,dtype=int), np.array(test,dtype=int)

@torch.no_grad()
def extract_firing(rows,args):
    model_args=type('Args',(),{'model_path':args.model_path,'dtype':args.dtype})()
    model,tokenizer,dtype=temp_train.build_model(model_args)
    hook_name,hook_module=topk_train.resolve_hook_module(model,args.hook_layer)
    ckpt=torch.load(args.sae_path,map_location='cpu')
    sae_args=ckpt['args']
    sae=temp_train.TopKMatryoshkaSae(ckpt['d_in'],ckpt['num_latents'],sae_args['k'],sae_args.get('high_frac',0.2)).cuda().eval()
    sae.load_state_dict(ckpt['sae'])
    run_args=type('Args',(),{'num_frames':args.num_frames,'max_patches_per_frame':args.max_patches_per_frame})()
    firing=np.zeros((len(rows),sae.num_latents),dtype=np.uint8)
    metas=[]; start=time.time()
    for i,row in enumerate(rows):
        try:
            h_grid=temp_train.extract_image_hidden(row,model,tokenizer,dtype,hook_module,run_args)
            x=h_grid.reshape(-1,h_grid.shape[-1])
            values,indices=sae.encode_topk(x)
            present=torch.zeros(sae.num_latents,device=indices.device,dtype=torch.bool)
            present[indices.reshape(-1).unique()]=True
            firing[i]=present.cpu().numpy().astype(np.uint8)
            metas.append({'ok':True,'id':row.get('id'),'label':row.get('label')})
        except Exception as exc:
            metas.append({'ok':False,'id':row.get('id'),'label':row.get('label'),'error':str(exc)})
        if (i+1)%args.progress_every==0:
            print(f'processed {i+1}/{len(rows)} elapsed={time.time()-start:.1f}s',flush=True)
    return firing,metas

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='/root/codex/data/sht_clip_32_160_conversations_filtered.json')
    p.add_argument('--sae-path',default='/root/codex/outputs/temporal_m_sae_internvl_sht_clips10k_filtered_8f_16x_k256_1ep/sae_final.pt')
    p.add_argument('--model-path',default='/root/autodl-tmp/get_hf/InternVL2')
    p.add_argument('--output-dir',default='/root/codex/outputs/temporal_m_sae_clip10k_filtered_switch_eval')
    p.add_argument('--hook-layer',type=int,default=12)
    p.add_argument('--num-frames',type=int,default=8)
    p.add_argument('--max-patches-per-frame',type=int,default=1)
    p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-frac',type=float,default=0.2)
    p.add_argument('--progress-every',type=int,default=100)
    args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    cache_path=out/'firing_presence_uint8.npz'; meta_path=out/'firing_meta.jsonl'
    rows=json.load(open(args.data,encoding='utf-8'))
    labels=np.array([1 if r.get('label')=='abnormal' else 0 for r in rows],dtype=int)
    if cache_path.exists() and meta_path.exists():
        firing=np.load(cache_path)['firing']
        metas=[json.loads(l) for l in open(meta_path,encoding='utf-8')]
        print(f'loaded cache {cache_path}',flush=True)
    else:
        firing,metas=extract_firing(rows,args)
        np.savez_compressed(cache_path,firing=firing)
        with open(meta_path,'w',encoding='utf-8') as f:
            for m in metas: f.write(json.dumps(m,ensure_ascii=False)+'\n')
        print(f'saved cache {cache_path}',flush=True)
    ok=np.array([m.get('ok',False) for m in metas],dtype=bool)
    valid_idx=np.where(ok)[0]
    y=labels[valid_idx]; h=firing[valid_idx].astype(bool)
    train_rel,test_rel=stratified_split(y,args.test_frac,args.seed)
    h_train,h_test=h[train_rel],h[test_rel]; y_train,y_test=y[train_rel],y[test_rel]
    p_abn=h_train[y_train==1].mean(axis=0); p_norm=h_train[y_train==0].mean(axis=0)
    delta=p_abn-p_norm; order=np.argsort(-delta)
    top_rows=[]
    for rank,feat in enumerate(order[:1000],1):
        top_rows.append({'rank':rank,'feature':int(feat),'delta_train':float(delta[feat]),'p_abnormal_train':float(p_abn[feat]),'p_normal_train':float(p_norm[feat]),'abnormal_train_count':int(h_train[y_train==1,feat].sum()),'normal_train_count':int(h_train[y_train==0,feat].sum())})
    with open(out/'train_delta_abnormal_top_features.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(top_rows[0].keys())); w.writeheader(); w.writerows(top_rows)
    results=[]
    for k in [1,3,10]:
        feats=order[:k]
        switch_score=h_test[:,feats].any(axis=1).astype(int)
        count_score=h_test[:,feats].sum(axis=1).astype(float)
        rec={'k':k,'features':[int(x) for x in feats],'train_deltas':[float(delta[x]) for x in feats],'switch_rule':'abnormal if any selected feature fires in the clip','switch_auc':auc_score(y_test,switch_score),'count_auc':auc_score(y_test,count_score),**binary_metrics(y_test,switch_score),'test_positive_rate_pred_abnormal':float(switch_score.mean())}
        results.append(rec)
    summary={'data':args.data,'sae_path':args.sae_path,'split':f'stratified mixed train/test, test_frac={args.test_frac}, seed={args.seed}','valid_rows':int(len(valid_idx)),'failed_rows':int((~ok).sum()),'label_counts_all':dict(Counter(['abnormal' if v else 'normal' for v in y])),'label_counts_train':dict(Counter(['abnormal' if v else 'normal' for v in y_train])),'label_counts_test':dict(Counter(['abnormal' if v else 'normal' for v in y_test])),'feature_selection':'Delta_f on train only: Pr(feature fires | abnormal) - Pr(feature fires | normal), h=1[max_t activation>0]','results':results,'top10_train_delta':top_rows[:10],'outputs':{'dir':str(out),'firing_cache':str(cache_path),'top_features_csv':str(out/'train_delta_abnormal_top_features.csv')}}
    with open(out/'switch_eval_summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    with open(out/'switch_eval_results.csv','w',newline='',encoding='utf-8') as f:
        fields=['k','features','train_deltas','switch_auc','count_auc','balanced_acc','accuracy','tpr_abnormal','tnr_normal','tp','tn','fp','fn','test_positive_rate_pred_abnormal']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
