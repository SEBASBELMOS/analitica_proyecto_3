from pathlib import Path
import json, hashlib, math, os, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer

ROOT = Path('/workspace/geovision-cali-hf')
EXPERIMENT = 'remoteclip_fusion_ksae_v10_v5b_gsplit_seed_sweep'
DATA = ROOT / 'outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit'
EMB_NPZ = ROOT / 'outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz'
OUT = ROOT / f'outputs/clip_training_{EXPERIMENT}'
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(json.dumps({'experiment': EXPERIMENT, 'device': str(DEVICE), 'data': str(DATA), 'emb_npz': str(EMB_NPZ)}, ensure_ascii=False), flush=True)

meta = pd.read_json(DATA / 'metadata.jsonl', lines=True)
classes = sorted(meta.candidate_class.unique()); c2i = {c:i for i,c in enumerate(classes)}; y = np.array([c2i[c] for c in meta.candidate_class]); splits = meta.split.values
npz = np.load(EMB_NPZ, allow_pickle=True); remote = npz['remoteclip_visual_512'].astype('float32')
if remote.shape[0] != len(meta): raise ValueError('embedding row mismatch')
if 'pair_id' in npz and not np.all(npz['pair_id'].astype(str) == meta['pair_id'].astype(str).to_numpy()): raise ValueError('pair_id alignment mismatch')

def aux_row(r):
    img = np.load(r.image_path).astype('float32'); red = img[:,:,3]; green = img[:,:,2]; nir = img[:,:,7]; swir = img[:,:,10]
    ndvi = (nir-red)/(nir+red+1e-6); ndbi = (swir-nir)/(swir+nir+1e-6); ndwi = (green-nir)/(green+nir+1e-6); arr = img.reshape(-1,12)
    vals = []; vals.extend(arr.mean(0)); vals.extend(arr.std(0)); vals.extend(np.percentile(arr,[10,50,90],axis=0).ravel())
    for idx in [ndvi,ndbi,ndwi]: vals.extend([np.nanmean(idx),np.nanstd(idx),np.nanpercentile(idx,10),np.nanpercentile(idx,50),np.nanpercentile(idx,90)])
    d = pd.to_datetime(r.date_day); doy = d.dayofyear
    vals.extend([float(d.year),np.sin(2*np.pi*doy/366),np.cos(2*np.pi*doy/366),float(r.ndvi_mean),float(r.ndbi_mean),float(r.ndwi_mean),float(r.scl_cloud_shadow_pct),float(r.scl_valid_visual_pct)])
    return np.array(vals,dtype='float32')

aux = np.stack([aux_row(r) for _, r in meta.iterrows()]); train_mask = splits == 'train'; sc_r = StandardScaler().fit(remote[train_mask]); sc_a = StandardScaler().fit(aux[train_mask]); X = np.concatenate([sc_r.transform(remote), sc_a.transform(aux)], axis=1).astype('float32')
class_text = {c: meta[(meta.split == 'train') & (meta.candidate_class == c)].text.iloc[0] for c in classes}
st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=str(DEVICE))
with torch.no_grad(): txt_base = torch.tensor(st.encode([class_text[c] for c in classes], normalize_embeddings=True), dtype=torch.float32, device=DEVICE)
TEXT_DIM = txt_base.shape[1]; IN = X.shape[1]; EMB = 256; SAE_H = 1024; X_t = torch.tensor(X, device=DEVICE); y_t = torch.tensor(y, device=DEVICE); idxs = {s: torch.tensor(np.where(splits == s)[0], device=DEVICE) for s in ['train','val','test']}

class SAE(nn.Module):
    def __init__(self, k_frac=0.15): super().__init__(); self.enc = nn.Sequential(nn.Linear(EMB, SAE_H), nn.ReLU()); self.dec = nn.Linear(SAE_H, EMB); self.k_frac = k_frac
    def forward(self, x): z = self.enc(x); k = max(1, int(z.shape[1]*self.k_frac)); vals, idx = torch.topk(z, k, dim=1); sparse = torch.zeros_like(z).scatter(1, idx, vals); return sparse, self.dec(sparse)

class M(nn.Module):
    def __init__(self, drop=0.20, k_frac=0.15):
        super().__init__(); self.img = nn.Sequential(nn.LayerNorm(IN), nn.Linear(IN,768), nn.GELU(), nn.Dropout(drop), nn.Linear(768,512), nn.GELU(), nn.Dropout(drop), nn.Linear(512,EMB)); self.txt = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM,512), nn.GELU(), nn.Linear(512,EMB)); self.sae_i = SAE(k_frac); self.sae_t = SAE(k_frac); self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))
    def forward_img(self,x): h = self.img(x); z,r = self.sae_i(h); return F.normalize(r,dim=-1),h,z,r
    def forward_txt(self): h = self.txt(txt_base); z,r = self.sae_t(h); return F.normalize(r,dim=-1),h,z,r

def metrics(model, split):
    model.eval(); ii = idxs[split]
    with torch.no_grad():
        tn, th, zt, rt = model.forward_txt(); im, ih, zi, ri = model.forward_img(X_t[ii]); logits = model.logit_scale.exp().clamp(max=100) * im @ tn.T; pred = logits.argmax(1); yy = y_t[ii]
        rep = classification_report(yy.cpu(), pred.cpu(), labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
        return {'recall_at_1_image_to_text': float((pred==yy).float().mean().item()), 'macro_recall': float(rep['macro avg']['recall']), 'sae_visual_sparsity_ratio': float((zi.abs()<0.01).float().mean().item()), 'sae_visual_recon_mse': float(F.mse_loss(ri, ih).item()), 'confusion_matrix': confusion_matrix(yy.cpu(), pred.cpu(), labels=list(range(len(classes)))).tolist(), 'classification_report': rep}

def run(cfg):
    torch.manual_seed(SEED + cfg['seed_offset'])
    model = M(cfg['drop'], cfg['k_frac']).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    best = None; best_score = -9; hist = []; train_idx = idxs['train']; bs = cfg['batch']; bad = 0
    for ep in range(1, cfg['epochs'] + 1):
        model.train(); perm = train_idx[torch.randperm(len(train_idx), device=DEVICE)]; losses = []
        for start in range(0, len(perm), bs):
            ii = perm[start:start+bs]; opt.zero_grad(set_to_none=True); tn, th, zt, rt = model.forward_txt(); im, ih, zi, ri = model.forward_img(X_t[ii]); logits = model.logit_scale.exp().clamp(max=100)*im@tn.T; ce = F.cross_entropy(logits, y_t[ii]); rec = F.mse_loss(ri, ih)+F.mse_loss(rt, th); l1 = zi.abs().mean()+zt.abs().mean(); loss = ce + cfg['alpha']*rec + cfg['l1']*l1; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(loss.item())
        if ep % 5 == 0 or ep == 1:
            val = metrics(model, 'val'); row = {'epoch': ep, 'loss': float(np.mean(losses)), 'val': val}; hist.append(row)
            score = val['recall_at_1_image_to_text'] + 0.05*min(val['sae_visual_sparsity_ratio'],0.90) - 0.05*max(0,val['sae_visual_recon_mse']-0.02)
            print(json.dumps({'cfg': cfg['name'], 'epoch': ep, 'loss': row['loss'], 'val_r1': val['recall_at_1_image_to_text'], 'val_sp': val['sae_visual_sparsity_ratio'], 'val_mse': val['sae_visual_recon_mse']}, ensure_ascii=False), flush=True)
            if score > best_score: best_score = score; best = {'state': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'hist': hist.copy(), 'cfg': cfg}; bad = 0
            else: bad += 1
            if ep >= 45 and bad >= 10: break
    model.load_state_dict(best['state'])
    out = {'run_name': EXPERIMENT + '_' + cfg['name'], 'dataset': str(DATA), 'embedding_npz': str(EMB_NPZ), 'config': cfg, 'history': best['hist'], 'val': metrics(model,'val'), 'train': metrics(model,'train'), 'test': metrics(model,'test')}
    path = OUT / f"{cfg['name']}_best.pt"; torch.save({'model_state_dict': model.state_dict(), 'classes': classes, 'config': cfg, 'dataset': str(DATA), 'embedding_npz': str(EMB_NPZ)}, path)
    out['checkpoint_best'] = str(path); out['checkpoint_best_md5'] = hashlib.md5(path.read_bytes()).hexdigest(); (OUT / f"metrics_{cfg['name']}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('DONE_CFG', cfg['name'], json.dumps({s:{k:out[s][k] for k in ['recall_at_1_image_to_text','macro_recall','sae_visual_sparsity_ratio','sae_visual_recon_mse']} for s in ['train','val','test']}, ensure_ascii=False), flush=True)
    return out

cfgs = []
for seed in [11, 14, 17, 21, 25, 33, 37, 41]:
    cfgs.append({'name': f'seed{seed}_k15_drop20_wd5e4', 'k_frac':0.15, 'l1':0.003, 'alpha':0.10, 'lr':2e-4, 'wd':5e-4, 'drop':0.20, 'epochs':90, 'batch':64, 'seed_offset':seed})
for seed in [45, 49, 53, 57]:
    cfgs.append({'name': f'seed{seed}_k12_drop25_wd1e3', 'k_frac':0.12, 'l1':0.004, 'alpha':0.10, 'lr':1.5e-4, 'wd':1e-3, 'drop':0.25, 'epochs':90, 'batch':64, 'seed_offset':seed})

allres = [run(c) for c in cfgs]
summary = [{'name': r['config']['name'], 'train_r1': r['train']['recall_at_1_image_to_text'], 'val_r1': r['val']['recall_at_1_image_to_text'], 'test_r1': r['test']['recall_at_1_image_to_text'], 'val_sp': r['val']['sae_visual_sparsity_ratio'], 'test_sp': r['test']['sae_visual_sparsity_ratio'], 'val_mse': r['val']['sae_visual_recon_mse'], 'test_mse': r['test']['sae_visual_recon_mse'], 'md5': r['checkpoint_best_md5']} for r in allres]
(OUT / f'summary_{EXPERIMENT}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print('SUMMARY_V10_GSPLIT_SEED_SWEEP', json.dumps(summary, ensure_ascii=False), flush=True)
