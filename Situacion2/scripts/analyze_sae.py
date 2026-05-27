from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


ROOT = Path("/workspace/geovision-cali-hf")
TRAIN_OUT = ROOT / "outputs/clip_training_remoteclip_fusion_ksae_v10_v5b_gsplit_seed57_full_logging"
CHECKPOINT = TRAIN_OUT / "seed57_k12_drop25_wd1e3_full_logging_best.pt"
DATA = ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit"
EMB_NPZ = ROOT / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz"
OUT = ROOT / "outputs/sae_analysis_v10_v5b_gsplit_seed57_full_logging"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB = 256
SAE_H = 1024


meta = pd.read_json(DATA / "metadata.jsonl", lines=True)
classes = sorted(meta.candidate_class.unique())
c2i = {c: i for i, c in enumerate(classes)}
y = np.array([c2i[c] for c in meta.candidate_class])
splits = meta.split.values
npz = np.load(EMB_NPZ, allow_pickle=True)
remote = npz["remoteclip_visual_512"].astype("float32")
if remote.shape[0] != len(meta):
    raise ValueError("embedding row mismatch")


def aux_row(r):
    img = np.load(r.image_path).astype("float32")
    red = img[:, :, 3]
    green = img[:, :, 2]
    nir = img[:, :, 7]
    swir = img[:, :, 10]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    arr = img.reshape(-1, 12)
    vals = []
    vals.extend(arr.mean(0))
    vals.extend(arr.std(0))
    vals.extend(np.percentile(arr, [10, 50, 90], axis=0).ravel())
    for idx in [ndvi, ndbi, ndwi]:
        vals.extend([np.nanmean(idx), np.nanstd(idx), np.nanpercentile(idx, 10), np.nanpercentile(idx, 50), np.nanpercentile(idx, 90)])
    d = pd.to_datetime(r.date_day)
    doy = d.dayofyear
    vals.extend([float(d.year), np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366), float(r.ndvi_mean), float(r.ndbi_mean), float(r.ndwi_mean), float(r.scl_cloud_shadow_pct), float(r.scl_valid_visual_pct)])
    return np.array(vals, dtype="float32")


aux = np.stack([aux_row(r) for _, r in meta.iterrows()])
train_mask = splits == "train"
sc_r = StandardScaler().fit(remote[train_mask])
sc_a = StandardScaler().fit(aux[train_mask])
X = np.concatenate([sc_r.transform(remote), sc_a.transform(aux)], axis=1).astype("float32")
class_text = {c: meta[(meta.split == "train") & (meta.candidate_class == c)].text.iloc[0] for c in classes}
st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=str(DEVICE))
with torch.no_grad():
    txt_base = torch.tensor(st.encode([class_text[c] for c in classes], normalize_embeddings=True), dtype=torch.float32, device=DEVICE)

TEXT_DIM = txt_base.shape[1]
IN = X.shape[1]
X_t = torch.tensor(X, device=DEVICE)
y_t = torch.tensor(y, device=DEVICE)
idxs = {s: torch.tensor(np.where(splits == s)[0], device=DEVICE) for s in ["train", "val", "test"]}


class SAE(nn.Module):
    def __init__(self, k_frac=0.15):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(EMB, SAE_H), nn.ReLU())
        self.dec = nn.Linear(SAE_H, EMB)
        self.k_frac = k_frac

    def forward(self, x):
        z = self.enc(x)
        k = max(1, int(z.shape[1] * self.k_frac))
        vals, idx = torch.topk(z, k, dim=1)
        sparse = torch.zeros_like(z).scatter(1, idx, vals)
        return sparse, self.dec(sparse)


class M(nn.Module):
    def __init__(self, drop=0.20, k_frac=0.15):
        super().__init__()
        self.img = nn.Sequential(nn.LayerNorm(IN), nn.Linear(IN, 768), nn.GELU(), nn.Dropout(drop), nn.Linear(768, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, EMB))
        self.txt = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Linear(TEXT_DIM, 512), nn.GELU(), nn.Linear(512, EMB))
        self.sae_i = SAE(k_frac)
        self.sae_t = SAE(k_frac)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward_img(self, x):
        h = self.img(x)
        z, r = self.sae_i(h)
        return F.normalize(r, dim=-1), h, z, r

    def forward_txt(self):
        h = self.txt(txt_base)
        z, r = self.sae_t(h)
        return F.normalize(r, dim=-1), h, z, r


def metrics_and_activations(model, split):
    model.eval()
    ii = idxs[split]
    with torch.no_grad():
        tn, _, _, _ = model.forward_txt()
        im, ih, zi, ri = model.forward_img(X_t[ii])
        logits = model.logit_scale.exp().clamp(max=100) * im @ tn.T
        pred = logits.argmax(1)
        yy = y_t[ii]
        rep = classification_report(yy.cpu(), pred.cpu(), labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
        metrics = {
            "split": split,
            "n": int(len(ii)),
            "recall_at_1": float((pred == yy).float().mean().item()),
            "macro_recall": float(rep["macro avg"]["recall"]),
            "sparsity_ratio": float((zi.abs() < 0.01).float().mean().item()),
            "recon_mse": float(F.mse_loss(ri, ih).item()),
            "active_units_mean": float((zi.abs() >= 0.01).float().sum(dim=1).float().mean().item()),
            "active_units_median": float((zi.abs() >= 0.01).float().sum(dim=1).float().median().item()),
            "dead_units_pct": float(((zi.abs() >= 0.01).float().sum(dim=0) == 0).float().mean().item() * 100),
        }
        return metrics, zi.detach().cpu().numpy(), pred.detach().cpu().numpy(), yy.detach().cpu().numpy(), ii.detach().cpu().numpy(), confusion_matrix(yy.cpu(), pred.cpu(), labels=list(range(len(classes)))).tolist(), rep


def main():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg = ckpt["config"]
    model = M(cfg["drop"], cfg["k_frac"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split_rows = []
    all_activation_rows = []
    all_pair_rows = []
    reports = {}
    confusions = {}
    for split in ["train", "val", "test"]:
        metrics, z, pred, yy, original_idx, cm, rep = metrics_and_activations(model, split)
        split_rows.append(metrics)
        reports[split] = rep
        confusions[split] = cm
        active_mask = np.abs(z) >= 0.01
        for local_i, idx in enumerate(original_idx):
            top_units = np.argsort(-np.abs(z[local_i]))[:12]
            all_pair_rows.append(
                {
                    "split": split,
                    "pair_id": meta.iloc[idx].pair_id,
                    "candidate_class": meta.iloc[idx].candidate_class,
                    "pred_class": classes[int(pred[local_i])],
                    "correct": bool(pred[local_i] == yy[local_i]),
                    "active_units_count": int(active_mask[local_i].sum()),
                    "top_units": ",".join([str(int(u)) for u in top_units]),
                    "top_values": ",".join([f"{float(z[local_i, u]):.6f}" for u in top_units]),
                }
            )
        for class_name in classes:
            class_mask = np.array([meta.iloc[idx].candidate_class == class_name for idx in original_idx])
            if class_mask.sum() == 0:
                continue
            mean_abs = np.abs(z[class_mask]).mean(axis=0)
            active_freq = active_mask[class_mask].mean(axis=0)
            for unit in np.argsort(-mean_abs)[:25]:
                all_activation_rows.append(
                    {
                        "split": split,
                        "candidate_class": class_name,
                        "unit": int(unit),
                        "mean_abs_activation": float(mean_abs[unit]),
                        "active_frequency": float(active_freq[unit]),
                    }
                )

    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(OUT / "sae_split_summary.csv", index=False, encoding="utf-8-sig")
    top_units = pd.DataFrame(all_activation_rows)
    top_units.to_csv(OUT / "sae_top_units_by_class.csv", index=False, encoding="utf-8-sig")
    pair_units = pd.DataFrame(all_pair_rows)
    pair_units.to_csv(OUT / "sae_pair_top_units.csv", index=False, encoding="utf-8-sig")

    test_top = top_units[top_units["split"] == "test"].copy()
    heat = test_top.pivot_table(index="candidate_class", columns="unit", values="mean_abs_activation", aggfunc="mean", fill_value=0)
    keep_units = test_top.groupby("unit")["mean_abs_activation"].max().sort_values(ascending=False).head(30).index
    heat = heat.reindex(columns=keep_units, fill_value=0)
    plt.figure(figsize=(14, 5))
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Activacion absoluta media")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=90)
    plt.title("Top unidades SAE visuales por clase - test")
    plt.tight_layout()
    plt.savefig(OUT / "sae_top_units_by_class_heatmap.png", dpi=170)
    plt.close()

    manifest = {
        "checkpoint": str(CHECKPOINT),
        "dataset": str(DATA),
        "embedding_npz": str(EMB_NPZ),
        "config": cfg,
        "classes": classes,
        "reports": reports,
        "confusion_matrices": confusions,
        "outputs": {
            "split_summary": str(OUT / "sae_split_summary.csv"),
            "top_units_by_class": str(OUT / "sae_top_units_by_class.csv"),
            "pair_top_units": str(OUT / "sae_pair_top_units.csv"),
            "heatmap": str(OUT / "sae_top_units_by_class_heatmap.png"),
        },
    }
    (OUT / "manifest_sae_analysis.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "split_summary": split_rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
