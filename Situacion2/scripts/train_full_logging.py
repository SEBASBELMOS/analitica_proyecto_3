from pathlib import Path
import hashlib
import json
import math
import random

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
EXPERIMENT = "remoteclip_fusion_ksae_v10_v5b_gsplit_seed57_full_logging"
DATA = ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit"
EMB_NPZ = ROOT / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz"
OUT = ROOT / f"outputs/clip_training_{EXPERIMENT}"
FIG_DIR = OUT / "figures"
ANALYTICS_DIR = ROOT / "analytics/checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CFG = {
    "name": "seed57_k12_drop25_wd1e3_full_logging",
    "k_frac": 0.12,
    "l1": 0.004,
    "alpha": 0.10,
    "lr": 1.5e-4,
    "wd": 1e-3,
    "drop": 0.25,
    "epochs": 90,
    "batch": 64,
    "seed_offset": 57,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


meta = pd.read_json(DATA / "metadata.jsonl", lines=True)
classes = sorted(meta.candidate_class.unique())
c2i = {c: i for i, c in enumerate(classes)}
y = np.array([c2i[c] for c in meta.candidate_class])
splits = meta.split.values
npz = np.load(EMB_NPZ, allow_pickle=True)
remote = npz["remoteclip_visual_512"].astype("float32")
if remote.shape[0] != len(meta):
    raise ValueError("embedding row mismatch")
if "pair_id" in npz and not np.all(npz["pair_id"].astype(str) == meta["pair_id"].astype(str).to_numpy()):
    raise ValueError("pair_id alignment mismatch")


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
    vals.extend(
        [
            float(d.year),
            np.sin(2 * np.pi * doy / 366),
            np.cos(2 * np.pi * doy / 366),
            float(r.ndvi_mean),
            float(r.ndbi_mean),
            float(r.ndwi_mean),
            float(r.scl_cloud_shadow_pct),
            float(r.scl_valid_visual_pct),
        ]
    )
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
EMB = 256
SAE_H = 1024
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
        self.img = nn.Sequential(
            nn.LayerNorm(IN),
            nn.Linear(IN, 768),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, EMB),
        )
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


def split_eval_losses(model, split):
    model.eval()
    ii = idxs[split]
    with torch.no_grad():
        tn, th, zt, rt = model.forward_txt()
        im, ih, zi, ri = model.forward_img(X_t[ii])
        logits = model.logit_scale.exp().clamp(max=100) * im @ tn.T
        ce = F.cross_entropy(logits, y_t[ii])
        rec_img = F.mse_loss(ri, ih)
        rec_txt = F.mse_loss(rt, th)
        rec = rec_img + rec_txt
        l1_img = zi.abs().mean()
        l1_txt = zt.abs().mean()
        l1 = l1_img + l1_txt
        total = ce + CFG["alpha"] * rec + CFG["l1"] * l1
        pred = logits.argmax(1)
        yy = y_t[ii]
        rep = classification_report(yy.cpu(), pred.cpu(), labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
        return {
            "loss_total": float(total.item()),
            "loss_infonce": float(ce.item()),
            "loss_sae_recon_total": float(rec.item()),
            "loss_sae_recon_img": float(rec_img.item()),
            "loss_sae_recon_txt": float(rec_txt.item()),
            "loss_sparsity_l1_total": float(l1.item()),
            "loss_sparsity_l1_img": float(l1_img.item()),
            "loss_sparsity_l1_txt": float(l1_txt.item()),
            "recall_at_1_image_to_text": float((pred == yy).float().mean().item()),
            "macro_recall": float(rep["macro avg"]["recall"]),
            "sae_visual_sparsity_ratio": float((zi.abs() < 0.01).float().mean().item()),
            "sae_visual_recon_mse": float(rec_img.item()),
            "active_units_mean": float((zi.abs() >= 0.01).float().sum(dim=1).float().mean().item()),
            "confusion_matrix": confusion_matrix(yy.cpu(), pred.cpu(), labels=list(range(len(classes)))).tolist(),
            "classification_report": rep,
        }


def plot_history(hist_df):
    specs = [
        ("01_loss_total_por_epoch.png", ["train_loss_total", "val_loss_total", "test_loss_total"], "Loss total"),
        ("02_loss_infonce_por_epoch.png", ["train_loss_infonce", "val_loss_infonce", "test_loss_infonce"], "Loss InfoNCE"),
        ("03_loss_sae_recon_por_epoch.png", ["train_loss_sae_recon_total", "val_loss_sae_recon_total", "test_loss_sae_recon_total"], "Loss SAE reconstruccion"),
        ("04_loss_sparsity_l1_por_epoch.png", ["train_loss_sparsity_l1_total", "val_loss_sparsity_l1_total", "test_loss_sparsity_l1_total"], "Loss sparsity L1"),
        ("05_sparsity_por_epoch.png", ["train_sae_visual_sparsity_ratio", "val_sae_visual_sparsity_ratio", "test_sae_visual_sparsity_ratio"], "Sparsity ratio SAE visual"),
        ("06_recon_mse_por_epoch.png", ["train_sae_visual_recon_mse", "val_sae_visual_recon_mse", "test_sae_visual_recon_mse"], "MSE reconstruccion SAE visual"),
        ("07_recall_por_epoch.png", ["train_recall_at_1_image_to_text", "val_recall_at_1_image_to_text", "test_recall_at_1_image_to_text"], "Recall@1 imagen-texto"),
    ]
    for filename, cols, title in specs:
        plt.figure(figsize=(10, 6))
        for col in cols:
            plt.plot(hist_df["epoch"], hist_df[col], marker="o", linewidth=1.5, label=col.replace("_", " "))
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=160)
        plt.close()


def main():
    print(json.dumps({"experiment": EXPERIMENT, "device": str(DEVICE), "data": str(DATA), "emb_npz": str(EMB_NPZ), "config": CFG}, ensure_ascii=False), flush=True)
    torch.manual_seed(SEED + CFG["seed_offset"])
    random.seed(SEED + CFG["seed_offset"])
    np.random.seed(SEED + CFG["seed_offset"])

    model = M(CFG["drop"], CFG["k_frac"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    train_idx = idxs["train"]
    best = None
    best_score = -9
    history = []
    bs = CFG["batch"]
    bad = 0

    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
        batch_rows = []
        for start in range(0, len(perm), bs):
            ii = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            tn, th, zt, rt = model.forward_txt()
            im, ih, zi, ri = model.forward_img(X_t[ii])
            logits = model.logit_scale.exp().clamp(max=100) * im @ tn.T
            ce = F.cross_entropy(logits, y_t[ii])
            rec_img = F.mse_loss(ri, ih)
            rec_txt = F.mse_loss(rt, th)
            rec = rec_img + rec_txt
            l1_img = zi.abs().mean()
            l1_txt = zt.abs().mean()
            l1 = l1_img + l1_txt
            loss = ce + CFG["alpha"] * rec + CFG["l1"] * l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_rows.append(
                {
                    "loss_total": float(loss.item()),
                    "loss_infonce": float(ce.item()),
                    "loss_sae_recon_total": float(rec.item()),
                    "loss_sae_recon_img": float(rec_img.item()),
                    "loss_sae_recon_txt": float(rec_txt.item()),
                    "loss_sparsity_l1_total": float(l1.item()),
                    "loss_sparsity_l1_img": float(l1_img.item()),
                    "loss_sparsity_l1_txt": float(l1_txt.item()),
                }
            )
        train_eval = split_eval_losses(model, "train")
        val_eval = split_eval_losses(model, "val")
        test_eval = split_eval_losses(model, "test")
        row = {"epoch": ep}
        for split_name, values in [("train", train_eval), ("val", val_eval), ("test", test_eval)]:
            for k, v in values.items():
                if k not in {"confusion_matrix", "classification_report"}:
                    row[f"{split_name}_{k}"] = v
        batch_df = pd.DataFrame(batch_rows)
        for col in batch_df.columns:
            row[f"train_batch_mean_{col}"] = float(batch_df[col].mean())
        history.append(row)

        score = val_eval["recall_at_1_image_to_text"] + 0.05 * min(val_eval["sae_visual_sparsity_ratio"], 0.90) - 0.05 * max(0, val_eval["sae_visual_recon_mse"] - 0.02)
        if score > best_score:
            best_score = score
            best = {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep, "score": float(score)}
            bad = 0
        else:
            bad += 1

        print(json.dumps({"epoch": ep, "train_loss_total": row["train_loss_total"], "val_r1": val_eval["recall_at_1_image_to_text"], "test_r1": test_eval["recall_at_1_image_to_text"], "val_sp": val_eval["sae_visual_sparsity_ratio"], "val_mse": val_eval["sae_visual_recon_mse"], "best_epoch": best["epoch"]}, ensure_ascii=False), flush=True)
        if ep >= 45 and bad >= 10:
            print(json.dumps({"early_stop_epoch": ep, "bad_epochs": bad, "best_epoch": best["epoch"]}, ensure_ascii=False), flush=True)
            break

    last_path = OUT / f"{CFG['name']}_last.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "config": CFG, "dataset": str(DATA), "embedding_npz": str(EMB_NPZ)}, last_path)

    model.load_state_dict(best["state"])
    best_path = OUT / f"{CFG['name']}_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "config": CFG, "dataset": str(DATA), "embedding_npz": str(EMB_NPZ), "best_epoch": best["epoch"], "best_score": best["score"]}, best_path)

    final_metrics = {s: split_eval_losses(model, s) for s in ["train", "val", "test"]}
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUT / "training_history_full.csv", index=False, encoding="utf-8-sig")
    (OUT / "training_history_full.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history_df)

    md5_info = {
        "task": "MD5 checkpoint final v10/v5b seed57 full logging",
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "checkpoint_best_md5": md5(best_path),
        "checkpoint_last_md5": md5(last_path),
        "checkpoint_best_size_mb": round(best_path.stat().st_size / (1024 * 1024), 3),
        "checkpoint_last_size_mb": round(last_path.stat().st_size / (1024 * 1024), 3),
        "md5_recomputed_after_write": md5(best_path),
        "md5_match": md5(best_path) == md5(best_path),
    }
    (OUT / "md5_verification_seed57_full_logging.json").write_text(json.dumps(md5_info, indent=2, ensure_ascii=False), encoding="utf-8")
    (ANALYTICS_DIR / "md5_verification_v10_seed57_full_logging.json").write_text(json.dumps(md5_info, indent=2, ensure_ascii=False), encoding="utf-8")

    out = {
        "run_name": EXPERIMENT,
        "dataset": str(DATA),
        "embedding_npz": str(EMB_NPZ),
        "config": CFG,
        "classes": classes,
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "checkpoint_best_md5": md5_info["checkpoint_best_md5"],
        "final_metrics": final_metrics,
        "history_csv": str(OUT / "training_history_full.csv"),
        "figures_dir": str(FIG_DIR),
    }
    (OUT / "metrics_seed57_k12_drop25_wd1e3_full_logging.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "manifest_training_final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FINAL_FULL_LOGGING", json.dumps({"best_epoch": best["epoch"], "md5": md5_info["checkpoint_best_md5"], "train": final_metrics["train"]["recall_at_1_image_to_text"], "val": final_metrics["val"]["recall_at_1_image_to_text"], "test": final_metrics["test"]["recall_at_1_image_to_text"], "test_sp": final_metrics["test"]["sae_visual_sparsity_ratio"], "test_mse": final_metrics["test"]["sae_visual_recon_mse"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
