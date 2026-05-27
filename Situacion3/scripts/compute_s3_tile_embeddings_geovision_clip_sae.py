from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import open_clip


BASE = Path("/workspace/geovision-cali-hf")
S2_V5B_META = BASE / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500/metadata.jsonl"
S2_V5B_REMOTE = BASE / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz"
S2_V5B_CKPT = BASE / "outputs/clip_training_remoteclip_fusion_ksae_v10_v5b_gsplit_seed_sweep/seed57_k12_drop25_wd1e3_best.pt"

SEED = 42
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def md5_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def s2_to_rgb(path: str) -> torch.Tensor:
    arr = np.load(path).astype("float32")
    rgb = arr[:, :, [3, 2, 1]]
    lo = np.nanpercentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")
    ten = torch.from_numpy(rgb).permute(2, 0, 1)
    ten = F.interpolate(ten.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return (ten - CLIP_MEAN) / CLIP_STD


def aux_row_from_image_and_meta(r) -> np.ndarray:
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
    vals.extend([
        float(d.year),
        np.sin(2 * np.pi * doy / 366),
        np.cos(2 * np.pi * doy / 366),
        float(getattr(r, "ndvi_mean", np.nanmean(ndvi))),
        float(getattr(r, "ndbi_mean", np.nanmean(ndbi))),
        float(getattr(r, "ndwi_mean", np.nanmean(ndwi))),
        float(getattr(r, "scl_cloud_shadow_pct", 0.0)),
        float(getattr(r, "scl_valid_visual_pct", 0.0)),
    ])
    return np.array(vals, dtype="float32")


class SAE(nn.Module):
    def __init__(self, emb=256, hidden=1024, k_frac=0.15):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(emb, hidden), nn.ReLU())
        self.dec = nn.Linear(hidden, emb)
        self.k_frac = k_frac

    def forward(self, x):
        z = self.enc(x)
        k = max(1, int(z.shape[1] * self.k_frac))
        vals, idx = torch.topk(z, k, dim=1)
        sparse = torch.zeros_like(z).scatter(1, idx, vals)
        return sparse, self.dec(sparse)


class VisualProjectorSAE(nn.Module):
    def __init__(self, in_dim=595, emb=256, drop=0.25, k_frac=0.12):
        super().__init__()
        self.img = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 768),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, emb),
        )
        self.sae_i = SAE(emb=emb, hidden=1024, k_frac=k_frac)

    def forward_img(self, x):
        h = self.img(x)
        z, r = self.sae_i(h)
        return h, z, r


def load_remoteclip(device: torch.device):
    remote_ckpt = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir=str(BASE / "checkpoints" / "remoteclip"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(remote_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, remote_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True, help="CSV metadata with accepted_s2_policy/image_path/date_day")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tiles", type=int, default=None)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    meta_all = pd.read_csv(args.metadata)
    meta = meta_all[meta_all["accepted_s2_policy"].fillna(False).astype(bool)].copy().reset_index(drop=True)
    meta = meta[meta["image_path"].notna()].copy().reset_index(drop=True)
    if args.max_tiles is not None:
        meta = meta.iloc[: args.max_tiles].copy().reset_index(drop=True)
    if "pair_id" not in meta.columns:
        meta["pair_id"] = meta["tile_id"].astype(str)
    meta["date_day"] = pd.to_datetime(meta["date_day"]).dt.strftime("%Y-%m-%d")
    print(json.dumps({"stage": "load", "metadata": str(args.metadata), "accepted_tiles": len(meta), "device": str(device)}, ensure_ascii=False), flush=True)

    s2_meta = pd.read_json(S2_V5B_META, lines=True)
    s2_remote = np.load(S2_V5B_REMOTE, allow_pickle=True)["remoteclip_visual_512"].astype("float32")

    clip_model, remote_ckpt = load_remoteclip(device)
    remote_embs = []
    paths = meta["image_path"].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(paths), args.batch):
            batch_paths = paths[start : start + args.batch]
            imgs = torch.stack([s2_to_rgb(p) for p in batch_paths]).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                z = clip_model.encode_image(imgs).float()
            z = F.normalize(z, dim=-1)
            remote_embs.append(z.cpu().numpy().astype("float32"))
            print(json.dumps({"stage": "remoteclip", "done": min(start + len(batch_paths), len(paths)), "total": len(paths)}, ensure_ascii=False), flush=True)
    remote = np.concatenate(remote_embs, axis=0).astype("float32") if remote_embs else np.zeros((0, 512), dtype="float32")

    aux_new = np.stack([aux_row_from_image_and_meta(r) for _, r in meta.iterrows()]).astype("float32") if len(meta) else np.zeros((0, 83), dtype="float32")
    aux_s2 = np.stack([aux_row_from_image_and_meta(r) for _, r in s2_meta.iterrows()]).astype("float32")
    train_mask = s2_meta["split"].to_numpy() == "train"
    sc_remote = StandardScaler().fit(s2_remote[train_mask])
    sc_aux = StandardScaler().fit(aux_s2[train_mask])
    X_595 = np.concatenate([sc_remote.transform(remote), sc_aux.transform(aux_new)], axis=1).astype("float32") if len(meta) else np.zeros((0, 595), dtype="float32")

    ckpt = torch.load(S2_V5B_CKPT, map_location="cpu")
    model = VisualProjectorSAE(in_dim=595, emb=256, drop=ckpt["config"]["drop"], k_frac=ckpt["config"]["k_frac"]).to(device)
    visual_state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("img.") or k.startswith("sae_i.")}
    missing, unexpected = model.load_state_dict(visual_state, strict=False)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_595, dtype=torch.float32, device=device)
        h_256, z_1024, r_256 = model.forward_img(x_t)
    h_256 = h_256.cpu().numpy().astype("float32")
    z_1024 = z_1024.cpu().numpy().astype("float32")
    r_256 = r_256.cpu().numpy().astype("float32")

    out_npz = args.out / "embeddings_geovision_clip_sae_256.npz"
    out_meta = args.out / "metadata_embeddings.csv"
    np.savez_compressed(
        out_npz,
        remoteclip_visual_512=remote,
        h_256=h_256,
        r_256=r_256,
        z_1024=z_1024,
        pair_id=meta["pair_id"].astype(str).to_numpy(),
        tile_id=meta["tile_id"].astype(str).to_numpy(),
        image_path=meta["image_path"].astype(str).to_numpy(),
        grid_id=meta["grid_id"].astype(str).to_numpy(),
        date_day=meta["date_day"].astype(str).to_numpy(),
    )
    meta.to_csv(out_meta, index=False)
    summary = {
        "metadata": str(args.metadata),
        "output_npz": str(out_npz),
        "n_tiles": int(len(meta)),
        "remoteclip_visual_512_shape": list(remote.shape),
        "h_256_shape": list(h_256.shape),
        "r_256_shape": list(r_256.shape),
        "z_1024_shape": list(z_1024.shape),
        "z_1024_sparsity_ratio": float((np.abs(z_1024) < 1e-8).mean()) if len(z_1024) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "remoteclip_checkpoint": str(remote_ckpt),
        "checkpoint": str(S2_V5B_CKPT),
        "checkpoint_md5": md5_file(S2_V5B_CKPT),
        "input_metadata_md5": md5_file(args.metadata),
    }
    (args.out / "summary_embeddings.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
