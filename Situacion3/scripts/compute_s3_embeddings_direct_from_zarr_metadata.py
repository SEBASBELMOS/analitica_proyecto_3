from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from numcodecs import Blosc
from sklearn.preprocessing import StandardScaler
import open_clip
import zarr


BASE = Path("/workspace/geovision-cali-hf")
S2_V5B_META = BASE / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500/metadata.jsonl"
S2_V5B_REMOTE = BASE / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz"
S2_V5B_CKPT = BASE / "outputs/clip_training_remoteclip_fusion_ksae_v10_v5b_gsplit_seed_sweep/seed57_k12_drop25_wd1e3_best.pt"

SEED = 42
TILE_SIZE = 64
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def md5_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_zarr_tile(row) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(row.zarr_path, mode="r")
    row_off = int(float(row.row_off))
    col_off = int(float(row.col_off))
    image_chw = root["image"][:, row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE]
    image = np.moveaxis(np.asarray(image_chw, dtype="float32"), 0, -1)
    scl = np.asarray(root["scl"][row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE], dtype="uint8")
    return image, scl


def image_to_remoteclip_rgb(image: np.ndarray) -> torch.Tensor:
    rgb = image[:, :, [3, 2, 1]]
    lo = np.nanpercentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")
    ten = torch.from_numpy(rgb).permute(2, 0, 1)
    ten = F.interpolate(ten.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return (ten - CLIP_MEAN) / CLIP_STD


def aux_from_image_and_meta(image: np.ndarray, r) -> np.ndarray:
    red = image[:, :, 3]
    green = image[:, :, 2]
    nir = image[:, :, 7]
    swir = image[:, :, 10]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    arr = image.reshape(-1, 12)
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


def aux_row_from_npy_meta(r) -> np.ndarray:
    image = np.load(r.image_path).astype("float32")
    return aux_from_image_and_meta(image, r)


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
            nn.LayerNorm(in_dim), nn.Linear(in_dim, 768), nn.GELU(), nn.Dropout(drop),
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, emb),
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


def load_metadata(paths: list[Path], max_tiles: int | None) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_metadata"] = str(path)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta = meta[meta["accepted_s2_policy"].fillna(False).astype(bool)].copy()
    meta = meta[meta["zarr_path"].notna()].copy()
    meta = meta.drop_duplicates(subset=["tile_id"]).reset_index(drop=True)
    meta["date_day"] = pd.to_datetime(meta["date_day"]).dt.strftime("%Y-%m-%d")
    if "pair_id" not in meta.columns:
        meta["pair_id"] = meta["tile_id"].astype(str)
    if max_tiles is not None:
        meta = meta.iloc[:max_tiles].copy().reset_index(drop=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume an existing output Zarr instead of overwriting it")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(args.metadata, args.max_tiles)
    n = len(meta)
    print(json.dumps({"stage": "load", "n_tiles": n, "device": str(device), "batch": args.batch}, ensure_ascii=False), flush=True)
    meta_out = args.out / "metadata_embeddings_direct_from_zarr.csv"
    meta.to_csv(meta_out, index=False)

    s2_meta = pd.read_json(S2_V5B_META, lines=True)
    s2_remote = np.load(S2_V5B_REMOTE, allow_pickle=True)["remoteclip_visual_512"].astype("float32")
    aux_s2 = np.stack([aux_row_from_npy_meta(r) for _, r in s2_meta.iterrows()]).astype("float32")
    train_mask = s2_meta["split"].to_numpy() == "train"
    sc_remote = StandardScaler().fit(s2_remote[train_mask])
    sc_aux = StandardScaler().fit(aux_s2[train_mask])

    clip_model, remote_ckpt = load_remoteclip(device)
    ckpt = torch.load(S2_V5B_CKPT, map_location="cpu")
    sae_model = VisualProjectorSAE(in_dim=595, emb=256, drop=ckpt["config"]["drop"], k_frac=ckpt["config"]["k_frac"]).to(device)
    visual_state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("img.") or k.startswith("sae_i.")}
    missing, unexpected = sae_model.load_state_dict(visual_state, strict=False)
    sae_model.eval()

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    zarr_path = args.out / "embeddings_direct_from_zarr.zarr"
    chunk_rows = min(max(args.batch, 1), max(n, 1))
    if args.resume and (zarr_path / ".zgroup").exists():
        root = zarr.open_group(str(zarr_path), mode="a")
        remote_ds = root["remoteclip_visual_512"]
        h_ds = root["h_256"]
        r_ds = root["r_256"]
        z_ds = root["z_1024"]
        if tuple(h_ds.shape) != (n, 256) or tuple(remote_ds.shape) != (n, 512):
            raise ValueError(f"Cannot resume: existing shapes do not match n={n}")
    else:
        root = zarr.open_group(str(zarr_path), mode="w")
        remote_ds = root.create_dataset("remoteclip_visual_512", shape=(n, 512), chunks=(chunk_rows, 512), dtype="float32", compressor=compressor)
        h_ds = root.create_dataset("h_256", shape=(n, 256), chunks=(chunk_rows, 256), dtype="float32", compressor=compressor)
        r_ds = root.create_dataset("r_256", shape=(n, 256), chunks=(chunk_rows, 256), dtype="float32", compressor=compressor)
        z_ds = root.create_dataset("z_1024", shape=(n, 1024), chunks=(chunk_rows, 1024), dtype="float32", compressor=compressor)

    def complete_until() -> int:
        arrays = [remote_ds, h_ds, r_ds, z_ds]
        max_full_chunks = n // chunk_rows
        done_chunks = 0
        for chunk_idx in range(max_full_chunks):
            key = f"{chunk_idx}.0"
            if all((Path(a.store.path) / a.path / key).exists() for a in arrays):
                done_chunks += 1
            else:
                break
        return done_chunks * chunk_rows

    start_at = complete_until() if args.resume else 0
    if start_at:
        print(json.dumps({"stage": "resume", "start_at": int(start_at), "total": n}, ensure_ascii=False), flush=True)

    with torch.no_grad():
        for start in range(start_at, n, args.batch):
            batch_meta = meta.iloc[start:start + args.batch]
            images = []
            aux_rows = []
            for row in batch_meta.itertuples(index=False):
                image, _ = read_zarr_tile(row)
                images.append(image_to_remoteclip_rgb(image))
                aux_rows.append(aux_from_image_and_meta(image, row))
            imgs = torch.stack(images).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                remote = clip_model.encode_image(imgs).float()
            remote = F.normalize(remote, dim=-1).cpu().numpy().astype("float32")
            aux = np.stack(aux_rows).astype("float32")
            X_595 = np.concatenate([sc_remote.transform(remote), sc_aux.transform(aux)], axis=1).astype("float32")
            x_t = torch.tensor(X_595, dtype=torch.float32, device=device)
            h_256, z_1024, r_256 = sae_model.forward_img(x_t)
            end = start + len(batch_meta)
            remote_ds[start:end, :] = remote
            h_ds[start:end, :] = h_256.cpu().numpy().astype("float32")
            r_ds[start:end, :] = r_256.cpu().numpy().astype("float32")
            z_ds[start:end, :] = z_1024.cpu().numpy().astype("float32")
            print(json.dumps({"stage": "batch", "done": end, "total": n}, ensure_ascii=False), flush=True)

    root.attrs.update({
        "n_tiles": int(n),
        "remoteclip_checkpoint": str(remote_ckpt),
        "checkpoint": str(S2_V5B_CKPT),
        "checkpoint_md5": md5_file(S2_V5B_CKPT),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    })
    summary = {
        "n_tiles": int(n),
        "output_zarr": str(args.out / "embeddings_direct_from_zarr.zarr"),
        "metadata": str(meta_out),
        "remoteclip_visual_512_shape": [int(n), 512],
        "h_256_shape": [int(n), 256],
        "r_256_shape": [int(n), 256],
        "z_1024_shape": [int(n), 1024],
        "checkpoint_md5": md5_file(S2_V5B_CKPT),
        "input_metadata_md5": {str(p): md5_file(p) for p in args.metadata if p.exists()},
    }
    (args.out / "summary_embeddings_direct_from_zarr.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
