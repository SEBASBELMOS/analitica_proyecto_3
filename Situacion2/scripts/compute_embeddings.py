from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import open_clip


ROOT = Path("/workspace/geovision-cali-hf")
DATA = Path(os.environ.get("DATASET_DIR", ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500"))
OUT = Path(os.environ.get("OUT_DIR", ROOT / "outputs/clip_training_remoteclip_v10_v5b_embeddings"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH = int(os.environ.get("BATCH", "32"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def s2_to_rgb(path: str) -> torch.Tensor:
    arr = np.load(path).astype("float32")
    rgb = arr[:, :, [3, 2, 1]]
    lo = np.nanpercentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")
    ten = torch.from_numpy(rgb).permute(2, 0, 1)
    ten = F.interpolate(ten.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return (ten - CLIP_MEAN) / CLIP_STD


def main() -> None:
    meta = pd.read_json(DATA / "metadata.jsonl", lines=True)
    print(json.dumps({"stage": "start", "dataset": str(DATA), "n": len(meta), "device": str(DEVICE)}, ensure_ascii=False), flush=True)

    remote_ckpt = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir=str(ROOT / "checkpoints" / "remoteclip"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")
    state = torch.load(remote_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    embs = []
    paths = meta["image_path"].tolist()
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            batch_paths = paths[start:start + BATCH]
            imgs = torch.stack([s2_to_rgb(p) for p in batch_paths]).to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                z = model.encode_image(imgs).float()
            z = F.normalize(z, dim=-1)
            embs.append(z.cpu().numpy().astype("float32"))
            print(json.dumps({"stage": "embedding", "done": min(start + len(batch_paths), len(paths)), "total": len(paths)}, ensure_ascii=False), flush=True)

    remote = np.concatenate(embs, axis=0).astype("float32")
    out_npz = OUT / "embeddings_remoteclip_v10_v5b.npz"
    np.savez_compressed(
        out_npz,
        remoteclip_visual_512=remote,
        pair_id=meta["pair_id"].astype(str).to_numpy(),
        image_path=meta["image_path"].astype(str).to_numpy(),
    )
    summary = {
        "dataset": str(DATA),
        "output_npz": str(out_npz),
        "shape": list(remote.shape),
        "dtype": str(remote.dtype),
        "remoteclip_checkpoint": remote_ckpt,
        "normalized": True,
    }
    (OUT / "embedding_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
