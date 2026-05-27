from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import zarr


POLLUTANTS = ["NO2", "SO2", "O3"]
HORIZONS = [1, 3, 7]


class SpatialTensorDataset(Dataset):
    def __init__(self, tensor_root: Path, indices: np.ndarray, y_mean: float, y_std: float):
        self.root = zarr.open_group(str(tensor_root), mode="r")
        self.x = self.root["X_h_256"]
        self.y = self.root["y_t1_t3_t7"]
        self.mask = self.root["y_mask"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        src_idx = int(self.indices[idx])
        x = np.asarray(self.x[src_idx], dtype="float32")
        y = np.asarray(self.y[src_idx], dtype="float32")
        mask = np.asarray(self.mask[src_idx], dtype="float32")
        y_scaled = (y - self.y_mean) / self.y_std
        y_scaled = np.where(mask > 0, y_scaled, 0.0).astype("float32")
        return torch.from_numpy(x), torch.from_numpy(y_scaled), torch.from_numpy(mask), torch.from_numpy(y)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, height: int, width: int, device: torch.device):
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device)
        return h, c


class StackedConvLSTM(nn.Module):
    def __init__(self, input_channels: int = 256, hidden_channels: int = 128, kernel_size: int = 3, num_layers: int = 2):
        super().__init__()
        layers = []
        for layer in range(num_layers):
            in_ch = input_channels if layer == 0 else hidden_channels
            layers.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            x = torch.flip(x, dims=[1])
        b, t, _, h, w = x.shape
        layer_input = x
        for cell in self.layers:
            state = cell.init_state(b, h, w, x.device)
            outputs = []
            for step in range(t):
                state = cell(layer_input[:, step], state)
                outputs.append(state[0])
            layer_input = torch.stack(outputs, dim=1)
        return layer_input[:, -1]


class BiConvLSTMForecast(nn.Module):
    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        num_layers: int = 2,
        horizons: int = 3,
        pollutants: int = 3,
    ):
        super().__init__()
        self.horizons = horizons
        self.pollutants = pollutants
        self.forward_net = StackedConvLSTM(input_channels, hidden_channels, kernel_size, num_layers)
        self.backward_net = StackedConvLSTM(input_channels, hidden_channels, kernel_size, num_layers)
        self.head = nn.Conv2d(2 * hidden_channels, horizons * pollutants, kernel_size=1)

    def forward(self, x: torch.Tensor):
        hf = self.forward_net(x, reverse=False)
        hb = self.backward_net(x, reverse=True)
        out = self.head(torch.cat([hf, hb], dim=1))
        b, _, h, w = out.shape
        return out.view(b, self.horizons, self.pollutants, h, w)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def regression_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    keep = mask.astype(bool)
    yp = pred[keep]
    yt = y[keep]
    if len(yt) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    err = yp - yt
    rmse = float(math.sqrt(float(np.mean(err**2))))
    mae = float(np.mean(np.abs(err)))
    if len(yt) > 1 and float(np.var(yt)) > 0:
        r2 = float(1.0 - np.sum(err**2) / np.sum((yt - np.mean(yt)) ** 2))
    else:
        r2 = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(yt))}


def detailed_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> list[dict[str, float | int | str]]:
    rows = []
    rows.append({"scope": "all", "pollutant": "all", "horizon_days": "all", **regression_metrics(pred, y, mask)})
    for p_idx, pollutant in enumerate(POLLUTANTS):
        rows.append({"scope": "pollutant", "pollutant": pollutant, "horizon_days": "all", **regression_metrics(pred[:, :, p_idx], y[:, :, p_idx], mask[:, :, p_idx])})
    for h_idx, horizon in enumerate(HORIZONS):
        rows.append({"scope": "horizon", "pollutant": "all", "horizon_days": horizon, **regression_metrics(pred[:, h_idx], y[:, h_idx], mask[:, h_idx])})
    for h_idx, horizon in enumerate(HORIZONS):
        for p_idx, pollutant in enumerate(POLLUTANTS):
            rows.append({"scope": "pollutant_horizon", "pollutant": pollutant, "horizon_days": horizon, **regression_metrics(pred[:, h_idx, p_idx], y[:, h_idx, p_idx], mask[:, h_idx, p_idx])})
    return rows


def split_indices(metadata: pd.DataFrame, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    meta = metadata.copy()
    meta["end_date"] = pd.to_datetime(meta["end_date"])
    order = np.argsort(meta["end_date"].to_numpy())
    if max_samples is not None:
        order = order[:max_samples]
        meta = meta.iloc[order].reset_index(drop=False).rename(columns={"index": "tensor_index"})
        ordered_indices = meta["tensor_index"].to_numpy(dtype=np.int64)
    else:
        meta = meta.reset_index(drop=False).rename(columns={"index": "tensor_index"})
        ordered_indices = order.astype(np.int64)
    n = len(ordered_indices)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return ordered_indices[:n_train], ordered_indices[n_train : n_train + n_val], ordered_indices[n_train + n_val :], meta


def target_scale(tensor_root: Path, train_idx: np.ndarray) -> tuple[float, float]:
    root = zarr.open_group(str(tensor_root), mode="r")
    y_arr = root["y_t1_t3_t7"]
    mask_arr = root["y_mask"]
    values = []
    for idx in train_idx:
        y = np.asarray(y_arr[int(idx)], dtype="float32")
        mask = np.asarray(mask_arr[int(idx)], dtype=bool)
        values.append(y[mask])
    vals = np.concatenate(values) if values else np.array([0.0], dtype="float32")
    std = float(np.std(vals))
    return float(np.mean(vals)), max(std, 1e-6)


def collect_predictions(model, loader, device, y_mean: float, y_std: float, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    preds, ys, masks = [], [], []
    for xb, yb_scaled, mb, yb_raw in loader:
        xb = xb.to(device, non_blocking=True)
        yb_scaled = yb_scaled.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred_scaled = model(xb)
            loss = masked_mse(pred_scaled, yb_scaled, mb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        pred_raw = pred_scaled.detach().cpu().numpy() * y_std + y_mean
        preds.append(pred_raw.astype("float32"))
        ys.append(yb_raw.numpy().astype("float32"))
        masks.append(mb.detach().cpu().numpy().astype("float32"))
    return float(np.mean(losses)) if losses else float("nan"), np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-zarr", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-channels", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test cap after temporal ordering")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(args.metadata)
    train_idx, val_idx, test_idx, split_meta = split_indices(meta, args.max_samples)
    split_meta.to_csv(args.out / "metadata_samples_with_tensor_index.csv", index=False)
    y_mean, y_std = target_scale(args.tensor_zarr, train_idx)

    loaders = {
        "train": DataLoader(SpatialTensorDataset(args.tensor_zarr, train_idx, y_mean, y_std), batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(SpatialTensorDataset(args.tensor_zarr, val_idx, y_mean, y_std), batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(SpatialTensorDataset(args.tensor_zarr, test_idx, y_mean, y_std), batch_size=args.batch_size, shuffle=False, num_workers=0),
    }

    root = zarr.open_group(str(args.tensor_zarr), mode="r")
    x_shape = list(root["X_h_256"].shape)
    y_shape = list(root["y_t1_t3_t7"].shape)
    model = BiConvLSTMForecast(
        input_channels=256,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        horizons=len(HORIZONS),
        pollutants=len(POLLUTANTS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = args.out / "convlstm_covariates_spatial_best.pt"
    history = []
    best_val = float("inf")
    bad = 0

    setup = {
        "tensor_zarr": str(args.tensor_zarr),
        "metadata": str(args.metadata),
        "device": str(device),
        "x_shape": x_shape,
        "y_shape": y_shape,
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "y_scale": {"mean": y_mean, "std": y_std},
        "architecture": {"bidirectional": True, "hidden": args.hidden_channels, "kernel": args.kernel_size, "layers": args.num_layers, "output": "B x 3 x 3 x H x W"},
    }
    (args.out / "training_setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "setup", **setup}), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = collect_predictions(model, loaders["train"], device, y_mean, y_std, optimizer=optimizer)
        val_loss, val_pred, val_y, val_mask = collect_predictions(model, loaders["val"], device, y_mean, y_std)
        val_metrics = regression_metrics(val_pred, val_y, val_mask)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps({"stage": "epoch", **row}), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({"model_state_dict": model.state_dict(), "config": setup, "epoch": epoch, "val_loss": val_loss}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                break

    pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_pred, test_y, test_mask = collect_predictions(model, loaders["test"], device, y_mean, y_std)
    metrics = detailed_metrics(test_pred, test_y, test_mask)
    pd.DataFrame(metrics).to_csv(args.out / "test_metrics_masked.csv", index=False)
    summary = {
        **setup,
        "best_epoch": int(ckpt["epoch"]),
        "best_val_loss": float(ckpt["val_loss"]),
        "test_loss_scaled": float(test_loss),
        "test_metrics_all": metrics[0],
        "checkpoint": str(best_path),
        "note_no2": "NO2 has only one available DAGMA station in current inputs; spatial LOO-CV for NO2 is not defensible and must be reported as a data limitation.",
    }
    (args.out / "summary_convlstm_bidirectional_spatial.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "summary", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
