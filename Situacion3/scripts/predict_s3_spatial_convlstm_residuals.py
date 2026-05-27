from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_s3_spatial_convlstm_from_zarr import BiConvLSTMForecast, HORIZONS, POLLUTANTS  # noqa: E402


BASE = Path("/workspace/geovision-cali-hf")
GRID_PATH = BASE / "outputs/situacion3/02_grilla_s2_features/grid_cali_005deg.parquet"
STATION_GRID_PATH = BASE / "outputs/situacion3/02_grilla_s2_features/station_to_s2_grid_mapping.csv"


def grid_id_from_rc(row_idx: int, col_idx: int) -> str:
    return f"g_{row_idx:03d}_{col_idx:03d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-zarr", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    arch = config["architecture"]
    y_mean = float(config["y_scale"]["mean"])
    y_std = float(config["y_scale"]["std"])

    model = BiConvLSTMForecast(
        input_channels=256,
        hidden_channels=int(arch["hidden"]),
        kernel_size=int(arch["kernel"]),
        num_layers=int(arch["layers"]),
        horizons=len(HORIZONS),
        pollutants=len(POLLUTANTS),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    root = zarr.open_group(str(args.tensor_zarr), mode="r")
    x_arr = root["X_h_256"]
    y_arr = root["y_t1_t3_t7"]
    mask_arr = root["y_mask"]
    row_min = int(root.attrs["row_min"])
    col_min = int(root.attrs["col_min"])

    samples = pd.read_csv(args.metadata)
    samples["end_date"] = pd.to_datetime(samples["end_date"]).dt.normalize()
    grid = pd.read_parquet(GRID_PATH)
    grid_lookup = grid.set_index("grid_id")[["lat", "lon"]].to_dict("index")
    station_map = pd.read_csv(STATION_GRID_PATH).rename(columns={"nearest_grid_id": "grid_id"})
    station_lookup = station_map.groupby("grid_id")["estacion"].apply(lambda s: "|".join(sorted(set(map(str, s))))).to_dict()

    rows = []
    prediction_zarr = args.out / "convlstm_predictions_all_samples.zarr"
    pred_root = zarr.open_group(str(prediction_zarr), mode="w")
    pred_ds = pred_root.create_dataset("prediction", shape=y_arr.shape, chunks=(1,) + tuple(y_arr.shape[1:]), dtype="float32", compressor=None)
    pred_root.attrs.update({"y_mean": y_mean, "y_std": y_std, "checkpoint": str(args.checkpoint)})

    with torch.no_grad():
        for start in range(0, x_arr.shape[0], args.batch_size):
            end = min(start + args.batch_size, x_arr.shape[0])
            x = torch.from_numpy(np.asarray(x_arr[start:end], dtype="float32")).to(device)
            pred_scaled = model(x).detach().cpu().numpy().astype("float32")
            pred = pred_scaled * y_std + y_mean
            pred_ds[start:end] = pred
            y = np.asarray(y_arr[start:end], dtype="float32")
            mask = np.asarray(mask_arr[start:end], dtype=bool)
            for local_i, sample_idx in enumerate(range(start, end)):
                coords = np.argwhere(mask[local_i])
                end_date = samples.iloc[sample_idx]["end_date"]
                for h_idx, p_idx, rr, cc in coords:
                    grid_id = grid_id_from_rc(row_min + int(rr), col_min + int(cc))
                    loc = grid_lookup.get(grid_id, {"lat": np.nan, "lon": np.nan})
                    observed = float(y[local_i, h_idx, p_idx, rr, cc])
                    predicted = float(pred[local_i, h_idx, p_idx, rr, cc])
                    rows.append({
                        "sample_index": int(sample_idx),
                        "sample_id": samples.iloc[sample_idx]["sample_id"],
                        "start_date": samples.iloc[sample_idx]["start_date"],
                        "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                        "target_date": (pd.Timestamp(end_date) + pd.Timedelta(days=int(HORIZONS[h_idx]))).strftime("%Y-%m-%d"),
                        "horizon_days": int(HORIZONS[h_idx]),
                        "pollutant": POLLUTANTS[p_idx],
                        "grid_id": grid_id,
                        "lat": float(loc["lat"]),
                        "lon": float(loc["lon"]),
                        "estacion": station_lookup.get(grid_id, ""),
                        "observed": observed,
                        "predicted_convlstm": predicted,
                        "residual_observed_minus_predicted": observed - predicted,
                    })
            print(json.dumps({"stage": "batch", "done": end, "total": int(x_arr.shape[0])}), flush=True)

    residuals = pd.DataFrame(rows)
    residuals.to_csv(args.out / "convlstm_predictions_residuals_long.csv", index=False)
    summary = {
        "tensor_zarr": str(args.tensor_zarr),
        "metadata": str(args.metadata),
        "checkpoint": str(args.checkpoint),
        "prediction_zarr": str(prediction_zarr),
        "residuals_csv": str(args.out / "convlstm_predictions_residuals_long.csv"),
        "n_samples": int(x_arr.shape[0]),
        "n_residual_rows": int(len(residuals)),
        "rows_by_pollutant": residuals.groupby("pollutant").size().to_dict() if len(residuals) else {},
        "rows_by_horizon": {str(k): int(v) for k, v in residuals.groupby("horizon_days").size().to_dict().items()} if len(residuals) else {},
    }
    (args.out / "summary_convlstm_predictions_residuals.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
