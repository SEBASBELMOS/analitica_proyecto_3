from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from pykrige.ok3d import OrdinaryKriging3D


BASE = Path("/workspace/geovision-cali-hf")
GRID_PATH = BASE / "outputs/situacion3/02_grilla_s2_features/grid_cali_005deg.parquet"
POLLUTANTS = ["NO2", "SO2", "O3"]
HORIZONS = [1, 3, 7]


def grid_id_from_rc(row_idx: int, col_idx: int) -> str:
    return f"g_{row_idx:03d}_{col_idx:03d}"


def experimental_variogram(df: pd.DataFrame, max_pairs: int = 20000, n_bins: int = 12) -> pd.DataFrame:
    if len(df) < 3:
        return pd.DataFrame(columns=["bin", "distance_min", "distance_max", "distance_mean", "semivariance", "n_pairs"])
    rng = np.random.default_rng(42)
    coords = df[["lon", "lat", "time_scaled"]].to_numpy("float64")
    vals = df["residual_observed_minus_predicted"].to_numpy("float64")
    n = len(df)
    if n * (n - 1) // 2 <= max_pairs:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=max_pairs)
        jj = rng.integers(0, n, size=max_pairs)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
    dist = np.linalg.norm(coords[ii] - coords[jj], axis=1)
    semi = 0.5 * (vals[ii] - vals[jj]) ** 2
    if len(dist) == 0 or float(dist.max()) == 0:
        return pd.DataFrame(columns=["bin", "distance_min", "distance_max", "distance_mean", "semivariance", "n_pairs"])
    bins = np.linspace(float(dist.min()), float(dist.max()), n_bins + 1)
    rows = []
    for b in range(n_bins):
        mask = (dist >= bins[b]) & (dist < bins[b + 1] if b < n_bins - 1 else dist <= bins[b + 1])
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "distance_min": float(bins[b]),
            "distance_max": float(bins[b + 1]),
            "distance_mean": float(dist[mask].mean()),
            "semivariance": float(semi[mask].mean()),
            "n_pairs": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-zarr", type=Path, required=True)
    ap.add_argument("--prediction-zarr", type=Path, required=True)
    ap.add_argument("--samples-metadata", type=Path, required=True)
    ap.add_argument("--residuals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variogram-model", default="linear", choices=["linear", "power", "gaussian", "spherical", "exponential"])
    ap.add_argument("--min-points", type=int, default=20)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tensor = zarr.open_group(str(args.tensor_zarr), mode="r")
    pred_root = zarr.open_group(str(args.prediction_zarr), mode="r")
    pred = pred_root["prediction"]
    row_min = int(tensor.attrs["row_min"])
    col_min = int(tensor.attrs["col_min"])
    _, _, _, h, w = pred.shape

    samples = pd.read_csv(args.samples_metadata)
    samples["end_date"] = pd.to_datetime(samples["end_date"]).dt.normalize()
    sample_idx = int(samples["end_date"].idxmax())
    sample = samples.iloc[sample_idx]
    residuals = pd.read_csv(args.residuals)
    residuals["target_date"] = pd.to_datetime(residuals["target_date"]).dt.normalize()
    residuals = residuals.dropna(subset=["lon", "lat", "residual_observed_minus_predicted"]).copy()
    t0 = residuals["target_date"].min()
    residuals["time_days"] = (residuals["target_date"] - t0).dt.days.astype(float)
    residuals["time_scaled"] = residuals["time_days"] / 365.25 * 0.05

    grid = pd.read_parquet(GRID_PATH)
    grid_lookup = grid.set_index("grid_id")[["lat", "lon"]].to_dict("index")
    grid_rows = []
    for rr in range(h):
        for cc in range(w):
            gid = grid_id_from_rc(row_min + rr, col_min + cc)
            loc = grid_lookup.get(gid)
            if loc is None:
                continue
            grid_rows.append({"rr": rr, "cc": cc, "grid_id": gid, "lat": float(loc["lat"]), "lon": float(loc["lon"])})
    grid_df = pd.DataFrame(grid_rows)

    map_rows = []
    variogram_rows = []
    model_rows = []
    for pollutant in POLLUTANTS:
        p_idx = POLLUTANTS.index(pollutant)
        for horizon in HORIZONS:
            h_idx = HORIZONS.index(horizon)
            sub = residuals[(residuals["pollutant"] == pollutant) & (residuals["horizon_days"] == horizon)].copy()
            target_date = pd.Timestamp(sample["end_date"]) + pd.Timedelta(days=int(horizon))
            target_time_scaled = float(((target_date - t0).days) / 365.25 * 0.05)
            pred_map = np.asarray(pred[sample_idx, h_idx, p_idx], dtype="float32")
            status = "ok"
            residual_hat = np.zeros(len(grid_df), dtype="float32")
            residual_var = np.full(len(grid_df), np.nan, dtype="float32")
            if len(sub) >= args.min_points and sub[["lon", "lat", "time_scaled"]].drop_duplicates().shape[0] >= 4:
                try:
                    ok3d = OrdinaryKriging3D(
                        sub["lon"].to_numpy(float),
                        sub["lat"].to_numpy(float),
                        sub["time_scaled"].to_numpy(float),
                        sub["residual_observed_minus_predicted"].to_numpy(float),
                        variogram_model=args.variogram_model,
                        verbose=False,
                        enable_plotting=False,
                    )
                    zhat, ss = ok3d.execute(
                        "points",
                        grid_df["lon"].to_numpy(float),
                        grid_df["lat"].to_numpy(float),
                        np.full(len(grid_df), target_time_scaled, dtype=float),
                    )
                    residual_hat = np.asarray(zhat, dtype="float32")
                    residual_var = np.asarray(ss, dtype="float32")
                except Exception as exc:
                    status = f"kriging_failed:{type(exc).__name__}:{exc}"
            else:
                status = f"insufficient_points:{len(sub)}"

            vg = experimental_variogram(sub)
            if len(vg):
                vg.insert(0, "pollutant", pollutant)
                vg.insert(1, "horizon_days", horizon)
                variogram_rows.append(vg)

            for i, g in enumerate(grid_df.itertuples(index=False)):
                conv = float(pred_map[int(g.rr), int(g.cc)])
                corr = float(residual_hat[i])
                map_rows.append({
                    "sample_index": sample_idx,
                    "end_date": pd.Timestamp(sample["end_date"]).strftime("%Y-%m-%d"),
                    "target_date": target_date.strftime("%Y-%m-%d"),
                    "pollutant": pollutant,
                    "horizon_days": horizon,
                    "grid_id": g.grid_id,
                    "lat": float(g.lat),
                    "lon": float(g.lon),
                    "prediction_convlstm": conv,
                    "kriged_residual": corr,
                    "prediction_corrected": conv + corr,
                    "kriging_variance": float(residual_var[i]) if np.isfinite(residual_var[i]) else np.nan,
                    "kriging_status": status,
                })
            model_rows.append({
                "pollutant": pollutant,
                "horizon_days": horizon,
                "n_residual_points": int(len(sub)),
                "n_unique_stations": int(sub["estacion"].nunique()),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "variogram_model": args.variogram_model,
                "status": status,
            })

    maps = pd.DataFrame(map_rows)
    models = pd.DataFrame(model_rows)
    maps.to_csv(args.out / "st_kriging_corrected_maps_long.csv", index=False)
    models.to_csv(args.out / "st_kriging_model_summary.csv", index=False)
    if variogram_rows:
        pd.concat(variogram_rows, ignore_index=True).to_csv(args.out / "experimental_variograms.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.out / "experimental_variograms.csv", index=False)
    summary = {
        "tensor_zarr": str(args.tensor_zarr),
        "prediction_zarr": str(args.prediction_zarr),
        "residuals": str(args.residuals),
        "maps_csv": str(args.out / "st_kriging_corrected_maps_long.csv"),
        "models_csv": str(args.out / "st_kriging_model_summary.csv"),
        "variograms_csv": str(args.out / "experimental_variograms.csv"),
        "sample_index": sample_idx,
        "end_date": pd.Timestamp(sample["end_date"]).strftime("%Y-%m-%d"),
        "n_map_rows": int(len(maps)),
        "n_maps": int(models.shape[0]),
        "statuses": models.groupby("status").size().to_dict(),
        "note_no2": "NO2 kriging is computed if numerically possible, but spatial validation is not defensible because only one NO2 station exists.",
    }
    (args.out / "summary_st_kriging_residual_maps.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
