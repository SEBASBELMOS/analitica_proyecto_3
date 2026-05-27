from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def experimental_variogram(df: pd.DataFrame, residual_col: str, max_pairs: int = 20000, n_bins: int = 12) -> pd.DataFrame:
    if len(df) < 3:
        return pd.DataFrame(columns=["bin", "distance_min", "distance_max", "distance_mean", "semivariance", "n_pairs"])
    rng = np.random.default_rng(42)
    coords = df[["lon", "lat", "time_scaled"]].to_numpy("float64")
    vals = df[residual_col].to_numpy("float64")
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


def variogram_classification(variograms: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pollutant, horizon), g in variograms.groupby(["pollutant", "horizon_days"]):
        g = g.sort_values("distance_mean")
        x = g["distance_mean"].to_numpy(float)
        y = g["semivariance"].to_numpy(float)
        if len(g) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            corr = np.nan
            rel_slope = np.nan
            cls = "no_evaluable"
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
            rel_slope = float((y[-1] - y[0]) / max(abs(np.nanmean(y)), 1e-9))
            cls = "sin_estructura_nugget_puro" if abs(corr) < 0.30 and abs(rel_slope) < 0.30 else "estructura_residual_detectable"
        rows.append({
            "pollutant": pollutant,
            "horizon_days": int(horizon),
            "distance_semivariance_corr": corr,
            "relative_semivariance_change": rel_slope,
            "classification": cls,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo-predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.loo_predictions)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    t0 = df["target_date"].min()
    df["time_days"] = (df["target_date"] - t0).dt.days.astype(float)
    df["time_scaled"] = df["time_days"] / 365.25 * 0.05
    df["residual_observed_minus_corrected"] = df["observed"] - df["predicted_corrected_loo"]
    df = df.dropna(subset=["lon", "lat", "time_scaled", "residual_observed_minus_corrected"])

    rows = []
    for (pollutant, horizon), sub in df.groupby(["pollutant", "horizon_days"]):
        vg = experimental_variogram(sub, "residual_observed_minus_corrected")
        if len(vg):
            vg.insert(0, "pollutant", pollutant)
            vg.insert(1, "horizon_days", int(horizon))
            rows.append(vg)
    variograms = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    classes = variogram_classification(variograms) if len(variograms) else pd.DataFrame()
    variograms.to_csv(args.out / "experimental_variograms_corrected_loo.csv", index=False)
    classes.to_csv(args.out / "variogram_residual_structure_classification.csv", index=False)
    summary = {
        "loo_predictions": str(args.loo_predictions),
        "variograms_csv": str(args.out / "experimental_variograms_corrected_loo.csv"),
        "classification_csv": str(args.out / "variogram_residual_structure_classification.csv"),
        "n_variogram_rows": int(len(variograms)),
        "classification_counts": classes["classification"].value_counts().to_dict() if len(classes) else {},
    }
    (args.out / "summary_loo_residual_variograms.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
