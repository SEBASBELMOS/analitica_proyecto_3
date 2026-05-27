from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


POLLUTANTS = ["SO2", "O3"]
HORIZONS = [1, 3, 7]


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    keep = np.isfinite(pred) & np.isfinite(y)
    pred = pred[keep]
    y = y[keep]
    if len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    err = pred - y
    rmse = float(math.sqrt(float(np.mean(err**2))))
    mae = float(np.mean(np.abs(err)))
    r2 = float("nan")
    if len(y) > 1 and float(np.var(y)) > 0:
        r2 = float(1.0 - np.sum(err**2) / np.sum((y - np.mean(y)) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y))}


def metric_row(scope: str, pollutant: str, horizon_days: int | str, df: pd.DataFrame) -> dict:
    base = regression_metrics(df["predicted_convlstm"].to_numpy(float), df["observed"].to_numpy(float))
    corr = regression_metrics(df["predicted_corrected_loo"].to_numpy(float), df["observed"].to_numpy(float))
    return {
        "scope": scope,
        "pollutant": pollutant,
        "horizon_days": horizon_days,
        **{f"base_{k}": v for k, v in base.items()},
        **{f"corrected_{k}": v for k, v in corr.items()},
        "coverage_95_sigma_kriging": float(df["inside_95_sigma_kriging"].mean()) if len(df) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo-predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.loo_predictions)
    pred = pred[pred["pollutant"].isin(POLLUTANTS)].copy()

    negative_corrected = pred["predicted_corrected_loo"] < 0
    negative_variance = pred["loo_kriging_variance"] < 0
    pred["used_convlstm_fallback"] = negative_corrected
    pred.loc[negative_corrected, "predicted_corrected_loo"] = pred.loc[negative_corrected, "predicted_convlstm"]
    negative_after_fallback = pred["predicted_corrected_loo"] < 0
    pred["used_zero_floor"] = negative_after_fallback
    pred["predicted_corrected_loo"] = pred["predicted_corrected_loo"].clip(lower=0)
    pred["loo_kriging_variance"] = pred["loo_kriging_variance"].clip(lower=0)
    pred["loo_kriging_sigma"] = np.sqrt(pred["loo_kriging_variance"])
    pred["inside_95_sigma_kriging"] = (
        np.isfinite(pred["loo_kriging_sigma"])
        & (np.abs(pred["observed"] - pred["predicted_corrected_loo"]) <= 1.96 * pred["loo_kriging_sigma"])
    )

    metric_rows = []
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            sub = pred[(pred["pollutant"] == pollutant) & (pred["horizon_days"] == horizon)]
            metric_rows.append(metric_row("pollutant_horizon", pollutant, int(horizon), sub))
    for pollutant in POLLUTANTS:
        sub = pred[pred["pollutant"] == pollutant]
        metric_rows.append(metric_row("pollutant", pollutant, "all", sub))
    metric_rows.append(metric_row("all", "SO2+O3", "all", pred))
    metrics = pd.DataFrame(metric_rows)

    pred.to_csv(args.out / "loo_spatial_predictions_long.csv", index=False)
    metrics.to_csv(args.out / "loo_spatial_metrics_summary.csv", index=False)
    summary = {
        "source_loo_predictions": str(args.loo_predictions),
        "predictions_csv": str(args.out / "loo_spatial_predictions_long.csv"),
        "metrics_csv": str(args.out / "loo_spatial_metrics_summary.csv"),
        "n_predictions": int(len(pred)),
        "n_negative_corrected_input": int(negative_corrected.sum()),
        "n_negative_variance_input": int(negative_variance.sum()),
        "n_convlstm_fallback": int(pred["used_convlstm_fallback"].sum()),
        "n_zero_floor_after_fallback": int(pred["used_zero_floor"].sum()),
        "n_negative_corrected_output": int((pred["predicted_corrected_loo"] < 0).sum()),
        "n_negative_variance_output": int((pred["loo_kriging_variance"] < 0).sum()),
        "n_zero_corrected_output": int((pred["predicted_corrected_loo"] == 0).sum()),
        "overall_coverage_95_sigma_kriging": float(pred["inside_95_sigma_kriging"].mean()) if len(pred) else float("nan"),
    }
    (args.out / "summary_loo_physical_bounds.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
