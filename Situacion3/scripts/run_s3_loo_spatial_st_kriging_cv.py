from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D


POLLUTANTS = ["SO2", "O3"]
HORIZONS = [1, 3, 7]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float("nan")
    if len(y_true) > 1 and float(np.var(y_true)) > 0:
        r2 = float(1.0 - np.sum(err**2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--residuals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variogram-model", default="linear", choices=["linear", "power", "gaussian", "spherical", "exponential"])
    ap.add_argument("--min-train-points", type=int, default=20)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.residuals)
    df = df[df["pollutant"].isin(POLLUTANTS)].copy()
    df = df.dropna(subset=["lon", "lat", "observed", "predicted_convlstm", "residual_observed_minus_predicted", "estacion"])
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    t0 = df["target_date"].min()
    df["time_days"] = (df["target_date"] - t0).dt.days.astype(float)
    df["time_scaled"] = df["time_days"] / 365.25 * 0.05

    rows = []
    fold_rows = []
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            sub = df[(df["pollutant"] == pollutant) & (df["horizon_days"] == horizon)].copy()
            stations = sorted(s for s in sub["estacion"].dropna().unique() if str(s).strip())
            for station in stations:
                test = sub[sub["estacion"] == station].copy()
                train = sub[sub["estacion"] != station].copy()
                status = "ok"
                pred_resid = np.full(len(test), np.nan, dtype="float64")
                pred_var = np.full(len(test), np.nan, dtype="float64")
                if len(train) < args.min_train_points or train["estacion"].nunique() < 2 or len(test) == 0:
                    status = f"insufficient_train_points:{len(train)}"
                else:
                    try:
                        ok3d = OrdinaryKriging3D(
                            train["lon"].to_numpy(float),
                            train["lat"].to_numpy(float),
                            train["time_scaled"].to_numpy(float),
                            train["residual_observed_minus_predicted"].to_numpy(float),
                            variogram_model=args.variogram_model,
                            verbose=False,
                            enable_plotting=False,
                        )
                        zhat, ss = ok3d.execute(
                            "points",
                            test["lon"].to_numpy(float),
                            test["lat"].to_numpy(float),
                            test["time_scaled"].to_numpy(float),
                        )
                        pred_resid = np.asarray(zhat, dtype="float64")
                        pred_var = np.asarray(ss, dtype="float64")
                    except Exception as exc:
                        status = f"kriging_failed:{type(exc).__name__}:{exc}"
                pred_corrected = test["predicted_convlstm"].to_numpy(float) + np.nan_to_num(pred_resid, nan=0.0)
                fold_metric_base = metrics(test["observed"].to_numpy(float), test["predicted_convlstm"].to_numpy(float))
                fold_metric_corr = metrics(test["observed"].to_numpy(float), pred_corrected)
                sigma = np.sqrt(np.clip(pred_var, 0, None))
                coverage = np.isfinite(pred_var) & (np.abs(test["observed"].to_numpy(float) - pred_corrected) <= 1.96 * sigma)
                coverage_95 = float(coverage.mean()) if len(coverage) else float("nan")
                fold_rows.append({
                    "pollutant": pollutant,
                    "horizon_days": int(horizon),
                    "left_out_station": station,
                    "status": status,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    **{f"base_{k}": v for k, v in fold_metric_base.items()},
                    **{f"corrected_{k}": v for k, v in fold_metric_corr.items()},
                    "coverage_95_sigma_kriging": coverage_95,
                })
                for i, (_, r) in enumerate(test.iterrows()):
                    rows.append({
                        "pollutant": pollutant,
                        "horizon_days": int(horizon),
                        "left_out_station": station,
                        "status": status,
                        "sample_index": int(r["sample_index"]),
                        "target_date": pd.Timestamp(r["target_date"]).strftime("%Y-%m-%d"),
                        "grid_id": r["grid_id"],
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "observed": float(r["observed"]),
                        "predicted_convlstm": float(r["predicted_convlstm"]),
                        "loo_kriged_residual": float(pred_resid[i]) if np.isfinite(pred_resid[i]) else np.nan,
                        "loo_kriging_variance": float(pred_var[i]) if np.isfinite(pred_var[i]) else np.nan,
                        "loo_kriging_sigma": float(np.sqrt(max(pred_var[i], 0))) if np.isfinite(pred_var[i]) else np.nan,
                        "predicted_corrected_loo": float(pred_corrected[i]),
                        "inside_95_sigma_kriging": bool(np.isfinite(pred_var[i]) and abs(float(r["observed"]) - float(pred_corrected[i])) <= 1.96 * np.sqrt(max(pred_var[i], 0))),
                    })

    pred = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    pred.to_csv(args.out / "loo_spatial_predictions_long.csv", index=False)
    folds.to_csv(args.out / "loo_spatial_fold_metrics.csv", index=False)

    metric_rows = []
    for keys, group in pred.groupby(["pollutant", "horizon_days"]):
        pollutant, horizon = keys
        base = metrics(group["observed"].to_numpy(float), group["predicted_convlstm"].to_numpy(float))
        corr = metrics(group["observed"].to_numpy(float), group["predicted_corrected_loo"].to_numpy(float))
        metric_rows.append({"scope": "pollutant_horizon", "pollutant": pollutant, "horizon_days": int(horizon), **{f"base_{k}": v for k, v in base.items()}, **{f"corrected_{k}": v for k, v in corr.items()}, "coverage_95_sigma_kriging": float(group["inside_95_sigma_kriging"].mean())})
    for pollutant, group in pred.groupby("pollutant"):
        base = metrics(group["observed"].to_numpy(float), group["predicted_convlstm"].to_numpy(float))
        corr = metrics(group["observed"].to_numpy(float), group["predicted_corrected_loo"].to_numpy(float))
        metric_rows.append({"scope": "pollutant", "pollutant": pollutant, "horizon_days": "all", **{f"base_{k}": v for k, v in base.items()}, **{f"corrected_{k}": v for k, v in corr.items()}, "coverage_95_sigma_kriging": float(group["inside_95_sigma_kriging"].mean())})
    base = metrics(pred["observed"].to_numpy(float), pred["predicted_convlstm"].to_numpy(float))
    corr = metrics(pred["observed"].to_numpy(float), pred["predicted_corrected_loo"].to_numpy(float))
    metric_rows.append({"scope": "all", "pollutant": "SO2+O3", "horizon_days": "all", **{f"base_{k}": v for k, v in base.items()}, **{f"corrected_{k}": v for k, v in corr.items()}, "coverage_95_sigma_kriging": float(pred["inside_95_sigma_kriging"].mean())})
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out / "loo_spatial_metrics_summary.csv", index=False)

    summary = {
        "residuals": str(args.residuals),
        "predictions_csv": str(args.out / "loo_spatial_predictions_long.csv"),
        "fold_metrics_csv": str(args.out / "loo_spatial_fold_metrics.csv"),
        "metrics_summary_csv": str(args.out / "loo_spatial_metrics_summary.csv"),
        "pollutants": POLLUTANTS,
        "horizons_days": HORIZONS,
        "n_predictions": int(len(pred)),
        "n_folds": int(len(folds)),
        "statuses": folds.groupby("status").size().to_dict() if len(folds) else {},
        "overall_base": base,
        "overall_corrected": corr,
        "overall_coverage_95_sigma_kriging": float(pred["inside_95_sigma_kriging"].mean()) if len(pred) else float("nan"),
        "note_no2": "NO2 is excluded from spatial LOO-CV because current DAGMA/SISAIRE inputs have one NO2 station only.",
    }
    (args.out / "summary_loo_spatial_st_kriging_cv.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
