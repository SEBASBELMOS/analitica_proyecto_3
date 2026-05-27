from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(df: pd.DataFrame, pred_col: str = "predicted_convlstm") -> dict[str, float | int]:
    y = df["observed"].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    err = p - y
    r2 = float("nan")
    if len(y) > 1 and np.var(y) > 0:
        r2 = float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "r2": r2, "n": int(len(df))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo-predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--so2-max-valid", type=float, default=60.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.loo_predictions)
    so2 = df[(df["pollutant"] == "SO2") & (df["horizon_days"] == 1)].copy()
    so2["qa_outlier_extreme_so2"] = so2["observed"] > args.so2_max_valid
    outliers = so2[so2["qa_outlier_extreme_so2"]].copy()
    clean = so2[~so2["qa_outlier_extreme_so2"]].copy()
    by_station = so2.groupby("left_out_station").apply(lambda g: pd.Series({
        "n": len(g),
        "n_qa_outliers": int(g["qa_outlier_extreme_so2"].sum()),
        "observed_mean": float(g["observed"].mean()),
        "observed_max": float(g["observed"].max()),
        "rmse_raw": metrics(g)["rmse"],
    })).reset_index()
    result = pd.DataFrame([
        {"evaluation": "raw", "filter": "none", **metrics(so2)},
        {"evaluation": "qa_filtered", "filter": f"observed <= {args.so2_max_valid:g}", **metrics(clean)},
    ])
    result["kpi_minimum"] = "RMSE <= 6 ug/m3"
    result["kpi_excellent"] = "RMSE <= 3 ug/m3"
    result["status"] = result["rmse"].apply(lambda x: "excelente" if x <= 3 else ("cumple_minimo" if x <= 6 else "no_cumple"))
    outliers.to_csv(args.out / "so2_t1_qa_outliers_extreme.csv", index=False)
    by_station.to_csv(args.out / "so2_t1_qa_by_station.csv", index=False)
    result.to_csv(args.out / "so2_t1_qa_filtered_kpi.csv", index=False)
    summary = {
        "loo_predictions": str(args.loo_predictions),
        "so2_max_valid": float(args.so2_max_valid),
        "n_raw": int(len(so2)),
        "n_outliers": int(len(outliers)),
        "raw": result[result["evaluation"] == "raw"].iloc[0].to_dict(),
        "qa_filtered": result[result["evaluation"] == "qa_filtered"].iloc[0].to_dict(),
        "interpretation": "SO2 T+1 fails the raw KPI due to four extreme Flora-2020 observations; after explicit QA sensitivity filter observed<=60, RMSE meets the minimum threshold. This should be reported as sensitivity analysis, not as replacement for the raw KPI unless the data owner accepts the QA exclusion.",
    }
    (args.out / "summary_so2_t1_qa_filtered_kpi.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
