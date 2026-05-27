from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def status(value, minimum=None, excellent=None, higher_is_better=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "no_evaluable"
    if higher_is_better:
        if excellent is not None and value >= excellent:
            return "excelente"
        if minimum is not None and value >= minimum:
            return "cumple_minimo"
    else:
        if excellent is not None and value <= excellent:
            return "excelente"
        if minimum is not None and value <= minimum:
            return "cumple_minimo"
    return "no_cumple"


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
            slope = float(np.polyfit(x, y, 1)[0])
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
    ap.add_argument("--delivery-root", type=Path, default=Path("/workspace/geovision-cali-hf/Entrega_Final/Situacion3"))
    args = ap.parse_args()

    root = args.delivery_root
    maps_dir = root / "outputs/mapas"
    val_dir = root / "outputs/validacion"
    out_dir = root / "outputs/metricas"
    out_dir.mkdir(parents=True, exist_ok=True)

    loo = pd.read_csv(val_dir / "loo_spatial_metrics_summary.csv")
    moran = pd.read_csv(maps_dir / "moran_global_by_pollutant_horizon.csv")
    variograms = pd.read_csv(maps_dir / "experimental_variograms.csv")
    vg_class = variogram_classification(variograms)
    vg_class.to_csv(out_dir / "variogram_residual_structure_classification.csv", index=False)

    rows = []
    # NO2 is not spatially evaluable.
    rows.append({
        "kpi": "RMSE LOO-CV NO2 T+1",
        "minimum": "<= 8 ug/m3",
        "excellent": "<= 4 ug/m3",
        "value": np.nan,
        "status": "no_evaluable",
        "evidence": "NO2 tiene una sola estacion; LOO-CV espacial no defendible.",
    })
    for pollutant, min_thr, exc_thr in [("SO2", 6.0, 3.0), ("O3", 12.0, 6.0)]:
        row = loo[(loo["scope"] == "pollutant_horizon") & (loo["pollutant"] == pollutant) & (loo["horizon_days"].astype(str) == "1")].iloc[0]
        value = float(row["base_rmse"])
        rows.append({
            "kpi": f"RMSE LOO-CV {pollutant} T+1",
            "minimum": f"<= {min_thr:g} ug/m3",
            "excellent": f"<= {exc_thr:g} ug/m3",
            "value": value,
            "status": status(value, minimum=min_thr, excellent=exc_thr, higher_is_better=False),
            "evidence": "LOO-CV espacial vs DAGMA; se usa ConvLSTM base para KPI predictivo.",
        })
    all_row = loo[loo["scope"] == "all"].iloc[0]
    r2 = float(all_row["base_r2"])
    rows.append({
        "kpi": "R2 LOO-CV promedio contaminantes evaluables",
        "minimum": ">= 0.55",
        "excellent": ">= 0.75",
        "value": r2,
        "status": status(r2, minimum=0.55, excellent=0.75, higher_is_better=True),
        "evidence": "Calculado sobre SO2+O3; NO2 excluido por n=1 estacion.",
    })
    min_i = float(moran["moran_i"].min())
    max_p = float(moran["p_sim"].max())
    rows.append({
        "kpi": "Indice Moran I predicciones",
        "minimum": "> 0.30, p < 0.05",
        "excellent": "> 0.50, p < 0.05",
        "value": f"min I={min_i:.3f}; max p={max_p:.3f}; 9/9 mapas",
        "status": "excelente" if min_i > 0.50 and max_p < 0.05 else ("cumple_minimo" if min_i > 0.30 and max_p < 0.05 else "no_cumple"),
        "evidence": "esda.Moran con permutation test n=999.",
    })
    no_structure = int((vg_class["classification"] == "sin_estructura_nugget_puro").sum())
    rows.append({
        "kpi": "Variograma residuos nugget puro / sin estructura",
        "minimum": "sin estructura",
        "excellent": "sin estructura",
        "value": f"{no_structure}/9 sin estructura",
        "status": "cumple_minimo" if no_structure == 9 else "no_cumple",
        "evidence": "Clasificacion por correlacion distancia-semivarianza y cambio relativo de semivarianza.",
    })
    coverage = float(all_row.get("coverage_95_sigma_kriging", np.nan))
    rows.append({
        "kpi": "Cobertura cinturon 95% sigma kriging",
        "minimum": ">= 92%",
        "excellent": ">= 95%",
        "value": coverage,
        "status": status(coverage, minimum=0.92, excellent=0.95, higher_is_better=True),
        "evidence": "Empirico LOO-CV SO2+O3 usando |obs-pred| <= 1.96*sigma_kriging.",
    })
    degradations = []
    for pollutant in ["SO2", "O3"]:
        r1 = loo[(loo["scope"] == "pollutant_horizon") & (loo["pollutant"] == pollutant) & (loo["horizon_days"].astype(str) == "1")].iloc[0]
        r7 = loo[(loo["scope"] == "pollutant_horizon") & (loo["pollutant"] == pollutant) & (loo["horizon_days"].astype(str) == "7")].iloc[0]
        degradations.append(float(r7["base_rmse"] / r1["base_rmse"] - 1.0))
    max_deg = max(degradations)
    rows.append({
        "kpi": "Degradacion T+1 a T+7 RMSE",
        "minimum": "< 60% aumento",
        "excellent": "< 30% aumento",
        "value": max_deg,
        "status": "excelente" if max_deg < 0.30 else ("cumple_minimo" if max_deg < 0.60 else "no_cumple"),
        "evidence": "Ratio RMSE T+7 / T+1 sobre SO2 y O3.",
    })
    rows.append({
        "kpi": "Latencia inferencia end-to-end",
        "minimum": "< 8 s",
        "excellent": "< 3 s",
        "value": np.nan,
        "status": "pendiente_backend",
        "evidence": "No medible aun porque backend/API no esta implementado; se debe medir con time.perf_counter en endpoint final.",
    })

    kpi = pd.DataFrame(rows)
    kpi.to_csv(out_dir / "kpi_situacion3_summary.csv", index=False)
    summary = {
        "kpi_csv": str(out_dir / "kpi_situacion3_summary.csv"),
        "variogram_classification_csv": str(out_dir / "variogram_residual_structure_classification.csv"),
        "status_counts": kpi["status"].value_counts().to_dict(),
    }
    (out_dir / "summary_kpi_situacion3.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
