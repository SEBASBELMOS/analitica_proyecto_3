"""Logica de negocio sobre los DataFrames de Situacion 3."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import HORIZONS, POLLUTANTS, TARGET_DATES, Situacion3Data


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def haversine_km_vec(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1 = np.radians(lat)
    p2 = np.radians(lats)
    dp = np.radians(lats - lat)
    dl = np.radians(lons - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def sanitize_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prediction_corrected" in out.columns:
        negative = out["prediction_corrected"] < 0
        if "prediction_convlstm" in out.columns:
            out.loc[negative, "prediction_corrected"] = out.loc[negative, "prediction_convlstm"]
        out["prediction_corrected"] = out["prediction_corrected"].clip(lower=0)
    if "kriging_variance" in out.columns:
        out["kriging_variance"] = out["kriging_variance"].clip(lower=0)
    if "sigma" in out.columns:
        out["sigma"] = out["sigma"].clip(lower=0)
    return out


class Situacion3Service:
    def __init__(self, data: Situacion3Data) -> None:
        self.data = data

    # --------- map ---------
    def get_map(self, pollutant: str, horizon_days: int) -> pd.DataFrame:
        m = self.data.maps
        df = m[(m["pollutant"] == pollutant) & (m["horizon_days"] == horizon_days)]
        return sanitize_physical_bounds(df)

    def attach_lisa_and_kmeans(self, df: pd.DataFrame, pollutant: str, horizon_days: int) -> pd.DataFrame:
        lisa = self.data.lisa
        lisa_sub = lisa[(lisa["pollutant"] == pollutant) & (lisa["horizon_days"] == horizon_days)][
            ["grid_id", "lisa_cluster"]
        ]
        km = self.data.kmeans_by_grid[["grid_id", "cluster"]].rename(columns={"cluster": "kmeans_cluster"})
        out = df.merge(lisa_sub, on="grid_id", how="left").merge(km, on="grid_id", how="left")
        return sanitize_physical_bounds(out)

    # --------- point ---------
    def nearest_cell(self, lat: float, lon: float, pollutant: str) -> Tuple[pd.Series, float]:
        m = self.data.maps
        sub = m[(m["pollutant"] == pollutant) & (m["horizon_days"] == HORIZONS[0])].reset_index(drop=True)
        d = haversine_km_vec(lat, lon, sub["lat"].values, sub["lon"].values)
        i = int(d.argmin())
        return sub.iloc[i], float(d[i])

    def horizons_for_cell(self, grid_id: str, pollutant: str) -> pd.DataFrame:
        m = self.data.maps
        df = m[(m["pollutant"] == pollutant) & (m["grid_id"] == grid_id)].sort_values("horizon_days")
        df = sanitize_physical_bounds(df)
        lisa_sub = self.data.lisa[self.data.lisa["pollutant"] == pollutant][
            ["grid_id", "horizon_days", "lisa_cluster"]
        ]
        km = self.data.kmeans_by_grid[["grid_id", "cluster"]].rename(columns={"cluster": "kmeans_cluster"})
        df = df.merge(lisa_sub, on=["grid_id", "horizon_days"], how="left").merge(km, on="grid_id", how="left")
        return df

    # --------- radius ---------
    def radius_summary(self, lat: float, lon: float, radius_km: float, pollutant: str) -> Tuple[int, List[dict]]:
        m = self.data.maps
        base = m[(m["pollutant"] == pollutant) & (m["horizon_days"] == HORIZONS[0])].reset_index(drop=True)
        d = haversine_km_vec(lat, lon, base["lat"].values, base["lon"].values)
        mask = d <= radius_km
        in_grid_ids = set(base.loc[mask, "grid_id"].tolist())
        n_cells = len(in_grid_ids)
        if n_cells == 0:
            return 0, []

        lisa_sub = self.data.lisa[self.data.lisa["pollutant"] == pollutant]
        km = self.data.kmeans_by_grid.set_index("grid_id")["cluster"].to_dict()

        out = []
        for h in HORIZONS:
            sub = m[(m["pollutant"] == pollutant) & (m["horizon_days"] == h) & (m["grid_id"].isin(in_grid_ids))]
            if sub.empty:
                continue
            sub = sanitize_physical_bounds(sub)
            lisa_h = lisa_sub[(lisa_sub["horizon_days"] == h) & (lisa_sub["grid_id"].isin(in_grid_ids))]
            lisa_counts = Counter(lisa_h["lisa_cluster"].fillna("not_significant").tolist())
            cluster_counts = Counter(int(km[g]) for g in in_grid_ids if g in km)

            out.append({
                "horizon_days": int(h),
                "target_date": TARGET_DATES[h],
                "mean_prediction": float(sub["prediction_corrected"].mean()),
                "min_prediction": float(sub["prediction_corrected"].min()),
                "max_prediction": float(sub["prediction_corrected"].max()),
                "mean_uncertainty_variance": float(sub["kriging_variance"].mean()),
                "max_uncertainty_variance": float(sub["kriging_variance"].max()),
                "mean_uncertainty_sigma": float(sub["sigma"].mean()),
                "lisa_counts": {str(k): int(v) for k, v in lisa_counts.items()},
                "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
            })
        return n_cells, out

    # --------- clusters ---------
    def clusters_summary(self) -> List[dict]:
        df = self.data.kmeans_summary.sort_values("risk_rank")
        return df.to_dict(orient="records")

    # --------- lisa ---------
    def lisa_map(self, pollutant: str, horizon_days: int) -> pd.DataFrame:
        l = self.data.lisa
        return l[(l["pollutant"] == pollutant) & (l["horizon_days"] == horizon_days)].copy()

    # --------- kpis ---------
    def kpis(self) -> pd.DataFrame:
        return self.data.kpis.copy()

    # --------- moran ---------
    def moran(self) -> pd.DataFrame:
        return self.data.moran.copy()


def validate_pollutant(p: str) -> str:
    if p not in POLLUTANTS:
        raise ValueError(f"pollutant invalido: {p}; debe ser uno de {POLLUTANTS}")
    return p


def validate_horizon(h: int) -> int:
    if h not in HORIZONS:
        raise ValueError(f"horizon_days invalido: {h}; debe ser uno de {HORIZONS}")
    return h
