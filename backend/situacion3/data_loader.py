"""Carga unica de los CSVs de Situacion 3 al startup del backend."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

POLLUTANTS = ("NO2", "SO2", "O3")
HORIZONS = (1, 3, 7)
BASE_DATE = "2024-12-16"
TARGET_DATES = {1: "2024-12-17", 3: "2024-12-19", 7: "2024-12-23"}

WARNINGS = [
    "NO2 tiene una sola estacion observacional; LOO-CV espacial no es defendible.",
    "KPIs finales usan ConvLSTM covariado + ST-Kriging + postproceso fisico consistente con mapas/API.",
    "Variograma residual no es nugget puro globalmente; persiste estructura residual en 5/6 combinaciones SO2/O3 LOO.",
    "Si la correccion ST-Kriging produce concentraciones negativas, la API usa la prediccion ConvLSTM base.",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "situacion3"


class Situacion3Data:
    """Contenedor singleton de los DataFrames de Situacion 3."""

    maps: pd.DataFrame
    lisa: pd.DataFrame
    kmeans_by_grid: pd.DataFrame
    kmeans_summary: pd.DataFrame
    moran: pd.DataFrame
    kpis: pd.DataFrame
    grid_meta: Dict[str, Any]

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "Situacion3Data":
        d = cls()
        d.maps = pd.read_csv(data_dir / "st_kriging_corrected_maps_long.csv")
        d.lisa = pd.read_csv(data_dir / "lisa_local_clusters_long.csv")
        d.kmeans_by_grid = pd.read_csv(data_dir / "kmeans_critical_profiles_by_grid.csv")
        d.kmeans_summary = pd.read_csv(data_dir / "kmeans_cluster_summary.csv")
        d.moran = pd.read_csv(data_dir / "moran_global_by_pollutant_horizon.csv")
        d.kpis = pd.read_csv(data_dir / "kpi_situacion3_summary.csv")

        d._sanitize_physical_bounds()
        d.maps["sigma"] = d.maps["kriging_variance"].clip(lower=0).pow(0.5)
        d.lisa["sigma"] = d.lisa["kriging_variance"].clip(lower=0).pow(0.5)

        d._validate_invariants()
        d.grid_meta = d._compute_grid_meta()
        return d

    def _sanitize_physical_bounds(self) -> None:
        """Apply physical bounds to values served by the API.

        Kriging residual correction can produce small negative concentrations near
        zero. In those cells, falling back to the positive ConvLSTM baseline is
        more informative than forcing a hard zero. PyKrige may also emit tiny
        negative variances from floating point precision. The model artifacts are
        kept unchanged on disk; API responses expose physically valid values.
        """
        for df in (self.maps, self.lisa):
            if "prediction_corrected" in df.columns:
                negative = df["prediction_corrected"] < 0
                if "prediction_convlstm" in df.columns:
                    df.loc[negative, "prediction_corrected"] = df.loc[negative, "prediction_convlstm"]
                df["prediction_corrected"] = df["prediction_corrected"].clip(lower=0)
            if "kriging_variance" in df.columns:
                df["kriging_variance"] = df["kriging_variance"].clip(lower=0)

    def _validate_invariants(self) -> None:
        assert set(self.maps["pollutant"].unique()) == set(POLLUTANTS), \
            f"pollutants mismatch: {self.maps['pollutant'].unique()}"
        assert set(self.maps["horizon_days"].unique()) == set(HORIZONS), \
            f"horizons mismatch: {self.maps['horizon_days'].unique()}"
        assert self.maps["prediction_corrected"].notna().all(), "prediction_corrected has NaN"
        assert self.maps["kriging_variance"].notna().all(), "kriging_variance has NaN"
        assert (self.maps["prediction_corrected"] >= 0).all(), "prediction_corrected has negative values"
        assert (self.maps["kriging_variance"] >= 0).all(), "kriging_variance has negative values"
        n_combos = self.maps.groupby(["pollutant", "horizon_days"]).ngroups
        assert n_combos == 9, f"expected 9 pollutant/horizon combos, got {n_combos}"

    def _compute_grid_meta(self) -> Dict[str, Any]:
        m = self.maps
        cells_per_combo = m.groupby(["pollutant", "horizon_days"]).size().unique()
        return {
            "n_cells": int(cells_per_combo[0]),
            "lat_min": float(m["lat"].min()),
            "lat_max": float(m["lat"].max()),
            "lon_min": float(m["lon"].min()),
            "lon_max": float(m["lon"].max()),
            "n_pollutants": len(POLLUTANTS),
            "n_horizons": len(HORIZONS),
            "n_maps": 9,
        }
