"""Carga del panel maestro DAGMA horario 2020-2024."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "panel"
    / "panel_geovision_cali_station_hourly_2020_2024_model_safe.parquet"
)

# Columnas mas usadas para el frontend (subset para acelerar carga)
CORE_COLUMNS = [
    "fecha_hora", "date", "hour", "municipio", "estacion", "lat", "lon",
    "no2_insitu", "so2_insitu", "o3_insitu",
    "temperature_2m_c", "relative_humidity_pct", "wind_speed_10m",
    "surface_pressure_hpa", "total_precipitation_mm_hourly",
    "aod_047_interp", "aod_055_interp",
]


class PanelData:
    """Singleton del panel maestro cargado al startup."""

    df: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls, path: Path = DATA_PATH) -> "PanelData":
        d = cls()
        if not path.exists():
            print(f"[WARN] Panel parquet no encontrado en {path}")
            d.df = pd.DataFrame()
            return d
        d.df = pd.read_parquet(path, columns=CORE_COLUMNS)
        d.df["estacion"] = d.df["estacion"].astype(str)
        print(f"[OK] Panel maestro cargado: {len(d.df):,} filas, {d.df['estacion'].nunique()} estaciones")
        return d

    def stations(self) -> list[dict]:
        if self.df is None or self.df.empty:
            return []
        agg = self.df.groupby("estacion").agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            municipio=("municipio", "first"),
            n_obs=("estacion", "count"),
            n_no2=("no2_insitu", "count"),
            n_so2=("so2_insitu", "count"),
            n_o3=("o3_insitu", "count"),
        ).reset_index()
        return agg.to_dict(orient="records")

    def latest_observations(self, station: str) -> Optional[dict]:
        """Ultima observacion horaria de una estacion (filtra valores no nulos)."""
        if self.df is None or self.df.empty:
            return None
        sub = self.df[self.df["estacion"] == station].sort_values("fecha_hora", ascending=False)
        if sub.empty:
            return None
        latest = sub.iloc[0]
        return {
            "estacion": str(latest["estacion"]),
            "fecha_hora": str(latest["fecha_hora"]),
            "no2_insitu": _f(latest["no2_insitu"]),
            "so2_insitu": _f(latest["so2_insitu"]),
            "o3_insitu": _f(latest["o3_insitu"]),
            "temperature_2m_c": _f(latest["temperature_2m_c"]),
            "relative_humidity_pct": _f(latest["relative_humidity_pct"]),
            "wind_speed_10m": _f(latest["wind_speed_10m"]),
        }

    def station_summary(self, station: str) -> Optional[dict]:
        """Estadisticas agregadas 2020-2024 de una estacion."""
        if self.df is None or self.df.empty:
            return None
        sub = self.df[self.df["estacion"] == station]
        if sub.empty:
            return None
        out = {"estacion": station, "n_obs": int(len(sub))}
        for gas in ["no2_insitu", "so2_insitu", "o3_insitu"]:
            s = sub[gas].dropna()
            if len(s) > 0:
                out[gas] = {
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "p50": float(s.median()),
                    "p95": float(s.quantile(0.95)),
                    "max": float(s.max()),
                }
            else:
                out[gas] = None
        return out


def _f(x) -> Optional[float]:
    if pd.isna(x):
        return None
    return float(x)
