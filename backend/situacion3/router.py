"""Router FastAPI con los 7 endpoints publicos de Situacion 3."""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .data_loader import (
    BASE_DATE,
    HORIZONS,
    POLLUTANTS,
    TARGET_DATES,
    WARNINGS,
    Situacion3Data,
)
from .schemas import (
    ClusterRow,
    ClustersResponse,
    GridMeta,
    HorizonPoint,
    HorizonRadius,
    KpiRow,
    KpisResponse,
    LisaCell,
    LisaResponse,
    MapCell,
    MapResponse,
    MetadataResponse,
    MoranResponse,
    MoranRow,
    NearestCell,
    PointResponse,
    RadiusResponse,
)
from .service import Situacion3Service

router = APIRouter(prefix="/api/situacion3", tags=["situacion3"])

_DATA: Optional[Situacion3Data] = None
_SERVICE: Optional[Situacion3Service] = None
_BENCHMARK_LATENCY_MS: Optional[float] = None


def _benchmark_radius_latency() -> float:
    """Mide la latencia de un /radius sintetico centrado en Cali para benchmark."""
    assert _SERVICE is not None
    samples = []
    for _ in range(5):
        t = time.perf_counter()
        _SERVICE.radius_summary(3.45, -76.52, 2.0, "O3")
        samples.append((time.perf_counter() - t) * 1000)
    return float(np.median(samples)) if samples else 0.0


def init_data() -> None:
    """Llamar una sola vez al startup."""
    global _DATA, _SERVICE, _BENCHMARK_LATENCY_MS
    _DATA = Situacion3Data.load()
    _SERVICE = Situacion3Service(_DATA)
    try:
        _BENCHMARK_LATENCY_MS = _benchmark_radius_latency()
        print(f"[OK] Benchmark /radius latency: {_BENCHMARK_LATENCY_MS:.2f} ms")
    except Exception as e:
        print(f"[WARN] benchmark fallo: {e}")
        _BENCHMARK_LATENCY_MS = None


def _svc() -> Situacion3Service:
    if _SERVICE is None:
        raise HTTPException(status_code=500, detail="Situacion 3 data not loaded")
    return _SERVICE


def _validate_pollutant(p: str) -> str:
    if p not in POLLUTANTS:
        raise HTTPException(status_code=400, detail=f"pollutant invalido: {p}")
    return p


def _validate_horizon(h: int) -> int:
    if h not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon_days invalido: {h}")
    return h


def _validate_latlon(lat: float, lon: float) -> None:
    g = _DATA.grid_meta
    margin = 0.05
    if not (g["lat_min"] - margin <= lat <= g["lat_max"] + margin):
        raise HTTPException(status_code=400, detail=f"lat fuera de rango: {lat}")
    if not (g["lon_min"] - margin <= lon <= g["lon_max"] + margin):
        raise HTTPException(status_code=400, detail=f"lon fuera de rango: {lon}")


def _safe_value(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == ""):
        return None
    return str(v)


# ---------------- endpoints ----------------

@router.get("/metadata", response_model=MetadataResponse)
def metadata():
    g = _DATA.grid_meta
    return MetadataResponse(
        pollutants=list(POLLUTANTS),
        horizons_days=list(HORIZONS),
        base_date=BASE_DATE,
        target_dates={str(k): v for k, v in TARGET_DATES.items()},
        grid=GridMeta(
            n_cells=g["n_cells"],
            lat_min=g["lat_min"],
            lat_max=g["lat_max"],
            lon_min=g["lon_min"],
            lon_max=g["lon_max"],
        ),
        warnings=WARNINGS,
    )


@router.get("/map", response_model=MapResponse)
def get_map(
    pollutant: str = Query(..., description="NO2 | SO2 | O3"),
    horizon_days: int = Query(..., description="1 | 3 | 7"),
):
    p = _validate_pollutant(pollutant)
    h = _validate_horizon(horizon_days)
    df = _svc().get_map(p, h)
    df = _svc().attach_lisa_and_kmeans(df, p, h)
    cells = [
        MapCell(
            grid_id=row["grid_id"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            prediction=float(row["prediction_corrected"]),
            uncertainty_variance=float(row["kriging_variance"]),
            uncertainty_sigma=float(row["sigma"]),
            kmeans_cluster=None if pd.isna(row.get("kmeans_cluster")) else int(row["kmeans_cluster"]),
            lisa_cluster=None if pd.isna(row.get("lisa_cluster")) else str(row["lisa_cluster"]),
        )
        for _, row in df.iterrows()
    ]
    return MapResponse(
        pollutant=p,
        horizon_days=h,
        target_date=TARGET_DATES[h],
        n_cells=len(cells),
        cells=cells,
    )


@router.get("/point", response_model=PointResponse)
def get_point(
    lat: float = Query(...),
    lon: float = Query(...),
    pollutant: str = Query(...),
):
    _validate_latlon(lat, lon)
    p = _validate_pollutant(pollutant)
    nearest, dist = _svc().nearest_cell(lat, lon, p)
    horizons_df = _svc().horizons_for_cell(nearest["grid_id"], p)
    horizons = [
        HorizonPoint(
            horizon_days=int(r["horizon_days"]),
            target_date=TARGET_DATES[int(r["horizon_days"])],
            prediction=float(r["prediction_corrected"]),
            uncertainty_variance=float(r["kriging_variance"]),
            uncertainty_sigma=float(r["sigma"]),
            kmeans_cluster=None if pd.isna(r.get("kmeans_cluster")) else int(r["kmeans_cluster"]),
            lisa_cluster=None if pd.isna(r.get("lisa_cluster")) else str(r["lisa_cluster"]),
        )
        for _, r in horizons_df.iterrows()
    ]
    return PointResponse(
        query={"lat": lat, "lon": lon, "pollutant": p},
        nearest_cell=NearestCell(
            grid_id=str(nearest["grid_id"]),
            lat=float(nearest["lat"]),
            lon=float(nearest["lon"]),
            distance_km=round(dist, 3),
        ),
        horizons=horizons,
    )


@router.get("/radius", response_model=RadiusResponse)
def get_radius(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(..., gt=0, le=20),
    pollutant: str = Query(...),
):
    t0 = time.perf_counter()
    _validate_latlon(lat, lon)
    p = _validate_pollutant(pollutant)
    n_cells, horizons_data = _svc().radius_summary(lat, lon, radius_km, p)
    if n_cells == 0:
        raise HTTPException(status_code=404, detail="No hay celdas dentro del radio especificado")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return RadiusResponse(
        query={"lat": lat, "lon": lon, "radius_km": radius_km, "pollutant": p, "elapsed_ms": elapsed_ms},
        n_cells=n_cells,
        horizons=[HorizonRadius(**h) for h in horizons_data],
    )


@router.get("/clusters", response_model=ClustersResponse)
def get_clusters():
    rows = _svc().clusters_summary()
    return ClustersResponse(
        k=len(rows),
        clusters=[
            ClusterRow(
                cluster=int(r["cluster"]),
                risk_rank=int(r["risk_rank"]),
                n_cells=int(r["n_cells"]),
                mean_prediction_profile=float(r["mean_prediction_profile"]),
                max_prediction_profile=float(r["max_prediction_profile"]),
                mean_uncertainty_profile=float(r["mean_uncertainty_profile"]),
                lat_mean=float(r["lat_mean"]),
                lon_mean=float(r["lon_mean"]),
            )
            for r in rows
        ],
    )


@router.get("/lisa", response_model=LisaResponse)
def get_lisa(
    pollutant: str = Query(...),
    horizon_days: int = Query(...),
):
    p = _validate_pollutant(pollutant)
    h = _validate_horizon(horizon_days)
    df = _svc().lisa_map(p, h)
    cells = [
        LisaCell(
            grid_id=str(r["grid_id"]),
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            lisa_cluster=str(r["lisa_cluster"]),
            local_i=float(r["local_i"]),
            p_sim=float(r["p_sim"]),
            significant=bool(r["significant"]),
        )
        for _, r in df.iterrows()
    ]
    return LisaResponse(pollutant=p, horizon_days=h, n_cells=len(cells), cells=cells)


@router.get("/kpis", response_model=KpisResponse)
def get_kpis():
    df = _svc().kpis()
    rows = []
    for _, r in df.iterrows():
        kpi_name = str(r["kpi"])
        value = _safe_value(r["value"])
        status = str(r["status"])
        evidence = _safe_value(r["evidence"])

        # Sobreescribir dinamicamente la latencia con la medicion real del backend
        if "latencia" in kpi_name.lower() and _BENCHMARK_LATENCY_MS is not None:
            value = f"{_BENCHMARK_LATENCY_MS:.2f} ms (mediana 5 reps /radius)"
            if _BENCHMARK_LATENCY_MS < 3000:
                status = "excelente"
            elif _BENCHMARK_LATENCY_MS < 8000:
                status = "cumple_minimo"
            else:
                status = "no_cumple"
            evidence = f"Medido al startup con time.perf_counter en /api/situacion3/radius."

        rows.append(KpiRow(
            kpi=kpi_name,
            minimum=_safe_value(r["minimum"]),
            excellent=_safe_value(r["excellent"]),
            value=value,
            status=status,
            evidence=evidence,
        ))
    return KpisResponse(kpis=rows, notes=WARNINGS)


@router.get("/moran", response_model=MoranResponse)
def get_moran():
    df = _svc().moran()
    rows = [MoranRow(**{k: r[k] for k in MoranRow.model_fields.keys()}) for _, r in df.iterrows()]
    return MoranResponse(rows=rows)
