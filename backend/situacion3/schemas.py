"""Schemas Pydantic v2 para los endpoints de Situacion 3."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

Pollutant = Literal["NO2", "SO2", "O3"]
HorizonDays = Literal[1, 3, 7]
LayerName = Literal["prediction", "uncertainty", "lisa", "kmeans"]


class GridMeta(BaseModel):
    n_cells: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class MetadataResponse(BaseModel):
    pollutants: List[str]
    horizons_days: List[int]
    base_date: str
    target_dates: Dict[str, str]
    grid: GridMeta
    warnings: List[str]


class MapCell(BaseModel):
    grid_id: str
    lat: float
    lon: float
    prediction: float
    uncertainty_variance: float
    uncertainty_sigma: float
    kmeans_cluster: Optional[int] = None
    lisa_cluster: Optional[str] = None


class MapResponse(BaseModel):
    pollutant: Pollutant
    horizon_days: HorizonDays
    target_date: str
    n_cells: int
    cells: List[MapCell]


class NearestCell(BaseModel):
    grid_id: str
    lat: float
    lon: float
    distance_km: float


class HorizonPoint(BaseModel):
    horizon_days: int
    target_date: str
    prediction: float
    uncertainty_variance: float
    uncertainty_sigma: float
    kmeans_cluster: Optional[int] = None
    lisa_cluster: Optional[str] = None


class PointResponse(BaseModel):
    query: Dict[str, Union[float, str]]
    nearest_cell: NearestCell
    horizons: List[HorizonPoint]


class HorizonRadius(BaseModel):
    horizon_days: int
    target_date: str
    mean_prediction: float
    min_prediction: float
    max_prediction: float
    mean_uncertainty_variance: float
    max_uncertainty_variance: float
    mean_uncertainty_sigma: float
    lisa_counts: Dict[str, int]
    cluster_counts: Dict[str, int]


class RadiusResponse(BaseModel):
    query: Dict[str, Union[float, str]]
    n_cells: int
    horizons: List[HorizonRadius]


class ClusterRow(BaseModel):
    cluster: int
    risk_rank: int
    n_cells: int
    mean_prediction_profile: float
    max_prediction_profile: float
    mean_uncertainty_profile: float
    lat_mean: float
    lon_mean: float


class ClustersResponse(BaseModel):
    k: int
    clusters: List[ClusterRow]


class LisaCell(BaseModel):
    grid_id: str
    lat: float
    lon: float
    lisa_cluster: str
    local_i: float
    p_sim: float
    significant: bool


class LisaResponse(BaseModel):
    pollutant: Pollutant
    horizon_days: HorizonDays
    n_cells: int
    cells: List[LisaCell]


class KpiRow(BaseModel):
    kpi: str
    minimum: Optional[str]
    excellent: Optional[str]
    value: Optional[str]
    status: str
    evidence: Optional[str]


class KpisResponse(BaseModel):
    kpis: List[KpiRow]
    notes: List[str]


class MoranRow(BaseModel):
    pollutant: str
    horizon_days: int
    moran_i: float
    expected_i: float
    p_sim: float
    z_sim: float
    n: int
    status: str


class MoranResponse(BaseModel):
    rows: List[MoranRow]
