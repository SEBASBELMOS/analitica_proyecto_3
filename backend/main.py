"""
GeoVision-CLIP Cali - Backend FastAPI
Endpoints: /predict, /validate, /health, /api/situacion3/*
"""
import math
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from situacion3 import router as sit3_router
from situacion3.data_loader import HORIZONS, POLLUTANTS, TARGET_DATES
from situacion3.service import sanitize_physical_bounds
from panel import router as panel_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    sit3_router.init_data()
    print("[OK] Situacion 3 data loaded")
    panel_router.init_data()
    yield


app = FastAPI(
    title="GeoVision-CLIP Cali - API",
    version="2.0.0",
    description=(
        "API de estimacion de contaminacion atmosferica (NO2, SO2, O3) sobre Cali. "
        "Pipeline: CLIP+SAE -> ConvLSTM bidireccional -> ST-Kriging corregido. "
        "Endpoints Situacion 3 sirven artefactos precomputados (no inferencia en request)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Estaciones DAGMA Cali (panel maestro 2020-2024, todas en municipio Cali)
# Extraidas de panel_geovision_cali_station_hourly_2020_2024_model_safe.parquet
# ---------------------------------------------------------------------------
DAGMA_STATIONS = [
    {"id": "BASE_AEREA",  "lat": 3.457128, "lon": -76.502303, "name": "Base Aerea"},
    {"id": "CANAVERALEJO","lat": 3.416366, "lon": -76.549613, "name": "Canaveralejo"},
    {"id": "COMPARTIR",   "lat": 3.428260, "lon": -76.466584, "name": "Compartir"},
    {"id": "ERA_OBRERO",  "lat": 3.457317, "lon": -76.506539, "name": "ERA - Obrero"},
    {"id": "ERMITA",      "lat": 3.455514, "lon": -76.530978, "name": "Ermita"},
    {"id": "FLORA",       "lat": 3.488218, "lon": -76.518058, "name": "Flora"},
    {"id": "PANCE",       "lat": 3.304517, "lon": -76.531252, "name": "Pance"},
    {"id": "TRANSITORIA_NAVARRO", "lat": 3.417183, "lon": -76.494960, "name": "Transitoria - Navarro"},
    {"id": "UNIVALLE",    "lat": 3.377911, "lon": -76.533811, "name": "Univalle"},
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_station(lat: float, lon: float) -> dict:
    return min(DAGMA_STATIONS, key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))


app.include_router(sit3_router.router)
app.include_router(panel_router.router)


# ---------------------------------------------------------------------------
# Schemas Pydantic publicos
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    lat: float = Field(..., ge=3.25, le=3.55, examples=[3.4372],
                       description="Latitud WGS84 (Cali: 3.25-3.55)")
    lon: float = Field(..., ge=-76.60, le=-76.40, examples=[-76.5225],
                       description="Longitud WGS84 (Cali: -76.60 a -76.40)")
    fecha: date = Field(..., examples=["2024-12-17"],
                        description="Fecha objetivo (YYYY-MM-DD). Sit3 cubre 2024-12-17, 19, 23.")
    horizonte: Optional[str] = Field("T+1", description="T+1, T+3 o T+7")


class PredictResponse(BaseModel):
    lat: float
    lon: float
    fecha: str
    horizonte: str
    NO2: float
    SO2: float
    O3: float
    uncertainty_sigma: dict
    modelo: str
    estacion_dagma_cercana: dict
    grid_cell: dict


class ValidateRequest(BaseModel):
    lat: float = Field(..., ge=3.25, le=3.55)
    lon: float = Field(..., ge=-76.60, le=-76.40)


class ValidateResponse(BaseModel):
    lat: float
    lon: float
    estacion_cercana: str
    distancia_km: float
    grid_cell: dict
    predicciones_celda: dict
    observaciones_estacion: Optional[dict] = None
    resumen_estacion: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    n_estaciones_dagma: int
    situacion3_loaded: bool
    n_grid_cells: int
    panel_loaded: bool
    panel_n_rows: int
    version: str


_HORIZONTE_MAP = {"T+1": 1, "T+3": 3, "T+7": 7}


# ---------------------------------------------------------------------------
# Endpoints publicos
# ---------------------------------------------------------------------------
@app.get("/", response_model=dict)
def root():
    return {
        "service": "GeoVision-CLIP Cali API",
        "version": "2.0.0",
        "endpoints_publicos": ["/predict", "/validate", "/health", "/docs"],
        "endpoints_situacion3": [
            "/api/situacion3/metadata",
            "/api/situacion3/map",
            "/api/situacion3/point",
            "/api/situacion3/radius",
            "/api/situacion3/clusters",
            "/api/situacion3/lisa",
            "/api/situacion3/kpis",
            "/api/situacion3/moran",
        ],
        "endpoints_panel": [
            "/api/panel/stations",
            "/api/panel/station/{name}/latest",
            "/api/panel/station/{name}/summary",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    sit3_loaded = sit3_router._DATA is not None
    n_cells = sit3_router._DATA.grid_meta["n_cells"] if sit3_loaded else 0
    panel_loaded = panel_router._PANEL is not None and not panel_router._PANEL.df.empty
    panel_n = int(len(panel_router._PANEL.df)) if panel_loaded else 0
    return HealthResponse(
        status="ok",
        n_estaciones_dagma=len(DAGMA_STATIONS),
        situacion3_loaded=sit3_loaded,
        n_grid_cells=n_cells,
        panel_loaded=panel_loaded,
        panel_n_rows=panel_n,
        version="2.1.0",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Devuelve la prediccion de los 3 gases (NO2, SO2, O3) en la celda Sit3 mas cercana
    al (lat, lon), para el horizonte solicitado. Lee directo de los artefactos
    precomputados de ConvLSTM+ST-Kriging (Manuel/Luis), sin inferencia online.
    """
    h_str = (req.horizonte or "T+1").upper()
    if h_str not in _HORIZONTE_MAP:
        raise HTTPException(status_code=400, detail=f"horizonte invalido: {req.horizonte}")
    h = _HORIZONTE_MAP[h_str]

    svc = sit3_router._svc()
    nearest, dist_km = svc.nearest_cell(req.lat, req.lon, "NO2")
    grid_id = str(nearest["grid_id"])

    preds: dict = {}
    sigmas: dict = {}
    for gas in POLLUTANTS:
        row = svc.data.maps[
            (svc.data.maps["pollutant"] == gas)
            & (svc.data.maps["horizon_days"] == h)
            & (svc.data.maps["grid_id"] == grid_id)
        ]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"sin prediccion para {gas}/{h_str}/{grid_id}")
        r = sanitize_physical_bounds(row).iloc[0]
        preds[gas] = float(r["prediction_corrected"])
        sigmas[gas] = float(r["sigma"])

    station = _nearest_station(req.lat, req.lon)
    return PredictResponse(
        lat=req.lat,
        lon=req.lon,
        fecha=TARGET_DATES[h],
        horizonte=h_str,
        NO2=round(preds["NO2"], 4),
        SO2=round(preds["SO2"], 4),
        O3=round(preds["O3"], 4),
        uncertainty_sigma={k: round(v, 4) for k, v in sigmas.items()},
        modelo="ConvLSTM bidireccional + ST-Kriging corregido (Sit3 precomputado)",
        estacion_dagma_cercana={
            "id": station["id"],
            "name": station["name"],
            "distancia_km": round(_haversine_km(req.lat, req.lon, station["lat"], station["lon"]), 3),
        },
        grid_cell={
            "grid_id": grid_id,
            "lat": float(nearest["lat"]),
            "lon": float(nearest["lon"]),
            "distance_km": round(dist_km, 3),
        },
    )


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest):
    station = _nearest_station(req.lat, req.lon)
    dist_km = _haversine_km(req.lat, req.lon, station["lat"], station["lon"])

    svc = sit3_router._svc()
    nearest, cell_dist = svc.nearest_cell(req.lat, req.lon, "NO2")
    grid_id = str(nearest["grid_id"])

    preds: dict = {}
    for gas in POLLUTANTS:
        gas_data = {}
        for h in HORIZONS:
            row = svc.data.maps[
                (svc.data.maps["pollutant"] == gas)
                & (svc.data.maps["horizon_days"] == h)
                & (svc.data.maps["grid_id"] == grid_id)
            ]
            if not row.empty:
                r = sanitize_physical_bounds(row).iloc[0]
                gas_data[f"T+{h}"] = {
                    "prediction": round(float(r["prediction_corrected"]), 4),
                    "sigma": round(float(r["sigma"]), 4),
                }
        preds[gas] = gas_data

    # Adjuntar observaciones reales del panel maestro (si la estacion esta en el panel)
    obs_reales = None
    resumen = None
    if panel_router._PANEL is not None:
        obs_reales = panel_router._PANEL.latest_observations(station["name"])
        resumen = panel_router._PANEL.station_summary(station["name"])

    return ValidateResponse(
        lat=req.lat,
        lon=req.lon,
        estacion_cercana=station["name"],
        distancia_km=round(dist_km, 3),
        grid_cell={
            "grid_id": grid_id,
            "lat": float(nearest["lat"]),
            "lon": float(nearest["lon"]),
            "distance_km": round(cell_dist, 3),
        },
        predicciones_celda=preds,
        observaciones_estacion=obs_reales,
        resumen_estacion=resumen,
    )
