"""Router /api/panel/* - panel maestro DAGMA horario 2020-2024."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .data_loader import PanelData

router = APIRouter(prefix="/api/panel", tags=["panel"])

_PANEL: Optional[PanelData] = None


def init_data() -> None:
    global _PANEL
    _PANEL = PanelData.load()


def _panel() -> PanelData:
    if _PANEL is None:
        raise HTTPException(status_code=500, detail="Panel data not loaded")
    return _PANEL


@router.get("/stations")
def list_stations():
    """Devuelve las 9 estaciones DAGMA con sus coordenadas y cobertura."""
    return {"n": len(_panel().stations()), "stations": _panel().stations()}


@router.get("/station/{station}/latest")
def station_latest(station: str):
    """Ultima observacion horaria de una estacion."""
    obs = _panel().latest_observations(station)
    if obs is None:
        raise HTTPException(status_code=404, detail=f"Estacion '{station}' no encontrada")
    return obs


@router.get("/station/{station}/summary")
def station_summary(station: str):
    """Estadisticas 2020-2024 de una estacion (mean/p50/p95/max por gas)."""
    s = _panel().station_summary(station)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Estacion '{station}' no encontrada")
    return s
