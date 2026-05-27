"""
Tests pytest del backend FastAPI GeoVision-CLIP Cali v2.
Cubre endpoints publicos (/predict, /validate, /health) y los 8 endpoints
de Situacion 3 (/api/situacion3/*).

Ejecutar: pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from main import DAGMA_STATIONS, _haversine_km, _nearest_station, app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Endpoints publicos basicos
# ---------------------------------------------------------------------------
def test_root_responde_metadata(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "GeoVision-CLIP Cali API"
    assert "/predict" in body["endpoints_publicos"]
    assert "/api/situacion3/metadata" in body["endpoints_situacion3"]


def test_health_responde_campos_esperados(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_estaciones_dagma"] == 9
    assert body["situacion3_loaded"] is True
    assert body["n_grid_cells"] == 1920
    assert body["panel_loaded"] is True
    assert body["panel_n_rows"] > 200000
    assert body["version"] == "2.1.0"


# ---------------------------------------------------------------------------
# /predict - delega a Situacion 3 (no mock)
# ---------------------------------------------------------------------------
def test_predict_coords_centro_cali_devuelve_sit3(client):
    payload = {"lat": 3.4372, "lon": -76.5225, "fecha": "2024-12-17", "horizonte": "T+1"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    for field in ["NO2", "SO2", "O3", "uncertainty_sigma", "modelo",
                  "estacion_dagma_cercana", "grid_cell", "fecha", "horizonte"]:
        assert field in body, f"Falta campo {field}"
    assert body["NO2"] > 0 and body["SO2"] > 0 and body["O3"] > 0
    for gas in ["NO2", "SO2", "O3"]:
        assert body["uncertainty_sigma"][gas] > 0
    assert "ConvLSTM" in body["modelo"]
    assert body["horizonte"] == "T+1"
    assert body["fecha"] == "2024-12-17"


def test_predict_horizonte_t3_devuelve_target_date_correcta(client):
    payload = {"lat": 3.45, "lon": -76.52, "fecha": "2024-12-19", "horizonte": "T+3"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["fecha"] == "2024-12-19"


def test_predict_lat_fuera_de_bounds_rechazado(client):
    payload = {"lat": 10.0, "lon": -76.5, "fecha": "2024-12-17"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_deterministico_misma_entrada(client):
    payload = {"lat": 3.4372, "lon": -76.5225, "fecha": "2024-12-17", "horizonte": "T+1"}
    r1 = client.post("/predict", json=payload)
    r2 = client.post("/predict", json=payload)
    assert r1.json()["NO2"] == r2.json()["NO2"]


# ---------------------------------------------------------------------------
# /validate
# ---------------------------------------------------------------------------
def test_validate_responde_estacion_cercana_y_3_horizontes(client):
    payload = {"lat": 3.30, "lon": -76.53}  # cerca de Pance (3.304, -76.531)
    r = client.post("/validate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["estacion_cercana"] == "Pance"
    assert body["distancia_km"] < 5
    for gas in ["NO2", "SO2", "O3"]:
        assert gas in body["predicciones_celda"]
        for h in ["T+1", "T+3", "T+7"]:
            assert h in body["predicciones_celda"][gas]
    # Panel: observaciones reales presentes
    assert body["resumen_estacion"] is not None
    assert body["resumen_estacion"]["estacion"] == "Pance"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def test_haversine_distancia_conocida():
    # Pance (3.3045, -76.5313) a Flora (3.4882, -76.5181) ~ 21 km
    d = _haversine_km(3.3045, -76.5313, 3.4882, -76.5181)
    assert 18 < d < 25


def test_dagma_tiene_9_estaciones():
    assert len(DAGMA_STATIONS) == 9
    nombres = {s["name"] for s in DAGMA_STATIONS}
    # las 9 reales del panel maestro (sin Yumbo)
    esperadas = {"Base Aerea", "Canaveralejo", "Compartir", "ERA - Obrero",
                 "Ermita", "Flora", "Pance", "Transitoria - Navarro", "Univalle"}
    assert nombres == esperadas, f"diferencia: {nombres ^ esperadas}"
    for s in DAGMA_STATIONS:
        assert "id" in s and "lat" in s and "lon" in s and "name" in s


# ---------------------------------------------------------------------------
# Situacion 3 - los 8 endpoints
# ---------------------------------------------------------------------------
def test_sit3_metadata(client):
    r = client.get("/api/situacion3/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["pollutants"] == ["NO2", "SO2", "O3"]
    assert body["horizons_days"] == [1, 3, 7]
    assert body["base_date"] == "2024-12-16"
    assert body["target_dates"]["1"] == "2024-12-17"
    assert body["grid"]["n_cells"] == 1920
    assert len(body["warnings"]) >= 3


def test_sit3_map_o3_t1_devuelve_1920_celdas(client):
    r = client.get("/api/situacion3/map", params={"pollutant": "O3", "horizon_days": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_cells"] == 1920
    assert body["target_date"] == "2024-12-17"
    cell = body["cells"][0]
    for k in ["grid_id", "lat", "lon", "prediction", "uncertainty_variance", "uncertainty_sigma"]:
        assert k in cell


def test_sit3_map_no_expone_valores_fisicos_negativos(client):
    for gas in ["NO2", "SO2", "O3"]:
        for horizon in [1, 3, 7]:
            r = client.get("/api/situacion3/map", params={"pollutant": gas, "horizon_days": horizon})
            assert r.status_code == 200
            cells = r.json()["cells"]
            assert all(c["prediction"] >= 0 for c in cells)
            assert all(c["uncertainty_variance"] >= 0 for c in cells)
            assert all(c["uncertainty_sigma"] >= 0 for c in cells)


def test_sit3_map_pollutant_invalido_rechaza(client):
    r = client.get("/api/situacion3/map", params={"pollutant": "CO", "horizon_days": 1})
    assert r.status_code == 400


def test_sit3_map_horizon_invalido_rechaza(client):
    r = client.get("/api/situacion3/map", params={"pollutant": "O3", "horizon_days": 5})
    assert r.status_code == 400


def test_sit3_point_devuelve_3_horizontes(client):
    r = client.get("/api/situacion3/point", params={"lat": 3.45, "lon": -76.53, "pollutant": "O3"})
    assert r.status_code == 200
    body = r.json()
    assert body["nearest_cell"]["distance_km"] < 5
    assert len(body["horizons"]) == 3
    assert {h["horizon_days"] for h in body["horizons"]} == {1, 3, 7}


def test_sit3_radius_2km_devuelve_celdas(client):
    r = client.get("/api/situacion3/radius",
                   params={"lat": 3.45, "lon": -76.53, "radius_km": 2, "pollutant": "O3"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_cells"] > 0
    assert len(body["horizons"]) == 3
    h0 = body["horizons"][0]
    assert h0["min_prediction"] <= h0["mean_prediction"] <= h0["max_prediction"]
    assert "lisa_counts" in h0 and "cluster_counts" in h0


def test_sit3_radius_sin_celdas_404(client):
    r = client.get("/api/situacion3/radius",
                   params={"lat": 3.40, "lon": -76.50, "radius_km": 0.001, "pollutant": "O3"})
    assert r.status_code == 404


def test_sit3_radius_radio_invalido_rechaza(client):
    r = client.get("/api/situacion3/radius",
                   params={"lat": 3.45, "lon": -76.53, "radius_km": 0, "pollutant": "O3"})
    assert r.status_code == 422


def test_sit3_clusters_devuelve_5(client):
    r = client.get("/api/situacion3/clusters")
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 5
    risk_ranks = {c["risk_rank"] for c in body["clusters"]}
    assert risk_ranks == {1, 2, 3, 4, 5}


def test_sit3_lisa_o3_t1(client):
    r = client.get("/api/situacion3/lisa", params={"pollutant": "O3", "horizon_days": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_cells"] == 1920
    valid = {"high_high", "low_low", "high_low", "low_high", "not_significant"}
    for cell in body["cells"][:50]:
        assert cell["lisa_cluster"] in valid


def test_sit3_kpis_devuelve_lista(client):
    r = client.get("/api/situacion3/kpis")
    assert r.status_code == 200
    body = r.json()
    assert len(body["kpis"]) > 0
    assert len(body["notes"]) >= 3
    valid_status = {"excelente", "cumple_minimo", "no_cumple", "no_evaluable", "pendiente_backend", "pendiente_medicion_formal"}
    for k in body["kpis"]:
        assert k["status"] in valid_status


def test_sit3_moran_9_filas(client):
    r = client.get("/api/situacion3/moran")
    assert r.status_code == 200
    body = r.json()
    assert len(body["rows"]) == 9
    for row in body["rows"]:
        assert row["moran_i"] > 0.30
        assert row["p_sim"] < 0.05


# ---------------------------------------------------------------------------
# Panel maestro - los 3 endpoints
# ---------------------------------------------------------------------------
def test_panel_stations_devuelve_9_sin_yumbo(client):
    r = client.get("/api/panel/stations")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 9
    nombres = {s["estacion"] for s in body["stations"]}
    assert "Pance" in nombres
    assert "Univalle" in nombres
    assert "Flora" in nombres
    # Yumbo NO debe estar en el panel
    assert not any("yumbo" in s.lower() for s in nombres)
    assert not any("acopi" in s.lower() for s in nombres)


def test_panel_station_latest(client):
    r = client.get("/api/panel/station/Pance/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["estacion"] == "Pance"
    assert "fecha_hora" in body


def test_panel_station_summary_pance(client):
    r = client.get("/api/panel/station/Pance/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["estacion"] == "Pance"
    assert body["n_obs"] > 30000
    # NO2 debe ser None en Pance (solo Univalle tiene NO2)
    assert body["no2_insitu"] is None
    # O3 si esta en Pance
    assert body["o3_insitu"] is not None
    assert body["o3_insitu"]["n"] > 30000


def test_panel_station_no_existe_404(client):
    r = client.get("/api/panel/station/InexistenteXYZ/summary")
    assert r.status_code == 404
