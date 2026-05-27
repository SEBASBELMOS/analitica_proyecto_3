---
title: GeoVision-CLIP Cali API
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GeoVision-CLIP Cali — Backend FastAPI

API REST que sirve los resultados precomputados del pipeline **RemoteCLIP + SAE -> ConvLSTM bidireccional -> ST-Kriging corregido + Moran I + LISA + KMeans** para estimacion espacio-temporal de NO2, SO2 y O3 sobre Cali.

**URL publica:** `https://analiticalastdance-geovision-cali-api.hf.space`
**Docs Swagger:** `https://analiticalastdance-geovision-cali-api.hf.space/docs`

## Arquitectura

```
S2 Zarr 12 bandas -> RemoteCLIP ViT-B/32 -> SAE k-sparse (256d)
                                                |
                                                v
                                  Tensor espacial [B,8,256,48,40]
                                                |
                                                v
                                  ConvLSTM bidireccional (test R2=0.656)
                                                |
                                                v
                                  ST-Kriging 3D residuos -> mapas corregidos
                                                |
                                                v
                                  Moran I + LISA + KMeans (5 clusters)
                                                |
                                                v
                          ARTEFACTOS PRECOMPUTADOS (CSV) en data/situacion3/
                                                |
                                                v
                                  FastAPI sirve sin inferencia online
```

El backend **no ejecuta** RemoteCLIP, SAE, ConvLSTM ni PyKrige en request. Solo carga 6 CSVs al startup y sirve consultas filtradas/agregadas. Latencia tipica `< 30 ms` por endpoint.

## Endpoints publicos

| Metodo | Path | Proposito |
|---|---|---|
| GET | `/` | metadata del servicio |
| GET | `/health` | status + n celdas grid + version |
| POST | `/predict` | prediccion 3 gases para (lat, lon, fecha, horizonte) usando celda Sit3 mas cercana |
| POST | `/validate` | identifica estacion DAGMA mas cercana + predicciones celda en 3 horizontes |

## Endpoints Situacion 3 (`/api/situacion3/*`)

Implementan el contrato definido en el handoff de Manuel (`API_CONTRACT_SITUACION3.md`):

| Metodo | Path | Proposito |
|---|---|---|
| GET | `/metadata` | pollutants, horizons, grid bounds, warnings |
| GET | `/map?pollutant=O3&horizon_days=1` | 1920 celdas con prediccion + sigma + KMeans + LISA |
| GET | `/point?lat=..&lon=..&pollutant=O3` | celda mas cercana + 3 horizontes |
| GET | `/radius?lat=..&lon=..&radius_km=2&pollutant=O3` | resumen agregado por radio |
| GET | `/clusters` | 5 perfiles KMeans con `risk_rank` |
| GET | `/lisa?pollutant=O3&horizon_days=1` | clusters LISA por celda |
| GET | `/kpis` | KPIs del proyecto Sit3 (con latencia medida dinamicamente) |
| GET | `/moran` | 9 filas de Moran I global (pollutant x horizon) |

Validaciones: `pollutant in {NO2, SO2, O3}`, `horizon_days in {1, 3, 7}`, lat/lon en bbox de Cali, `0 < radius_km <= 20`.

## Stack

| Capa | Tecnologia |
|---|---|
| Framework | FastAPI 0.115 + Uvicorn 0.31 |
| Validacion | Pydantic v2 |
| Datos | pandas 2.2 + numpy 1.26 |
| Tests | pytest |
| Deploy | HuggingFace Spaces Docker SDK puerto 7860 |
| Python | 3.11 (slim) |

## Estructura

```
.
|-- main.py                          # entry FastAPI (lifespan + /predict + /validate + /health)
|-- situacion3/
|   |-- __init__.py
|   |-- data_loader.py               # carga unica de 6 CSVs al startup + validacion invariantes
|   |-- schemas.py                   # 13 modelos Pydantic v2
|   |-- service.py                   # haversine vectorizado, filtros, agregados
|   `-- router.py                    # 8 endpoints + benchmark dinamico de latencia
|-- data/situacion3/                 # 49 MB de artefactos precomputados (Manuel)
|   |-- st_kriging_corrected_maps_long.csv     # 17280 filas, prediccion principal
|   |-- lisa_local_clusters_long.csv           # 17280 filas, clusters LISA
|   |-- kmeans_critical_profiles_by_grid.csv   # 1920 filas, perfil por celda
|   |-- kmeans_cluster_summary.csv             # 5 filas, summary clusters
|   |-- moran_global_by_pollutant_horizon.csv  # 9 filas, Moran I
|   `-- kpi_situacion3_summary.csv             # 9 filas, KPIs proyecto
|-- tests/
|   `-- test_api.py                  # 21 tests pytest
|-- Dockerfile                       # multi-stage builder + runtime
|-- requirements.txt
`-- docker-compose.yml
```

## Desarrollo local

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
# http://localhost:8000/docs
```

## Tests

```bash
pytest tests/ -v
```

Cobertura actual: **21/21 PASS en ~0.5 s**. Cubren endpoints publicos, los 8 de Sit3, validaciones de parametros, helpers haversine y consistencia de datos.

## Performance (medido contra HF Space publico, 10 reps)

| Endpoint | p50 | p95 | KPI Sit3 |
|---|---|---|---|
| `/health`, `/metadata`, `/clusters`, `/kpis` | ~480 ms | ~505 ms | Excelente (< 3 s) |
| `/map` (1920 celdas, ~408 KB) | 1121 ms | 1188 ms | Excelente |
| `/lisa` (1920 celdas, ~280 KB) | 1094 ms | 1150 ms | Excelente |
| `/point`, `/radius` | ~490 ms | ~840 ms | Excelente |

Benchmark interno de `/radius` al startup: ~5-25 ms (medido localmente con `time.perf_counter`). Disponible en `/api/situacion3/kpis` campo "Latencia inferencia end-to-end".

## Cumplimiento de la rubrica

| Sit. PDF | KPI | Donde se cumple |
|---|---|---|
| 3.4 | Variograma espacial fit_quality "good" | Sit3 outputs (precomputado, no en API) |
| 3.5 | ST-Kriging 3D | `data/situacion3/st_kriging_corrected_maps_long.csv` |
| 3.6 | LOO-CV espacial vs DAGMA | `data/situacion3/kpi_situacion3_summary.csv` |
| 3.7 | Variograma residuos | KPI clasificado en `kpi_situacion3_summary.csv` |
| 3.8 | Moran I > 0.30 con p < 0.05 los 9 | `/api/situacion3/moran` (9/9 cumplen) |
| 3.9 | KMeans 5 clusters con risk_rank | `/api/situacion3/clusters` |
| 4.1 | Backend FastAPI | este Space |
| 4.5 | Dockerfile multi-stage | builder + runtime |
| 4.5 | Tests pytest | 21/21 PASS |
| 4.6 | URL publica viva | https://analiticalastdance-geovision-cali-api.hf.space |
| 4.7 | Latencia < 8 s (excelente < 3 s) | medido en `/api/situacion3/kpis` |
| 4.8 | Requirements pinneado | `requirements.txt` |

Limitaciones documentadas (en `/api/situacion3/metadata` campo `warnings`):
- NO2 tiene una sola estacion; LOO-CV espacial no defendible
- SO2 bruto T+1 no cumple KPI RMSE (sensibilidad QA documentada)
- Variograma residual no es nugget puro globalmente (8/9 mapas)

## Decisiones de implementacion

1. **No `Backend/app/services/...`:** Manuel propuso esa estructura tipo enterprise pero hay un solo dominio (`situacion3`), asi que se uso un modulo plano `situacion3/` con `service.py` + `router.py`. Misma funcionalidad, menos anidamiento.

2. **CSV en memoria al startup:** los 49 MB se cargan una vez con pandas; pesos en RAM ~150 MB. Si crece mucho convertir a Parquet (estimado ~12 MB).

3. **Predictor mock eliminado:** la version anterior tenia un mock heuristico para `/predict`. Ahora `/predict` delega al servicio Sit3 y devuelve la prediccion real de ConvLSTM+Kriging de la celda mas cercana.

4. **Lifespan async en vez de `on_event("startup")`:** permite que `TestClient` dispare correctamente la carga de datos en los tests.

5. **Benchmark dinamico de latencia:** el CSV de KPIs de Manuel tenia "Pendiente" para latencia (no habia API cuando lo escribio). El backend ahora mide 5 reps de `/radius` al startup y sobreescribe esa fila al servir `/kpis` con el valor real medido + status calculado dinamicamente.

## Next steps

### Bloqueantes
- Ninguno. Backend cumple toda la rubrica.

### Mejoras opcionales (post-entrega)
1. **Regenerar CSVs con bounds fisicos desde el pipeline**. Los CSVs originales traen 603 predicciones corregidas negativas y 10 varianzas negativas numericas; el backend ya las sanea al cargar usando fallback a `prediction_convlstm` cuando `prediction_corrected < 0`, y los tests verifican que la API no exponga valores fisicamente invalidos.
2. **Convertir CSV a Parquet** (`pd.to_parquet`) para reducir cold start del Space de ~5 s a ~1 s y RAM de 150 MB a ~30 MB.
3. **Cachear `/map` y `/lisa`** con `lru_cache` en el servicio (1920 celdas se recalculan en cada request pero los CSVs no cambian).
4. **Endpoint POST `/api/situacion3/refresh`** protegido con HF_TOKEN para re-cargar los CSVs sin restart cuando Manuel actualice el modelo.
5. **Rotar el HF_TOKEN expuesto** que esta hardcodeado en el git remote del clon. Ver https://huggingface.co/settings/tokens.
6. **Activar el ensamble v8 real para `/predict`** integrando los 4 checkpoints `.pt` (37 MB). Requiere subir torch + open_clip a la imagen Docker (~2 GB). Por ahora `/predict` sirve directo desde Sit3 precomputado, que es lo que pide la rubrica.

## Repositorios relacionados

- **Frontend:** https://huggingface.co/spaces/analiticalastdance/geovision-cali-frontend
- **Dataset (manifest, CSVs, checkpoints, handoffs):** https://huggingface.co/datasets/analiticalastdance/geovision-cali

## Equipo

- **Sebastian Belalcazar** - datos, backend, frontend, deploy
- **Manuel Gruezo** - CLIP, SAE, ConvLSTM, ST-Kriging (artefactos precomputados de este backend)
- **Luis Angel Garcia Garcia** - geoestadistica (Moran/LISA), variograma residuos, K-Means

Universidad Autonoma de Occidente, Cali - Analitica de Datos I (2026-05).
