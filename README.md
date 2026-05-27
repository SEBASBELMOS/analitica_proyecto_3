# GeoVision-CLIP Cali

> Estimacion de Contaminacion Atmosferica en Puntos No Muestreados mediante Deep Learning + Estadistica Geoespacial Avanzada

**Proyecto Final** - Analitica de Datos, Universidad Autonoma de Occidente - 2026
**Grupo:** Sebastian Belalcazar, Manuel Gruezo, Luis Garcia

---

## 1. Resumen

Sistema hibrido que combina (i) **GeoVision-CLIP+SAE** (aprendizaje multimodal contrastivo sobre Sentinel-2 + descripciones en espanol), (ii) **ConvLSTM** espacio-temporal, y (iii) **ST-Kriging** geoestadistico, para estimar NO2, SO2 y O3 en cualquier punto del area metropolitana de Cali con cuantificacion de incertidumbre.

---

## 2. Estructura del Repositorio

```
analitica/
+-- README.md                              (este archivo)
+-- requirements.txt                       (deps globales del proyecto)
|
+-- pipeline.ipynb                         (Sit. 1: descarga S5P + S2 + manifest MD5)
+-- dataset_pairs_2_1_v2.ipynb             (Sit. 2.1+2.2: pares CLIP v2, 12 bandas, Zarr propio)
+-- dataset_pairs_2_1_v3.ipynb             (Sit. 2.1+2.2: pares CLIP v3, 12 bandas, S2 de Manuel)
+-- eda_pairs_2_1.ipynb                    (EDA dataset CLIP)
+-- inspect_manuel_repo.ipynb              (auditoria del repo S2 de Manuel)
+-- build_s5p_panel_tabular.ipynb          (Sit. 1: panel tabular S5P para Luis)
|
+-- fastapi_app/                           (Sit. 4: backend)
|   +-- main.py                            (endpoints /predict /validate /health)
|   +-- requirements.txt
|   +-- Dockerfile                         (multi-stage builder + runtime)
|   +-- docker-compose.yml                 (backend + frontend)
|   +-- .dockerignore
|
+-- docs/                                  (documentacion adicional)
|   +-- DEPLOY_HF_SPACES.md                (como desplegar el backend)
|   +-- RUNPOD_SETUP.md                    (como entrenar el modelo)
|
+-- manifest.json                          (copia local del manifest MD5 del dataset)
+-- ProyectoFinal_GeoVisionCLIP_Cali.pdf   (enunciado del proyecto)
```

---

## 3. Datasets en HuggingFace

### Repo principal: `analiticalastdance/geovision-cali`

```
analiticalastdance/geovision-cali/
+-- panel_zarr/
|   +-- sentinel5p/{NO2,SO2,O3}/             (Zarr 1826 dias, 29x24 grilla)
|   +-- sentinel2/{r0c0..r4c3, j0c0..j0c3}/  (24 tiles Zarr 12 bandas 10m)
+-- panel_tabular/
|   +-- sentinel5p.parquet                   (1.27M filas x 21 cols, para Luis)
|   +-- manifest_sentinel5p.json
+-- pairs_dataset_v2/                        (1000 pares CLIP 12 bandas, Zarr propio)
+-- pairs_dataset_v3/                        (1000 pares CLIP 12 bandas, S2 Manuel) OFICIAL
+-- manifest/manifest.json                   (8907 archivos MD5, 49.27 GB total)
```

### Repo S2 de Manuel: `analiticalastdance/geovision-cali-s2`

S2 L2A particionado por (year, month, MGRS-tile), 12 bandas, formato Zarr uint16, CRS EPSG:32618.

---

## 4. Reproducibilidad

**Semilla aleatoria:** `SEED=42` en todos los notebooks y scripts.

### Requisitos
- Python 3.11+
- Token de HuggingFace con acceso de lectura al repo `analiticalastdance/geovision-cali`
- (Opcional) Cuenta GEE para regenerar datos desde cero

### Instalacion

```bash
pip install -r requirements.txt
```

### Verificacion de integridad

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id='analiticalastdance/geovision-cali',
    filename='manifest/manifest.json',
    repo_type='dataset',
)
mf = json.load(open(path))
print(f"Archivos: {mf['summary']['total_files']}, GB: {mf['summary']['total_size_gb']:.2f}")
```

---

## 5. Tareas y Estado

| Sit. | Tarea | Responsable | Estado |
|------|-------|-------------|--------|
| 1.1-1.2 | Credenciales GEE + S5P | Sebastian | LISTO |
| 1.3 | S2 (12 bandas, MGRS) | Manuel | LISTO |
| 1.4 | ERA5 + MODIS | Luis | EN CURSO |
| 1.5 | DAGMA/SISAIRE | Luis | EN CURSO |
| 1.6-1.7 | Zarr + manifest MD5 | Sebastian | LISTO |
| 1.8 | EDA 8 viz del panel | Luis | PENDIENTE |
| 2.1-2.2 | Dataset CLIP + Split | Sebastian | LISTO |
| 2.3-2.6 | CLIP+SAE | Manuel | EN ESPERA |
| 2.7-2.10 | Recall + checkpoint | Sebastian (post-Manuel) | BLOQUEADO |
| 2.8-2.9 | AFE + AFC | Luis (post-Manuel) | BLOQUEADO |
| 3.1-3.3 | ConvLSTM | Manuel | PENDIENTE |
| 3.4-3.9 | Variograma + Kriging + Moran | Luis | PENDIENTE |
| 4.1 | Backend FastAPI | Sebastian | LISTO (mock) |
| 4.2-4.4 | Frontend React | Luis | PENDIENTE |
| 4.5 | Docker multi-stage | Sebastian | LISTO |
| 4.6 | Deploy HF Spaces | Sebastian | LISTO (URL: https://analiticalastdance-geovision-cali-api.hf.space) |
| 4.7 | Reporte IEEE | Todos | PENDIENTE |
| 4.8 | Reproducibilidad | Sebastian | LISTO |

---

## 6. Stack Tecnologico

| Componente | Tecnologia | Estado |
|------------|-----------|--------|
| Descarga datos | Google Earth Engine | LISTO |
| Almacenamiento | HuggingFace Datasets (Git LFS) | LISTO |
| Formato raster | Zarr v2 (zarr<3 obligatorio) | LISTO |
| Formato tabular | Parquet (snappy) | LISTO |
| Procesamiento | Colab + NumPy + xarray + pyproj | LISTO |
| Modelo CLIP | RemoteCLIP ViT-B/32 | EN ESPERA |
| Texto encoder | XLM-RoBERTa multilingue | EN ESPERA |
| SAE | PyTorch nn.Module | EN ESPERA |
| ConvLSTM | PyTorch nn.Module | EN ESPERA |
| Geoestadistica | PyKrige + PySAL | EN ESPERA |
| Backend | FastAPI + Uvicorn | LISTO (mock) |
| Frontend | React + Vite + Leaflet | PENDIENTE |
| Contenedor | Docker multi-stage + docker-compose | LISTO |
| Despliegue | HuggingFace Spaces (SDK Docker) | LISTO - URL publica activa |
| Entrenamiento | RunPod (NVIDIA RTX 4070, 12 GB VRAM) | EN ESPERA |
| Trazabilidad | manifest MD5 + SEED=42 | LISTO |

---

## 7. Quick Start

### Probar la API publica (desplegada en HF Spaces)

```bash
curl https://analiticalastdance-geovision-cali-api.hf.space/health

curl -X POST https://analiticalastdance-geovision-cali-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": 3.4372, "lon": -76.5225, "fecha": "2024-06-15"}'
```

Swagger UI publico: https://analiticalastdance-geovision-cali-api.hf.space/docs

### Levantar el backend localmente

```bash
cd fastapi_app
docker-compose up --build
# Abre http://localhost:7860
# Swagger UI en http://localhost:7860/docs
```

### Cargar dataset CLIP desde HF

```python
from huggingface_hub import snapshot_download
import json
import numpy as np

local = snapshot_download(
    repo_id='analiticalastdance/geovision-cali',
    repo_type='dataset',
    token='hf_...',
    allow_patterns='pairs_dataset_v3/**',
)

# Leer metadata
with open(f'{local}/pairs_dataset_v3/metadata.jsonl') as f:
    pairs = [json.loads(l) for l in f]

# Cargar una imagen
img = np.load(f'{local}/pairs_dataset_v3/{pairs[0]["image"]}')  # (64, 64, 12) float32
print(pairs[0]['text'])
```

---

## 8. Documentacion Adicional

- **Contexto integral del proyecto:** [HANDOFF.md](HANDOFF.md) - para nuevas sesiones
- **Contexto vivo dia a dia:** [geovision_context.md](geovision_context.md)
- **Despliegue en HF Spaces:** [docs/DEPLOY_HF_SPACES.md](docs/DEPLOY_HF_SPACES.md)
- **Setup de entrenamiento en RunPod:** [docs/RUNPOD_SETUP.md](docs/RUNPOD_SETUP.md)
- **Enunciado original:** [ProyectoFinal_GeoVisionCLIP_Cali.pdf](ProyectoFinal_GeoVisionCLIP_Cali.pdf)

---

## 9. Licencia y Cita

Proyecto academico - Universidad Autonoma de Occidente, Ingenieria de Datos e IA - 2026

Datos satelitales: Copernicus (Sentinel-2, Sentinel-5P), NASA (MODIS), ECMWF (ERA5), IDEAM/DAGMA Cali (estaciones in-situ).

Para citar este trabajo:

```
Belalcazar S., Gruezo M., Garcia L. (2026). GeoVision-CLIP Cali: Estimacion de
Contaminacion Atmosferica mediante Deep Learning + Estadistica Geoespacial.
Proyecto final Analitica de Datos, UAO.
```
