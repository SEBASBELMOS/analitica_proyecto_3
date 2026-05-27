# Evidencias Situacion 1

Esta carpeta contiene artefactos livianos para entrega y defensa de Situacion 1. No incluye datasets pesados; conserva manifests, logs y tablas de calidad que documentan la trazabilidad del panel.

Para la revision completa del zip final, usar tambien `../INDICE_EVIDENCIAS.md` y `../tablas/12_cumplimiento_rubrica_situacion1.csv`.

## Estructura

- `manifests/`: manifests JSON/JSONL con rutas, hashes, dimensiones, fuentes, volumen y cobertura.
- `logs/`: logs relevantes de descarga/conversion/procesamiento Sentinel-2.
- `tablas_fuente/`: resumenes CSV de calidad, cobertura y nulos por fuente.

## Evidencia Principal De Volumen >= 50 GB

Archivo:

`manifests/scene_manifest.jsonl`

Resumen auditado:

- 198 escenas Sentinel-2 L2A OK.
- 12 bandas por escena.
- Periodo: 2020-01-02 a 2024-12-16.
- Tiles MGRS: 18NUJ y 18NUK.
- Volumen total: 93.1 GB decimales, equivalente a 86.7 GiB.

Log complementario:

`logs/sentinel2_12band_raw_download_full.log`

Este log muestra la descarga acumulada y cruza el umbral de 50 GB durante la ingesta Sentinel-2.

## Uso En El Informe

Estos archivos soportan:

- Manifest con trazabilidad y hashes.
- Evidencia de volumen superior a 50 GB.
- Auditoria de calidad por fuente.
- Discusion de diferencia entre datos raw/cache, features y panel final comprimido.

Nota: algunos manifests preservan rutas originales de generacion en Windows, WSL o cache Hugging Face. Esas rutas son metadata historica; las rutas evaluables del zip final estan resumidas en `../INDICE_EVIDENCIAS.md`.
