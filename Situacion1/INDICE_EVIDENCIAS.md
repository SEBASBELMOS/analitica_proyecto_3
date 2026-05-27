# Indice de Evidencias Finales - Situacion 1

Este indice define la version evaluable de Situacion 1 para el zip final. La evidencia pesada no se duplica dentro de esta carpeta; se referencia mediante manifests, tablas de auditoria y rutas del proyecto.

## Raiz Evaluable

- Ruta absoluta: `/workspace/geovision-cali-hf/Entrega_Final/Situacion1`.
- Ruta relativa del proyecto: `Entrega_Final/Situacion1`.
- Notebook reproducible: `EDA_consolidado_situacion1.ipynb`.
- Tabla formal de cumplimiento: `tablas/12_cumplimiento_rubrica_situacion1.csv`.

## Orden Recomendado de Revision

1. `EDA_consolidado_situacion1.ipynb`.
2. `tablas/12_cumplimiento_rubrica_situacion1.csv`.
3. `evidencias/README.md`.

## Mapa Rubrica a Evidencia

| Rubrica / entregable | Evidencia final |
|---|---|
| Volumen >= 50 GB | `tablas/resumen_sentinel2_raw_93gb.csv`, `evidencias/manifests/scene_manifest.jsonl`, `manifest/manifest.json` |
| Manifest con MD5 | `evidencias/manifests/manifest_panel_geovision_cali_2020_2024.json`, manifests por fuente en `evidencias/manifests/` |
| Panel analitico final | `panel_tabular/panel_geovision_cali_station_hourly_2020_2024_model_safe.parquet` |
| Auditoria model_safe sin leakage S5P directo | `tablas/10_panel_final_revision_leakage.csv`, `tablas/10_panel_final_columnas_excluir_modelado.csv` |
| EDA >= 8 visualizaciones | `figuras/` |
| EDA interpretado y reproducible | `EDA_consolidado_situacion1.ipynb` |
| MODIS observado vs interpolado | `figuras/07_modis_origen_valores_aod.png`, `tablas/07_modis_origen_valores_aod.csv` |
| MODIS AOD interpolado | `figuras/07_modis_distribucion_aod_interpolado.png`, `figuras/07_modis_serie_diaria_aod_055_origen.png` |
| Cobertura espacial | `figuras/04_cobertura_espacial_fuentes_situacion1.png`, `tablas/resumen_cobertura_espacial_fuentes.csv` |
| Cobertura temporal | `figuras/10_panel_final_cobertura_mensual.png`, tablas mensuales en `tablas/` |
| Calidad, nulos y outliers | `tablas/10_panel_final_calidad_variables.csv`, `tablas/10_panel_final_top20_nulos.csv`, `tablas/10_panel_final_outliers_iqr_variables_clave.csv` |
| ETL y trazabilidad Sentinel-2 | `evidencias/logs/sentinel2_12band_raw_download_full.log`, `evidencias/logs/scene_zarr_build.log`, `evidencias/manifests/scene_zarr_manifest.jsonl` |

## Notas de Control

- El umbral de 50 GB se verifica sobre ingesta raw/cache Sentinel-2 y manifest global, no sobre el panel final comprimido.
- El panel `model_safe` tiene 278618 filas, 102 columnas y MD5 `617962658552dec5c9e3bce6ef69bd1b`.
- El manifest `evidencias/manifests/manifest_panel_geovision_cali_2020_2024.json` conserva metadata del panel completo y de `model_safe`; por eso lista columnas S5P directas en la seccion del panel completo. La verificacion de leakage del archivo `model_safe` se audita aparte en `tablas/10_panel_final_revision_leakage.csv`.
- Algunos manifests preservan rutas originales de generacion en Windows, WSL o cache Hugging Face. Para evaluacion del zip, usar las rutas relativas indicadas en este indice y en `tablas/12_cumplimiento_rubrica_situacion1.csv`.
- No se conserva evidencia directa de Dask/Spark; se documenta procesamiento por fecha, tile y banda con logs Sentinel-2 como limitacion metodologica.
