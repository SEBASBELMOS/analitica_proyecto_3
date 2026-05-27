# Entrega Final - Situacion 3

Carpeta consolidada de la Situacion 3 del proyecto GeoVisionCLIP Cali.

## Notebook Principal

Los notebooks finales ejecutables y autocontenidos de la entrega son:

```text
notebooks/99_consolidado_situacion3_rubrica.ipynb
notebooks/07_convlstm_bidireccional_rubrica.ipynb
```

`99_consolidado_situacion3_rubrica.ipynb` es el notebook principal de revision. `07_convlstm_bidireccional_rubrica.ipynb` documenta el entrenamiento real final del ConvLSTM usando el checkpoint copiado en `checkpoints/`.

Incluye evidencia de:

- Sentinel-2 Zarr con SCL.
- Metadata limpia y filtros de calidad.
- Embeddings GeoVisionCLIP/SAE de 256 dimensiones.
- Tensor espacial `B x 8 x 273 x H x W` con embeddings + covariables S2/SCL/calendario/lat-lon.
- ConvLSTM bidireccional de rubrica.
- ST-Kriging 3D de residuos.
- LOO-CV espacial para `O3` y `SO2`, incluyendo version fisicamente valida con la misma regla usada en mapas/API.
- Moran I, LISA y K-Means.
- Limitacion metodologica de `NO2` por una sola estacion.

## Estructura

```text
notebooks/                         Notebooks y markdowns de evidencia
scripts/                           Scripts usados para reproducir el pipeline
outputs/metricas/                  Summaries, historia de entrenamiento y metricas ConvLSTM
outputs/validacion/                Residuos y LOO-CV espacial base
outputs/validacion_covariates/     LOO-CV espacial covariado
outputs/validacion_covariates_physical_bounds/ LOO-CV covariado con postproceso fisico
outputs/mapas/                     Mapas corregidos fisicamente, Moran/LISA/KMeans y GeoJSON
outputs/figuras/                   PNG finales embebidos en el notebook consolidado
artefactos_pesados_symlinks/       Symlinks a Zarr/checkpoints pesados sin duplicar datos
```

## Resultados Clave

Embeddings:

```text
116542 tiles aceptados -> h_256 [116542, 256]
```

Tensor espacial:

```text
X:          [119, 8, 273, 48, 40]
y:          [119, 3, 3, 48, 40]
targets:    2895
```

ConvLSTM espacial covariado:

```text
hidden=128, kernel=3, layers=2, bidirectional=True
test RMSE=7.54, MAE=5.46, R2=0.700
```

ST-Kriging y postproceso fisico:

```text
9/9 mapas generados con OrdinaryKriging3D
NO2, SO2, O3 x T+1, T+3, T+7
603 correcciones negativas en mapas reemplazadas por ConvLSTM; 10 varianzas negativas numericas a 0
0 predicciones negativas y 0 varianzas negativas en mapas finales
```

LOO-CV espacial:

```text
SO2/O3: 33 folds OK
NO2: excluido de LOO-CV espacial por n=1 estacion
```

Moran/LISA/K-Means:

```text
Moran I significativo en 9/9 mapas
LISA significativos: 10254 celdas-horizonte-contaminante
K-Means: 5 perfiles criticos
```

KPIs formales:

```text
O3 RMSE LOO-CV T+1 fisico: 8.515, cumple minimo
SO2 RMSE LOO-CV T+1 fisico: 3.895, cumple minimo
NO2 LOO-CV espacial: no evaluable por n=1 estacion
R2 LOO-CV SO2+O3 fisico: 0.807, excelente
Moran I: excelente en 9/9 mapas
Cobertura 95% sigma kriging: cumple minimo (93.85%)
Variograma nugget puro/sin estructura: no cumple globalmente (1/6 SO2/O3 LOO fisico)
Degradacion T+1 a T+7: excelente
Latencia backend: 28.75 ms, excelente (mediana 5 reps/radius en /api/situacion3/radius)
```

Tabla KPI fisica final:

```text
outputs/metricas/kpi_situacion3_physical_bounds_summary.csv
```

Barrido de modelos de variograma LOO-CV:

```text
outputs/metricas/42_loo_variogram_model_comparison_physical_bounds.csv
outputs/validacion_variogram_sweep/
```

Se probaron `linear`, `power`, `gaussian`, `spherical` y `exponential`. Ninguno elimina la estructura residual global; `linear` se mantiene como opcion final por menor RMSE y mayor R2 corregido.

## Archivos De Mapas

Mapas corregidos largos:

```text
outputs/mapas/st_kriging_corrected_maps_long.csv
```

GeoJSON:

```text
outputs/mapas/kmeans_critical_profiles.geojson
outputs/mapas/lisa_local_clusters.geojson
```

Figuras PNG finales:

```text
outputs/figuras/prediction_corrected/   9 mapas de concentracion corregida
outputs/figuras/kriging_variance/       9 mapas de incertidumbre
outputs/figuras/lisa/                   9 mapas LISA
outputs/figuras/kmeans/                 1 mapa K-Means
outputs/figuras/variograms/             variogramas experimentales base
```

Total:

```text
28 PNG finales fisicamente validos
```

## Checkpoint Final

El checkpoint final del ConvLSTM espacial covariado esta copiado dentro de la entrega:

```text
checkpoints/convlstm_covariates_spatial_best.pt
MD5: 4be0b8360b37a6d30ed4c386c9c79d0d
```

## Nota Sobre Artefactos Pesados

Los `Zarr` y checkpoints no se duplican para no exceder cuota. Estan enlazados desde:

```text
artefactos_pesados_symlinks/
```

Si se mueve esta carpeta fuera del workspace, copiar manualmente los destinos de esos symlinks o regenerarlos con los scripts incluidos.
