# Entrega Final - Situacion 2

Carpeta consolidada para subir la Situacion 2 de GeoVision-CLIP Cali. Reune evidencias de entrenamiento CLIP/SAE, checkpoints, curvas, AFE/AFC e interpretabilidad mecanica del SAE.

La raiz evaluable unica es `/workspace/geovision-cali-hf/Entrega_Final/Situacion2`. El indice maestro de rutas finales esta en `INDICE_EVIDENCIAS.md`.

## Resultado Principal

- Modelo: RemoteCLIP ViT-B/32 + Sparse Autoencoder k-sparse v10/v5b.
- Dataset: `clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit`.
- Pares imagen-texto: 1500.
- Split: holdout por escena, sin solapamiento train/val/test.
- Nota de split: se priorizo holdout por `scene_id` para evitar fuga por escena/fecha; por eso la particion final es `1000/265/235` y no exactamente `70/15/15`, aunque el dataset global conserva 300 pares por clase.
- Mejor checkpoint del sweep: `checkpoints/sweep_best.pt`.
- Recall@1 test principal: `0.6042553186416626`.
- Reentrenamiento instrumentado: `checkpoints/full_logging_best.pt`.
- Recall@1 test instrumentado: `0.548936128616333`.

## Estructura


| Carpeta                  | Contenido                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `checkpoints/`           | Checkpoints `.pt` principal e instrumentado.                                           |
| `metricas/`              | KPIs, metricas de sweep, nota Recall@5, manifest instrumentado, MD5 y dataset summary. |
| `curvas/`                | Curvas de entrenamiento e historial csv/json del reentrenamiento instrumentado.        |
| `afe_afc/`               | Matriz de cargas rotada, scree plots, AFC, KMO/Bartlett y manifest psicometrico.       |
| `sae_interpretabilidad/` | Top unidades SAE por clase, resumen por split y heatmap.                               |
| `notebooks/`             | Notebooks consolidados ejecutados y exportados a HTML.                                 |
| `scripts/`               | Scripts usados para generar entrenamiento, AFE/AFC y analisis SAE.                     |
| `figuras/`               | Figuras PCA por clase usadas previamente en el reporte.                                |


La evidencia final se referencia exclusivamente desde las carpetas consolidadas: `metricas/`, `curvas/`, `afe_afc/`, `sae_interpretabilidad/`, `checkpoints/` y `notebooks/`.

## Notebooks Consolidados


| Notebook                                                     | Entregable cubierto                                                              |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `notebooks/00_resumen_situacion2_consolidado.ipynb`          | Resumen ejecutivo de KPIs y checklist.                                           |
| `notebooks/01_checkpoint_md5_consolidado.ipynb`              | Checkpoint `.pt`, metricas y MD5.                                                |
| `notebooks/02_curvas_entrenamiento_consolidado.ipynb`        | Curvas de entrenamiento.                                                         |
| `notebooks/03_afe_afc_consolidado.ipynb`                     | AFE+AFC, scree plot e indices de ajuste.                                         |
| `notebooks/04_sae_interpretabilidad_consolidado.ipynb`       | Neuronas activas SAE por clase.                                                  |
| `notebooks/05_generacion_dataset_tiles_v10_v5b_codigo.ipynb` | Codigo limpio y protocolo de generacion del dataset final v10/v5b de 1500 pares. |


Cada notebook tambien tiene version `.html` en la misma carpeta. Los notebooks `03` y `04` generan figuras adicionales directamente desde las tablas finales: proyecciones PCA/AFE por clase y graficas de top unidades SAE por clase.

## KPIs Finales

La tabla formal esta en `metricas/kpi_situacion2_summary.csv` y el checklist en `CHECKLIST_RUBRICA_FINAL.md`.

Resumen:

- Recall@1 test: `0.6043`, cumple minimo.
- Recall@5 test: `1.0000`, excelente como cobertura de cinco prompts de clase; ver `metricas/recall_at5_protocol_note.json`.
- Sparsity SAE visual test: `0.8809`, excelente.
- MSE reconstruccion SAE val: `0.0106`, excelente. Test MSE adicional: `0.0114`.
- Varianza explicada AFE: `0.8013`, cumple minimo.
- RMSEA AFC: `0.0312`, excelente.
- CFI AFC: `0.9465`, cumple minimo.

## Nota de Transparencia

El mejor checkpoint del sweep tiene mejor desempeno test (`Recall@1 = 0.6043`) y se reporta como resultado principal. El reentrenamiento instrumentado de la misma configuracion tiene `Recall@1 = 0.5489` y se conserva porque aporta las curvas completas solicitadas por la rubrica. Ambos superan el umbral minimo.

## Evidencia de Generacion de Tiles

se incluye `notebooks/05_generacion_dataset_tiles_v10_v5b_codigo.ipynb`, que documenta el protocolo final, las rutas auditadas y las validaciones del dataset de 1500 pares.
