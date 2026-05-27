# Indice de Evidencias Finales - Situacion 2

Este indice define la version unica evaluable de Situacion 2. La evidencia final se referencia exclusivamente desde las carpetas consolidadas de esta entrega.

## Raiz Evaluable

- Ruta absoluta: `/workspace/geovision-cali-hf/Entrega_Final/Situacion2`.
- Ruta relativa del proyecto: `Entrega_Final/Situacion2`.
- Tabla formal de KPIs: `metricas/kpi_situacion2_summary.csv`.
- Checklist contra rubrica: `CHECKLIST_RUBRICA_FINAL.md`.

## Orden Recomendado de Revision

1. `README.md`.
2. `CHECKLIST_RUBRICA_FINAL.md`.
3. `notebooks/00_resumen_situacion2_consolidado.html`.
4. `notebooks/01_checkpoint_md5_consolidado.html`.
5. `notebooks/02_curvas_entrenamiento_consolidado.html`.
6. `notebooks/03_afe_afc_consolidado.html`.
7. `notebooks/04_sae_interpretabilidad_consolidado.html`.
8. `notebooks/05_generacion_dataset_tiles_v10_v5b_codigo.html`.

## Mapa Rubrica a Evidencia

| Rubrica / entregable | Evidencia final unica |
|---|---|
| KPIs consolidados | `metricas/kpi_situacion2_summary.csv` |
| Recall@1 principal | `metricas/sweep_best_metrics.json` |
| Recall@5 y nota metodologica | `metricas/recall_at5_protocol_note.json` |
| Checkpoint principal | `checkpoints/sweep_best.pt` |
| MD5 checkpoint principal | `metricas/sweep_best_md5.json` |
| Checkpoint instrumentado | `checkpoints/full_logging_best.pt` |
| MD5 checkpoint instrumentado | `metricas/full_logging_md5.json` |
| Curvas de entrenamiento | `curvas/01_loss_total_por_epoch.png` a `curvas/07_recall_por_epoch.png` |
| Historial de entrenamiento | `curvas/full_logging_history.csv`, `curvas/full_logging_history.json` |
| AFE/AFC KPIs de rubrica | `afe_afc/manifest_afe_afc.json`, `afe_afc/check_kpis_afc_rubrica.csv` |
| Diagnostico AFC adicional | `afe_afc/diagnostico_afc_srmr.csv` |
| Cargas rotadas | `afe_afc/afe_cargas_rotadas.csv` |
| Figuras AFE/AFC | `afe_afc/figures/` |
| PCA/AFE por clase generado en notebook | `afe_afc/figures/05_pca_factor_scatter_pc1_pc2_por_clase_generado_notebook.png`, `afe_afc/figures/06_pca_factor_scatter_multipanel_por_clase_generado_notebook.png` |
| KMO/Bartlett | `afe_afc/kmo_bartlett_summary.csv`, `afe_afc/kmo_bartlett_summary.json` |
| Interpretabilidad SAE | `sae_interpretabilidad/split_summary.csv`, `sae_interpretabilidad/top_units_by_class.csv` |
| Graficas SAE por clase generadas en notebook | `sae_interpretabilidad/top_units_by_class_test_bars_generado_notebook.png`, `sae_interpretabilidad/top_units_by_class_heatmap_generado_notebook.png` |
| Matriz de confusion test generada en notebook | `sae_interpretabilidad/confusion_matrix_test_generado_notebook.png` |
| Heatmap SAE por clase original | `sae_interpretabilidad/top_units_by_class_heatmap.png` |
| Auditoria dataset final | `metricas/dataset_summary.json` |
| Protocolo limpio dataset | `notebooks/05_generacion_dataset_tiles_v10_v5b_codigo.ipynb` |

## Notas de Control

- `Recall@1 = 0.6043` corresponde al mejor checkpoint del sweep y es el resultado principal.
- `Recall@1 = 0.5489` corresponde al reentrenamiento instrumentado que aporta curvas completas.
- `Recall@5 = 1.0` se interpreta como cobertura bajo cinco prompts de clase, no como ranking fino pair-level.
- `SRMR = 0.1458` se reporta como diagnostico adicional; no es KPI duro de la rubrica.
- El split final usa holdout por escena sin solapamiento (`1000/265/235`) para reducir fuga espacial/temporal; esto reemplaza el split estratificado exacto `70/15/15` pedido como referencia operacional.
