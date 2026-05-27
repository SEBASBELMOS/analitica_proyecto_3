# Checklist Rubrica Final - Situacion 2

## KPIs


| KPI                            | Umbral minimo | Excelente | Valor final | Estado    | Evidencia                                |
| ------------------------------ | ------------- | --------- | ----------- | --------- | ---------------------------------------- |
| Recall@1 imagen-texto test     | >= 0.45       | >= 0.65   | 0.6043      | Cumple    | `metricas/sweep_best_metrics.json`       |
| Recall@5 imagen-texto test     | >= 0.70       | >= 0.85   | 1.0000      | Excelente | `metricas/recall_at5_protocol_note.json` |
| Sparsity ratio SAE visual test | >= 0.70       | >= 0.85   | 0.8809      | Excelente | `metricas/sweep_best_metrics.json`       |
| MSE reconstruccion SAE val     | <= 0.05       | <= 0.02   | 0.0106      | Excelente | `metricas/sweep_best_metrics.json`       |
| Varianza explicada AFE         | >= 80%        | >= 90%    | 0.8013      | Cumple    | `afe_afc/check_kpis_afc_rubrica.csv`     |
| RMSEA AFC                      | < 0.08        | < 0.05    | 0.0312      | Excelente | `afe_afc/manifest_afe_afc.json`          |
| CFI AFC                        | > 0.90        | > 0.95    | 0.9465      | Cumple    | `afe_afc/manifest_afe_afc.json`          |


## Entregables


| Entregable                              | Estado | Evidencia                                                                        |
| --------------------------------------- | ------ | -------------------------------------------------------------------------------- |
| Checkpoint `.pt` con MD5 verificable    | Cumple | `checkpoints/sweep_best.pt`, `metricas/sweep_best_md5.json`                      |
| Curvas de entrenamiento                 | Cumple | `curvas/01_*.png` a `curvas/07_*.png`, `curvas/full_logging_history.csv`         |
| Reporte AFE+AFC                         | Cumple | `afe_afc/`, `notebooks/03_afe_afc_consolidado.ipynb`                             |
| Analisis neuronas activas SAE por clase | Cumple | `sae_interpretabilidad/`, `notebooks/04_sae_interpretabilidad_consolidado.ipynb` |


## Nota Metodologica

El resultado principal de retrieval es el mejor checkpoint del sweep v10/v5b (`Recall@1 = 0.6043`). El reentrenamiento instrumentado de la misma configuracion obtiene `Recall@1 = 0.5489` y aporta las curvas completas de entrenamiento solicitadas por la rubrica.
