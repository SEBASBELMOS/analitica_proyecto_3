from __future__ import annotations

import json
import textwrap
from pathlib import Path

import nbformat as nbf


BASE = Path("/workspace/geovision-cali-hf")
NB_PATH = BASE / "manuel/notebooks/Situacion3/Rubrica/99_consolidado_situacion3_rubrica.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md("""
        # Situación 3 - Consolidado De Rúbrica

        Este notebook consolida la evidencia computacional de Situación 3: Sentinel-2 Zarr, metadata SCL, embeddings GeoVisionCLIP/SAE, tensor espacial, ConvLSTM bidireccional, ST-Kriging de residuos, LOO-CV espacial y mapas finales.
        """),
        code("""
        from pathlib import Path
        import json
        import pandas as pd
        import numpy as np

        BASE = Path('/workspace/geovision-cali-hf')
        RUB = BASE / 'outputs/situacion3/rubrica'

        def read_json(path):
            path = Path(path)
            return json.loads(path.read_text()) if path.exists() else {'missing': str(path)}

        def show_json(obj):
            print(json.dumps(obj, indent=2, ensure_ascii=False))
        """),
        md("""
        ## 1. Cache Sentinel-2 Zarr

        Fuente Sentinel-2 transformada a Zarr con 12 bandas y SCL. Esto reemplaza los JP2 crudos para trabajar dentro de la cuota de disco.
        """),
        code("""
        zarr_summary = read_json(RUB / '12_sentinel2_zarr_cache_cali_bbox/summary_zarr_scene_cache.json')
        show_json(zarr_summary)
        """),
        md("""
        ## 2. Metadata Limpia Desde Zarr

        La metadata filtra qué ventanas se convierten a embeddings. Evita generar `.npy` masivos y conserva trazabilidad SCL/calidad.
        """),
        code("""
        meta_paths = sorted((RUB / '16_metadata_tiles_from_zarr_cache/full').glob('chunk_*_of_008/metadata_tiles_scl.csv'))
        rows = []
        for p in meta_paths:
            df = pd.read_csv(p)
            ok = df['accepted_s2_policy'].fillna(False).astype(bool)
            rows.append({'chunk': p.parent.name, 'rows': len(df), 'accepted': int(ok.sum()), 'rejected': int((~ok).sum()), 'duplicates': int(df['tile_id'].duplicated().sum())})
        metadata_summary = pd.DataFrame(rows)
        display(metadata_summary)
        display(metadata_summary[['rows','accepted','rejected','duplicates']].sum().to_frame('total').T)
        """),
        md("""
        ## 3. Embeddings GeoVisionCLIP/SAE

        Cada ventana aceptada se lee desde Zarr en memoria y se convierte a embedding `h_256` usando RemoteCLIP + SAE de Situación 2.
        """),
        code("""
        emb_summary = read_json(RUB / '17_embeddings_full_clean_direct_from_zarr/summary_embeddings_direct_from_zarr.json')
        show_json(emb_summary)
        """),
        md("""
        ## 4. Tensor Espacial Para ConvLSTM

        Tensor `B x 8 x 256 x H x W`, targets `B x 3 x 3 x H x W` para horizontes T+1, T+3, T+7 y contaminantes NO2, SO2, O3.
        """),
        code("""
        tensor_summary = read_json(RUB / '18_spatial_embedding_tensors_full_clean/summary_spatial_embedding_tensors_len8.json')
        show_json(tensor_summary)
        """),
        md("""
        ## 5. ConvLSTM Bidireccional

        Arquitectura de rúbrica: bidireccional, `hidden=128`, `kernel=3`, `2 capas`, salida `B x 3 x 3 x H x W`.
        """),
        code("""
        conv_summary = read_json(MET / 'summary_convlstm_bidirectional_spatial.json')
        show_json(conv_summary)
        metrics = pd.read_csv(MET / 'test_metrics_masked.csv')
        display(metrics)
        hist = pd.read_csv(MET / 'training_history.csv')
        display(hist)
        """),
        md("""
        ## 6. Residuos ConvLSTM

        Residuos observados menos predichos en estaciones DAGMA/SISAIRE, base para ST-Kriging.
        """),
        code("""
        residual_summary = read_json(RUB / '20_convlstm_spatial_predictions_residuals/summary_convlstm_predictions_residuals.json')
        show_json(residual_summary)
        residuals = pd.read_csv(RUB / '20_convlstm_spatial_predictions_residuals/convlstm_predictions_residuals_long.csv')
        display(residuals.head())
        display(residuals.groupby(['pollutant','horizon_days']).size().rename('n').reset_index())
        """),
        md("""
        ## 7. ST-Kriging De Residuos Y Mapas Finales

        Se usa `PyKrige.OrdinaryKriging3D` sobre `(lon, lat, tiempo)` para corregir mapas ConvLSTM y obtener varianza de kriging.
        """),
        code("""
        krig_summary = read_json(RUB / '21_st_kriging_residual_corrected_maps/summary_st_kriging_residual_maps.json')
        show_json(krig_summary)
        model_summary = pd.read_csv(RUB / '21_st_kriging_residual_corrected_maps/st_kriging_model_summary.csv')
        display(model_summary)
        maps = pd.read_csv(RUB / '21_st_kriging_residual_corrected_maps/st_kriging_corrected_maps_long.csv')
        display(maps.groupby(['pollutant','horizon_days']).agg(
            rows=('grid_id','size'),
            corrected_min=('prediction_corrected','min'),
            corrected_mean=('prediction_corrected','mean'),
            corrected_max=('prediction_corrected','max'),
            variance_mean=('kriging_variance','mean'),
        ).reset_index())
        """),
        md("""
        ## 8. LOO-CV Espacial

        Validación leave-one-station-out para O3 y SO2. NO2 se excluye de LOO-CV espacial porque solo tiene una estación disponible.
        """),
        code("""
        loo_summary_path = RUB / '22_loo_spatial_st_kriging_cv/summary_loo_spatial_st_kriging_cv.json'
        if loo_summary_path.exists():
            loo_summary = read_json(loo_summary_path)
            show_json(loo_summary)
            display(pd.read_csv(RUB / '22_loo_spatial_st_kriging_cv/loo_spatial_metrics_summary.csv'))
            display(pd.read_csv(RUB / '22_loo_spatial_st_kriging_cv/loo_spatial_fold_metrics.csv').head(30))
        else:
            print('LOO-CV todavía no ejecutado:', loo_summary_path)
        """),
        md("""
        ## 9. Limitación NO2

        NO2 se modela y se mapea, pero no tiene LOO-CV espacial defendible por disponibilidad observacional (`n=1` estación).
        """),
        code("""
        no2_note = BASE / 'manuel/notebooks/Situacion3/Rubrica/limitacion_no2_validacion_espacial.md'
        print(no2_note.read_text() if no2_note.exists() else 'Nota NO2 no encontrada')
        """),
        md("""
        ## 10. Moran I, LISA Y K-Means

        Autocorrelación espacial global/local y clustering de perfiles críticos sobre los 9 mapas corregidos.
        """),
        code("""
        spatial_summary_path = RUB / '23_moran_lisa_kmeans_mapas_finales/summary_moran_lisa_kmeans.json'
        if spatial_summary_path.exists():
            spatial_summary = read_json(spatial_summary_path)
            show_json(spatial_summary)
            display(pd.read_csv(RUB / '23_moran_lisa_kmeans_mapas_finales/moran_global_by_pollutant_horizon.csv'))
            display(pd.read_csv(RUB / '23_moran_lisa_kmeans_mapas_finales/kmeans_cluster_summary.csv'))
            lisa = pd.read_csv(RUB / '23_moran_lisa_kmeans_mapas_finales/lisa_local_clusters_long.csv')
            display(lisa.groupby(['pollutant','horizon_days','lisa_cluster']).size().rename('n').reset_index())
        else:
            print('Moran/LISA/KMeans todavía no ejecutado:', spatial_summary_path)
        """),
        md("""
        ## Pendiente Visual

        - Exportar PNG/HTML finales para los 9 mapas y mapas de incertidumbre.
        - Integrar las figuras seleccionadas al documento final.
        """),
    ]
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(NB_PATH)


if __name__ == "__main__":
    main()
