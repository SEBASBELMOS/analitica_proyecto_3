from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/geovision-cali-hf")
OUT_DIR = ROOT / "outputs/pca_afe_afc_kmo_bartlett"
MATRIZ = "remoteclip_visual_512"

CASOS = [
    {
        "modelo": "v8",
        "npz": ROOT / "outputs/clip_training_remoteclip_vitb32_sae_v6/embeddings_remoteclip_vitb32_sae_v6.npz",
        "metadata": ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_1000_v2/metadata.jsonl",
        "filtrar_cero": True,
        "nota": "baseline v8; se filtran tiles visualmente cero por metadata image_max/image_std como en el análisis PCA/AFE/AFC principal",
    },
    {
        "modelo": "v10_v5b",
        "npz": ROOT / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz",
        "metadata": ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit/metadata.jsonl",
        "filtrar_cero": True,
        "nota": "modelo robusto v10/v5b; no se esperan filas cero",
    },
]


def interpretar_kmo(kmo: float) -> str:
    if kmo < 0.50:
        return "malo"
    if kmo < 0.60:
        return "debil"
    if kmo < 0.70:
        return "aceptable_bajo"
    if kmo < 0.80:
        return "bueno"
    if kmo < 0.90:
        return "muy_bueno"
    return "excelente"


def calcular_bartlett_manual(x_std: np.ndarray) -> tuple[float, int, float]:
    n, p = x_std.shape
    corr = np.corrcoef(x_std, rowvar=False)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        corr = corr + np.eye(p) * 1e-8
        sign, logdet = np.linalg.slogdet(corr)
    chi_square = float(-(n - 1 - (2 * p + 5) / 6) * logdet)
    dof = int(p * (p - 1) / 2)
    p_value = float(chi2.sf(chi_square, dof))
    return chi_square, dof, p_value


def cargar_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filas = []

    for caso in CASOS:
        npz = np.load(caso["npz"], allow_pickle=True)
        x = npz[MATRIZ].astype("float64")
        metadata = cargar_metadata(caso["metadata"])
        if len(metadata) != len(x):
            raise ValueError(f"{caso['modelo']}: metadata y embeddings tienen longitudes distintas")
        if "image_std" in metadata.columns:
            filas_cero = np.isclose(metadata["image_max"].astype(float), 0.0) | np.isclose(metadata["image_std"].astype(float), 0.0)
        else:
            filas_cero = np.isclose(metadata["image_max"].astype(float), 0.0) | np.isclose(metadata["image_mean"].astype(float), 0.0)
        if caso["filtrar_cero"]:
            x_valid = x[~filas_cero]
        else:
            x_valid = x

        x_std = StandardScaler().fit_transform(x_valid)
        kmo_por_variable, kmo_global = calculate_kmo(x_std)

        try:
            bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(x_std)
            bartlett_dof = int(x_std.shape[1] * (x_std.shape[1] - 1) / 2)
            bartlett_origen = "factor_analyzer"
        except Exception:
            bartlett_chi2, bartlett_dof, bartlett_p = calcular_bartlett_manual(x_std)
            bartlett_origen = "manual"

        kmo_vars = pd.DataFrame(
            {
                "modelo": caso["modelo"],
                "variable": [f"emb_dim_{i:04d}" for i in range(x_std.shape[1])],
                "kmo_variable": kmo_por_variable,
            }
        )
        kmo_vars.to_csv(OUT_DIR / f"kmo_por_variable_{caso['modelo']}.csv", index=False, encoding="utf-8-sig")

        filas.append(
            {
                "modelo": caso["modelo"],
                "input_npz": str(caso["npz"]),
                "metadata": str(caso["metadata"]),
                "matriz": MATRIZ,
                "shape_original": list(x.shape),
                "shape_usada": list(x_valid.shape),
                "filas_cero_excluidas": int(filas_cero.sum()) if caso["filtrar_cero"] else 0,
                "kmo_global": float(kmo_global),
                "kmo_interpretacion": interpretar_kmo(float(kmo_global)),
                "kmo_min_variable": float(np.nanmin(kmo_por_variable)),
                "kmo_mediana_variable": float(np.nanmedian(kmo_por_variable)),
                "kmo_max_variable": float(np.nanmax(kmo_por_variable)),
                "bartlett_chi2": float(bartlett_chi2),
                "bartlett_dof": int(bartlett_dof),
                "bartlett_p_value": float(bartlett_p),
                "bartlett_significativo_0_05": bool(bartlett_p < 0.05),
                "bartlett_origen": bartlett_origen,
                "nota": caso["nota"],
            }
        )

    resumen = pd.DataFrame(filas)
    resumen.to_csv(OUT_DIR / "resumen_kmo_bartlett_embeddings.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "resumen_kmo_bartlett_embeddings.json").write_text(
        json.dumps(filas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
