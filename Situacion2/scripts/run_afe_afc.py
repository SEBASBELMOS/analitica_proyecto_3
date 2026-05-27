from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from semopy import Model, calc_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/geovision-cali-hf")
EMB_PATH = ROOT / "outputs/clip_training_remoteclip_v10_v5b_embeddings/embeddings_remoteclip_v10_v5b.npz"
METADATA_PATH = ROOT / "outputs/clip_dataset_final_step_by_step/clip_s2_12band_pdf_classes_v5b_high_purity_1500_gsplit/metadata.jsonl"
OUT_DIR = ROOT / "outputs/pca_afe_afc_remoteclip_v10_v5b_desde_cero"
FIG_DIR = OUT_DIR / "figures"
MATRIZ_OFICIAL = "remoteclip_visual_512"
MAX_MODELOS_AFC = 10000


def varimax(phi: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-6):
    p, k = phi.shape
    rotation = np.eye(k)
    d_anterior = 0.0
    for _ in range(q):
        lambda_ = phi @ rotation
        u, s, vh = np.linalg.svd(
            phi.T @ (lambda_ ** 3 - (gamma / p) * lambda_ @ np.diag(np.diag(lambda_.T @ lambda_)))
        )
        rotation = u @ vh
        d_actual = s.sum()
        if d_anterior and d_actual / d_anterior < 1 + tol:
            break
        d_anterior = d_actual
    return phi @ rotation, rotation


def correr_pca(x_std: np.ndarray, threshold: float = 0.80):
    n_componentes = min(x_std.shape[0], x_std.shape[1])
    pca = PCA(n_components=n_componentes, random_state=42)
    scores = pca.fit_transform(x_std)
    acumulada = np.cumsum(pca.explained_variance_ratio_)
    m = int(np.searchsorted(acumulada, threshold) + 1)
    return pca, scores, acumulada, m


def cargar_metadata() -> pd.DataFrame:
    rows = []
    with METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def construir_modelo_afc(asignacion: dict[str, list[str]]) -> str:
    return "\n".join([f"{constructo} =~ {inds[0]} + {inds[1]}" for constructo, inds in asignacion.items()])


def calcular_srmr(modelo: Model, datos: pd.DataFrame) -> float:
    cov_obs = np.cov(datos.values, rowvar=False)
    std_obs = np.sqrt(np.diag(cov_obs))
    corr_obs = cov_obs / np.outer(std_obs, std_obs)
    sigma = modelo.calc_sigma()[0]
    std_imp = np.sqrt(np.diag(sigma))
    corr_imp = sigma / np.outer(std_imp, std_imp)
    mask = np.triu(np.ones_like(corr_obs, dtype=bool), k=1)
    return float(np.sqrt(np.mean((corr_obs[mask] - corr_imp[mask]) ** 2)))


def ajustar_afc(asignacion: dict[str, list[str]], datos_base: pd.DataFrame):
    variables = [v for inds in asignacion.values() for v in inds]
    if len(set(variables)) < len(variables):
        return None
    datos = datos_base[variables].copy()
    desc = construir_modelo_afc(asignacion)
    modelo = Model(desc)
    modelo.fit(datos)
    stats = calc_stats(modelo).loc["Value"]
    return {
        "modelo_desc": desc,
        "CFI": float(stats["CFI"]),
        "RMSEA": float(stats["RMSEA"]),
        "chi2": float(stats["chi2"]),
        "DoF": float(stats["DoF"]),
        "AIC": float(stats["AIC"]),
        "BIC": float(stats["BIC"]),
        "SRMR": calcular_srmr(modelo, datos),
    }


def ranking_constructo(promedios: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col in promedios.columns:
        no2 = promedios.loc["contaminacion_alta_NO2", col]
        so2 = promedios.loc["contaminacion_alta_SO2", col]
        o3 = promedios.loc["ozono_anomalo", col]
        veg = promedios.loc["vegetacion_densa", col]
        urb = promedios.loc["suelo_urbano", col]
        atmosfericas = np.mean([no2, so2, o3])
        cobertura = np.mean([veg, urb])
        out.append(
            {
                "factor": col,
                "Carga_Antropogenica": abs(np.mean([no2, so2]) - veg),
                "Estres_Vegetal": abs(veg - np.mean([no2, so2, o3, urb])),
                "Densidad_Urbana": abs(urb - veg),
                "Volatilidad_Atmosferica": abs(atmosfericas - cobertura),
            }
        )
    return pd.DataFrame(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(EMB_PATH, allow_pickle=True)
    metadata = cargar_metadata()
    x_full = npz[MATRIZ_OFICIAL].astype("float32")
    pair_ids = pd.Series(npz["pair_id"].astype(str), name="pair_id")

    meta = pd.DataFrame({"pair_id": pair_ids})
    meta = meta.merge(metadata, on="pair_id", how="left", validate="one_to_one")
    if meta["candidate_class"].isna().any():
        faltantes = int(meta["candidate_class"].isna().sum())
        raise ValueError(f"Hay {faltantes} embeddings sin metadata gsplit por pair_id.")

    fila_cero = np.isclose(np.abs(x_full).sum(axis=1), 0.0)
    x_valid = x_full[~fila_cero]
    meta_valid = meta.loc[~fila_cero].reset_index(drop=True)

    x_std = StandardScaler().fit_transform(x_valid)
    pca, scores, acumulada, m_principal = correr_pca(x_std)
    var80 = float(acumulada[m_principal - 1])

    x_full_std = StandardScaler().fit_transform(x_full)
    _, _, acumulada_full, m_full = correr_pca(x_full_std)

    loadings = pca.components_[:m_principal].T * np.sqrt(pca.explained_variance_[:m_principal])
    cargas_rotadas, rotacion = varimax(loadings)
    scores_rotados = scores[:, :m_principal] @ rotacion

    factor_cols = [f"Factor_{i}" for i in range(1, m_principal + 1)]
    score_cols = [f"Factor_{i}_score" for i in range(1, m_principal + 1)]
    emb_cols = [f"emb_dim_{i:04d}" for i in range(x_valid.shape[1])]

    cargas_df = pd.DataFrame(cargas_rotadas, index=emb_cols, columns=factor_cols)
    cargas_df.to_csv(OUT_DIR / "01_afe_cargas_rotadas_varimax_remoteclip_v10_v5b.csv", encoding="utf-8-sig")

    top_rows = []
    for factor in factor_cols:
        top = cargas_df[factor].abs().sort_values(ascending=False).head(8).index
        for dim in top:
            top_rows.append(
                {
                    "factor": factor,
                    "dimension": dim,
                    "carga": float(cargas_df.loc[dim, factor]),
                    "carga_abs": float(abs(cargas_df.loc[dim, factor])),
                }
            )
    pd.DataFrame(top_rows).to_csv(OUT_DIR / "02_top_dimensiones_por_factor.csv", index=False, encoding="utf-8-sig")

    factor_scores = pd.DataFrame(scores_rotados, columns=score_cols)
    for col in ["pair_id", "split", "candidate_class", "scene_id"]:
        factor_scores[col] = meta_valid[col].values
    factor_scores.to_parquet(OUT_DIR / "03_factor_scores_remoteclip_v10_v5b.parquet", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(acumulada_full) + 1), acumulada_full, label=f"completa n={x_full.shape[0]}")
    plt.plot(np.arange(1, len(acumulada) + 1), acumulada, label=f"filtrada n={x_valid.shape[0]}")
    plt.axhline(0.80, color="red", linestyle="--", label="80%")
    plt.axvline(m_principal, color="gray", linestyle=":", label=f"m principal={m_principal}")
    plt.xlabel("Componentes")
    plt.ylabel("Varianza acumulada")
    plt.title("PCA RemoteCLIP v10/v5b: varianza acumulada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_pca_remoteclip_v10_v5b_completa_vs_filtrada.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 8))
    vista = cargas_df.iloc[:80, : min(20, m_principal)]
    plt.imshow(vista.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Carga rotada")
    plt.yticks(range(len(vista.index)), vista.index, fontsize=6)
    plt.xticks(range(len(vista.columns)), vista.columns, rotation=90)
    plt.title("Cargas rotadas Varimax v10/v5b: primeras 80 dimensiones")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_heatmap_cargas_rotadas_v10_v5b.png", dpi=160)
    plt.close()

    promedios_clase = factor_scores.groupby("candidate_class")[score_cols].mean()
    ranking_factores = ranking_constructo(promedios_clase)
    ranking_factores.to_csv(OUT_DIR / "04_ranking_factores_por_constructo.csv", index=False, encoding="utf-8-sig")

    constructos = ["Carga_Antropogenica", "Estres_Vegetal", "Densidad_Urbana", "Volatilidad_Atmosferica"]
    candidatos_score = {c: ranking_factores.sort_values(c, ascending=False)["factor"].head(8).tolist() for c in constructos}
    rankings = {c: {factor: i for i, factor in enumerate(candidatos_score[c])} for c in constructos}
    combos = {c: list(itertools.combinations(candidatos_score[c], 2)) for c in constructos}

    asignaciones_candidatas = []
    for ca in combos["Carga_Antropogenica"]:
        for ev in combos["Estres_Vegetal"]:
            for du in combos["Densidad_Urbana"]:
                for va in combos["Volatilidad_Atmosferica"]:
                    asignacion = {
                        "Carga_Antropogenica": list(ca),
                        "Estres_Vegetal": list(ev),
                        "Densidad_Urbana": list(du),
                        "Volatilidad_Atmosferica": list(va),
                    }
                    variables = [v for inds in asignacion.values() for v in inds]
                    if len(set(variables)) < len(variables):
                        continue
                    puntaje = sum(rankings[c][v] for c, inds in asignacion.items() for v in inds)
                    asignaciones_candidatas.append((puntaje, asignacion))

    asignaciones_candidatas = sorted(asignaciones_candidatas, key=lambda x: x[0])[:MAX_MODELOS_AFC]

    resultados_afc = []
    datos_afc_base = factor_scores[score_cols].copy()
    for model_id, (puntaje, asignacion) in enumerate(asignaciones_candidatas, start=1):
        try:
            res = ajustar_afc(asignacion, datos_afc_base)
            if res is not None:
                res["model_id"] = model_id
                res["puntaje_ranking"] = puntaje
                res["asignacion"] = json.dumps(asignacion, ensure_ascii=False)
                resultados_afc.append(res)
        except Exception:
            continue

    resultados_afc = pd.DataFrame(resultados_afc)
    if resultados_afc.empty:
        raise RuntimeError("No se pudo ajustar ningún modelo AFC.")
    resultados_afc["cumple_pdf"] = (resultados_afc["RMSEA"] < 0.08) & (resultados_afc["CFI"] > 0.90)
    resultados_afc["cumple_srmr"] = resultados_afc["SRMR"] < 0.08
    resultados_afc = resultados_afc.sort_values(
        ["cumple_pdf", "CFI", "RMSEA", "SRMR"], ascending=[False, False, True, True]
    )
    resultados_afc.to_csv(OUT_DIR / "05_busqueda_afc_modelos.csv", index=False, encoding="utf-8-sig")

    mejor_afc = resultados_afc.iloc[0].copy()
    modelo_final = Model(mejor_afc["modelo_desc"])
    variables_finales = []
    for linea in mejor_afc["modelo_desc"].splitlines():
        variables_finales.extend([v.strip() for v in linea.split("=~")[1].split("+")])
    modelo_final.fit(datos_afc_base[variables_finales])
    estimaciones_finales = modelo_final.inspect()
    estimaciones_finales.to_csv(OUT_DIR / "06_afc_final_estimaciones.csv", index=False, encoding="utf-8-sig")

    check_final = pd.DataFrame(
        [
            {"criterio": "Varianza acumulada >= 80%", "valor": var80, "cumple": var80 >= 0.80},
            {"criterio": "RMSEA < 0.08", "valor": float(mejor_afc["RMSEA"]), "cumple": float(mejor_afc["RMSEA"]) < 0.08},
            {"criterio": "CFI > 0.90", "valor": float(mejor_afc["CFI"]), "cumple": float(mejor_afc["CFI"]) > 0.90},
        ]
    )
    check_final.to_csv(OUT_DIR / "07_check_final_pdf.csv", index=False, encoding="utf-8-sig")

    cumple_pdf = bool(var80 >= 0.80 and float(mejor_afc["RMSEA"]) < 0.08 and float(mejor_afc["CFI"]) > 0.90)
    conclusion = (
        f"El análisis principal usa remoteclip_visual_512 de v10/v5b con {x_valid.shape[0]} tiles válidos y {x_valid.shape[1]} dimensiones. "
        f"PCA/AFE retiene {m_principal} factores para alcanzar {var80:.4f} de varianza acumulada. "
        f"El mejor AFC obtiene RMSEA={float(mejor_afc['RMSEA']):.4f}, CFI={float(mejor_afc['CFI']):.4f} y SRMR={float(mejor_afc['SRMR']):.4f}. "
        + (
            "Según los criterios explícitos del PDF, hay evidencia mínima defendible de validez de constructo."
            if cumple_pdf
            else "No cumple todos los criterios explícitos del PDF; la evidencia debe reportarse como parcial."
        )
    )
    (OUT_DIR / "08_conclusion.txt").write_text(conclusion, encoding="utf-8")

    manifest = {
        "script": str(Path(__file__).resolve()),
        "input_npz": str(EMB_PATH),
        "metadata": str(METADATA_PATH),
        "input_oficial": MATRIZ_OFICIAL,
        "input_shape_original": list(x_full.shape),
        "input_shape_filtrado": list(x_valid.shape),
        "filas_embedding_cero_excluidas": int(fila_cero.sum()),
        "componentes_80_completo": int(m_full),
        "componentes_80_filtrado_principal": int(m_principal),
        "varianza_80_filtrado_principal": var80,
        "max_modelos_afc": MAX_MODELOS_AFC,
        "modelos_afc_ajustados": int(len(resultados_afc)),
        "modelo_afc_final": mejor_afc["modelo_desc"],
        "metricas_afc_final": {
            "CFI": float(mejor_afc["CFI"]),
            "RMSEA": float(mejor_afc["RMSEA"]),
            "SRMR": float(mejor_afc["SRMR"]),
            "chi2": float(mejor_afc["chi2"]),
            "DoF": float(mejor_afc["DoF"]),
        },
        "cumple_pdf": cumple_pdf,
        "nota": "Metadata y Sentinel-5P se usan para trazabilidad/interpretación, no como variables PCA.",
    }
    (OUT_DIR / "09_manifest_pca_afe_afc_v10_v5b.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(conclusion)


if __name__ == "__main__":
    main()
