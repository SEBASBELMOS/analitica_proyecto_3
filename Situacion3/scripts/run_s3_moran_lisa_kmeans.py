from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


POLLUTANTS = ["NO2", "SO2", "O3"]
HORIZONS = [1, 3, 7]


def lisa_label(q: int, significant: bool) -> str:
    if not significant:
        return "not_significant"
    return {1: "high_high", 2: "low_high", 3: "low_low", 4: "high_low"}.get(int(q), "unknown")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-neighbors", type=int, default=8)
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k-clusters", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    maps = pd.read_csv(args.maps)
    maps = maps.dropna(subset=["lon", "lat", "prediction_corrected", "kriging_variance"]).copy()
    grid = maps[["grid_id", "lat", "lon"]].drop_duplicates("grid_id").sort_values("grid_id").reset_index(drop=True)
    coords = grid[["lon", "lat"]].to_numpy(float)
    k = min(args.k_neighbors, max(1, len(grid) - 1))
    weights = KNN.from_array(coords, k=k)
    weights.transform = "r"

    moran_rows = []
    lisa_rows = []
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            sub = maps[(maps["pollutant"] == pollutant) & (maps["horizon_days"] == horizon)]
            values = grid[["grid_id", "lat", "lon"]].merge(
                sub[["grid_id", "prediction_corrected", "kriging_variance"]], on="grid_id", how="left"
            )
            y = values["prediction_corrected"].to_numpy(float)
            if np.nanstd(y) == 0 or np.isnan(y).any():
                moran_rows.append({"pollutant": pollutant, "horizon_days": horizon, "moran_i": np.nan, "p_sim": np.nan, "z_sim": np.nan, "n": int(len(y)), "status": "invalid_values"})
                continue
            mi = Moran(y, weights, permutations=args.permutations)
            moran_rows.append({
                "pollutant": pollutant,
                "horizon_days": int(horizon),
                "moran_i": float(mi.I),
                "expected_i": float(mi.EI),
                "p_sim": float(mi.p_sim),
                "z_sim": float(mi.z_sim),
                "n": int(len(y)),
                "status": "ok",
            })
            local = Moran_Local(y, weights, permutations=args.permutations)
            for i, r in values.iterrows():
                sig = bool(local.p_sim[i] <= args.alpha)
                lisa_rows.append({
                    "pollutant": pollutant,
                    "horizon_days": int(horizon),
                    "grid_id": r["grid_id"],
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "prediction_corrected": float(r["prediction_corrected"]),
                    "kriging_variance": float(r["kriging_variance"]),
                    "local_i": float(local.Is[i]),
                    "p_sim": float(local.p_sim[i]),
                    "quadrant": int(local.q[i]),
                    "significant": sig,
                    "lisa_cluster": lisa_label(int(local.q[i]), sig),
                })

    moran_df = pd.DataFrame(moran_rows)
    lisa_df = pd.DataFrame(lisa_rows)
    moran_df.to_csv(args.out / "moran_global_by_pollutant_horizon.csv", index=False)
    lisa_df.to_csv(args.out / "lisa_local_clusters_long.csv", index=False)

    wide = grid.copy()
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            sub = maps[(maps["pollutant"] == pollutant) & (maps["horizon_days"] == horizon)][["grid_id", "prediction_corrected", "kriging_variance"]].copy()
            sub = sub.rename(columns={
                "prediction_corrected": f"{pollutant}_t{horizon}_pred",
                "kriging_variance": f"{pollutant}_t{horizon}_var",
            })
            wide = wide.merge(sub, on="grid_id", how="left")
    feature_cols = [c for c in wide.columns if c.endswith("_pred") or c.endswith("_var")]
    X = wide[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.k_clusters, n_init=50, random_state=42)
    wide["cluster"] = km.fit_predict(Xs)

    pred_cols = [c for c in feature_cols if c.endswith("_pred")]
    wide["mean_prediction_profile"] = wide[pred_cols].mean(axis=1)
    wide["max_prediction_profile"] = wide[pred_cols].max(axis=1)
    wide["mean_uncertainty_profile"] = wide[[c for c in feature_cols if c.endswith("_var")]].mean(axis=1)
    cluster_summary = wide.groupby("cluster").agg(
        n_cells=("grid_id", "size"),
        mean_prediction_profile=("mean_prediction_profile", "mean"),
        max_prediction_profile=("max_prediction_profile", "mean"),
        mean_uncertainty_profile=("mean_uncertainty_profile", "mean"),
        lat_mean=("lat", "mean"),
        lon_mean=("lon", "mean"),
    ).reset_index()
    cluster_summary["risk_rank"] = cluster_summary["mean_prediction_profile"].rank(ascending=False, method="dense").astype(int)
    wide.to_csv(args.out / "kmeans_critical_profiles_by_grid.csv", index=False)
    cluster_summary.to_csv(args.out / "kmeans_cluster_summary.csv", index=False)

    gdf = gpd.GeoDataFrame(wide, geometry=[Point(xy) for xy in zip(wide["lon"], wide["lat"])], crs="EPSG:4326")
    gdf.to_file(args.out / "kmeans_critical_profiles.geojson", driver="GeoJSON")
    lisa_gdf = gpd.GeoDataFrame(lisa_df, geometry=[Point(xy) for xy in zip(lisa_df["lon"], lisa_df["lat"])], crs="EPSG:4326")
    lisa_gdf.to_file(args.out / "lisa_local_clusters.geojson", driver="GeoJSON")

    summary = {
        "maps": str(args.maps),
        "n_grid_cells": int(len(grid)),
        "k_neighbors": int(k),
        "permutations": int(args.permutations),
        "alpha": float(args.alpha),
        "k_clusters": int(args.k_clusters),
        "moran_csv": str(args.out / "moran_global_by_pollutant_horizon.csv"),
        "lisa_csv": str(args.out / "lisa_local_clusters_long.csv"),
        "kmeans_csv": str(args.out / "kmeans_critical_profiles_by_grid.csv"),
        "cluster_summary_csv": str(args.out / "kmeans_cluster_summary.csv"),
        "geojson_outputs": [str(args.out / "kmeans_critical_profiles.geojson"), str(args.out / "lisa_local_clusters.geojson")],
        "moran_significant_count": int((moran_df["p_sim"] <= args.alpha).sum()),
        "lisa_significant_count": int(lisa_df["significant"].sum()),
    }
    (args.out / "summary_moran_lisa_kmeans.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
