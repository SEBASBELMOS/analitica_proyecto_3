from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLLUTANTS = ["NO2", "SO2", "O3"]
HORIZONS = [1, 3, 7]


def pivot_grid(df: pd.DataFrame, value_col: str):
    lat_vals = np.sort(df["lat"].unique())
    lon_vals = np.sort(df["lon"].unique())
    arr = df.pivot(index="lat", columns="lon", values=value_col).reindex(index=lat_vals, columns=lon_vals).to_numpy()
    extent = [float(lon_vals.min()), float(lon_vals.max()), float(lat_vals.min()), float(lat_vals.max())]
    return arr, extent


def save_heatmap(df: pd.DataFrame, value_col: str, title: str, out: Path, cmap: str, cbar_label: str):
    arr, extent = pivot_grid(df, value_col)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, aspect="auto")
    ax.scatter(df["lon"], df["lat"], s=0.05, c="none")
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_lisa(df: pd.DataFrame, title: str, out: Path):
    colors = {
        "high_high": "#d7191c",
        "low_low": "#2c7bb6",
        "high_low": "#fdae61",
        "low_high": "#abd9e9",
        "not_significant": "#d9d9d9",
        "unknown": "#969696",
    }
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for label, sub in df.groupby("lisa_cluster"):
        ax.scatter(sub["lon"], sub["lat"], s=4, c=colors.get(label, "#969696"), label=label, alpha=0.9, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="best", fontsize=7, frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_kmeans(df: pd.DataFrame, out: Path):
    palette = {
        1: "#b2182b",  # mas critico
        2: "#ef8a62",
        3: "#fddbc7",
        4: "#a6dba0",
        5: "#1b7837",  # menor riesgo relativo
    }
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    if "risk_rank" not in df.columns:
        if "mean_prediction_profile" not in df.columns:
            raise ValueError("KMeans input must include risk_rank or mean_prediction_profile")
        ranking = (
            df.groupby("cluster")["mean_prediction_profile"]
            .mean()
            .rank(ascending=False, method="dense")
            .astype(int)
            .to_dict()
        )
        df = df.copy()
        df["risk_rank"] = df["cluster"].map(ranking)
    for rank in sorted(df["risk_rank"].dropna().unique()):
        rank = int(rank)
        sub = df[df["risk_rank"] == rank]
        clusters = ", ".join(str(int(c)) for c in sorted(sub["cluster"].dropna().unique()))
        label = f"rank {rank} (cluster {clusters}, n={len(sub)})"
        ax.scatter(
            sub["lon"],
            sub["lat"],
            c=palette.get(rank, "#969696"),
            s=12,
            linewidths=0,
            alpha=0.95,
            label=label,
        )
    ax.set_title("K-Means de perfiles criticos - Situacion 3")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(title="Ranking de riesgo", loc="upper right", fontsize=7, title_fontsize=8, frameon=True)
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--lisa", type=Path, required=True)
    ap.add_argument("--kmeans", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    maps = pd.read_csv(args.maps)
    lisa = pd.read_csv(args.lisa)
    kmeans = pd.read_csv(args.kmeans)

    outputs = []
    for pollutant in POLLUTANTS:
        for horizon in HORIZONS:
            sub = maps[(maps["pollutant"] == pollutant) & (maps["horizon_days"] == horizon)].copy()
            target_date = sub["target_date"].iloc[0]
            pred_out = args.out / "prediction_corrected" / f"map_{pollutant}_t{horizon}_prediction_corrected.png"
            var_out = args.out / "kriging_variance" / f"map_{pollutant}_t{horizon}_kriging_variance.png"
            save_heatmap(
                sub,
                "prediction_corrected",
                f"{pollutant} T+{horizon} corregido - {target_date}",
                pred_out,
                "viridis",
                "Concentracion corregida",
            )
            save_heatmap(
                sub,
                "kriging_variance",
                f"{pollutant} T+{horizon} incertidumbre kriging - {target_date}",
                var_out,
                "magma",
                "Varianza kriging",
            )
            outputs.extend([str(pred_out), str(var_out)])

            lisa_sub = lisa[(lisa["pollutant"] == pollutant) & (lisa["horizon_days"] == horizon)].copy()
            lisa_out = args.out / "lisa" / f"lisa_{pollutant}_t{horizon}.png"
            save_lisa(lisa_sub, f"LISA {pollutant} T+{horizon}", lisa_out)
            outputs.append(str(lisa_out))

    kmeans_out = args.out / "kmeans" / "kmeans_critical_profiles.png"
    save_kmeans(kmeans, kmeans_out)
    outputs.append(str(kmeans_out))

    summary = {
        "maps": str(args.maps),
        "lisa": str(args.lisa),
        "kmeans": str(args.kmeans),
        "out": str(args.out),
        "n_png": len(outputs),
        "png_outputs": outputs,
    }
    (args.out / "summary_final_map_figures.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
