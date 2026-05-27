from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variograms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.variograms)
    outputs = []
    for (pollutant, horizon), sub in df.groupby(["pollutant", "horizon_days"]):
        sub = sub.sort_values("distance_mean")
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(sub["distance_mean"], sub["semivariance"], marker="o", linewidth=1.5)
        for _, r in sub.iterrows():
            ax.annotate(str(int(r["n_pairs"])), (r["distance_mean"], r["semivariance"]), fontsize=6, alpha=0.7)
        ax.set_title(f"Variograma experimental residuos {pollutant} T+{int(horizon)}")
        ax.set_xlabel("Distancia espacio-temporal escalada")
        ax.set_ylabel("Semivarianza")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = args.out / f"variogram_{pollutant}_t{int(horizon)}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(out))

    summary = {"variograms": str(args.variograms), "out": str(args.out), "n_png": len(outputs), "png_outputs": outputs}
    (args.out / "summary_variogram_figures.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
