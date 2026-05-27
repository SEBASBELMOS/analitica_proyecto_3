from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/workspace/geovision-cali-hf")
EMB_DIR = BASE / "outputs/situacion3/rubrica/06_embeddings_tiles_estaciones_geovision_clip_sae"
OUT = BASE / "outputs/situacion3/rubrica/08_station_embedding_sequences"
OUT.mkdir(parents=True, exist_ok=True)

META_PATH = EMB_DIR / "metadata_embeddings_estaciones.csv"
EMB_PATH = EMB_DIR / "embeddings_estaciones_geovision_clip_sae_256.npz"
DAGMA_PATH = BASE / "outputs/situacion3/01_panel_dagma/dagma_sisaire_daily_long.parquet"

SEQ_LEN = 8
HORIZONS = [1, 3, 7]


def main() -> None:
    meta = pd.read_csv(META_PATH)
    meta["date_day"] = pd.to_datetime(meta["date_day"]).dt.normalize()
    emb = np.load(EMB_PATH, allow_pickle=True)
    h_256 = emb["h_256"].astype("float32")
    r_256 = emb["r_256"].astype("float32")
    pair_id = emb["pair_id"].astype(str)

    if len(meta) != len(h_256):
        raise ValueError(f"Metadata/embedding length mismatch: {len(meta)} vs {len(h_256)}")
    if not np.array_equal(meta["pair_id"].astype(str).to_numpy(), pair_id):
        raise ValueError("pair_id order mismatch between metadata and npz")

    dagma = pd.read_parquet(DAGMA_PATH)
    dagma["date"] = pd.to_datetime(dagma["date"]).dt.normalize()
    dagma = dagma.dropna(subset=["valor_mean"]).copy()
    target_lookup = {
        (r.estacion, r.contaminante, r.date): float(r.valor_mean)
        for r in dagma.itertuples(index=False)
    }

    rows = []
    x_h = []
    x_r = []
    source_indices = []

    meta_ordered = meta.reset_index().rename(columns={"index": "embedding_index"})
    for station, g in meta_ordered.groupby("estacion"):
        g = g.sort_values("date_day").reset_index(drop=True)
        for end_pos in range(SEQ_LEN - 1, len(g)):
            seq = g.iloc[end_pos - SEQ_LEN + 1:end_pos + 1]
            end_date = seq.iloc[-1]["date_day"]
            idx = seq["embedding_index"].to_numpy(dtype=int)
            for pollutant in sorted(dagma.loc[dagma["estacion"] == station, "contaminante"].unique()):
                ys = []
                ok = True
                for horizon in HORIZONS:
                    y = target_lookup.get((station, pollutant, end_date + pd.Timedelta(days=horizon)))
                    if y is None or not np.isfinite(y):
                        ok = False
                        break
                    ys.append(float(y))
                if not ok:
                    continue
                rows.append({
                    "sample_id": f"{station}|{pollutant}|{end_date.date()}",
                    "estacion": station,
                    "grid_id": seq.iloc[-1]["grid_id"],
                    "contaminante": pollutant,
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "start_date": seq.iloc[0]["date_day"].strftime("%Y-%m-%d"),
                    "n_s2_dates": int(SEQ_LEN),
                    "s2_dates": "|".join(seq["date_day"].dt.strftime("%Y-%m-%d")),
                    "target_t_plus_1": ys[0],
                    "target_t_plus_3": ys[1],
                    "target_t_plus_7": ys[2],
                })
                x_h.append(h_256[idx])
                x_r.append(r_256[idx])
                source_indices.append(idx)

    samples = pd.DataFrame(rows)
    if len(samples):
        X_h = np.stack(x_h).astype("float32")
        X_r = np.stack(x_r).astype("float32")
        source_indices_arr = np.stack(source_indices).astype("int32")
        y = samples[["target_t_plus_1", "target_t_plus_3", "target_t_plus_7"]].to_numpy(dtype="float32")
    else:
        X_h = np.zeros((0, SEQ_LEN, 256), dtype="float32")
        X_r = np.zeros((0, SEQ_LEN, 256), dtype="float32")
        source_indices_arr = np.zeros((0, SEQ_LEN), dtype="int32")
        y = np.zeros((0, len(HORIZONS)), dtype="float32")

    np.savez_compressed(
        OUT / "station_sequences_len8_embeddings_256_targets_t1_t3_t7.npz",
        X_h_256=X_h,
        X_r_256=X_r,
        y_t1_t3_t7=y,
        source_embedding_indices=source_indices_arr,
        sample_id=samples["sample_id"].astype(str).to_numpy() if len(samples) else np.array([], dtype=str),
        contaminante=samples["contaminante"].astype(str).to_numpy() if len(samples) else np.array([], dtype=str),
        estacion=samples["estacion"].astype(str).to_numpy() if len(samples) else np.array([], dtype=str),
    )
    samples.to_csv(OUT / "metadata_station_sequences_len8.csv", index=False)

    by_pollutant = samples.groupby("contaminante").size().to_dict() if len(samples) else {}
    by_station = samples.groupby("estacion").size().to_dict() if len(samples) else {}
    summary = {
        "seq_len": SEQ_LEN,
        "horizons_days": HORIZONS,
        "n_input_embeddings": int(len(meta)),
        "n_samples": int(len(samples)),
        "X_h_256_shape": list(X_h.shape),
        "X_r_256_shape": list(X_r.shape),
        "y_shape": list(y.shape),
        "samples_by_pollutant": {str(k): int(v) for k, v in by_pollutant.items()},
        "samples_by_station": {str(k): int(v) for k, v in by_station.items()},
    }
    (OUT / "summary_station_sequences_len8.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
