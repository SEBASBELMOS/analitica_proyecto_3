from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


BASE = Path("/workspace/geovision-cali-hf")
DAGMA_PATH = BASE / "outputs/situacion3/01_panel_dagma/dagma_sisaire_daily_long.parquet"
STATION_GRID_PATH = BASE / "outputs/situacion3/02_grilla_s2_features/station_to_s2_grid_mapping.csv"

SEQ_LEN = 8
HORIZONS = [1, 3, 7]
POLLUTANTS = ["NO2", "SO2", "O3"]
POLLUTANT_TO_IDX = {p: i for i, p in enumerate(POLLUTANTS)}


def grid_rc(grid_id: str) -> tuple[int, int]:
    m = re.fullmatch(r"g_(\d{3})_(\d{3})", str(grid_id))
    if not m:
        raise ValueError(f"Invalid grid_id: {grid_id}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, required=True, help="NPZ or Zarr group with h_256/r_256 embeddings")
    ap.add_argument("--metadata", type=Path, required=True, help="metadata_embeddings.csv aligned to embeddings")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedding-key", default="h_256", choices=["h_256", "r_256"])
    ap.add_argument("--min-input-coverage", type=float, default=0.25, help="Minimum fraction of cell-date inputs present per sample")
    ap.add_argument("--output-format", choices=["npz", "zarr"], default="npz")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata)
    meta["date_day"] = pd.to_datetime(meta["date_day"]).dt.normalize()
    if args.embeddings.suffix == ".npz":
        emb_npz = np.load(args.embeddings, allow_pickle=True)
        emb = emb_npz[args.embedding_key].astype("float32")
    else:
        emb_root = zarr.open_group(str(args.embeddings), mode="r")
        emb = emb_root[args.embedding_key]
    if len(meta) != len(emb):
        raise ValueError(f"metadata/embedding length mismatch: {len(meta)} vs {len(emb)}")

    rc = meta["grid_id"].map(grid_rc)
    meta["grid_r"] = [x[0] for x in rc]
    meta["grid_c"] = [x[1] for x in rc]
    r_min, r_max = int(meta["grid_r"].min()), int(meta["grid_r"].max())
    c_min, c_max = int(meta["grid_c"].min()), int(meta["grid_c"].max())
    h = r_max - r_min + 1
    w = c_max - c_min + 1

    grid_ids = sorted(meta["grid_id"].unique(), key=grid_rc)
    grid_to_hw = {}
    for gid in grid_ids:
        r, c = grid_rc(gid)
        grid_to_hw[gid] = (r - r_min, c - c_min)

    lookup = {}
    for i, row in meta.reset_index().iterrows():
        lookup[(row.grid_id, row.date_day)] = int(row["index"])

    dates = sorted(meta["date_day"].unique())
    station_map = pd.read_csv(STATION_GRID_PATH)
    station_map = station_map.rename(columns={"nearest_grid_id": "grid_id"})
    station_map = station_map[station_map["grid_id"].isin(grid_to_hw)].copy()

    dagma = pd.read_parquet(DAGMA_PATH)
    dagma["date"] = pd.to_datetime(dagma["date"]).dt.normalize()
    dagma = dagma.dropna(subset=["valor_mean"]).copy()
    target_lookup = {
        (r.estacion, r.contaminante, r.date): float(r.valor_mean)
        for r in dagma.itertuples(index=False)
    }

    rows = []
    sample_payloads = []
    for end_pos in range(SEQ_LEN - 1, len(dates)):
        seq_dates = dates[end_pos - SEQ_LEN + 1 : end_pos + 1]
        end_date = seq_dates[-1]
        X = np.zeros((SEQ_LEN, 256, h, w), dtype="float32")
        input_mask = np.zeros((SEQ_LEN, 1, h, w), dtype="float32")
        present = 0
        for t, d in enumerate(seq_dates):
            for gid, (rr, cc) in grid_to_hw.items():
                idx = lookup.get((gid, d))
                if idx is None:
                    continue
                X[t, :, rr, cc] = emb[idx]
                input_mask[t, 0, rr, cc] = 1.0
                present += 1
        coverage = present / float(SEQ_LEN * len(grid_to_hw)) if grid_to_hw else 0.0
        if coverage < args.min_input_coverage:
            continue

        y = np.zeros((len(HORIZONS), len(POLLUTANTS), h, w), dtype="float32")
        y_mask = np.zeros_like(y, dtype="float32")
        target_count = 0
        for s in station_map.itertuples(index=False):
            rr, cc = grid_to_hw[s.grid_id]
            for p in POLLUTANTS:
                p_idx = POLLUTANT_TO_IDX[p]
                for h_idx, horizon in enumerate(HORIZONS):
                    value = target_lookup.get((s.estacion, p, end_date + pd.Timedelta(days=horizon)))
                    if value is None or not np.isfinite(value):
                        continue
                    y[h_idx, p_idx, rr, cc] = value
                    y_mask[h_idx, p_idx, rr, cc] = 1.0
                    target_count += 1
        if target_count == 0:
            continue
        sample_id = f"seq_{pd.Timestamp(seq_dates[0]).date()}_{pd.Timestamp(end_date).date()}"
        sample_payloads.append((X, input_mask, y, y_mask))
        rows.append({
            "sample_id": sample_id,
            "start_date": pd.Timestamp(seq_dates[0]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
            "input_coverage": coverage,
            "n_input_present": int(present),
            "n_input_possible": int(SEQ_LEN * len(grid_to_hw)),
            "n_targets": int(target_count),
        })

    samples = pd.DataFrame(rows)
    n_samples = len(sample_payloads)
    x_shape = [n_samples, SEQ_LEN, 256, h, w]
    input_mask_shape = [n_samples, SEQ_LEN, 1, h, w]
    y_shape = [n_samples, len(HORIZONS), len(POLLUTANTS), h, w]
    y_mask_shape = y_shape
    output_path = args.out / f"spatial_embedding_tensors_len8_{args.embedding_key}_targets_t1_t3_t7.{args.output_format}"
    total_targets = 0

    if args.output_format == "npz":
        if sample_payloads:
            X_arr = np.stack([p[0] for p in sample_payloads]).astype("float32")
            input_mask_arr = np.stack([p[1] for p in sample_payloads]).astype("float32")
            y_arr = np.stack([p[2] for p in sample_payloads]).astype("float32")
            y_mask_arr = np.stack([p[3] for p in sample_payloads]).astype("float32")
        else:
            X_arr = np.zeros(tuple(x_shape), dtype="float32")
            input_mask_arr = np.zeros(tuple(input_mask_shape), dtype="float32")
            y_arr = np.zeros(tuple(y_shape), dtype="float32")
            y_mask_arr = np.zeros_like(y_arr)
        total_targets = int(y_mask_arr.sum())
        np.savez_compressed(
            output_path,
            X_h_256=X_arr,
            input_mask=input_mask_arr,
            y_t1_t3_t7=y_arr,
            y_mask=y_mask_arr,
            sample_id=samples["sample_id"].astype(str).to_numpy() if len(samples) else np.array([], dtype=str),
            grid_id=np.array(grid_ids, dtype=str),
            pollutants=np.array(POLLUTANTS, dtype=str),
            horizons_days=np.array(HORIZONS, dtype="int16"),
            row_min=np.array([r_min], dtype="int16"),
            col_min=np.array([c_min], dtype="int16"),
        )
    else:
        root = zarr.open_group(str(output_path), mode="w")
        chunks_x = (1, SEQ_LEN, 256, min(h, 16), min(w, 16))
        chunks_mask = (1, SEQ_LEN, 1, min(h, 16), min(w, 16))
        chunks_y = (1, len(HORIZONS), len(POLLUTANTS), min(h, 16), min(w, 16))
        x_arr = root.create_dataset("X_h_256", shape=tuple(x_shape), chunks=chunks_x, dtype="float32", compressor=None)
        input_mask_arr = root.create_dataset("input_mask", shape=tuple(input_mask_shape), chunks=chunks_mask, dtype="float32", compressor=None)
        y_arr = root.create_dataset("y_t1_t3_t7", shape=tuple(y_shape), chunks=chunks_y, dtype="float32", compressor=None)
        y_mask_arr = root.create_dataset("y_mask", shape=tuple(y_mask_shape), chunks=chunks_y, dtype="float32", compressor=None)
        root.create_dataset("grid_id", data=np.array(grid_ids, dtype=str), dtype=str)
        root.create_dataset("pollutants", data=np.array(POLLUTANTS, dtype=str), dtype=str)
        root.create_dataset("horizons_days", data=np.array(HORIZONS, dtype="int16"))
        root.attrs.update({"row_min": int(r_min), "col_min": int(c_min), "embedding_key": args.embedding_key})
        for i, (X, input_mask, y, y_mask) in enumerate(sample_payloads):
            x_arr[i] = X
            input_mask_arr[i] = input_mask
            y_arr[i] = y
            y_mask_arr[i] = y_mask
            total_targets += int(y_mask.sum())

    samples.to_csv(args.out / "metadata_spatial_embedding_tensors_len8.csv", index=False)
    summary = {
        "embeddings": str(args.embeddings),
        "metadata": str(args.metadata),
        "output_format": args.output_format,
        "output_path": str(output_path),
        "embedding_key": args.embedding_key,
        "seq_len": SEQ_LEN,
        "horizons_days": HORIZONS,
        "pollutants": POLLUTANTS,
        "n_embedding_rows": int(len(meta)),
        "n_grid_cells_with_embeddings": int(len(grid_ids)),
        "grid_shape_hw": [int(h), int(w)],
        "grid_row_range": [int(r_min), int(r_max)],
        "grid_col_range": [int(c_min), int(c_max)],
        "n_station_cells_inside_extent": int(len(station_map)),
        "min_input_coverage": float(args.min_input_coverage),
        "n_samples": int(len(samples)),
        "X_shape": x_shape,
        "input_mask_shape": input_mask_shape,
        "y_shape": y_shape,
        "y_mask_shape": y_mask_shape,
        "total_targets": int(total_targets),
    }
    (args.out / "summary_spatial_embedding_tensors_len8.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
