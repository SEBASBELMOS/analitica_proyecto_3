from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.transform import rowcol
from rasterio.warp import transform as rio_transform
import zarr


BASE = Path("/workspace/geovision-cali-hf")
ZARR_CACHE = BASE / "outputs/situacion3/rubrica/12_sentinel2_zarr_cache_cali_bbox"
DEFAULT_OUT = BASE / "outputs/situacion3/rubrica/16_metadata_tiles_from_zarr_cache"
PATHS = {
    "station_to_grid": BASE / "outputs/situacion3/02_grilla_s2_features/station_to_s2_grid_mapping.csv",
    "grid": BASE / "outputs/situacion3/02_grilla_s2_features/grid_cali_005deg.parquet",
    "s2_features": BASE / "outputs/situacion3/02_grilla_s2_features/s2_grid_features_with_quality.parquet",
    "zarr_manifest": ZARR_CACHE / "zarr_scene_cache_manifest.csv",
}

TILE_SIZE = 64
SCL_CLOUD_SHADOW = {3, 8, 9, 10}
SCL_VALID_VISUAL = {4, 5, 6, 7}
SCL_INVALID_EXTRA = {0, 1, 2, 11}
MAX_SCL_CLOUD_SHADOW_PCT = 30.0
OUTPUT_COLUMNS = [
    "tile_id", "scope", "radius_km", "grid_id", "lat", "lon", "date_day", "nearest_station", "estacion", "scope_distance_km",
    "accepted_s2_policy", "reject_reason", "window_inside", "raster_crs", "row_px", "col_px", "row_off", "col_off",
    "scene_id", "mgrs_tile", "zarr_path", "image_min", "image_max", "image_mean", "image_std", "ndvi_mean", "ndbi_mean", "ndwi_mean",
    "scl_cloud_shadow_pct", "scl_valid_visual_pct", "scl_invalid_extra_pct", "zero_pct", "high_reflectance_pct",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(path: Path, payload: dict) -> None:
    line = json.dumps({"time": utc_now(), **payload}, ensure_ascii=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def select_grid_cells(scope: str, radius_km: float):
    grid = pd.read_parquet(PATHS["grid"])
    stations = pd.read_csv(PATHS["station_to_grid"])
    if scope == "full":
        cells = grid.copy()
        cells["scope_distance_km"] = np.nan
        return cells
    if scope == "station":
        cells = stations[["estacion", "nearest_grid_id", "grid_lat", "grid_lon", "distance_km"]].drop_duplicates().copy()
        return cells.rename(columns={"nearest_grid_id": "grid_id", "grid_lat": "lat", "grid_lon": "lon", "distance_km": "scope_distance_km"})
    distances = []
    nearest_station = []
    for _, cell in grid.iterrows():
        d = haversine_km(cell.lat, cell.lon, stations["station_lat"].to_numpy(), stations["station_lon"].to_numpy())
        i = int(np.argmin(d))
        distances.append(float(d[i]))
        nearest_station.append(stations.iloc[i]["estacion"])
    cells = grid.copy()
    cells["scope_distance_km"] = distances
    cells["nearest_station"] = nearest_station
    return cells[cells["scope_distance_km"] <= radius_km].copy().reset_index(drop=True)


def build_pairs(cells, dates, chunk_index: int, n_chunks: int, max_pairs: int | None):
    pairs = [(cell, date) for _, cell in cells.iterrows() for date in dates]
    if n_chunks > 1:
        pairs = [p for i, p in enumerate(pairs) if i % n_chunks == chunk_index]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def open_scene(row):
    root = zarr.open_group(row.zarr_path, mode="r")
    vals = list(root.attrs["transform"])
    transform = Affine(*vals[:6]) if len(vals) == 9 else Affine(*vals)
    return root, transform, root.attrs["crs"]


def empty_row(base, reason: str):
    row = {k: np.nan for k in OUTPUT_COLUMNS}
    row.update(base)
    row.update({"accepted_s2_policy": False, "reject_reason": reason, "window_inside": False})
    return row


def tile_stats(image, scl):
    red = image[:, :, 3]
    green = image[:, :, 2]
    nir = image[:, :, 7]
    swir = image[:, :, 10]
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndbi = (swir - nir) / (swir + nir + 1e-6)
    ndwi = (green - nir) / (green + nir + 1e-6)
    return {
        "image_min": float(np.nanmin(image)),
        "image_max": float(np.nanmax(image)),
        "image_mean": float(np.nanmean(image)),
        "image_std": float(np.nanstd(image)),
        "ndvi_mean": float(np.nanmean(ndvi)),
        "ndbi_mean": float(np.nanmean(ndbi)),
        "ndwi_mean": float(np.nanmean(ndwi)),
        "scl_cloud_shadow_pct": float(np.isin(scl, list(SCL_CLOUD_SHADOW)).mean() * 100),
        "scl_valid_visual_pct": float(np.isin(scl, list(SCL_VALID_VISUAL)).mean() * 100),
        "scl_invalid_extra_pct": float(np.isin(scl, list(SCL_INVALID_EXTRA)).mean() * 100),
        "zero_pct": float((image == 0).mean() * 100),
        "high_reflectance_pct": float((image > 1.2).mean() * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["station", "zone", "full"], default="full")
    ap.add_argument("--radius-km", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    tag = args.scope if args.scope != "zone" else f"zone_{args.radius_km:g}km"
    out = args.out / tag / f"chunk_{args.chunk_index:03d}_of_{args.n_chunks:03d}"
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata_tiles_scl.csv"
    log_path = out / "generation_progress.log"

    zmanifest = pd.read_csv(PATHS["zarr_manifest"])
    zmanifest = zmanifest[zmanifest["zarr_path"].notna()].copy()
    zmanifest["date"] = pd.to_datetime(zmanifest["date"]).dt.normalize()
    zmanifest = zmanifest.sort_values(["date", "mgrs_tile", "scene_id"])
    s2_dates = pd.read_parquet(PATHS["s2_features"], columns=["date"])
    s2_dates["date"] = pd.to_datetime(s2_dates["date"]).dt.normalize()
    dates = sorted(set(s2_dates["date"].drop_duplicates()) & set(zmanifest["date"].drop_duplicates()))
    cells = select_grid_cells(args.scope, args.radius_km)
    pairs = build_pairs(cells, dates, args.chunk_index, args.n_chunks, args.max_pairs)

    done_ids = set()
    if metadata_path.exists():
        done_ids = set(pd.read_csv(metadata_path, usecols=["tile_id"])["tile_id"].astype(str))
    scene_cache = {}
    records = []
    accepted = 0
    rejected = 0
    log(log_path, {"message": "start", "scope": args.scope, "n_cells": len(cells), "n_dates": len(dates), "n_pairs_this_chunk": len(pairs), "already_done": len(done_ids)})

    for i, (cell, date) in enumerate(pairs, start=1):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        tile_id = f"s3meta_{tag}_{cell.grid_id}_{date_str}".replace(".", "p")
        if tile_id in done_ids:
            continue
        base = {
            "tile_id": tile_id, "scope": args.scope, "radius_km": args.radius_km if args.scope == "zone" else np.nan,
            "grid_id": cell.grid_id, "lat": float(cell.lat), "lon": float(cell.lon), "date_day": date_str,
            "nearest_station": getattr(cell, "nearest_station", np.nan), "estacion": getattr(cell, "estacion", np.nan),
            "scope_distance_km": getattr(cell, "scope_distance_km", np.nan),
        }
        scenes = zmanifest[zmanifest["date"] == date]
        wrote = False
        best_reject = "no_scene_window_inside"
        best_reject_row = None
        for scene in scenes.itertuples(index=False):
            if scene.scene_id not in scene_cache:
                scene_cache[scene.scene_id] = open_scene(scene)
            root, transform, crs = scene_cache[scene.scene_id]
            xs, ys = rio_transform("EPSG:4326", crs, [float(cell.lon)], [float(cell.lat)])
            row_px, col_px = rowcol(transform, xs[0], ys[0])
            row_off = int(row_px - TILE_SIZE // 2)
            col_off = int(col_px - TILE_SIZE // 2)
            height, width = root["scl"].shape
            if not (row_off >= 0 and col_off >= 0 and row_off + TILE_SIZE <= height and col_off + TILE_SIZE <= width):
                continue
            image = np.moveaxis(np.asarray(root["image"][:, row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE], dtype="float32"), 0, -1)
            scl = np.asarray(root["scl"][row_off:row_off + TILE_SIZE, col_off:col_off + TILE_SIZE], dtype="uint8")
            stats = tile_stats(image, scl)
            shape_ok = image.shape == (TILE_SIZE, TILE_SIZE, 12) and scl.shape == (TILE_SIZE, TILE_SIZE)
            finite = bool(np.isfinite(image).all())
            accepted_policy = bool(shape_ok and finite and stats["zero_pct"] < 99 and stats["image_std"] > 1e-6 and stats["scl_cloud_shadow_pct"] <= MAX_SCL_CLOUD_SHADOW_PCT)
            reasons = []
            if not shape_ok: reasons.append("shape_not_ok")
            if not finite: reasons.append("non_finite")
            if stats["zero_pct"] >= 99 or stats["image_std"] <= 1e-6: reasons.append("zero_pct_ge_99|image_std_le_1e-6")
            if stats["scl_cloud_shadow_pct"] > MAX_SCL_CLOUD_SHADOW_PCT: reasons.append("scl_cloud_shadow_pct_gt_30")
            row = {k: np.nan for k in OUTPUT_COLUMNS}
            row.update(base)
            row.update(stats)
            row.update({
                "accepted_s2_policy": accepted_policy, "reject_reason": "accepted" if accepted_policy else "|".join(reasons),
                "window_inside": True, "raster_crs": crs, "row_px": int(row_px), "col_px": int(col_px), "row_off": row_off, "col_off": col_off,
                "scene_id": scene.scene_id, "mgrs_tile": scene.mgrs_tile, "zarr_path": scene.zarr_path,
            })
            if accepted_policy:
                records.append(row)
                accepted += 1
                wrote = True
                break
            best_reject = row["reject_reason"] or best_reject
            best_reject_row = row
        if not wrote:
            rejected += 1
            if not any(r["tile_id"] == tile_id for r in records):
                records.append(best_reject_row if best_reject_row is not None else empty_row(base, best_reject))
        if len(records) >= 5000:
            pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(metadata_path, mode="a", header=not metadata_path.exists(), index=False)
            records.clear()
        if i == 1 or i % 10000 == 0:
            log(log_path, {"message": "progress", "i": i, "n": len(pairs), "accepted_new": accepted, "rejected_new": rejected})
    if records:
        pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(metadata_path, mode="a", header=not metadata_path.exists(), index=False)
    meta = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    ok = meta["accepted_s2_policy"].fillna(False).astype(bool)
    summary = {"scope": args.scope, "chunk_index": args.chunk_index, "n_chunks": args.n_chunks, "n_pairs_this_chunk": len(pairs), "n_records_total_chunk": int(len(meta)), "n_accepted_total_chunk": int(ok.sum()), "n_rejected_total_chunk": int((~ok).sum())}
    (out / "summary_metadata_tiles_scl.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(log_path, {"message": "done", **summary})


if __name__ == "__main__":
    main()
