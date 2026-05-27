from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform
from rasterio.windows import Window, from_bounds


os.environ.setdefault("GDAL_CACHEMAX", "64")
os.environ.setdefault("OPJ_NUM_THREADS", "1")
os.environ.setdefault("GDAL_NUM_THREADS", "1")

BASE = Path("/workspace/geovision-cali-hf")
RAW = BASE / "data/sentinel2-sentinel-5p/sentinel2p-raw"
DEFAULT_OUT = BASE / "outputs/situacion3/rubrica/07_tiles_grilla_temporal_scl_resumible"

PATHS = {
    "station_to_grid": BASE / "outputs/situacion3/02_grilla_s2_features/station_to_s2_grid_mapping.csv",
    "grid": BASE / "outputs/situacion3/02_grilla_s2_features/grid_cali_005deg.parquet",
    "s2_features": BASE / "outputs/situacion3/02_grilla_s2_features/s2_grid_features_with_quality.parquet",
    "scene_inventory": BASE / "outputs/situacion3/rubrica/05_auditoria_fuentes_raster_sentinel2/raw_s2_scene_completeness.csv",
}

TILE_SIZE = 64
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
SCL_CLOUD_SHADOW = {3, 8, 9, 10}
SCL_VALID_VISUAL = {4, 5, 6, 7}
SCL_INVALID_EXTRA = {0, 1, 2, 11}
MAX_SCL_CLOUD_SHADOW_PCT = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(path: Path, payload: dict) -> None:
    line = json.dumps({"time": utc_now(), **payload}, ensure_ascii=False)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def find_band_file(scene_dir, band):
    matches = sorted(Path(scene_dir).rglob(f"*_{band}_*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No band {band} in {scene_dir}")
    return matches[0]


def find_scl_file(scene_dir):
    matches = sorted(Path(scene_dir).rglob("*_SCL_*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No SCL in {scene_dir}")
    return matches[0]


def inspect_point(scene_row, lon, lat):
    band_path = find_band_file(scene_row.scene_dir, "B04")
    try:
        with rasterio.Env(GDAL_CACHEMAX=64, OPJ_NUM_THREADS="1", GDAL_NUM_THREADS="1"):
            with rasterio.open(band_path) as src:
                xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
                row_px, col_px = src.index(xs[0], ys[0])
                row_off = int(row_px - TILE_SIZE // 2)
                col_off = int(col_px - TILE_SIZE // 2)
                inside = row_off >= 0 and col_off >= 0 and row_off + TILE_SIZE <= src.height and col_off + TILE_SIZE <= src.width
                return {
                    "window_inside": bool(inside),
                    "reason": "ok" if inside else "outside_or_edge",
                    "raster_crs": str(src.crs),
                    "raster_width": int(src.width),
                    "raster_height": int(src.height),
                    "row_px": int(row_px),
                    "col_px": int(col_px),
                    "row_off": row_off,
                    "col_off": col_off,
                }
    except Exception as exc:
        return {"window_inside": False, "reason": f"{type(exc).__name__}: {exc}"}


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


def read_tile(scene_row, row_off, col_off):
    scene_dir = Path(scene_row.scene_dir)
    b04_path = find_band_file(scene_dir, "B04")
    ref_window = Window(col_off=int(col_off), row_off=int(row_off), width=TILE_SIZE, height=TILE_SIZE)
    with rasterio.open(b04_path) as ref_src:
        ref_bounds = rasterio.windows.bounds(ref_window, ref_src.transform)
        ref_crs = ref_src.crs
    arrays = []
    for band in BANDS:
        band_path = find_band_file(scene_dir, band)
        with rasterio.open(band_path) as src:
            if src.crs != ref_crs:
                raise ValueError(f"CRS mismatch {band}: {src.crs} != {ref_crs}")
            win = from_bounds(*ref_bounds, transform=src.transform)
            arr = src.read(1, window=win, out_shape=(TILE_SIZE, TILE_SIZE), resampling=Resampling.bilinear).astype("float32") / 10000.0
            arrays.append(arr)
    image = np.stack(arrays, axis=-1).astype("float32")
    scl_path = find_scl_file(scene_dir)
    with rasterio.open(scl_path) as src:
        win = from_bounds(*ref_bounds, transform=src.transform)
        scl = src.read(1, window=win, out_shape=(TILE_SIZE, TILE_SIZE), resampling=Resampling.nearest).astype("uint8")
    return image, scl, tile_stats(image, scl)


def select_grid_cells(scope: str, radius_km: float):
    grid = pd.read_parquet(PATHS["grid"])
    stations = pd.read_csv(PATHS["station_to_grid"])
    if scope == "full":
        cells = grid.copy()
        cells["scope_distance_km"] = np.nan
        return cells
    if scope == "station":
        station_cells = stations[["estacion", "nearest_grid_id", "grid_lat", "grid_lon", "distance_km"]].drop_duplicates().copy()
        station_cells = station_cells.rename(columns={"nearest_grid_id": "grid_id", "grid_lat": "lat", "grid_lon": "lon", "distance_km": "scope_distance_km"})
        return station_cells[["grid_id", "lat", "lon", "estacion", "scope_distance_km"]]
    if scope != "zone":
        raise ValueError(f"Unsupported scope: {scope}")
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


def build_pairs(cells, dates, scene_inventory, chunk_index: int, n_chunks: int, max_pairs: int | None):
    pairs = []
    for _, cell in cells.iterrows():
        for date in dates:
            pairs.append((cell, date))
    if n_chunks > 1:
        pairs = [p for i, p in enumerate(pairs) if i % n_chunks == chunk_index]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["station", "zone", "full"], default="zone")
    ap.add_argument("--radius-km", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunk-index", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    tag = args.scope if args.scope != "zone" else f"zone_{args.radius_km:g}km"
    out = args.out / tag / f"chunk_{args.chunk_index:03d}_of_{args.n_chunks:03d}"
    img_dir = out / "images"
    scl_dir = out / "scl"
    img_dir.mkdir(parents=True, exist_ok=True)
    scl_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata_tiles_scl.csv"
    log_path = out / "generation_progress.log"

    s2_dates = pd.read_parquet(PATHS["s2_features"], columns=["date"])
    s2_dates["date"] = pd.to_datetime(s2_dates["date"]).dt.normalize()
    dates = sorted(s2_dates["date"].drop_duplicates())
    scene_inventory = pd.read_csv(PATHS["scene_inventory"])
    scene_inventory = scene_inventory[scene_inventory["complete_12band_scl"].astype(bool)].copy()
    scene_inventory["date"] = pd.to_datetime(scene_inventory["date"]).dt.normalize()
    cells = select_grid_cells(args.scope, args.radius_km)
    pairs = build_pairs(cells, dates, scene_inventory, args.chunk_index, args.n_chunks, args.max_pairs)

    done_ids = set()
    if metadata_path.exists():
        old = pd.read_csv(metadata_path, usecols=["tile_id"])
        done_ids = set(old["tile_id"].astype(str))

    log(log_path, {"message": "start", "scope": args.scope, "radius_km": args.radius_km, "n_cells": len(cells), "n_dates": len(dates), "n_pairs_this_chunk": len(pairs), "already_done": len(done_ids)})
    records = []
    accepted = 0
    rejected = 0

    for i, (cell, date) in enumerate(pairs, start=1):
        date_str = date.strftime("%Y-%m-%d")
        tile_id = f"s3_{tag}_{cell.grid_id}_{date_str}".replace(".", "p")
        if tile_id in done_ids:
            continue

        scenes = scene_inventory[scene_inventory["date"] == date]
        base = {
            "tile_id": tile_id,
            "scope": args.scope,
            "radius_km": args.radius_km if args.scope == "zone" else np.nan,
            "grid_id": cell.grid_id,
            "lat": float(cell.lat),
            "lon": float(cell.lon),
            "date_day": date_str,
            "nearest_station": getattr(cell, "nearest_station", None),
            "estacion": getattr(cell, "estacion", None),
            "scope_distance_km": getattr(cell, "scope_distance_km", np.nan),
        }
        best_reject = "no_scene_window_inside"
        wrote = False
        for _, scene in scenes.iterrows():
            insp = inspect_point(scene, cell.lon, cell.lat)
            if not insp.get("window_inside", False):
                continue
            row = {**base, **insp, "scene_dir": scene.scene_dir, "mgrs_tile": scene.mgrs_tile}
            try:
                image, scl, stats = read_tile(scene, insp["row_off"], insp["col_off"])
                row.update(stats)
                shape_ok = image.shape == (TILE_SIZE, TILE_SIZE, 12) and scl.shape == (TILE_SIZE, TILE_SIZE)
                finite = bool(np.isfinite(image).all())
                accepted_policy = bool(shape_ok and finite and row["zero_pct"] < 99 and row["image_std"] > 1e-6 and row["scl_cloud_shadow_pct"] <= MAX_SCL_CLOUD_SHADOW_PCT)
                if accepted_policy:
                    image_path = img_dir / f"{tile_id}.npy"
                    scl_path = scl_dir / f"{tile_id}_scl.npy"
                    np.save(image_path, image)
                    np.save(scl_path, scl)
                    row.update({"accepted_s2_policy": True, "reject_reason": "accepted", "image_path": str(image_path), "scl_path": str(scl_path)})
                    accepted += 1
                    wrote = True
                    records.append(row)
                    break
                reasons = []
                if not shape_ok:
                    reasons.append("shape_not_ok")
                if not finite:
                    reasons.append("non_finite")
                if row["zero_pct"] >= 99 or row["image_std"] <= 1e-6:
                    reasons.append("zero_pct_ge_99|image_std_le_1e-6")
                if row["scl_cloud_shadow_pct"] > MAX_SCL_CLOUD_SHADOW_PCT:
                    reasons.append("scl_cloud_shadow_pct_gt_30")
                row.update({"accepted_s2_policy": False, "reject_reason": "|".join(reasons), "image_path": None, "scl_path": None})
                best_reject = row["reject_reason"] or best_reject
            except Exception as exc:
                row.update({"accepted_s2_policy": False, "reject_reason": f"read_error:{type(exc).__name__}", "error_message": str(exc), "image_path": None, "scl_path": None})
                best_reject = row["reject_reason"]
        if not wrote:
            rejected += 1
            if not any(r["tile_id"] == tile_id for r in records):
                records.append({**base, "accepted_s2_policy": False, "reject_reason": best_reject, "image_path": None, "scl_path": None})
        if len(records) >= 200:
            pd.DataFrame(records).to_csv(metadata_path, mode="a", header=not metadata_path.exists(), index=False)
            records.clear()
        if i == 1 or i % 500 == 0:
            log(log_path, {"message": "progress", "i": i, "n": len(pairs), "accepted_new": accepted, "rejected_new": rejected})

    if records:
        pd.DataFrame(records).to_csv(metadata_path, mode="a", header=not metadata_path.exists(), index=False)

    meta = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    summary = {
        "scope": args.scope,
        "radius_km": args.radius_km if args.scope == "zone" else None,
        "chunk_index": args.chunk_index,
        "n_chunks": args.n_chunks,
        "n_cells_scope": int(len(cells)),
        "n_dates": int(len(dates)),
        "n_pairs_this_chunk": int(len(pairs)),
        "n_records_total_chunk": int(len(meta)),
        "n_accepted_total_chunk": int(meta["accepted_s2_policy"].fillna(False).astype(bool).sum()) if len(meta) else 0,
        "n_rejected_total_chunk": int((~meta["accepted_s2_policy"].fillna(False).astype(bool)).sum()) if len(meta) else 0,
    }
    (out / "summary_tiles_scl.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    log(log_path, {"message": "done", **summary})


if __name__ == "__main__":
    main()
