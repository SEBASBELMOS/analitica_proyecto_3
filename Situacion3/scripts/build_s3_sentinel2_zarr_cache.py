from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numcodecs import Blosc
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform as rio_transform
from rasterio.windows import from_bounds, transform as window_transform
import zarr


os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("OPJ_NUM_THREADS", "2")
os.environ.setdefault("GDAL_NUM_THREADS", "2")

BASE = Path("/workspace/geovision-cali-hf")
SCENE_INVENTORY = BASE / "outputs/situacion3/rubrica/05_auditoria_fuentes_raster_sentinel2/raw_s2_scene_completeness.csv"
DEFAULT_OUT = BASE / "outputs/situacion3/rubrica/12_sentinel2_zarr_cache_cali_bbox"
BBOX_WGS84 = (-76.60, 3.30, -76.40, 3.55)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def find_band_file(scene_dir: str | Path, band: str) -> Path:
    matches = sorted(Path(scene_dir).rglob(f"*_{band}_*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No band {band} in {scene_dir}")
    return matches[0]


def find_scl_file(scene_dir: str | Path) -> Path:
    matches = sorted(Path(scene_dir).rglob("*_SCL_*.jp2"))
    if not matches:
        raise FileNotFoundError(f"No SCL in {scene_dir}")
    return matches[0]


def scene_zarr_path(out: Path, scene_id: str) -> Path:
    safe = scene_id.replace("/", "_")
    return out / "scenes" / f"{safe}.zarr"


def build_one_scene(row: dict, out: str, force: bool = False) -> dict:
    out_path = Path(out)
    scene_id = row["scene_id"]
    zpath = scene_zarr_path(out_path, scene_id)
    summary_path = zpath / ".cache_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8")) | {"skipped": True}

    zpath.parent.mkdir(parents=True, exist_ok=True)
    if zpath.exists() and force:
        import shutil
        shutil.rmtree(zpath)

    b04_path = find_band_file(row["scene_dir"], "B04")
    with rasterio.Env(GDAL_CACHEMAX=512, OPJ_NUM_THREADS="2", GDAL_NUM_THREADS="2"):
        with rasterio.open(b04_path) as ref:
            xs, ys = rio_transform("EPSG:4326", ref.crs, [BBOX_WGS84[0], BBOX_WGS84[2]], [BBOX_WGS84[1], BBOX_WGS84[3]])
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            win = from_bounds(minx, miny, maxx, maxy, transform=ref.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, ref.width, ref.height))
            height, width = int(win.height), int(win.width)
            ref_bounds = rasterio.windows.bounds(win, ref.transform)
            crop_transform = window_transform(win, ref.transform)
            crs = str(ref.crs)

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        root = zarr.open_group(str(zpath), mode="w")
        image = root.create_dataset("image", shape=(len(BANDS), height, width), chunks=(1, 512, 512), dtype="float32", compressor=compressor)
        scl_ds = root.create_dataset("scl", shape=(height, width), chunks=(512, 512), dtype="uint8", compressor=compressor)

        for bi, band in enumerate(BANDS):
            band_path = find_band_file(row["scene_dir"], band)
            with rasterio.open(band_path) as src:
                band_win = from_bounds(*ref_bounds, transform=src.transform)
                arr = src.read(1, window=band_win, out_shape=(height, width), resampling=Resampling.bilinear).astype("float32") / 10000.0
                image[bi, :, :] = arr

        with rasterio.open(find_scl_file(row["scene_dir"])) as src:
            scl_win = from_bounds(*ref_bounds, transform=src.transform)
            scl = src.read(1, window=scl_win, out_shape=(height, width), resampling=Resampling.nearest).astype("uint8")
            scl_ds[:, :] = scl

    left, bottom, right, top = array_bounds(height, width, crop_transform)
    attrs = {
        "scene_id": scene_id,
        "date": str(row["date"]),
        "mgrs_tile": str(row["mgrs_tile"]),
        "scene_dir": str(row["scene_dir"]),
        "bands": BANDS,
        "bbox_wgs84": BBOX_WGS84,
        "crs": crs,
        "transform": tuple(crop_transform),
        "bounds_projected": [float(left), float(bottom), float(right), float(top)],
        "shape_hw": [height, width],
        "created_at_utc": utc_now(),
    }
    root.attrs.update(attrs)
    summary = {
        "scene_id": scene_id,
        "date": str(row["date"]),
        "mgrs_tile": str(row["mgrs_tile"]),
        "zarr_path": str(zpath),
        "shape_image": [len(BANDS), height, width],
        "shape_scl": [height, width],
        "size_mb": round(sum(f.stat().st_size for f in zpath.rglob("*") if f.is_file()) / 1024 / 1024, 3),
        "created_at_utc": attrs["created_at_utc"],
        "skipped": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(SCENE_INVENTORY)
    inv = inv[inv["complete_12band_scl"].astype(bool)].copy().sort_values(["date", "mgrs_tile", "scene_id"])
    if args.max_scenes is not None:
        inv = inv.iloc[: args.max_scenes].copy()
    rows = inv.to_dict(orient="records")
    print(json.dumps({"stage": "start", "n_scenes": len(rows), "workers": args.workers, "out": str(args.out)}, ensure_ascii=False), flush=True)

    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one_scene, r, str(args.out), args.force) for r in rows]
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                s = fut.result()
                summaries.append(s)
                print(json.dumps({"stage": "scene_done", "done": i, "total": len(rows), **s}, ensure_ascii=False), flush=True)
            except Exception as exc:
                err = {"stage": "scene_error", "done": i, "total": len(rows), "error": f"{type(exc).__name__}: {exc}"}
                summaries.append(err)
                print(json.dumps(err, ensure_ascii=False), flush=True)

    df = pd.DataFrame(summaries)
    df.to_csv(args.out / "zarr_scene_cache_manifest.csv", index=False)
    ok = df[df.get("zarr_path", pd.Series(dtype=str)).notna()] if len(df) else df
    summary = {
        "out": str(args.out),
        "n_requested": len(rows),
        "n_cached": int(len(ok)),
        "n_errors": int(len(df) - len(ok)),
        "total_size_mb": float(ok["size_mb"].sum()) if len(ok) and "size_mb" in ok else 0.0,
        "manifest": str(args.out / "zarr_scene_cache_manifest.csv"),
    }
    (args.out / "summary_zarr_scene_cache.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
