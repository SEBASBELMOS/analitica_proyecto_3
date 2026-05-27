from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("/workspace/geovision-cali-hf/outputs/situacion3/rubrica/07_tiles_grilla_temporal_scl_resumible")


def summarize_csv(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "accepted": 0, "rejected": 0, "cells": 0, "dates": 0}
    df = pd.read_csv(path)
    ok = df["accepted_s2_policy"].fillna(False).astype(bool) if "accepted_s2_policy" in df else pd.Series([], dtype=bool)
    return {
        "exists": True,
        "rows": int(len(df)),
        "accepted": int(ok.sum()),
        "rejected": int((~ok).sum()),
        "cells": int(df["grid_id"].nunique()) if len(df) and "grid_id" in df else 0,
        "dates": int(pd.to_datetime(df["date_day"]).nunique()) if len(df) and "date_day" in df else 0,
        "accepted_fraction": float(ok.mean()) if len(df) else None,
    }


def main() -> None:
    rows = []
    zone = BASE / "zone_1km/chunk_000_of_001/metadata_tiles_scl.csv"
    rows.append({"scope": "zone_1km", "chunk": "000_of_001", **summarize_csv(zone)})
    for i in range(8):
        p = BASE / f"full/chunk_{i:03d}_of_008/metadata_tiles_scl.csv"
        rows.append({"scope": "full", "chunk": f"{i:03d}_of_008", **summarize_csv(p)})
    df = pd.DataFrame(rows)
    out = BASE / "tile_generation_progress_summary.csv"
    df.to_csv(out, index=False)
    totals = df.groupby("scope")[["rows", "accepted", "rejected"]].sum().reset_index()
    payload = {"summary_csv": str(out), "rows": rows, "totals": totals.to_dict(orient="records")}
    (BASE / "tile_generation_progress_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
