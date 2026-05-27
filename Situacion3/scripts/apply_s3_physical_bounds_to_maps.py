from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    maps = pd.read_csv(args.maps)

    negative_prediction = maps["prediction_corrected"] < 0
    negative_variance = maps["kriging_variance"] < 0
    n_negative_prediction = int(negative_prediction.sum())
    n_negative_variance = int(negative_variance.sum())

    if "prediction_convlstm" in maps.columns:
        maps.loc[negative_prediction, "prediction_corrected"] = maps.loc[negative_prediction, "prediction_convlstm"]
        prediction_rule = "fallback_prediction_convlstm_when_prediction_corrected_negative"
    else:
        maps["prediction_corrected"] = maps["prediction_corrected"].clip(lower=0)
        prediction_rule = "clip_prediction_corrected_to_zero"

    maps["prediction_corrected"] = maps["prediction_corrected"].clip(lower=0)
    maps["kriging_variance"] = maps["kriging_variance"].clip(lower=0)

    out_csv = args.out / "st_kriging_corrected_maps_long.csv"
    maps.to_csv(out_csv, index=False)

    summary = {
        "source_maps": str(args.maps),
        "output_maps": str(out_csv),
        "prediction_rule": prediction_rule,
        "variance_rule": "clip_kriging_variance_to_zero",
        "n_rows": int(len(maps)),
        "n_negative_prediction_corrected_input": n_negative_prediction,
        "n_negative_kriging_variance_input": n_negative_variance,
        "n_negative_prediction_corrected_output": int((maps["prediction_corrected"] < 0).sum()),
        "n_negative_kriging_variance_output": int((maps["kriging_variance"] < 0).sum()),
        "n_zero_prediction_corrected_output": int((maps["prediction_corrected"] == 0).sum()),
        "min_prediction_corrected_output": float(maps["prediction_corrected"].min()),
        "min_kriging_variance_output": float(maps["kriging_variance"].min()),
    }
    (args.out / "summary_physical_bounds_maps.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
