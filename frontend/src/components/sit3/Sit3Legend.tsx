// Leyenda flotante sobre el mapa segun la capa activa.
import type { Sit3Cluster, Sit3Layer, Sit3MapResponse } from "../../types/situacion3";
import { LISA_COLORS, LISA_LABELS, legendStops } from "./colors";

interface Props {
  layer: Sit3Layer;
  pollutant: string;
  mapData: Sit3MapResponse | null;
  clusters: Sit3Cluster[] | null;
}

function percentile(values: number[], p: number): number {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export default function Sit3Legend({ layer, pollutant, mapData, clusters }: Props) {
  if (!mapData) return null;

  if (layer === "prediction" || layer === "uncertainty") {
    const arr = layer === "uncertainty"
      ? mapData.cells.map((c) => c.uncertainty_sigma)
      : mapData.cells.map((c) => c.prediction);
    const rawMin = Math.min(...arr);
    const vmin = layer === "uncertainty" ? percentile(arr, 0.02) : Math.max(0, rawMin);
    const vmax = layer === "uncertainty" ? percentile(arr, 0.98) : Math.max(...arr);
    const stops = legendStops(layer === "uncertainty" ? "uncertainty" : "prediction");
    const gradient = `linear-gradient(to right, ${stops.join(", ")})`;
    const title =
      layer === "prediction"
        ? `${pollutant} (ug/m³) - prediccion corregida`
        : `${pollutant} - incertidumbre (sigma)`;
    return (
      <div className="absolute bottom-4 left-4 z-[1000] bg-white/95 backdrop-blur px-3 py-2 rounded-lg shadow-lg text-xs border border-slate-200">
        <div className="font-bold mb-1.5 text-slate-800">{title}</div>
        <div
          className="w-44 h-3 rounded border border-slate-300"
          style={{ background: gradient }}
        ></div>
        <div className="flex justify-between mt-1 text-[10px] font-mono text-slate-600">
          <span>{layer === "uncertainty" ? `p2 ${vmin.toFixed(1)}` : vmin.toFixed(1)}</span>
          <span>{((vmin + vmax) / 2).toFixed(1)}</span>
          <span>{layer === "uncertainty" ? `p98 ${vmax.toFixed(1)}` : vmax.toFixed(1)}</span>
        </div>
      </div>
    );
  }

  if (layer === "lisa") {
    return (
      <div className="absolute bottom-4 left-4 z-[1000] bg-white/95 backdrop-blur px-3 py-2 rounded-lg shadow-lg text-xs border border-slate-200">
        <div className="font-bold mb-1.5 text-slate-800">Clusters LISA</div>
        <div className="space-y-0.5">
          {(Object.keys(LISA_COLORS) as Array<keyof typeof LISA_COLORS>).map((k) => (
            <div key={k} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-sm border border-slate-300"
                style={{ backgroundColor: LISA_COLORS[k] }}
              ></span>
              <span className="text-slate-700">{LISA_LABELS[k]}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (layer === "kmeans" && clusters) {
    const sorted = [...clusters].sort((a, b) => a.risk_rank - b.risk_rank);
    const palette = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
    return (
      <div className="absolute bottom-4 left-4 z-[1000] bg-white/95 backdrop-blur px-3 py-2 rounded-lg shadow-lg text-xs border border-slate-200">
        <div className="font-bold mb-1.5 text-slate-800">KMeans (ranking de riesgo)</div>
        <div className="space-y-0.5">
          {sorted.map((c, i) => (
            <div key={c.cluster} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-sm border border-slate-300"
                style={{ backgroundColor: palette[Math.min(i, palette.length - 1)] }}
              ></span>
              <span className="text-slate-700">
                #{c.cluster} - rank {c.risk_rank} ({c.n_cells} celdas)
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
}
