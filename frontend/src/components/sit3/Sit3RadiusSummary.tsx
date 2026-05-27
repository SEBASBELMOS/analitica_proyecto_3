// Tarjeta de resumen del area consultada por radio:
// promedio, min, max, incertidumbre, conteos LISA y KMeans por horizonte.
import type { Sit3RadiusResponse } from "../../types/situacion3";
import { LISA_COLORS, LISA_LABELS } from "./colors";
import type { LisaCluster } from "../../types/situacion3";

interface Props {
  data: Sit3RadiusResponse | null;
  loading?: boolean;
  pollutant: string;
}

export default function Sit3RadiusSummary({ data, loading, pollutant }: Props) {
  if (loading) {
    return (
      <div className="p-4 border-t border-slate-200">
        <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-2">Resumen del area</h3>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <div className="w-3 h-3 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin"></div>
          Calculando...
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="p-4 border-t border-slate-200">
        <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-2">Resumen del area</h3>
        <p className="text-xs text-slate-500 dark:text-slate-500 italic">
          Selecciona un punto en el mapa y ajusta el radio para ver el resumen.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">Resumen del area</h3>
        <span className="text-xs text-slate-500">
          {data.n_cells} celdas{data.query.elapsed_ms ? ` - ${data.query.elapsed_ms} ms` : ""}
        </span>
      </div>

      {data.horizons.map((h) => {
        const lisaTotal = Object.values(h.lisa_counts).reduce((a, b) => a + b, 0) || 1;
        const top3Clusters = Object.entries(h.cluster_counts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3);

        return (
          <div
            key={h.horizon_days}
            className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-slate-800 dark:text-white">T+{h.horizon_days}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">{h.target_date}</span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-[10px] text-slate-500 dark:text-slate-500 uppercase">min</div>
                <div className="font-mono font-semibold text-slate-800 dark:text-white">
                  {h.min_prediction.toFixed(1)}
                </div>
              </div>
              <div className="bg-white dark:bg-slate-900 rounded border border-indigo-200 dark:border-indigo-700 py-1">
                <div className="text-[10px] text-indigo-600 dark:text-indigo-400 uppercase font-bold">media</div>
                <div className="font-mono font-bold text-indigo-700 dark:text-indigo-300">
                  {h.mean_prediction.toFixed(1)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-500 dark:text-slate-500 uppercase">max</div>
                <div className="font-mono font-semibold text-slate-800 dark:text-white">
                  {h.max_prediction.toFixed(1)}
                </div>
              </div>
            </div>

            <div className="text-xs text-slate-600 dark:text-slate-400 flex items-center justify-between">
              <span>Incert. media (sigma):</span>
              <span className="font-mono font-semibold">
                {h.mean_uncertainty_sigma.toFixed(2)}
              </span>
            </div>

            {top3Clusters.length > 0 && (
              <div className="text-xs text-slate-600 dark:text-slate-300">
                <div className="mb-0.5">Clusters KMeans (top 3):</div>
                <div className="flex flex-wrap gap-1">
                  {top3Clusters.map(([clusterId, count]) => (
                    <span
                      key={clusterId}
                      className="font-mono text-[10px] bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 px-1.5 py-0.5 rounded"
                    >
                      #{clusterId}: {count}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Mini barra LISA */}
            <div>
              <div className="text-[10px] text-slate-500 dark:text-slate-500 uppercase mb-1">LISA</div>
              <div className="flex h-3 rounded overflow-hidden border border-slate-200">
                {(Object.keys(LISA_COLORS) as LisaCluster[]).map((key) => {
                  const count = h.lisa_counts[key] ?? 0;
                  const pct = (count / lisaTotal) * 100;
                  if (pct === 0) return null;
                  return (
                    <div
                      key={key}
                      style={{ width: `${pct}%`, backgroundColor: LISA_COLORS[key] }}
                      title={`${LISA_LABELS[key]}: ${count}`}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}

      <div className="text-[10px] text-slate-500 dark:text-slate-500 italic">
        {pollutant} - prediccion corregida (ConvLSTM + Kriging residual)
      </div>
    </div>
  );
}
