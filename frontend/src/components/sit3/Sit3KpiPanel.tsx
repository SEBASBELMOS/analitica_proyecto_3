// Panel de KPIs del proyecto Situacion 3 con estados visuales codificados por color.
import type { KpiStatus, Sit3Kpi } from "../../types/situacion3";

interface Props {
  kpis: Sit3Kpi[] | null;
  loading?: boolean;
}

const STATUS_STYLE: Record<KpiStatus, { bg: string; text: string; label: string }> = {
  excelente: { bg: "bg-emerald-100 dark:bg-emerald-900/40 border-emerald-300", text: "text-emerald-800", label: "Excelente" },
  cumple_minimo: { bg: "bg-lime-50 dark:bg-lime-900/40 border-lime-300", text: "text-lime-800", label: "Cumple" },
  no_cumple: { bg: "bg-rose-100 dark:bg-rose-900/40 border-rose-300", text: "text-rose-800", label: "No cumple" },
  no_evaluable: { bg: "bg-slate-100 dark:bg-slate-800 border-slate-300", text: "text-slate-700", label: "No evaluable" },
  pendiente_backend: { bg: "bg-amber-100 dark:bg-amber-900/30 border-amber-300", text: "text-amber-800", label: "Pendiente" },
  pendiente_medicion_formal: { bg: "bg-amber-100 dark:bg-amber-900/30 border-amber-300", text: "text-amber-800", label: "Pendiente" },
};

const STATUS_ORDER: Record<KpiStatus, number> = {
  excelente: 0,
  cumple_minimo: 1,
  pendiente_backend: 2,
  pendiente_medicion_formal: 2,
  no_evaluable: 3,
  no_cumple: 4,
};

export default function Sit3KpiPanel({ kpis, loading }: Props) {
  if (loading) {
    return (
      <div className="p-4 border-t border-slate-200">
        <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-2">KPIs Situacion 3</h3>
        <p className="text-xs text-slate-500">Cargando...</p>
      </div>
    );
  }
  if (!kpis) return null;

  const orderedKpis = [...kpis].sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status]);

  return (
    <div className="p-4 border-t border-slate-200">
      <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-3">KPIs Situacion 3</h3>
      <div className="space-y-1.5">
        {orderedKpis.map((k) => {
          const style = STATUS_STYLE[k.status];
          return (
            <div
              key={k.kpi}
              className={`border rounded-md p-2 ${style.bg}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="text-xs font-medium text-slate-800 dark:text-white flex-1">{k.kpi}</div>
                <span
                  className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${style.text} bg-white/70 whitespace-nowrap`}
                >
                  {style.label}
                </span>
              </div>
              {k.value && (
                <div className="text-[10px] text-slate-600 dark:text-slate-400 mt-0.5 font-mono">
                  {k.value} (min: {k.minimum ?? "-"} / exc: {k.excellent ?? "-"})
                </div>
              )}
              {k.evidence && (
                <div className="text-[10px] text-slate-500 dark:text-slate-500 mt-0.5 italic line-clamp-2">
                  {k.evidence}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
