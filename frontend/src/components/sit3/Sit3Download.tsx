// Boton de descarga CSV del mapa visible (gas + horizonte actual).
// Cumple Sit. 4.4 del PDF (descarga CSV del area consultada).
import Papa from "papaparse";
import type { Sit3MapResponse, Sit3RadiusResponse } from "../../types/situacion3";

interface Props {
  mapData: Sit3MapResponse | null;
  radiusData: Sit3RadiusResponse | null;
}

export default function Sit3Download({ mapData, radiusData }: Props) {
  const downloadMap = () => {
    if (!mapData) return;
    const rows = mapData.cells.map((c) => ({
      grid_id: c.grid_id,
      lat: c.lat,
      lon: c.lon,
      pollutant: mapData.pollutant,
      horizon_days: mapData.horizon_days,
      target_date: mapData.target_date,
      prediction_ug_m3: c.prediction,
      uncertainty_variance: c.uncertainty_variance,
      uncertainty_sigma: c.uncertainty_sigma,
      kmeans_cluster: c.kmeans_cluster ?? "",
      lisa_cluster: c.lisa_cluster ?? "",
    }));
    const csv = Papa.unparse(rows);
    const ts = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
    const name = `geovision_${mapData.pollutant}_T${mapData.horizon_days}_${ts}.csv`;
    triggerDownload(csv, name);
  };

  const downloadRadius = () => {
    if (!radiusData) return;
    const rows = radiusData.horizons.map((h) => ({
      lat: radiusData.query.lat,
      lon: radiusData.query.lon,
      radius_km: radiusData.query.radius_km,
      pollutant: radiusData.query.pollutant,
      n_cells: radiusData.n_cells,
      horizon_days: h.horizon_days,
      target_date: h.target_date,
      mean_prediction: h.mean_prediction,
      min_prediction: h.min_prediction,
      max_prediction: h.max_prediction,
      mean_uncertainty_variance: h.mean_uncertainty_variance,
      max_uncertainty_variance: h.max_uncertainty_variance,
      mean_uncertainty_sigma: h.mean_uncertainty_sigma,
      lisa_high_high: h.lisa_counts["high_high"] ?? 0,
      lisa_low_low: h.lisa_counts["low_low"] ?? 0,
      lisa_high_low: h.lisa_counts["high_low"] ?? 0,
      lisa_low_high: h.lisa_counts["low_high"] ?? 0,
      lisa_not_significant: h.lisa_counts["not_significant"] ?? 0,
    }));
    const csv = Papa.unparse(rows);
    const ts = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
    const name = `geovision_radio_${radiusData.query.pollutant}_${ts}.csv`;
    triggerDownload(csv, name);
  };

  const triggerDownload = (csv: string, name: string) => {
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-2">
      <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-1">Exportar CSV</h3>
      <button
        onClick={downloadMap}
        disabled={!mapData}
        className={`w-full text-xs px-3 py-2 rounded-lg font-medium transition-all ${
          mapData
            ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm"
            : "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
        }`}
      >
        {mapData
          ? `Mapa completo (${mapData.n_cells} celdas, ${mapData.pollutant} T+${mapData.horizon_days})`
          : "Mapa (cargando...)"}
      </button>
      <button
        onClick={downloadRadius}
        disabled={!radiusData}
        className={`w-full text-xs px-3 py-2 rounded-lg font-medium transition-all ${
          radiusData
            ? "bg-slate-700 dark:bg-slate-700 text-white hover:bg-slate-800 dark:hover:bg-slate-600 shadow-sm"
            : "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
        }`}
      >
        {radiusData
          ? `Resumen del area (${radiusData.n_cells} celdas, 3 horizontes)`
          : "Resumen del area (selecciona punto)"}
      </button>
      <p className="text-[10px] text-slate-500 dark:text-slate-400 italic mt-1">
        Cumple Sit. 4.4 del PDF (descarga CSV).
      </p>
    </div>
  );
}
