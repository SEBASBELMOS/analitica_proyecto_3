// Controles del panel Situacion 3:
// gas, horizonte, capa visible, radio de consulta, coordenadas seleccionadas.
import type {
  Sit3HorizonDays,
  Sit3Layer,
  Sit3Pollutant,
} from "../../types/situacion3";
import { HORIZON_LABELS, POLLUTANT_LABELS } from "../../types/situacion3";

interface Props {
  pollutant: Sit3Pollutant;
  horizon: Sit3HorizonDays;
  layer: Sit3Layer;
  radiusKm: number;
  lat: number | null;
  lon: number | null;
  onPollutantChange: (p: Sit3Pollutant) => void;
  onHorizonChange: (h: Sit3HorizonDays) => void;
  onLayerChange: (l: Sit3Layer) => void;
  onRadiusChange: (r: number) => void;
  onClearSelection: () => void;
  loading?: boolean;
}

const LAYER_LABELS: Record<Sit3Layer, string> = {
  prediction: "Concentracion",
  uncertainty: "Incertidumbre (sigma)",
  lisa: "Clusters LISA",
  kmeans: "Perfiles KMeans",
};

export default function Sit3Controls({
  pollutant,
  horizon,
  layer,
  radiusKm,
  lat,
  lon,
  onPollutantChange,
  onHorizonChange,
  onLayerChange,
  onRadiusChange,
  onClearSelection,
  loading,
}: Props) {
  return (
    <div className="p-4 space-y-4 text-sm">
      <h2 className="font-bold text-base text-slate-900">Controles</h2>

      {/* Pollutant pills */}
      <div>
        <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 uppercase tracking-wide">
          Contaminante
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {(["NO2", "SO2", "O3"] as const).map((p) => (
            <button
              key={p}
              onClick={() => onPollutantChange(p)}
              className={`px-3 py-2 rounded-lg font-medium text-sm transition-all ${
                pollutant === p
                  ? "bg-slate-900 text-white shadow-md"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200"
              }`}
              title={POLLUTANT_LABELS[p]}
            >
              {p}
            </button>
          ))}
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">
          {POLLUTANT_LABELS[pollutant]}
        </div>
      </div>

      {/* Horizonte pills */}
      <div>
        <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 uppercase tracking-wide">
          Horizonte temporal
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {([1, 3, 7] as const).map((h) => (
            <button
              key={h}
              onClick={() => onHorizonChange(h)}
              className={`px-3 py-2 rounded-lg font-medium text-sm transition-all ${
                horizon === h
                  ? "bg-slate-900 text-white shadow-md"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200"
              }`}
            >
              T+{h}
            </button>
          ))}
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-500 mt-1">{HORIZON_LABELS[horizon]}</div>
      </div>

      {/* Capa */}
      <div>
        <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 uppercase tracking-wide">
          Capa visible
        </label>
        <div className="grid grid-cols-2 gap-1.5">
          {(Object.keys(LAYER_LABELS) as Sit3Layer[]).map((l) => (
            <button
              key={l}
              onClick={() => onLayerChange(l)}
              className={`px-2 py-1.5 rounded-md text-xs font-medium transition-all ${
                layer === l
                  ? "bg-indigo-600 text-white shadow"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200"
              }`}
            >
              {LAYER_LABELS[l]}
            </button>
          ))}
        </div>
      </div>

      {/* Radio slider */}
      <div>
        <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1.5 uppercase tracking-wide">
          Radio de consulta
        </label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0.5}
            max={10}
            step={0.5}
            value={radiusKm}
            onChange={(e) => onRadiusChange(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="text-sm font-mono font-semibold w-14 text-right">
            {radiusKm.toFixed(1)} km
          </span>
        </div>
      </div>

      {/* Coordenadas seleccionadas */}
      <div className="bg-slate-50 dark:bg-slate-800 rounded-lg p-3 border border-slate-200">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide">
            Punto seleccionado
          </span>
          {(lat !== null || lon !== null) && (
            <button
              onClick={onClearSelection}
              className="text-xs text-rose-600 dark:text-rose-400 hover:text-rose-700 dark:text-rose-400 font-medium"
            >
              Limpiar
            </button>
          )}
        </div>
        {lat !== null && lon !== null ? (
          <div className="font-mono text-xs text-slate-700 dark:text-slate-200">
            <div>lat: {lat.toFixed(5)}</div>
            <div>lon: {lon.toFixed(5)}</div>
          </div>
        ) : (
          <div className="text-xs text-slate-500 dark:text-slate-500 italic">
            Haz click en el mapa para seleccionar
          </div>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs text-indigo-600">
          <div className="w-3 h-3 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin"></div>
          Cargando mapa...
        </div>
      )}
    </div>
  );
}
