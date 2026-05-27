// Pagina principal de Situacion 3: mapa + controles + resumen + KPIs + advertencias.
import { useEffect, useMemo, useState } from "react";

import {
  getSit3Clusters,
  getSit3Kpis,
  getSit3LisaCached,
  getSit3MapCached,
  getSit3Metadata,
  getSit3Radius,
} from "../api/situacion3";
import Sit3Controls from "../components/sit3/Sit3Controls";
import Sit3Download from "../components/sit3/Sit3Download";
import Sit3KpiPanel from "../components/sit3/Sit3KpiPanel";
import Sit3Legend from "../components/sit3/Sit3Legend";
import Sit3Map from "../components/sit3/Sit3Map";
import Sit3RadiusSummary from "../components/sit3/Sit3RadiusSummary";
import Sit3Warnings from "../components/sit3/Sit3Warnings";
import type {
  Sit3Cluster,
  Sit3HorizonDays,
  Sit3KpisResponse,
  Sit3Layer,
  Sit3LisaResponse,
  Sit3MapResponse,
  Sit3Metadata,
  Sit3Pollutant,
  Sit3RadiusResponse,
} from "../types/situacion3";

export default function Situacion3Page() {
  // estado del usuario
  const [pollutant, setPollutant] = useState<Sit3Pollutant>("O3");
  const [horizon, setHorizon] = useState<Sit3HorizonDays>(1);
  const [layer, setLayer] = useState<Sit3Layer>("prediction");
  const [radiusKm, setRadiusKm] = useState<number>(2);
  const [selectedLat, setSelectedLat] = useState<number | null>(null);
  const [selectedLon, setSelectedLon] = useState<number | null>(null);

  // datos remotos
  const [metadata, setMetadata] = useState<Sit3Metadata | null>(null);
  const [mapData, setMapData] = useState<Sit3MapResponse | null>(null);
  const [lisaData, setLisaData] = useState<Sit3LisaResponse | null>(null);
  const [clusters, setClusters] = useState<Sit3Cluster[] | null>(null);
  const [kpis, setKpis] = useState<Sit3KpisResponse | null>(null);
  const [radiusData, setRadiusData] = useState<Sit3RadiusResponse | null>(null);

  // estados de carga
  const [mapLoading, setMapLoading] = useState(false);
  const [radiusLoading, setRadiusLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // carga inicial
  useEffect(() => {
    Promise.all([getSit3Metadata(), getSit3Kpis(), getSit3Clusters()])
      .then(([m, k, c]) => {
        setMetadata(m);
        setKpis(k);
        setClusters(c.clusters);
      })
      .catch((e) => setError(`Error inicial: ${e?.message ?? "desconocido"}`));
  }, []);

  // refresca mapa al cambiar gas/horizonte
  useEffect(() => {
    setMapLoading(true);
    setError(null);
    Promise.all([
      getSit3MapCached(pollutant, horizon),
      getSit3LisaCached(pollutant, horizon),
    ])
      .then(([m, l]) => {
        setMapData(m);
        setLisaData(l);
      })
      .catch((e) => setError(`Error mapa: ${e?.message ?? "desconocido"}`))
      .finally(() => setMapLoading(false));
  }, [pollutant, horizon]);

  // consulta /radius cuando hay punto seleccionado
  useEffect(() => {
    if (selectedLat === null || selectedLon === null) {
      setRadiusData(null);
      return;
    }
    setRadiusLoading(true);
    getSit3Radius(selectedLat, selectedLon, radiusKm, pollutant)
      .then(setRadiusData)
      .catch((e) => {
        const detail = e?.response?.data?.detail ?? e?.message ?? "Error en /radius";
        if (e?.response?.status === 404) {
          setRadiusData(null);
        }
        setError(`Radio: ${detail}`);
      })
      .finally(() => setRadiusLoading(false));
  }, [selectedLat, selectedLon, radiusKm, pollutant]);

  const lisaCells = useMemo(() => lisaData?.cells ?? null, [lisaData]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Mapa principal */}
      <section className="flex-1 relative bg-slate-200 dark:bg-slate-700">
        <Sit3Map
          mapData={mapData}
          lisaData={lisaCells}
          clusters={clusters}
          metadata={metadata}
          layer={layer}
          pollutant={pollutant}
          horizon={horizon}
          selectedLat={selectedLat}
          selectedLon={selectedLon}
          radiusKm={radiusKm}
          onClick={(lat, lon) => {
            setSelectedLat(lat);
            setSelectedLon(lon);
          }}
        />

        {mapLoading && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[1000] bg-white/95 dark:bg-slate-900/95 dark:text-slate-100 backdrop-blur px-4 py-2 rounded-lg shadow-lg flex items-center gap-2 text-sm">
            <div className="w-4 h-4 border-2 border-indigo-300 border-t-indigo-600 rounded-full animate-spin"></div>
            <span>Cargando 1920 celdas...</span>
          </div>
        )}

        {error && (
          <div className="absolute top-4 right-4 z-[1000] bg-rose-100 dark:bg-rose-900/40 border border-rose-300 dark:border-rose-700 text-rose-800 dark:text-rose-200 px-3 py-2 rounded-lg shadow-md text-xs max-w-xs">
            {error}
            <button onClick={() => setError(null)} className="ml-2 font-bold">
              x
            </button>
          </div>
        )}

        <Sit3Legend layer={layer} pollutant={pollutant} mapData={mapData} clusters={clusters} />
      </section>

      {/* Panel lateral */}
      <aside className="w-[28rem] bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-700 overflow-y-auto flex flex-col">
        <Sit3Controls
          pollutant={pollutant}
          horizon={horizon}
          layer={layer}
          radiusKm={radiusKm}
          lat={selectedLat}
          lon={selectedLon}
          onPollutantChange={setPollutant}
          onHorizonChange={setHorizon}
          onLayerChange={setLayer}
          onRadiusChange={setRadiusKm}
          onClearSelection={() => {
            setSelectedLat(null);
            setSelectedLon(null);
          }}
          loading={mapLoading}
        />
        <Sit3RadiusSummary data={radiusData} loading={radiusLoading} pollutant={pollutant} />
        <Sit3Download mapData={mapData} radiusData={radiusData} />
        <Sit3KpiPanel kpis={kpis?.kpis ?? null} />
        <Sit3Warnings warnings={metadata?.warnings ?? []} />
      </aside>
    </div>
  );
}
