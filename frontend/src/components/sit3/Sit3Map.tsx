// Mapa Leaflet Sit3: renderiza las 1920 celdas de la grilla como rectangulos
// coloreados segun la capa activa (prediccion, incertidumbre, LISA o KMeans).
// Click en mapa -> selecciona punto y dibuja circulo de radio.
import { useEffect, useMemo } from "react";
import {
  Circle,
  MapContainer,
  Marker,
  Rectangle,
  Tooltip,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";

import { DAGMA_STATIONS } from "../../data/stations";
import { CALI_CENTER } from "../../types/api";
import type {
  Sit3Cluster,
  Sit3HorizonDays,
  Sit3Layer,
  Sit3LisaCell,
  Sit3MapCell,
  Sit3MapResponse,
  Sit3Metadata,
  Sit3Pollutant,
} from "../../types/situacion3";
import {
  LISA_COLORS,
  kmeansColorByRisk,
  predictionColor,
  uncertaintyColor,
} from "./colors";

interface Props {
  mapData: Sit3MapResponse | null;
  lisaData: Sit3LisaCell[] | null;
  clusters: Sit3Cluster[] | null;
  metadata: Sit3Metadata | null;
  layer: Sit3Layer;
  pollutant: Sit3Pollutant;
  horizon: Sit3HorizonDays;
  selectedLat: number | null;
  selectedLon: number | null;
  radiusKm: number;
  onClick: (lat: number, lon: number) => void;
}

const selectionIcon = L.divIcon({
  className: "",
  html: `<div style="width:18px;height:18px;background:#fff;border:3px solid #4f46e5;border-radius:50%;box-shadow:0 2px 6px rgba(0,0,0,0.4);"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

const stationIcon = L.divIcon({
  className: "",
  html: `<div style="width:14px;height:14px;background:#10b981;border:3px solid white;border-radius:50%;box-shadow:0 0 0 1px rgba(0,0,0,0.4),0 2px 4px rgba(0,0,0,0.35);"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

// Tamano de la celda en grados (basado en grilla 48x40 sobre bbox)
// lat_step = (3.5425 - 3.3075) / 47 ~= 0.005; lon_step = (-76.4025 - -76.5975) / 39 ~= 0.005
const HALF_LAT = 0.0025;
const HALF_LON = 0.0025;

function percentile(values: number[], p: number): number {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function ClickHandler({ onClick }: { onClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function FitBoundsOnMount({ metadata }: { metadata: Sit3Metadata | null }) {
  const map = useMap();
  useEffect(() => {
    if (!metadata) return;
    const bounds: [[number, number], [number, number]] = [
      [metadata.grid.lat_min - HALF_LAT, metadata.grid.lon_min - HALF_LON],
      [metadata.grid.lat_max + HALF_LAT, metadata.grid.lon_max + HALF_LON],
    ];
    const timers = [50, 200, 500, 800].map((d) =>
      setTimeout(() => {
        map.invalidateSize();
        map.fitBounds(bounds, { padding: [30, 30] });
      }, d),
    );
    return () => timers.forEach(clearTimeout);
  }, [map, metadata]);
  return null;
}

export default function Sit3Map({
  mapData,
  lisaData,
  clusters,
  metadata,
  layer,
  pollutant,
  horizon,
  selectedLat,
  selectedLon,
  radiusKm,
  onClick,
}: Props) {
  // Calcular vmin/vmax para escala de color. En incertidumbre usamos percentiles
  // para no perder contraste cuando hay pocos valores extremos o sigma casi plana.
  const { vmin, vmax } = useMemo(() => {
    if (!mapData) return { vmin: 0, vmax: 1 };
    const arr = layer === "uncertainty"
      ? mapData.cells.map((c) => c.uncertainty_sigma)
      : mapData.cells.map((c) => c.prediction);
    const rawMin = Math.min(...arr);
    const rawMax = Math.max(...arr);
    if (layer === "uncertainty") {
      return {
        vmin: percentile(arr, 0.02),
        vmax: percentile(arr, 0.98),
      };
    }
    return {
      vmin: Math.max(0, rawMin),
      vmax: rawMax,
    };
  }, [mapData, layer]);

  const riskByCluster = useMemo(() => {
    const m = new Map<number, number>();
    if (clusters) clusters.forEach((c) => m.set(c.cluster, c.risk_rank));
    return m;
  }, [clusters]);

  // LISA por grid_id para lookup rapido
  const lisaByGrid = useMemo(() => {
    const m = new Map<string, Sit3LisaCell>();
    if (lisaData) lisaData.forEach((c) => m.set(c.grid_id, c));
    return m;
  }, [lisaData]);

  function cellColor(cell: Sit3MapCell): string {
    if (layer === "prediction") return predictionColor(cell.prediction, vmin, vmax);
    if (layer === "uncertainty") return uncertaintyColor(cell.uncertainty_sigma, vmin, vmax);
    if (layer === "lisa") {
      const li = lisaByGrid.get(cell.grid_id);
      return li ? LISA_COLORS[li.lisa_cluster] : "#cccccc";
    }
    if (layer === "kmeans") {
      return cell.kmeans_cluster !== null
        ? kmeansColorByRisk(cell.kmeans_cluster, riskByCluster)
        : "#cccccc";
    }
    return "#cccccc";
  }

  return (
    <div className="h-full w-full relative">
      <MapContainer
        center={CALI_CENTER}
        zoom={12}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom
        preferCanvas
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FitBoundsOnMount metadata={metadata} />
        <ClickHandler onClick={onClick} />

        {/* BBox referencia - usa el grid real del backend */}
        {metadata && (
          <Rectangle
            bounds={[
              [metadata.grid.lat_min - HALF_LAT, metadata.grid.lon_min - HALF_LON],
              [metadata.grid.lat_max + HALF_LAT, metadata.grid.lon_max + HALF_LON],
            ]}
            pathOptions={{ color: "#4f46e5", weight: 2, fill: false, dashArray: "4,4" }}
          />
        )}

        {/* 1920 celdas de la grilla */}
        {mapData?.cells.map((cell) => (
          <Rectangle
            key={cell.grid_id}
            bounds={[
              [cell.lat - HALF_LAT, cell.lon - HALF_LON],
              [cell.lat + HALF_LAT, cell.lon + HALF_LON],
            ]}
            pathOptions={{
              color: cellColor(cell),
              weight: 0,
              fillColor: cellColor(cell),
              fillOpacity: 0.65,
            }}
          >
            <Tooltip direction="top" offset={[0, -2]} sticky>
              <div className="text-xs font-mono">
                <div className="font-bold text-sm">{cell.grid_id}</div>
                <div>{pollutant} T+{horizon}</div>
                <div>Prediccion: {cell.prediction.toFixed(2)} ug/m3</div>
                <div>sigma: {cell.uncertainty_sigma.toFixed(2)}</div>
                {cell.kmeans_cluster !== null && (
                  <div>Cluster KMeans: {cell.kmeans_cluster}</div>
                )}
                {cell.lisa_cluster && (
                  <div>LISA: {cell.lisa_cluster}</div>
                )}
              </div>
            </Tooltip>
          </Rectangle>
        ))}

        {/* Estaciones DAGMA */}
        {DAGMA_STATIONS.map((s) => (
          <Marker key={s.id} position={[s.lat, s.lon]} icon={stationIcon}>
            <Tooltip direction="top" offset={[0, -6]}>
              <div className="text-xs">
                <div className="font-bold">{s.name}</div>
                <div className="text-slate-500">DAGMA {s.id}</div>
              </div>
            </Tooltip>
          </Marker>
        ))}

        {/* Punto seleccionado + radio */}
        {selectedLat !== null && selectedLon !== null && (
          <>
            <Marker position={[selectedLat, selectedLon]} icon={selectionIcon} />
            <Circle
              center={[selectedLat, selectedLon]}
              radius={radiusKm * 1000}
              pathOptions={{ color: "#4f46e5", weight: 2, fillOpacity: 0.08 }}
            />
          </>
        )}
      </MapContainer>
    </div>
  );
}
