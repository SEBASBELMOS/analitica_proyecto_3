// 9 estaciones DAGMA Cali (panel maestro 2020-2024, todas en municipio Cali)
// Sincronizadas con geovision-cali-api/main.py DAGMA_STATIONS
// Fuente: panel_geovision_cali_station_hourly_2020_2024_model_safe.parquet
import type { DagmaStation } from "../types/api";

export const DAGMA_STATIONS: DagmaStation[] = [
  { id: "BASE_AEREA",          name: "Base Aerea",            lat: 3.457128, lon: -76.502303 },
  { id: "CANAVERALEJO",        name: "Canaveralejo",          lat: 3.416366, lon: -76.549613 },
  { id: "COMPARTIR",           name: "Compartir",             lat: 3.428260, lon: -76.466584 },
  { id: "ERA_OBRERO",          name: "ERA - Obrero",          lat: 3.457317, lon: -76.506539 },
  { id: "ERMITA",              name: "Ermita",                lat: 3.455514, lon: -76.530978 },
  { id: "FLORA",               name: "Flora",                 lat: 3.488218, lon: -76.518058 },
  { id: "PANCE",               name: "Pance",                 lat: 3.304517, lon: -76.531252 },
  { id: "TRANSITORIA_NAVARRO", name: "Transitoria - Navarro", lat: 3.417183, lon: -76.494960 },
  { id: "UNIVALLE",            name: "Univalle",              lat: 3.377911, lon: -76.533811 },
];
