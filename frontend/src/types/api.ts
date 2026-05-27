// Tipos TypeScript que mirror exactamente los schemas Pydantic del backend FastAPI.
// Mantener sincronizados con `fastapi_app/main.py`.

export interface DagmaStation {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface PredictRequest {
  lat: number;       // 3.25 - 3.55 (Cali + Jamundi)
  lon: number;       // -76.60 - -76.40
  fecha: string;     // YYYY-MM-DD
  horizonte?: "T+1" | "T+3" | "T+7";
}

export interface PredictResponse {
  lat: number;
  lon: number;
  fecha: string;
  horizonte: string;
  NO2: number;
  SO2: number;
  O3: number;
  uncertainty_sigma: { NO2: number; SO2: number; O3: number };
  modelo: string;
  estacion_dagma_cercana: {
    id: string;
    name: string;
    distancia_km: number;
  };
  grid_cell: {
    grid_id: string;
    lat: number;
    lon: number;
    distance_km: number;
  };
}

export interface ValidateRequest {
  lat: number;
  lon: number;
}

export interface ValidateResponse {
  lat: number;
  lon: number;
  estacion_cercana: string;
  distancia_km: number;
  grid_cell: {
    grid_id: string;
    lat: number;
    lon: number;
    distance_km: number;
  };
  predicciones_celda: Record<string, Record<string, { prediction: number; sigma: number }>>;
}

export interface HealthResponse {
  status: string;
  n_estaciones_dagma: number;
  situacion3_loaded: boolean;
  n_grid_cells: number;
  version: string;
}

export type Pollutant = "NO2" | "SO2" | "O3";
export type Horizon = "T+1" | "T+3" | "T+7";

// Bounds del bbox oficial Cali (sincronizado con main.py del backend)
export const CALI_BOUNDS = {
  latMin: 3.25,
  latMax: 3.55,
  lonMin: -76.60,
  lonMax: -76.40,
} as const;

export const CALI_CENTER: [number, number] = [3.42, -76.52];

// Umbrales OMS (referencia para colorear severidad)
export const WHO_THRESHOLDS = {
  NO2: { warning: 25, danger: 50 },  // ug/m^3 (anual)
  SO2: { warning: 20, danger: 40 },  // ug/m^3
  O3:  { warning: 60, danger: 100 }, // ug/m^3
} as const;
