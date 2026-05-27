// Tipos TypeScript para los endpoints /api/situacion3/*
// Mantener sincronizados con fastapi_app/situacion3/schemas.py

export type Sit3Pollutant = "NO2" | "SO2" | "O3";
export type Sit3HorizonDays = 1 | 3 | 7;
export type Sit3Layer = "prediction" | "uncertainty" | "lisa" | "kmeans";
export type LisaCluster =
  | "high_high"
  | "low_low"
  | "high_low"
  | "low_high"
  | "not_significant";

export interface Sit3Metadata {
  pollutants: Sit3Pollutant[];
  horizons_days: Sit3HorizonDays[];
  base_date: string;
  target_dates: Record<string, string>;
  grid: {
    n_cells: number;
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  warnings: string[];
}

export interface Sit3MapCell {
  grid_id: string;
  lat: number;
  lon: number;
  prediction: number;
  uncertainty_variance: number;
  uncertainty_sigma: number;
  kmeans_cluster: number | null;
  lisa_cluster: LisaCluster | null;
}

export interface Sit3MapResponse {
  pollutant: Sit3Pollutant;
  horizon_days: Sit3HorizonDays;
  target_date: string;
  n_cells: number;
  cells: Sit3MapCell[];
}

export interface Sit3HorizonPoint {
  horizon_days: number;
  target_date: string;
  prediction: number;
  uncertainty_variance: number;
  uncertainty_sigma: number;
  kmeans_cluster: number | null;
  lisa_cluster: LisaCluster | null;
}

export interface Sit3PointResponse {
  query: { lat: number; lon: number; pollutant: string };
  nearest_cell: {
    grid_id: string;
    lat: number;
    lon: number;
    distance_km: number;
  };
  horizons: Sit3HorizonPoint[];
}

export interface Sit3HorizonRadius {
  horizon_days: number;
  target_date: string;
  mean_prediction: number;
  min_prediction: number;
  max_prediction: number;
  mean_uncertainty_variance: number;
  max_uncertainty_variance: number;
  mean_uncertainty_sigma: number;
  lisa_counts: Record<string, number>;
  cluster_counts: Record<string, number>;
}

export interface Sit3RadiusResponse {
  query: {
    lat: number;
    lon: number;
    radius_km: number;
    pollutant: string;
    elapsed_ms?: number;
  };
  n_cells: number;
  horizons: Sit3HorizonRadius[];
}

export interface Sit3Cluster {
  cluster: number;
  risk_rank: number;
  n_cells: number;
  mean_prediction_profile: number;
  max_prediction_profile: number;
  mean_uncertainty_profile: number;
  lat_mean: number;
  lon_mean: number;
}

export interface Sit3ClustersResponse {
  k: number;
  clusters: Sit3Cluster[];
}

export interface Sit3LisaCell {
  grid_id: string;
  lat: number;
  lon: number;
  lisa_cluster: LisaCluster;
  local_i: number;
  p_sim: number;
  significant: boolean;
}

export interface Sit3LisaResponse {
  pollutant: Sit3Pollutant;
  horizon_days: Sit3HorizonDays;
  n_cells: number;
  cells: Sit3LisaCell[];
}

export type KpiStatus =
  | "excelente"
  | "cumple_minimo"
  | "no_cumple"
  | "no_evaluable"
  | "pendiente_backend"
  | "pendiente_medicion_formal";

export interface Sit3Kpi {
  kpi: string;
  minimum: string | null;
  excellent: string | null;
  value: string | null;
  status: KpiStatus;
  evidence: string | null;
}

export interface Sit3KpisResponse {
  kpis: Sit3Kpi[];
  notes: string[];
}

export interface Sit3MoranRow {
  pollutant: string;
  horizon_days: number;
  moran_i: number;
  expected_i: number;
  p_sim: number;
  z_sim: number;
  n: number;
  status: string;
}

export interface Sit3MoranResponse {
  rows: Sit3MoranRow[];
}

export const POLLUTANT_LABELS: Record<Sit3Pollutant, string> = {
  NO2: "Dioxido de nitrogeno (NO₂)",
  SO2: "Dioxido de azufre (SO₂)",
  O3: "Ozono (O₃)",
};

export const HORIZON_LABELS: Record<Sit3HorizonDays, string> = {
  1: "T+1 dia",
  3: "T+3 dias",
  7: "T+7 dias",
};
