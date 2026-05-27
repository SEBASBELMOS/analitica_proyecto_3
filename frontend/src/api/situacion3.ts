// Cliente tipado para los 8 endpoints /api/situacion3/*
import { api } from "./client";
import type {
  Sit3ClustersResponse,
  Sit3HorizonDays,
  Sit3KpisResponse,
  Sit3LisaResponse,
  Sit3MapResponse,
  Sit3Metadata,
  Sit3MoranResponse,
  Sit3PointResponse,
  Sit3Pollutant,
  Sit3RadiusResponse,
} from "../types/situacion3";

const BASE = "/api/situacion3";

export async function getSit3Metadata(): Promise<Sit3Metadata> {
  const { data } = await api.get<Sit3Metadata>(`${BASE}/metadata`);
  return data;
}

export async function getSit3Map(
  pollutant: Sit3Pollutant,
  horizon_days: Sit3HorizonDays,
): Promise<Sit3MapResponse> {
  const { data } = await api.get<Sit3MapResponse>(`${BASE}/map`, {
    params: { pollutant, horizon_days },
  });
  return data;
}

export async function getSit3Point(
  lat: number,
  lon: number,
  pollutant: Sit3Pollutant,
): Promise<Sit3PointResponse> {
  const { data } = await api.get<Sit3PointResponse>(`${BASE}/point`, {
    params: { lat, lon, pollutant },
  });
  return data;
}

export async function getSit3Radius(
  lat: number,
  lon: number,
  radius_km: number,
  pollutant: Sit3Pollutant,
): Promise<Sit3RadiusResponse> {
  const { data } = await api.get<Sit3RadiusResponse>(`${BASE}/radius`, {
    params: { lat, lon, radius_km, pollutant },
  });
  return data;
}

export async function getSit3Clusters(): Promise<Sit3ClustersResponse> {
  const { data } = await api.get<Sit3ClustersResponse>(`${BASE}/clusters`);
  return data;
}

export async function getSit3Lisa(
  pollutant: Sit3Pollutant,
  horizon_days: Sit3HorizonDays,
): Promise<Sit3LisaResponse> {
  const { data } = await api.get<Sit3LisaResponse>(`${BASE}/lisa`, {
    params: { pollutant, horizon_days },
  });
  return data;
}

export async function getSit3Kpis(): Promise<Sit3KpisResponse> {
  const { data } = await api.get<Sit3KpisResponse>(`${BASE}/kpis`);
  return data;
}

export async function getSit3Moran(): Promise<Sit3MoranResponse> {
  const { data } = await api.get<Sit3MoranResponse>(`${BASE}/moran`);
  return data;
}

// Cache simple en memoria por (pollutant, horizon) para mapas pesados
const _mapCache = new Map<string, Sit3MapResponse>();
const _lisaCache = new Map<string, Sit3LisaResponse>();

export async function getSit3MapCached(
  pollutant: Sit3Pollutant,
  horizon_days: Sit3HorizonDays,
): Promise<Sit3MapResponse> {
  const key = `${pollutant}-${horizon_days}`;
  if (_mapCache.has(key)) return _mapCache.get(key)!;
  const r = await getSit3Map(pollutant, horizon_days);
  _mapCache.set(key, r);
  return r;
}

export async function getSit3LisaCached(
  pollutant: Sit3Pollutant,
  horizon_days: Sit3HorizonDays,
): Promise<Sit3LisaResponse> {
  const key = `${pollutant}-${horizon_days}`;
  if (_lisaCache.has(key)) return _lisaCache.get(key)!;
  const r = await getSit3Lisa(pollutant, horizon_days);
  _lisaCache.set(key, r);
  return r;
}
