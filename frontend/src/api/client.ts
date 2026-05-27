// Cliente API para el backend FastAPI desplegado en HuggingFace Spaces.
// Endpoint configurable via env VITE_API_BASE_URL (default: deploy publico).
import axios from "axios";
import type {
  PredictRequest, PredictResponse,
  ValidateRequest, ValidateResponse,
  HealthResponse,
} from "../types/api";

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "https://analiticalastdance-geovision-cali-api.hf.space";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000, // 20s para /map con 1920 celdas
  headers: { "Content-Type": "application/json" },
});

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const { data } = await api.post<PredictResponse>("/predict", req);
  return data;
}

export async function validate(req: ValidateRequest): Promise<ValidateResponse> {
  const { data } = await api.post<ValidateResponse>("/validate", req);
  return data;
}

export { API_BASE_URL };
