// Paletas de color para los mapas Sit3.
// Viridis: prediccion (verde-amarillo). Magma: incertidumbre (morado-naranja).
// LISA y KMeans: categoricas.
import type { LisaCluster } from "../../types/situacion3";

// Viridis discretizado a 10 stops (perceptualmente uniforme, accesible)
const VIRIDIS = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
  "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
];

// Magma discretizado a 10 stops
const MAGMA = [
  "#000004", "#1c1044", "#4f127b", "#812581", "#b5367a",
  "#e55964", "#fb8761", "#fec287", "#fcfdbf", "#fcfdbf",
];

// YlOrRd alternativo para "concentracion peligrosa" (cuando se quiera contraste vs OMS)
const YLORRD = [
  "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
  "#fc4e2a", "#e31a1c", "#bd0026", "#800026", "#67001f",
];

function pickColor(palette: string[], value: number, vmin: number, vmax: number): string {
  if (!isFinite(value) || vmax === vmin) return palette[0];
  const t = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
  const idx = Math.min(palette.length - 1, Math.floor(t * palette.length));
  return palette[idx];
}

export function predictionColor(value: number, vmin: number, vmax: number): string {
  return pickColor(YLORRD, value, vmin, vmax);
}

export function uncertaintyColor(value: number, vmin: number, vmax: number): string {
  return pickColor(MAGMA, value, vmin, vmax);
}

export function viridisColor(value: number, vmin: number, vmax: number): string {
  return pickColor(VIRIDIS, value, vmin, vmax);
}

export const LISA_COLORS: Record<LisaCluster, string> = {
  high_high: "#d7191c",          // rojo: hotspot
  low_low: "#2c7bb6",            // azul: coldspot
  high_low: "#fdae61",           // naranja: outlier alto rodeado bajo
  low_high: "#abd9e9",           // celeste: outlier bajo rodeado alto
  not_significant: "#e0e0e0",    // gris claro
};

export const LISA_LABELS: Record<LisaCluster, string> = {
  high_high: "Hotspot (HH)",
  low_low: "Coldspot (LL)",
  high_low: "Outlier alto (HL)",
  low_high: "Outlier bajo (LH)",
  not_significant: "No significativo",
};

// Paleta categorica KMeans 5 clusters, ordenados por risk_rank
// risk_rank 1 (mas critico) -> rojo intenso; 5 (mas bajo) -> verde
export const KMEANS_COLORS: Record<number, string> = {
  0: "#90c590",  // (placeholder; el orden real depende de risk_rank)
  1: "#d7191c",
  2: "#fdae61",
  3: "#ffffbf",
  4: "#abd9e9",
};

// Helper para mapear cluster -> color usando risk_rank
export function kmeansColorByRisk(
  cluster: number,
  riskByCluster: Map<number, number>,
): string {
  const rank = riskByCluster.get(cluster);
  if (!rank) return "#cccccc";
  const palette = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
  return palette[Math.min(rank, palette.length) - 1];
}

// Escala de colores en stops para leyenda gradiente
export function legendStops(palette: "prediction" | "uncertainty" | "viridis"): string[] {
  if (palette === "prediction") return YLORRD;
  if (palette === "uncertainty") return MAGMA;
  return VIRIDIS;
}
