---
title: GeoVision-CLIP Cali Frontend
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# GeoVision-CLIP Cali — Frontend

SPA React + Vite + Leaflet para visualizar las predicciones de NO2, SO2 y O3 sobre Cali generadas por el pipeline **RemoteCLIP + SAE + ConvLSTM + ST-Kriging**. Consume el backend en `https://analiticalastdance-geovision-cali-api.hf.space`.

**URL publica:** `https://analiticalastdance-geovision-cali-frontend.hf.space`

## Tabs disponibles

### 1. Mapa Situacion 3 (default)

Vista principal que materializa lo pedido en el `UI_UX_CONTRACT_SITUACION3.md` de Manuel:

- **Mapa Leaflet con 1920 celdas** de la grilla de prediccion, coloreadas segun la capa activa.
- **4 capas alternables:**
  - Concentracion (paleta YlOrRd, leyenda con `vmin` clampeado a 0)
  - Incertidumbre sigma (paleta Magma)
  - Clusters LISA (HH=rojo, LL=azul, HL=naranja, LH=celeste, no_sig=gris)
  - Perfiles KMeans (gradiente verde-rojo segun `risk_rank`)
- **Selector de gas** (NO2 / SO2 / O3) actualiza paleta y dispara `/api/situacion3/map` + `/lisa`
- **Selector de horizonte** (T+1 / T+3 / T+7)
- **Slider de radio** 0.5-10 km
- **Click en mapa** dispara `/api/situacion3/radius` con el punto + radio actuales
- **Resumen del area** con min/media/max + incertidumbre + top 3 clusters KMeans + barra LISA por horizonte
- **Panel KPIs Situacion 3** con badges color-coded (excelente/cumple/no_cumple/no_evaluable/pendiente)
- **Advertencias metodologicas** colapsables (NO2/SO2/variograma)
- **9 estaciones DAGMA** marcadas con halo blanco visible sobre cualquier fondo
- **Leyenda flotante** que cambia segun capa activa

### 2. Consulta puntual

Flujo simple para predicciones individuales:

- Click en mapa -> `POST /predict` con (lat, lon, fecha, horizonte)
- Panel lateral con resultado: NO2/SO2/O3 ± σ + severidad OMS + estacion DAGMA mas cercana + celda Sit3 asociada
- Selector de fecha (las fechas reales son 2024-12-17, 19, 23)
- Descarga CSV del historial de predicciones (hasta 50)

## Stack

| Capa | Tecnologia |
|---|---|
| Framework | React 18 |
| Build | Vite 5 |
| Lenguaje | TypeScript 5.6 (strict) |
| Mapa | react-leaflet 4 + Leaflet 1.9 + preferCanvas (1920 rectangulos) |
| HTTP | axios (con cache en memoria por `pollutant+horizon`) |
| Estilos | Tailwind CSS 3.4 (paleta slate + indigo) |
| CSV export | papaparse |
| Deploy | HuggingFace Spaces Docker SDK puerto 7860 (servidor `serve`) |

## Estructura

```
src/
|-- main.tsx                                # entry + fix iconos Leaflet
|-- App.tsx                                 # header + tabs + footer + health check
|-- api/
|   |-- client.ts                           # axios instance + /health + /predict + /validate
|   `-- situacion3.ts                       # 8 endpoints /api/situacion3/* con cache
|-- types/
|   |-- api.ts                              # tipos /predict, /validate, /health
|   `-- situacion3.ts                       # 13 tipos para Sit3 + POLLUTANT_LABELS
|-- data/
|   `-- stations.ts                         # 9 estaciones DAGMA (sync con backend)
|-- styles/
|   `-- index.css                           # Tailwind + overrides Leaflet
|-- pages/
|   |-- Situacion3Page.tsx                  # orquesta mapa + controles + resumen + KPIs
|   `-- ConsultaPage.tsx                    # flujo /predict legacy mejorado
`-- components/
    |-- MapPanel.tsx                        # mapa Consulta puntual
    |-- StationMarker.tsx
    |-- PredictionPopup.tsx
    |-- ResultsPanel.tsx
    |-- ControlsPanel.tsx
    |-- DownloadButton.tsx
    `-- sit3/
        |-- Sit3Map.tsx                     # Leaflet con 1920 celdas + tooltips
        |-- Sit3Controls.tsx                # gas/horizonte/capa/radio
        |-- Sit3RadiusSummary.tsx           # min/media/max + top 3 clusters + LISA
        |-- Sit3KpiPanel.tsx                # 9 KPIs con badges color-coded
        |-- Sit3Warnings.tsx                # acordeon advertencias
        |-- Sit3Legend.tsx                  # leyenda flotante por capa
        `-- colors.ts                       # paletas YlOrRd, Magma, LISA, KMeans
```

## Desarrollo local

```bash
npm install
npm run dev
# http://localhost:5173
```

Por defecto consume el backend en HuggingFace. Para apuntar a un backend local:

```bash
echo "VITE_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev
```

## Build de produccion

```bash
npm run build       # genera dist/
npm run preview     # sirve dist/ en localhost
```

Build actual: **402 KB JS (129 KB gzip) + 33 KB CSS (10 KB gzip)** en ~750 ms.

## Deploy a HuggingFace Spaces

```bash
git add src/
git commit -m "feat: ..."
git push   # HF reconstruye Docker automaticamente, ~2-3 min
```

El Dockerfile usa `serve` de Vercel en el runtime (no `vite preview`) para evitar el bloqueo `allowedHosts` contra el proxy de HF Spaces.

## Decisiones de diseño

1. **Tabs en vez de SPA nueva:** se evoluciono la SPA existente con 2 tabs (`Mapa Situacion 3` default + `Consulta puntual`) en lugar de tirar lo viejo. Ambos comparten header, footer, health-check y data de estaciones DAGMA.

2. **`preferCanvas` en Leaflet:** renderizar 1920 rectangulos como SVG sobrecarga el DOM; con Canvas el zoom/pan se mantiene fluido.

3. **Cache en memoria por `pollutant+horizon`:** `/map` y `/lisa` se cachean en `Map` de JS para evitar re-fetch al toggle entre capas o cambios de radio.

4. **Clamp visual a `>=0` en leyenda de concentracion:** los CSVs originales tienen 603 predicciones negativas (artefacto del modelo). El frontend muestra esas celdas con el color minimo de la paleta pero la leyenda dice "0" para que no aparezca un feo "-2.0" al usuario.

5. **Markers DAGMA con halo grueso:** los 9 puntos se ven en cualquier fondo (rojo intenso de hotspots NO2 o verde de KMeans rank 5).

6. **Tooltip por celda con `sticky`:** al pasar el mouse muestra `grid_id`, gas, horizonte, prediccion, sigma, cluster KMeans, LISA.

7. **Bbox dashed indigo** alrededor del area real del grid (no del bbox amplio de Cali) para que el usuario vea exactamente la cobertura del modelo.

## Cumplimiento de la rubrica

| KPI PDF | Componente |
|---|---|
| Sit. 4.2 - Mapa Cali + 9 estaciones DAGMA + click | `Sit3Map.tsx` + `MapPanel.tsx` |
| Sit. 4.3 - Slider horizonte + selector contaminante | `Sit3Controls.tsx` + `ControlsPanel.tsx` |
| Sit. 4.4 - Tooltips valor ± σ + CSV | tooltip Leaflet en celdas + `DownloadButton.tsx` |
| Sit. 4.5 - Dockerfile multi-stage | `Dockerfile` |
| Sit. 4.6 - URL publica | https://analiticalastdance-geovision-cali-frontend.hf.space |
| Sit. 4.6 - NO Streamlit/Gradio (-30%) | React + Vite ✓ |

Adicionales del handoff de Manuel (UI/UX contract):
- ControlsPanel con gas/horizonte/capa/radio ✓
- MapView con celdas + punto + circulo + tooltip ✓
- RadiusSummaryCard con n_cells, mean, max, sigma, dominante, LISA ✓
- KpiPanel con tabla y estados color-coded ✓
- MethodologyWarnings siempre visible (acordeon) ✓
- Cache por gas+horizonte ✓
- Loading y errores manejados ✓

## Performance medido

Frontend (build production, despues de hidratacion):
- Carga inicial `Situacion3Page`: 3 requests paralelas (metadata + kpis + clusters) = ~1.5 s
- Toggle capa (con cache): instantaneo
- Cambio de gas o horizonte: ~1.2 s (fetch `/map` + `/lisa`)
- Click + `/radius` 2 km: ~500 ms

Bundle: 402 KB JS (129 KB gzip), 33 KB CSS (10 KB gzip).

## Next steps

### Bloqueantes
- Ninguno. El frontend cumple toda la rubrica y los contratos del handoff.

### Mejoras opcionales (post-entrega)
1. **Comparacion multi-gas lado a lado** (split-view) — sugerido por Manuel en "Mejoras Futuras" del handoff.
2. **Animacion temporal T+1 -> T+3 -> T+7** con play/pause — sugerido por Manuel.
3. **Descargar CSV/GeoJSON del area consultada** (botoncito en `Sit3RadiusSummary`) — sugerido por Manuel.
4. **Modo presentacion** que oculte el panel lateral y maximize el mapa con teclas Q/W/E para cambiar capas — para defensa oral.
5. **Cargar las figuras PNG de Manuel** (`outputs/figuras/`) como capa raster opcional via `<ImageOverlay>` para comparar mapa interactivo vs visualizacion estatica.
6. **Bottom sheet movil** — el layout actual asume desktop. En movil el panel lateral tapa el mapa.
7. **Cuando Manuel re-ejecute el modelo y arregle los 613 outliers negativos**, quitar el clamp visual `Math.max(0, vmin)` de `Sit3Map.tsx` y `Sit3Legend.tsx` (volver a usar `rawMin`).
8. **Tests E2E con Playwright** o **Vitest + React Testing Library** para los componentes Sit3.

## Repositorios relacionados

- **Backend API:** https://huggingface.co/spaces/analiticalastdance/geovision-cali-api
- **Dataset (manifest, CSVs, checkpoints, handoffs):** https://huggingface.co/datasets/analiticalastdance/geovision-cali

## Equipo

- **Sebastian Belalcazar** - datos, backend, frontend, deploy
- **Manuel Gruezo** - CLIP, SAE, ConvLSTM, ST-Kriging (handoffs `README_HANDOFF_SITUACION3.md`, `UI_UX_CONTRACT_SITUACION3.md`)
- **Luis Angel Garcia Garcia** - geoestadistica, K-Means, frontend pulido

Universidad Autonoma de Occidente, Cali - Analitica de Datos I (2026-05).
