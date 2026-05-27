# ============================================================
# Multi-stage Dockerfile para GeoVision-CLIP Cali — Frontend
# Cumple requisito PDF: imagen multi-stage (Sit. 4.5)
# Para deploy en HuggingFace Spaces (Docker SDK) puerto 7860
# ============================================================

# ---- Stage 1: builder (compila el bundle de Vite) ----
FROM node:20-alpine AS builder

WORKDIR /build

# Copia solo manifests primero (cache de capas mas eficiente)
COPY package.json package-lock.json* ./
RUN npm ci || npm install

# Copia el codigo y construye
COPY . .
RUN npm run build


# ---- Stage 2: runtime (servidor estatico ligero) ----
# Usamos `serve` de Vercel en lugar de `vite preview` porque vite preview
# tiene proteccion `allowedHosts` que rechaza requests desde HF Spaces
# (DNS rebinding protection que no se desactiva facil).
# `serve` es un servidor estatico SPA sin esa proteccion, ideal para deploy.
FROM node:20-alpine AS runtime

LABEL org.opencontainers.image.title="GeoVision-CLIP Cali Frontend"
LABEL org.opencontainers.image.description="Mapa interactivo de contaminacion atmosferica en Cali"
LABEL org.opencontainers.image.source="https://huggingface.co/spaces/analiticalastdance/geovision-cali-frontend"

WORKDIR /app

# Instalar serve globalmente (chico, ~10 MB)
RUN npm install -g serve

# Copiar SOLO el bundle compilado (sin node_modules ni package.json)
COPY --from=builder /build/dist ./dist

# Usuario no-root
RUN addgroup -S app && adduser -S app -G app && chown -R app:app /app
USER app

EXPOSE 7860

# `serve` sirve la carpeta dist/ en puerto 7860, con fallback SPA (-s)
# que devuelve index.html para rutas client-side
CMD ["serve", "-s", "dist", "-l", "7860"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --spider -q http://localhost:7860 || exit 1
