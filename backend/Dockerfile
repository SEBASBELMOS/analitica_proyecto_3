# ============================================================
# Multi-stage Dockerfile para GeoVision-CLIP Cali — Backend API
# Cumple requisito PDF: "imagen multi-stage" (Sit. 4.5)
# ============================================================

# ---- Stage 1: builder (instala dependencias compiladas) ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar herramientas de build (luego se descartan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar deps en un venv aislado para copiar después
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime (imagen final, ligera) ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="GeoVision-CLIP Cali API"
LABEL org.opencontainers.image.description="API de estimación de contaminación atmosférica para Cali"
LABEL org.opencontainers.image.source="https://huggingface.co/datasets/analiticalastdance/geovision-cali"

# Usuario no-root por seguridad
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Copiar solo el venv del builder (sin las herramientas de compilación)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Crear directorio para checkpoints (volumen)
RUN mkdir -p /app/models && chown -R app:app /app

# Copiar código de la app
COPY --chown=app:app main.py .
COPY --chown=app:app situacion3/ ./situacion3/
COPY --chown=app:app panel/ ./panel/
COPY --chown=app:app data/ ./data/

USER app

EXPOSE 7860

# HuggingFace Spaces requiere puerto 7860 por convención
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

# Healthcheck Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1
