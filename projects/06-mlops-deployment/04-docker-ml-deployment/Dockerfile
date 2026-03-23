# ===========================================================================
# Multi-stage Dockerfile for ML Model Deployment
# ===========================================================================
# Stage 1: "trainer"  – installs deps and trains the model
# Stage 2: "serving"  – slim runtime image that serves predictions
#
# Build:
#   docker build -t ml-model-api .
#
# Run:
#   docker run -p 8000:8000 ml-model-api
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1 – Training
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS trainer

WORKDIR /build

# Install only what is needed for training
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Train the model and persist artefacts to /build/models
RUN python train.py --output /build/models

# ---------------------------------------------------------------------------
# Stage 2 – Serving
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS serving

LABEL maintainer="datascience-portfolio"
LABEL description="Containerised ML model serving with FastAPI"

WORKDIR /app

# Install runtime dependencies (lighter than training deps)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy trained model artefacts from the trainer stage
COPY --from=trainer /build/models /app/models

# Environment variables
ENV MODEL_DIR=/app/models
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check (Docker HEALTHCHECK instruction)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
