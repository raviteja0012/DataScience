# syntax=docker/dockerfile:1.6
# =============================================================================
# Multi-Source Data Integration - Multi-stage Dockerfile
# =============================================================================
# Stage 1: builder  - installs Python dependencies into an isolated prefix
# Stage 2: runtime  - minimal image that runs the data integration pipeline
# =============================================================================

# -----------------------------------------------------------------------------
# Builder stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="Ravi Potluru"
LABEL stage="builder"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time OS deps - some database drivers compile native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt


# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Ravi Potluru"
LABEL project="multi-source-data-integration"
LABEL description="Multi-source data integration pipeline (Oracle / MSSQL -> Snowflake)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    DEMO_MODE=true

# Create non-root user (UID 1000)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Pull in installed Python deps from the builder stage
COPY --from=builder /install /usr/local

# Copy project source with proper ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser setup.py /app/setup.py
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt

# Pre-create writable output directory and hand it to the non-root user
RUN mkdir -p /app/output /app/reports \
    && chown -R appuser:appuser /app

USER appuser

# Lightweight healthcheck - confirms the pipeline module imports cleanly
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import src.main" || exit 1

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["run-pipeline", "--demo"]
