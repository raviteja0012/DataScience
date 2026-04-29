# syntax=docker/dockerfile:1.6
# =============================================================================
# Payment Intelligence Agent - Multi-stage Dockerfile
# =============================================================================
# Stage 1: builder  - compiles native deps and installs Python packages
# Stage 2: runtime  - slim image that serves the Streamlit app on port 8501
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

# Build-time OS deps - faiss / sentence-transformers need a C/C++ toolchain
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
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
LABEL project="payment-intelligence-agent"
LABEL description="Streamlit-based payment intelligence agent with RAG and anomaly detection"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO \
    DEMO_MODE=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Runtime OS deps required by faiss / numpy at execution time
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Pull installed Python deps from builder
COPY --from=builder /install /usr/local

# Copy project source with proper ownership
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser setup.py /app/setup.py
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt

# Cache directory for sentence-transformers model downloads
RUN mkdir -p /app/.cache /app/embeddings_cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Streamlit ships an internal health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
