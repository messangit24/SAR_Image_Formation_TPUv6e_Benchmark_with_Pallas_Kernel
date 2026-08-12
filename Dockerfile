# TPUbench: Multi-Stage Production Container for CPU & Cloud TPU Workloads
FROM python:3.12-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/home/tpubench/.local/bin:$PATH"

# Install system dependencies for HDF4, scientific libraries, and Cloud tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libhdf4-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create non-root system user and working directory (Wiz IaC Security: Rule 9e90c869-0c5e-40c1-976a-d6a2a3fa96c3)
RUN groupadd --system --gid 10001 tpubench && \
    useradd --system --uid 10001 --gid tpubench --create-home --home-dir /home/tpubench tpubench && \
    mkdir -p /workspace /workspace/out && \
    chown -R tpubench:tpubench /workspace /home/tpubench

WORKDIR /workspace

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Synchronize dependencies with uv
RUN uv sync --frozen --no-install-project

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md README_MODIS.md ./

# Install project package in editable mode
RUN uv pip install --no-deps -e .

# Install Cloud TPU runtime support for JAX
RUN uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html || true

# Set workspace permissions for non-root user
RUN chown -R tpubench:tpubench /workspace

# Switch to non-root user
USER 10001:10001

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "src.modis.distributed_batch", "--synthetic-granules", "8"]

