# Multi-stage Docker build for NIfTI Model Project
# Optimized for both CPU and GPU execution with minimal final image size

# =============================================================================
# Stage 1: Base Python Environment with System Dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for medical imaging and ML
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    # Medical imaging libraries
    libffi-dev \
    libssl-dev \
    # Networking
    curl \
    wget \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# =============================================================================
# Stage 2: Dependencies Installation
# =============================================================================
FROM base as dependencies

# Set work directory
WORKDIR /app

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Copy source code (required for building the package)
COPY niftilearn/ ./niftilearn/

# Install production dependencies only
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 3: Application Build
# =============================================================================
FROM dependencies as builder

# Copy additional files
COPY config_examples/ ./config_examples/
COPY bin/ ./bin/

# Compile Python bytecode for faster startup
RUN python -m compileall niftilearn/

# =============================================================================
# Stage 4: Runtime Image (Final)
# =============================================================================
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOME=/app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries only
    libgomp1 \
    libffi8 \
    libssl3 \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r niftiuser && useradd -r -g niftiuser -d /app -s /bin/bash niftiuser

# Set work directory
WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Copy application from builder stage
COPY --from=builder --chown=niftiuser:niftiuser /app/niftilearn ./niftilearn
COPY --from=builder --chown=niftiuser:niftiuser /app/config_examples ./config_examples
COPY --from=builder --chown=niftiuser:niftiuser /app/bin ./bin

# Create necessary directories
RUN mkdir -p data outputs logs && \
    chown -R niftiuser:niftiuser /app

# Update PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER niftiuser

# Set default working directory for data processing
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import niftilearn; print('OK')" || exit 1

# No entrypoint defined - allows flexible usage
# Users can run: docker run <image> niftilearn train --config config.yaml
# Or: docker run -it <image> bash