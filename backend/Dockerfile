# ============================================
# Multi-stage build for Python Backend
# ============================================

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# ============================================
# Dependencies Stage
# ============================================
FROM base as dependencies

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    gfortran \
    # Additional utilities
    wget \
    && rm -rf /var/lib/apt/lists/*

# Try to install optional packages (may not exist in all Debian versions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxvidcore-dev \
    libx264-dev \
    || true \
    && rm -rf /var/lib/apt/lists/*

# Try to install BLAS library (prefer OpenBLAS over ATLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    || apt-get install -y --no-install-recommends libatlas-base-dev \
    || true \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Production Stage
# ============================================
FROM base as production

# Install runtime dependencies for OpenCV (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libavcodec61 \
    libavformat61 \
    libswscale8 \
    libv4l-0 \
    && rm -rf /var/lib/apt/lists/*

# Try to install BLAS runtime library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    || apt-get install -y --no-install-recommends libatlas3-base \
    || true \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs \
    /app/uploads \
    /app/uploads/inference_results \
    /app/uploads/save_frame \
    /app/uploads/templates \
    /app/uploads/visualizations \
    /app/uploads/avatar \
    /app/camera_settings \
    /app/inference_state

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
