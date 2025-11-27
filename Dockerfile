# Multi-stage build for ML CI/CD Showcase
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-monitoring.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-monitoring.txt

# Copy source code and scripts
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY serve.py .
COPY train.py .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data

# Set Python path
ENV PYTHONPATH=/app

# Default command (for base)
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Development stage
FROM base as development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["/bin/bash"]

# Production stage (API serving)
FROM base as production
# Expose API port
EXPOSE 8000
# Run the API server
CMD ["python", "serve.py"]