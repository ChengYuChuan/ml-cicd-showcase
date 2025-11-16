# Multi-stage build for ML CI/CD Showcase
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Development stage
FROM base as development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["/bin/bash"]

# Production stage (minimal)
FROM base as production
# Only copy necessary files
RUN pip install --no-cache-dir pytest
CMD ["python", "-c", "from src.models.cnn_classifier import CNNClassifier; from src.models.rag_system import RAGSystem; print('ML models loaded successfully!')"]
