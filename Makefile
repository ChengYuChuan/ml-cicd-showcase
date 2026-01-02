.PHONY: help install install-dev test test-fast test-cov lint format clean docker-build docker-test train mlflow-up mlflow-down train-mlflow demo

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup & Dependencies:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          - Run all tests"
	@echo "  make test-fast     - Run fast tests only"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Train models"
	@echo "  make train-mlflow  - Train models with MLflow tracking"
	@echo ""
	@echo "Services:"
	@echo "  make serve         - Start API server locally"
	@echo "  make mlflow-up     - Start MLflow server"
	@echo "  make all-up        - Start all services (API + Monitoring + MLflow)"
	@echo "  make all-down      - Stop all services"
	@echo ""
	@echo "Demo:"
	@echo "  make demo          - Run full interactive demo"
	@echo "  make demo-quick    - Run quick demo (services only)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-test   - Run tests in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  make traffic       - Generate test traffic"
	@echo "  make clean         - Clean temporary files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint:
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

docker-build:
	docker-compose build

docker-test:
	docker-compose run ml-test

train:
	python train.py --model both

train-cnn:
	python train.py --model cnn

train-rag:
	python train.py --model rag

serve:
	python serve.py

serve-dev:
	uvicorn src.serving.app:app --reload --port 8000

monitoring-up:
	docker-compose up -d ml-api prometheus grafana
	@echo "Services available at:"
	@echo "  API: http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

monitoring-down:
	docker-compose down

traffic:
	python scripts/generate_traffic.py

# MLflow commands
mlflow-up:
	docker-compose up -d mlflow
	@echo "MLflow server starting..."
	@sleep 3
	@echo "MLflow UI available at: http://localhost:5000"

mlflow-down:
	docker-compose stop mlflow

train-mlflow:
	python train.py --model cnn --mlflow

train-mlflow-register:
	python train.py --model cnn --mlflow --register

# Start all services
all-up:
	docker-compose up -d mlflow ml-api prometheus grafana
	@echo ""
	@echo "All services starting..."
	@sleep 5
	@echo ""
	@echo "Services available at:"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"

all-down:
	docker-compose down

# Demo commands
demo:
	@chmod +x scripts/demo.sh
	./scripts/demo.sh all

demo-quick:
	@chmod +x scripts/demo.sh
	./scripts/demo.sh services

demo-cleanup:
	@chmod +x scripts/demo.sh
	./scripts/demo.sh cleanup