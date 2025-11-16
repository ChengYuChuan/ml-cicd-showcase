.PHONY: help install install-dev test test-fast test-cov lint format clean docker-build docker-test train

help:
	@echo "Available commands:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-fast     - Run fast tests only"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-test   - Run tests in Docker"
	@echo "  make train         - Train models"

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
