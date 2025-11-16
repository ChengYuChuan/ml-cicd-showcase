# Quick Start Guide

This guide will get you up and running with the ML CI/CD Showcase in under 10 minutes.

## Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Anthropic API key for RAG system

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ChengYuChuan/ml-cicd-showcase.git
cd ml-cicd-showcase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key (optional for RAG)
# ANTHROPIC_API_KEY=your_key_here
```

## Step 3: Quick Test

### Option A: Run Tests

```bash
# Run all tests (skip slow ones)
pytest tests/ -v -m "not slow"

# Run specific tests
pytest tests/test_cnn.py -v
```

### Option B: Train Models

```bash
# Train both models
python train.py --model both

# Train only CNN
python train.py --model cnn --epochs 2

# Setup only RAG
python train.py --model rag
```

## Step 4: Explore the Code

### CNN Classifier Example

```python
from src.models.cnn_classifier import CNNClassifier
from src.config import CNNConfig

# Create and train
config = CNNConfig(num_epochs=2)
model = CNNClassifier(config)
metrics = model.train()

print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

### RAG System Example

```python
from src.models.rag_system import RAGSystem
from src.config import RAGConfig

# Setup RAG
config = RAGConfig()
rag = RAGSystem(config)

# Add knowledge
docs = ["Python is a programming language."]
rag.ingest_documents(docs)

# Query
result = rag.predict("What is Python?")
print(result['answer'])
```

## Step 5: Run in Docker (Optional)

```bash
# Build and run tests
docker-compose run ml-test

# Development environment
docker-compose run ml-dev
```

## Next Steps

- ✅ Read the full [README.md](README.md)
- ✅ Explore the [architecture documentation](docs/architecture.md)
- ✅ Check out the [CI/CD pipeline](.github/workflows/ci.yml)
- ✅ Try adding a new model type

## Troubleshooting

### MNIST Download Issues

If MNIST download fails:
```bash
# Manually download to data/MNIST/
# The dataset will be automatically downloaded on first run
```

### Anthropic API Key

If you don't have an API key:
- Skip RAG tests: `pytest tests/test_cnn.py tests/test_integration.py`
- The CNN classifier works without any API key

### Import Errors

Make sure you're in the project root and have activated the virtual environment:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Common Commands

```bash
# Format code
black src/ tests/

# Run linting
flake8 src/ tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Install pre-commit hooks
pre-commit install
```

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Review the test files for usage examples

---

**Estimated time to complete**: 5-10 minutes

**You're all set!** 🚀
