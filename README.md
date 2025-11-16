# ML CI/CD Showcase

[![CI/CD Pipeline](https://github.com/yourusername/ml-cicd-showcase/workflows/ML%20Models%20CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/ml-cicd-showcase/actions)
[![codecov](https://codecov.io/gh/yourusername/ml-cicd-showcase/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/ml-cicd-showcase)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional showcase project demonstrating **ML/MLOps skills** with a **unified CI/CD pipeline** for multiple model types.

## 🎯 Project Overview

This project demonstrates:
- ✅ **Unified ML Framework**: Abstract base class for consistent model handling
- ✅ **Two Model Types**: 
  - CNN Classifier (MNIST, ~50K parameters)
  - RAG System (ChromaDB + Claude)
- ✅ **Complete CI/CD Pipeline**: Automated testing, validation, and deployment
- ✅ **Code Quality**: Black, Flake8, MyPy, pytest with >80% coverage
- ✅ **Containerization**: Multi-stage Docker builds
- ✅ **Performance Monitoring**: Automated benchmarks and thresholds

**Perfect for**: Showcasing MLOps skills in job applications for ML Engineer / MLOps roles.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│           BaseMLModel (Abstract)                │
│  - train()  - predict()  - evaluate()           │
│  - save()   - load()     - get_metrics()        │
└──────────────┬──────────────────────┬───────────┘
               │                      │
       ┌───────▼────────┐    ┌───────▼──────────┐
       │ CNNClassifier  │    │   RAGSystem      │
       │                │    │                  │
       │ • TinyConvNet  │    │ • SentenceT5     │
       │ • MNIST        │    │ • ChromaDB       │
       │ • PyTorch      │    │ • Claude API     │
       └────────────────┘    └──────────────────┘
```

### Key Design Principle
**Unified Interface** → Both models implement the same abstract base class, allowing the CI/CD pipeline to handle them identically.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Anthropic API key (for RAG system)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-cicd-showcase.git
cd ml-cicd-showcase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Set up API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Quick Test

```bash
# Run all tests
pytest tests/ -v

# Run specific model tests
pytest tests/test_cnn.py -v
pytest tests/test_rag.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 💻 Usage Examples

### CNN Classifier

```python
from src.models.cnn_classifier import CNNClassifier
from src.config import CNNConfig

# Configure and train
config = CNNConfig(num_epochs=3, batch_size=64)
model = CNNClassifier(config)

# Train on MNIST
metrics = model.train()
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save model
model.save_model("models/cnn_mnist.pth")

# Make predictions
import torch
sample = torch.randn(1, 1, 28, 28)
prediction = model.predict(sample)
```

### RAG System

```python
from src.models.rag_system import RAGSystem
from src.config import RAGConfig

# Initialize RAG
config = RAGConfig()
rag = RAGSystem(config)

# Ingest documents
documents = [
    "Python is a programming language.",
    "Machine learning uses data to improve.",
]
rag.ingest_documents(documents)

# Query the system
result = rag.predict("What is Python?")
print(f"Answer: {result['answer']}")
print(f"Context: {result['context']}")
```

## 🧪 Testing

### Test Structure
```
tests/
├── conftest.py          # Shared fixtures
├── test_cnn.py          # CNN-specific tests
├── test_rag.py          # RAG-specific tests
└── test_integration.py  # Cross-model tests
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_cnn.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"

# Run only fast tests in CI
pytest tests/ -m "not slow" --maxfail=1
```

## 🔄 CI/CD Pipeline

### Pipeline Stages

1. **Code Quality** (< 2 min)
   - Black formatting
   - Flake8 linting
   - MyPy type checking
   - isort import sorting

2. **CNN Tests** (< 5 min)
   - Unit tests
   - Training validation
   - Performance thresholds (>85% accuracy)

3. **RAG Tests** (< 3 min)
   - Unit tests
   - Retrieval quality validation
   - API integration tests

4. **Integration Tests** (< 5 min)
   - Cross-model compatibility
   - End-to-end workflows
   - Coverage report generation

5. **Docker Build** (< 3 min)
   - Multi-stage build
   - Image testing
   - Artifact upload

6. **Benchmarks** (< 5 min)
   - Performance metrics
   - Latency measurements
   - Model size validation

### Performance Thresholds

| Model | Metric | Threshold | Current |
|-------|--------|-----------|---------|
| CNN   | Accuracy | >85% | ~95% |
| CNN   | Latency | <100ms | ~15ms |
| CNN   | Size | <1MB | ~0.2MB |
| RAG   | Precision | >30% | ~60% |
| RAG   | Latency | <5s | ~2s |

## 🐳 Docker Usage

### Build and Run

```bash
# Build all stages
docker-compose build

# Development environment
docker-compose run ml-dev

# Run tests in Docker
docker-compose run ml-test

# Production deployment
docker-compose up ml-prod
```

### Manual Docker Commands

```bash
# Build image
docker build -t ml-cicd-showcase .

# Run tests
docker run --rm ml-cicd-showcase pytest tests/ -v

# Interactive shell
docker run -it --rm ml-cicd-showcase /bin/bash
```

## 📊 Model Performance

### CNN Classifier
- **Architecture**: TinyConvNet (~50K parameters)
- **Dataset**: MNIST (60K train, 10K test)
- **Training Time**: ~3 min (3 epochs, CPU)
- **Test Accuracy**: ~95%
- **Inference**: ~15ms per image
- **Model Size**: 0.2MB

### RAG System
- **Embedding**: all-MiniLM-L6-v2 (80MB)
- **Vector DB**: ChromaDB (local)
- **LLM**: Claude Sonnet 4
- **Retrieval**: ~30ms per query
- **Generation**: ~2s per answer
- **Precision**: ~60% on test queries

## 📁 Project Structure

```
ml-cicd-showcase/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── src/
│   ├── models/
│   │   ├── base_model.py       # Abstract base class
│   │   ├── cnn_classifier.py   # CNN implementation
│   │   └── rag_system.py       # RAG implementation
│   ├── utils/
│   │   └── metrics.py          # Utility functions
│   └── config.py               # Configuration management
├── tests/
│   ├── conftest.py             # pytest fixtures
│   ├── test_cnn.py             # CNN tests
│   ├── test_rag.py             # RAG tests
│   └── test_integration.py     # Integration tests
├── data/
│   ├── sample_images/          # Sample data
│   └── knowledge_base/         # RAG documents
├── models/                     # Saved models
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml              # Modern Python config
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Dev dependencies
└── README.md
```

## 🎓 Key Learnings & Design Decisions

### 1. Unified Interface Pattern
**Why**: Different model types (CNN vs RAG) can use the same CI/CD pipeline
**How**: Abstract `BaseMLModel` class with standardized methods
**Benefit**: Easy to add new models without changing infrastructure

### 2. Lightweight Models for CI
**Why**: GitHub Actions has limited compute and time
**How**: TinyConvNet (<50K params) instead of ResNet18 (11M params)
**Benefit**: Tests complete in <5 minutes instead of >30 minutes

### 3. Modular Configuration
**Why**: Different environments need different settings
**How**: Dataclass-based configs (CNNConfig, RAGConfig)
**Benefit**: Type-safe, easy to test, environment-specific

### 4. Comprehensive Testing
**Why**: Catch bugs before deployment
**How**: Unit, integration, and performance tests with >80% coverage
**Benefit**: Confidence in code quality and model performance

## 🛠️ Development

### Setup Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Code Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

### Adding a New Model

1. Create new model class inheriting from `BaseMLModel`
2. Implement required methods: `train()`, `predict()`, `evaluate()`
3. Add configuration dataclass in `config.py`
4. Create test file in `tests/`
5. Pipeline automatically handles it! 🎉

## 📈 Future Enhancements

- [ ] Add DVC for data/model versioning
- [ ] Implement model registry (MLflow)
- [ ] Add more model types (transformer, GNN)
- [ ] Set up automated deployment
- [ ] Add performance monitoring dashboard
- [ ] Implement A/B testing framework

## 🤝 Contributing

This is a showcase project for job applications. If you'd like to suggest improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 💡 For Recruiters / Hiring Managers

This project demonstrates:

✅ **MLOps Best Practices**: CI/CD, testing, containerization, monitoring
✅ **Software Engineering**: Clean code, design patterns, type hints, documentation
✅ **ML Knowledge**: Model selection, training, evaluation, optimization
✅ **Production Ready**: Error handling, logging, performance monitoring
✅ **Modern Tools**: GitHub Actions, Docker, pytest, type hints, pre-commit hooks

**Time Investment**: ~1 week (as planned) ✓
**Lines of Code**: ~2000+ (production quality)
**Test Coverage**: >80%
**Documentation**: Comprehensive

**Tech Stack**:
- **Languages**: Python 3.10
- **ML Frameworks**: PyTorch, Sentence Transformers
- **Vector DB**: ChromaDB
- **LLM**: Claude (Anthropic)
- **Testing**: pytest, pytest-cov
- **CI/CD**: GitHub Actions
- **Containerization**: Docker, docker-compose
- **Code Quality**: Black, Flake8, MyPy, isort, pre-commit

**Questions?** Feel free to reach out or open an issue!
