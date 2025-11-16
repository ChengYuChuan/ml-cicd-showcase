# Architecture Documentation

## System Overview

The ML CI/CD Showcase implements a **unified framework** for training, testing, and deploying different types of machine learning models through a consistent interface.

## Design Principles

### 1. Unified Interface Pattern

All models inherit from `BaseMLModel`, providing:
- Consistent method signatures
- Standardized metrics
- Uniform save/load operations
- Common validation logic

**Benefits:**
- Single CI/CD pipeline handles all models
- Easy to add new model types
- Consistent developer experience

### 2. Separation of Concerns

```
┌─────────────────┐
│   Configuration │  ← CNNConfig, RAGConfig
├─────────────────┤
│   Model Layer   │  ← TinyConvNet, RAGSystem
├─────────────────┤
│   Utilities     │  ← Metrics, validation
├─────────────────┤
│   Tests         │  ← Unit, integration, e2e
└─────────────────┘
```

### 3. Test-Driven Development

- Unit tests for individual components
- Integration tests for workflows
- Performance tests for thresholds
- CI/CD automation

## Component Architecture

### BaseMLModel (Abstract Base Class)

```python
class BaseMLModel(ABC):
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, float]:
        """Training logic - returns metrics"""
        
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Inference logic"""
        
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Evaluation logic - returns metrics"""
        
    def save_model(self, path: Path) -> None:
        """Save model to disk"""
        
    def load_model(self, path: Path) -> None:
        """Load model from disk"""
        
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics"""
        
    def validate_performance(self, thresholds: Dict) -> bool:
        """Validate against thresholds"""
```

### CNN Classifier Architecture

```
Input (28x28x1)
    ↓
Conv1 (16 filters, 3x3)
    ↓
ReLU + MaxPool
    ↓
Conv2 (32 filters, 3x3)
    ↓
ReLU + MaxPool
    ↓
Flatten
    ↓
FC1 (64 neurons)
    ↓
Dropout (0.25)
    ↓
FC2 (10 neurons)
    ↓
Output (10 classes)
```

**Key Features:**
- ~50,000 parameters (lightweight)
- <1MB model size
- ~15ms inference time
- >95% accuracy on MNIST

### RAG System Architecture

```
User Query
    ↓
┌─────────────────────────┐
│  Embedding Model        │
│  (all-MiniLM-L6-v2)     │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│  Vector Database        │
│  (ChromaDB)             │
└───────────┬─────────────┘
            ↓
    Retrieve Top-K
            ↓
┌─────────────────────────┐
│  LLM (Claude)           │
│  Generate Answer        │
└───────────┬─────────────┘
            ↓
        Response
```

**Key Features:**
- Local vector database (no external deps)
- ~80MB embedding model
- ~2s end-to-end latency
- Configurable retrieval (top-k)

## CI/CD Pipeline Architecture

```
┌──────────────────────────────────────────────┐
│               GitHub Actions                  │
├──────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐                           │
│  │ Code Quality │ ← Black, Flake8, MyPy      │
│  └──────┬───────┘                           │
│         ↓                                    │
│  ┌──────────────┐   ┌──────────────┐        │
│  │  Test CNN    │   │  Test RAG    │        │
│  └──────┬───────┘   └──────┬───────┘        │
│         └───────┬───────────┘                │
│                 ↓                            │
│         ┌──────────────┐                     │
│         │ Integration  │                     │
│         └──────┬───────┘                     │
│                ↓                             │
│         ┌──────────────┐                     │
│         │ Docker Build │                     │
│         └──────┬───────┘                     │
│                ↓                             │
│         ┌──────────────┐                     │
│         │  Benchmarks  │                     │
│         └──────────────┘                     │
└──────────────────────────────────────────────┘
```

### Pipeline Stages

1. **Code Quality** (Parallel)
   - Format checking
   - Linting
   - Type checking

2. **Model Testing** (Parallel)
   - CNN: Unit + performance tests
   - RAG: Unit + integration tests

3. **Integration** (Sequential)
   - Cross-model compatibility
   - End-to-end workflows
   - Coverage reporting

4. **Docker** (Sequential)
   - Multi-stage build
   - Image testing
   - Artifact upload

5. **Benchmarks** (Optional)
   - Performance metrics
   - Regression detection

## Data Flow

### Training Flow

```
Configuration
    ↓
Model Initialization
    ↓
Data Loading
    ↓
Training Loop
    ↓
Evaluation
    ↓
Metrics Collection
    ↓
Model Persistence
```

### Inference Flow

```
Input Data
    ↓
Preprocessing
    ↓
Model Prediction
    ↓
Postprocessing
    ↓
Output
```

## Technology Stack

### Core ML
- **PyTorch**: Deep learning framework
- **Sentence Transformers**: Text embeddings
- **ChromaDB**: Vector database
- **Anthropic SDK**: LLM integration

### Development
- **pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **pre-commit**: Git hooks

### DevOps
- **GitHub Actions**: CI/CD
- **Docker**: Containerization
- **Codecov**: Coverage reporting

## Design Decisions

### Why TinyConvNet instead of ResNet18?

| Aspect | TinyConvNet | ResNet18 |
|--------|-------------|----------|
| Parameters | ~50K | ~11M |
| Training time (1 epoch) | ~1 min | ~10 min |
| Model size | 0.2MB | ~45MB |
| CI friendly | ✓ | ✗ |

**Decision**: TinyConvNet for fast CI/CD cycles.

### Why ChromaDB instead of Pinecone/Weaviate?

| Aspect | ChromaDB | Cloud Services |
|--------|----------|----------------|
| Setup | pip install | API keys + config |
| Cost | Free | $ per month |
| CI Testing | ✓ Local | ✗ Need service |
| Privacy | ✓ Local | Cloud storage |

**Decision**: ChromaDB for local development and testing.

### Why Abstract Base Class?

**Alternatives considered:**
1. Protocol (typing.Protocol)
2. Duck typing
3. Abstract base class ✓

**Rationale**: ABC provides:
- Explicit interface contracts
- Runtime verification
- Better IDE support
- Clear documentation

## Scalability Considerations

### Current Limitations
- Single model training (no distributed)
- CPU-only in CI (no GPU)
- Local vector database (not distributed)

### Future Enhancements
1. **Distributed Training**
   - PyTorch DDP
   - Ray integration

2. **Production Deployment**
   - Model serving (TorchServe, FastAPI)
   - Kubernetes deployment
   - A/B testing framework

3. **Monitoring**
   - MLflow tracking
   - Prometheus metrics
   - Grafana dashboards

## Security Considerations

### API Keys
- Stored in GitHub Secrets
- Never committed to repo
- Loaded via environment variables

### Dependencies
- Pinned versions in requirements.txt
- Regular security updates
- Vulnerability scanning (future)

## Performance Optimization

### CNN Optimizations
- Batch processing
- Mixed precision training (future)
- Model quantization (future)

### RAG Optimizations
- Embedding caching
- Batch retrieval
- Query result caching

## Testing Strategy

### Test Pyramid

```
      ┌─────┐
      │ E2E │       ← Integration tests
      ├─────┤
      │     │
      │Unit │       ← Unit tests
      │     │
      └─────┘
```

### Coverage Goals
- Unit tests: >80%
- Integration tests: Key workflows
- E2E tests: Complete pipelines

## Monitoring & Observability

### Metrics Tracked
- Model accuracy
- Inference latency
- Training time
- Model size
- Code coverage

### Logging
- Structured logging
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Contextual information

## Conclusion

This architecture prioritizes:
1. **Developer experience**: Unified interface, clear patterns
2. **CI/CD efficiency**: Lightweight models, fast tests
3. **Maintainability**: Clean code, comprehensive tests
4. **Extensibility**: Easy to add new models

The design balances **production readiness** with **showcase simplicity**, making it ideal for demonstrating MLOps skills in job applications.
