"""FastAPI application for serving ML models with Prometheus metrics."""
import time
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import CNNConfig, RAGConfig
from src.models.cnn_classifier import CNNClassifier
from src.models.rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for CNN and RAG model predictions with monitoring",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
predictions_total = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model_type', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_type']
)

active_requests = Gauge(
    'ml_active_requests',
    'Number of active requests',
    ['model_type']
)

error_total = Counter(
    'ml_errors_total',
    'Total number of errors',
    ['model_type', 'error_type']
)

# Pydantic models for request/response validation
class CNNPredictionRequest(BaseModel):
    """Request schema for CNN prediction."""
    data: List[List[float]] = Field(
        ..., 
        description="Flattened 28x28 image data (784 values)",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [[0.0] * 784]
            }
        }


class CNNPredictionResponse(BaseModel):
    """Response schema for CNN prediction."""
    prediction: int = Field(..., description="Predicted digit (0-9)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query."""
    text: str = Field(..., min_length=1, description="Query text")
    top_k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""
    answer: str
    context: List[str]
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]


# Global model instances
cnn_model = None
rag_model = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global cnn_model, rag_model
    
    print("Initializing models...")
    
    # Load CNN model
    try:
        cnn_config = CNNConfig()
        cnn_model = CNNClassifier(cnn_config)
        
        model_path = Path("models/cnn_mnist.pth")
        if model_path.exists():
            cnn_model.load_model(model_path)
            print("✓ CNN model loaded from disk")
            
            # Update accuracy metric if available
            metrics = cnn_model.get_metrics()
            if 'accuracy' in metrics:
                model_accuracy.labels(model_type='cnn').set(metrics['accuracy'])
        else:
            print("⚠ CNN model not found, using untrained model")
            
    except Exception as e:
        print(f"✗ Failed to load CNN model: {e}")
        cnn_model = None
    
    # Load RAG model (only if API key is available)
    try:
        import os
        if os.getenv('ANTHROPIC_API_KEY'):
            rag_config = RAGConfig(collection_name="api_knowledge_base")
            rag_model = RAGSystem(rag_config)
            
            # Try to load saved state
            rag_path = Path("models/rag_state.json")
            if rag_path.exists():
                rag_model.load_model(rag_path)
                print("✓ RAG model loaded from disk")
            else:
                print("⚠ RAG model state not found, initializing empty")
        else:
            print("⚠ ANTHROPIC_API_KEY not set, RAG endpoint disabled")
            rag_model = None
            
    except Exception as e:
        print(f"✗ Failed to load RAG model: {e}")
        rag_model = None
    
    print("Startup complete!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Model API",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "cnn": cnn_model is not None and cnn_model.is_trained,
            "rag": rag_model is not None and rag_model.is_trained
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.post("/predict/cnn", response_model=CNNPredictionResponse)
async def predict_cnn(request: CNNPredictionRequest):
    """
    CNN prediction endpoint.
    
    Predicts a digit (0-9) from a 28x28 grayscale image.
    """
    if cnn_model is None:
        error_total.labels(model_type='cnn', error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="CNN model not loaded")
    
    active_requests.labels(model_type='cnn').inc()
    
    try:
        start_time = time.time()
        
        # Convert input to tensor
        try:
            # Expect flattened 784 values or 28x28 array
            data = request.data
            if len(data) == 1 and len(data[0]) == 784:
                # Flattened format
                image_tensor = torch.tensor(data[0]).reshape(1, 1, 28, 28).float()
            elif len(data) == 28 and all(len(row) == 28 for row in data):
                # 28x28 format
                image_tensor = torch.tensor(data).reshape(1, 1, 28, 28).float()
            else:
                raise ValueError("Invalid input shape. Expected 784 values or 28x28 array")
                
        except Exception as e:
            error_total.labels(model_type='cnn', error_type='invalid_input').inc()
            raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")
        
        # Make prediction
        prediction = cnn_model.predict(image_tensor)
        
        # Record metrics
        latency = time.time() - start_time
        prediction_latency.labels(model_type='cnn').observe(latency)
        predictions_total.labels(model_type='cnn', status='success').inc()
        
        return CNNPredictionResponse(
            prediction=int(prediction[0]),
            latency_ms=round(latency * 1000, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_total.labels(model_type='cnn', error_type='prediction_failed').inc()
        predictions_total.labels(model_type='cnn', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        active_requests.labels(model_type='cnn').dec()


@app.post("/predict/rag", response_model=RAGQueryResponse)
async def predict_rag(request: RAGQueryRequest):
    """
    RAG query endpoint.
    
    Retrieves relevant context and generates an answer using Claude.
    """
    if rag_model is None:
        error_total.labels(model_type='rag', error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=503, 
            detail="RAG model not loaded. Check ANTHROPIC_API_KEY."
        )
    
    if not rag_model.is_trained:
        error_total.labels(model_type='rag', error_type='no_documents').inc()
        raise HTTPException(
            status_code=400,
            detail="RAG model has no documents. Use POST /ingest to add documents."
        )
    
    active_requests.labels(model_type='rag').inc()
    
    try:
        start_time = time.time()
        
        # Make prediction
        result = rag_model.predict(request.text)
        
        # Record metrics
        latency = time.time() - start_time
        prediction_latency.labels(model_type='rag').observe(latency)
        predictions_total.labels(model_type='rag', status='success').inc()
        
        return RAGQueryResponse(
            answer=result['answer'],
            context=result['context'],
            latency_ms=round(latency * 1000, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_total.labels(model_type='rag', error_type='prediction_failed').inc()
        predictions_total.labels(model_type='rag', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    finally:
        active_requests.labels(model_type='rag').dec()


@app.post("/ingest", response_model=Dict[str, str])
async def ingest_documents(documents: List[str]):
    """
    Ingest documents into the RAG system.
    
    This is a utility endpoint for adding documents to the vector database.
    """
    if rag_model is None:
        raise HTTPException(status_code=503, detail="RAG model not loaded")
    
    try:
        metrics = rag_model.ingest_documents(documents)
        return {
            "status": "success",
            "message": f"Ingested {metrics['num_documents']} documents",
            "ingestion_time_sec": str(metrics['ingestion_time_sec'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)