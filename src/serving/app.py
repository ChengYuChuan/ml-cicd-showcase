"""Minimal FastAPI app with Prometheus metrics."""
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import torch
from pathlib import Path

from src.models.cnn_classifier import CNNClassifier
from src.config import CNNConfig

app = FastAPI(title="ML Model API")

# Prometheus metrics
predictions_total = Counter(
    'ml_predictions_total', 
    'Total predictions',
    ['model', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency',
    ['model']
)

# Load model
cnn_config = CNNConfig()
cnn = CNNClassifier(cnn_config)
model_path = Path("models/cnn_mnist.pth")
if model_path.exists():
    cnn.load_model(model_path)

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

@app.post("/predict")
def predict(data: dict):
    """CNN prediction endpoint."""
    try:
        start = time.time()
        
        # Convert to tensor
        image = torch.tensor(data['data']).reshape(1, 1, 28, 28)
        
        # Predict
        pred = cnn.predict(image)
        
        # Metrics
        latency = time.time() - start
        prediction_latency.labels(model='cnn').observe(latency)
        predictions_total.labels(model='cnn', status='success').inc()
        
        return {
            "prediction": int(pred[0]),
            "latency_ms": round(latency * 1000, 2)
        }
    except Exception as e:
        predictions_total.labels(model='cnn', status='error').inc()
        return {"error": str(e)}, 500

@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}