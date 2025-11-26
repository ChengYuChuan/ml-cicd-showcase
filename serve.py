#!/usr/bin/env python3
"""Start the ML model serving application with monitoring."""
import uvicorn
import sys
from pathlib import Path

# 確保可以 import src
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting ML Model API with Prometheus metrics...")
    print("API docs available at: http://localhost:8000/docs")
    print("Metrics available at: http://localhost:8000/metrics")
    
    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )