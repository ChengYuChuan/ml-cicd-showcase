#!/usr/bin/env python3
"""Quick training script for demonstrating the models with optional MLflow tracking."""
import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import CNNConfig, MLflowConfig, RAGConfig
from src.models.cnn_classifier import CNNClassifier
from src.models.rag_system import RAGSystem
from src.utils.metrics import format_metrics


def get_mlflow_tracker(use_mlflow: bool):
    """Get MLflow tracker if enabled."""
    if not use_mlflow:
        return None

    try:
        from src.tracking.mlflow_tracker import MLflowTracker

        config = MLflowConfig()
        tracker = MLflowTracker(config)
        print(f"MLflow tracking enabled: {config.tracking_uri}")
        return tracker
    except Exception as e:
        print(f"Warning: Could not initialize MLflow tracker: {e}")
        print("Continuing without MLflow tracking...")
        return None


def train_cnn(
    epochs: int = 3,
    save: bool = True,
    use_mlflow: bool = False,
    register_model: bool = False,
):
    """Train CNN classifier with optional MLflow tracking."""
    print("\n" + "=" * 50)
    print("Training CNN Classifier on MNIST")
    print("=" * 50 + "\n")

    tracker = get_mlflow_tracker(use_mlflow)
    config = CNNConfig(num_epochs=epochs)
    model = CNNClassifier(config, mlflow_tracker=tracker)

    # Use train_with_tracking if MLflow is enabled, otherwise regular train
    if tracker:
        metrics = model.train_with_tracking(
            run_name="cnn-mnist-training",
            tags={"model_type": "cnn", "dataset": "mnist"},
            register_model=register_model,
            registered_model_name="cnn-mnist",
        )
    else:
        metrics = model.train()

    print("\n" + format_metrics(metrics))

    if save:
        save_path = Path("models/cnn_mnist.pth")
        model.save_model(save_path)
        print(f"\n✓ Model saved to {save_path}")

        # Log artifact to MLflow if tracking
        if tracker:
            try:
                tracker.log_artifact(str(save_path))
                print("✓ Model artifact logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not log artifact to MLflow: {e}")

    return model, metrics


def setup_rag(
    save: bool = True,
    use_mlflow: bool = False,
):
    """Setup RAG system with sample documents and optional MLflow tracking."""
    print("\n" + "=" * 50)
    print("Setting up RAG System")
    print("=" * 50 + "\n")

    tracker = get_mlflow_tracker(use_mlflow)
    config = RAGConfig(collection_name="demo_knowledge_base")
    rag = RAGSystem(config, mlflow_tracker=tracker)

    # Load sample documents
    docs_file = Path("data/knowledge_base/sample_documents.md")
    if docs_file.exists():
        print(f"Loading documents from {docs_file}...")
        with open(docs_file, "r") as f:
            content = f.read()
        # Split by headers
        documents = [
            doc.strip()
            for doc in content.split("## ")
            if doc.strip() and not doc.startswith("#")
        ]
    else:
        # Fallback to hardcoded documents
        documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual information.",
        ]

    print(f"Ingesting {len(documents)} documents...")

    # Use train_with_tracking if MLflow is enabled
    if tracker:
        metrics = rag.train_with_tracking(
            run_name="rag-document-ingestion",
            tags={"model_type": "rag", "embedding_model": config.embedding_model},
            documents=documents,
        )
    else:
        metrics = rag.ingest_documents(documents)

    print("\n" + format_metrics(metrics))

    # Test query
    print("\n--- Testing Query ---")
    query = "What is Python?"
    result = rag.predict(query)
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")

    if save:
        save_path = Path("models/rag_state.json")
        rag.save_model(save_path)
        print(f"\n✓ RAG state saved to {save_path}")

    return rag, metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick training script for ML models with MLflow support"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "rag", "both"],
        default="both",
        help="Which model to train/setup",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs for CNN"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save models")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLflow Model Registry (requires --mlflow)",
    )

    args = parser.parse_args()

    save = not args.no_save
    use_mlflow = args.mlflow
    register_model = args.register

    if register_model and not use_mlflow:
        print("Warning: --register requires --mlflow. Enabling MLflow tracking.")
        use_mlflow = True

    try:
        if args.model in ["cnn", "both"]:
            train_cnn(
                epochs=args.epochs,
                save=save,
                use_mlflow=use_mlflow,
                register_model=register_model,
            )

        if args.model in ["rag", "both"]:
            setup_rag(save=save, use_mlflow=use_mlflow)

        print("\n" + "=" * 50)
        print("✓ All tasks completed successfully!")
        if use_mlflow:
            print("✓ Experiments logged to MLflow")
            print("  View at: http://localhost:5000")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
