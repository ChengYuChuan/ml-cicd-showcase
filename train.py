#!/usr/bin/env python3
"""Quick training script for demonstrating the models."""
import argparse
import sys
from pathlib import Path

from src.config import CNNConfig, RAGConfig
from src.models.cnn_classifier import CNNClassifier
from src.models.rag_system import RAGSystem
from src.utils.metrics import format_metrics


def train_cnn(epochs: int = 3, save: bool = True):
    """Train CNN classifier."""
    print("\n" + "=" * 50)
    print("Training CNN Classifier on MNIST")
    print("=" * 50 + "\n")

    config = CNNConfig(num_epochs=epochs)
    model = CNNClassifier(config)

    metrics = model.train()

    print("\n" + format_metrics(metrics))

    if save:
        save_path = Path("models/cnn_mnist.pth")
        model.save_model(save_path)
        print(f"\n✓ Model saved to {save_path}")

    return model, metrics


def setup_rag(save: bool = True):
    """Setup RAG system with sample documents."""
    print("\n" + "=" * 50)
    print("Setting up RAG System")
    print("=" * 50 + "\n")

    config = RAGConfig(collection_name="demo_knowledge_base")
    rag = RAGSystem(config)

    # Load sample documents
    docs_file = Path("data/knowledge_base/sample_documents.md")
    if docs_file.exists():
        print(f"Loading documents from {docs_file}...")
        with open(docs_file, "r") as f:
            content = f.read()
        # Split by headers
        documents = [
            doc.strip() for doc in content.split("## ") if doc.strip() and not doc.startswith("#")
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
    parser = argparse.ArgumentParser(description="Quick training script for ML models")
    parser.add_argument(
        "--model",
        choices=["cnn", "rag", "both"],
        default="both",
        help="Which model to train/setup",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for CNN")
    parser.add_argument("--no-save", action="store_true", help="Don't save models")

    args = parser.parse_args()

    save = not args.no_save

    try:
        if args.model in ["cnn", "both"]:
            train_cnn(epochs=args.epochs, save=save)

        if args.model in ["rag", "both"]:
            setup_rag(save=save)

        print("\n" + "=" * 50)
        print("✓ All tasks completed successfully!")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
