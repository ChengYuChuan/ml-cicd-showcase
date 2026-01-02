#!/usr/bin/env python3
"""
MLflow Demo Script - Demonstrates MLflow tracking and model registry features.

This script showcases:
1. Experiment tracking with parameters and metrics
2. Epoch-level metric logging for training curves
3. Model artifact logging
4. Model registration and stage transitions
5. Loading models from registry

Usage:
    python scripts/demo_mlflow.py [--tracking-uri URI]

Requirements:
    - MLflow server running (make mlflow-up)
    - Dependencies installed (make install)
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_mlflow_server(tracking_uri: str) -> bool:
    """Check if MLflow server is running."""
    import requests

    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def demo_experiment_tracking(tracker):
    """Demonstrate basic experiment tracking."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Experiment Tracking")
    print("=" * 60)

    with tracker.start_run(run_name="demo-experiment-tracking"):
        # Log parameters
        params = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 10,
            "optimizer": "Adam",
            "model_type": "CNN",
        }
        tracker.log_params(params)
        print(f"Logged parameters: {params}")

        # Simulate training and log metrics
        print("\nSimulating training...")
        for epoch in range(5):
            # Simulate metrics
            train_loss = 2.0 / (epoch + 1) + 0.1
            train_acc = 0.5 + epoch * 0.1
            val_loss = 2.2 / (epoch + 1) + 0.15
            val_acc = 0.45 + epoch * 0.1

            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            tracker.log_metrics(metrics, step=epoch)
            print(f"  Epoch {epoch + 1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
            time.sleep(0.5)

        # Log final metrics
        final_metrics = {
            "final_accuracy": 0.95,
            "final_loss": 0.15,
            "training_time_sec": 120.5,
        }
        tracker.log_metrics(final_metrics)
        print(f"\nFinal metrics: {final_metrics}")

        # Set tags
        tracker.set_tags({
            "dataset": "MNIST",
            "framework": "PyTorch",
            "environment": "demo",
        })
        print("Tags added: dataset=MNIST, framework=PyTorch, environment=demo")

    print("\n✓ Experiment tracking demo complete!")
    print(f"  View at: {tracker.config.tracking_uri}")


def demo_model_training_with_tracking():
    """Demonstrate actual model training with MLflow tracking."""
    print("\n" + "=" * 60)
    print("Demo 2: CNN Training with MLflow Tracking")
    print("=" * 60)

    from src.config import CNNConfig, MLflowConfig
    from src.models.cnn_classifier import CNNClassifier
    from src.tracking.mlflow_tracker import MLflowTracker

    # Initialize tracker and model
    mlflow_config = MLflowConfig()
    tracker = MLflowTracker(mlflow_config)
    cnn_config = CNNConfig(num_epochs=2)  # Quick training for demo
    model = CNNClassifier(cnn_config, mlflow_tracker=tracker)

    print(f"\nTraining CNN with {cnn_config.num_epochs} epochs...")
    print("MLflow will track:")
    print("  - Configuration parameters")
    print("  - Per-epoch loss and accuracy")
    print("  - Final evaluation metrics")

    # Train with tracking
    metrics = model.train_with_tracking(
        run_name="cnn-demo-training",
        tags={"demo": "true", "purpose": "showcase"},
        register_model=False,  # Don't register for this demo
    )

    print(f"\n✓ Training complete!")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  View training curves at: {mlflow_config.tracking_uri}")

    return model, tracker


def demo_model_registry(tracker, model):
    """Demonstrate model registry features."""
    print("\n" + "=" * 60)
    print("Demo 3: Model Registry")
    print("=" * 60)

    from src.tracking.model_registry import ModelRegistry

    registry = ModelRegistry(tracker.config.tracking_uri)

    # Register the model
    print("\nRegistering model to MLflow Model Registry...")

    with tracker.start_run(run_name="model-registration-demo"):
        # Log the model
        tracker.log_model(
            model.model,
            artifact_path="model",
            registered_name="demo-cnn-model",
        )
        run_id = tracker.active_run_id

    print(f"  Model registered as: demo-cnn-model")
    print(f"  Run ID: {run_id}")

    # Get latest version
    latest = registry.get_latest_version("demo-cnn-model")
    if latest:
        print(f"  Version: {latest.version}")
        print(f"  Stage: {latest.current_stage}")

        # Transition to staging
        print("\nTransitioning model to Staging...")
        registry.promote_to_staging("demo-cnn-model", int(latest.version))
        print("  ✓ Model is now in Staging")

        # Optionally promote to production
        print("\nTransitioning model to Production...")
        registry.promote_to_production("demo-cnn-model", int(latest.version))
        print("  ✓ Model is now in Production")

    print("\n✓ Model registry demo complete!")
    print(f"  View models at: {tracker.config.tracking_uri}/#/models")


def demo_search_runs(tracker):
    """Demonstrate searching for runs."""
    print("\n" + "=" * 60)
    print("Demo 4: Searching Experiment Runs")
    print("=" * 60)

    print("\nSearching for recent runs...")
    runs = tracker.search_runs(max_results=5)

    if runs:
        print(f"\nFound {len(runs)} recent runs:")
        print("-" * 80)
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "unnamed")
            status = run.info.status
            metrics = run.data.metrics

            print(f"\nRun: {run_name}")
            print(f"  ID: {run.info.run_id[:8]}...")
            print(f"  Status: {status}")
            if metrics:
                print(f"  Metrics: {dict(list(metrics.items())[:3])}...")
    else:
        print("No runs found yet. Run some experiments first!")

    print("\n✓ Search demo complete!")


def demo_compare_runs(tracker):
    """Demonstrate comparing multiple runs."""
    print("\n" + "=" * 60)
    print("Demo 5: Comparing Experiment Runs")
    print("=" * 60)

    # Create a few runs with different parameters
    print("\nCreating comparison runs with different learning rates...")

    learning_rates = [0.001, 0.01, 0.1]
    results = []

    for lr in learning_rates:
        with tracker.start_run(run_name=f"comparison-lr-{lr}"):
            tracker.log_params({"learning_rate": lr})

            # Simulate different results based on learning rate
            if lr == 0.001:
                acc = 0.92
            elif lr == 0.01:
                acc = 0.95
            else:
                acc = 0.85

            tracker.log_metrics({"accuracy": acc, "loss": 1 - acc})
            results.append((lr, acc))
            print(f"  LR={lr}: accuracy={acc:.2f}")

    print("\nComparison results:")
    print("-" * 40)
    best_lr, best_acc = max(results, key=lambda x: x[1])
    print(f"Best learning rate: {best_lr} (accuracy: {best_acc:.2f})")

    print("\n✓ Comparison demo complete!")
    print(f"  Compare visually at: {tracker.config.tracking_uri}")


def main():
    parser = argparse.ArgumentParser(
        description="MLflow Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all demos
    python scripts/demo_mlflow.py

    # Use custom tracking URI
    python scripts/demo_mlflow.py --tracking-uri http://localhost:5000

    # Skip model training (faster)
    python scripts/demo_mlflow.py --skip-training
        """,
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip actual model training demo",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("   MLflow Feature Demo")
    print("=" * 60)
    print(f"\nTracking URI: {args.tracking_uri}")

    # Check if MLflow server is running
    print("\nChecking MLflow server...")
    if not check_mlflow_server(args.tracking_uri):
        print("\n❌ MLflow server is not running!")
        print("   Start it with: make mlflow-up")
        print("   Or: docker-compose up -d mlflow")
        sys.exit(1)
    print("✓ MLflow server is running")

    # Import after path setup
    from src.config import MLflowConfig
    from src.tracking.mlflow_tracker import MLflowTracker

    # Initialize tracker
    config = MLflowConfig(tracking_uri=args.tracking_uri)
    tracker = MLflowTracker(config)

    try:
        # Demo 1: Basic experiment tracking
        demo_experiment_tracking(tracker)

        # Demo 2: Actual model training (optional)
        model = None
        if not args.skip_training:
            model, tracker = demo_model_training_with_tracking()
        else:
            print("\n[Skipping model training demo]")

        # Demo 3: Model registry (requires trained model)
        if model is not None:
            try:
                demo_model_registry(tracker, model)
            except Exception as e:
                print(f"\n⚠ Model registry demo skipped: {e}")

        # Demo 4: Search runs
        demo_search_runs(tracker)

        # Demo 5: Compare runs
        demo_compare_runs(tracker)

        # Summary
        print("\n" + "=" * 60)
        print("   Demo Complete!")
        print("=" * 60)
        print(f"""
Next steps:
  1. Open MLflow UI: {args.tracking_uri}
  2. Explore the experiments and runs
  3. View training curves in the metrics tab
  4. Check the model registry for registered models

Useful MLflow UI sections:
  - Experiments: {args.tracking_uri}/#/experiments
  - Models: {args.tracking_uri}/#/models
  - Compare runs: Select multiple runs and click "Compare"
        """)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise


if __name__ == "__main__":
    main()
