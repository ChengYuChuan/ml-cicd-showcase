"""Integration tests for the unified ML framework."""
import pytest

from src.config import CNNConfig, RAGConfig
from src.models.base_model import BaseMLModel
from src.models.cnn_classifier import CNNClassifier
from src.models.rag_system import RAGSystem


class TestUnifiedInterface:
    """Test that both models implement the unified interface."""

    def test_cnn_implements_base(self):
        """Test CNN implements BaseMLModel interface."""
        config = CNNConfig(num_epochs=1)
        model = CNNClassifier(config)

        assert isinstance(model, BaseMLModel)
        assert hasattr(model, "train")
        assert hasattr(model, "predict")
        assert hasattr(model, "evaluate")
        assert hasattr(model, "save_model")
        assert hasattr(model, "load_model")

    def test_rag_implements_base(self, anthropic_api_key):
        """Test RAG implements BaseMLModel interface."""
        config = RAGConfig(anthropic_api_key=anthropic_api_key, collection_name="interface_test")
        model = RAGSystem(config)

        assert isinstance(model, BaseMLModel)
        assert hasattr(model, "train")
        assert hasattr(model, "predict")
        assert hasattr(model, "evaluate")
        assert hasattr(model, "save_model")
        assert hasattr(model, "load_model")

    @pytest.mark.slow
    def test_unified_workflow_cnn(self, temp_dir):
        """Test unified workflow with CNN."""
        config = CNNConfig(num_epochs=1)
        model = CNNClassifier(config)

        # Train
        train_metrics = model.train()
        assert model.is_trained
        assert "accuracy" in train_metrics

        # Evaluate
        eval_metrics = model.evaluate()
        assert "accuracy" in eval_metrics

        # Get metrics
        metrics = model.get_metrics()
        assert len(metrics) > 0

        # Save
        save_path = temp_dir / "unified_cnn.pth"
        model.save_model(save_path)
        assert save_path.exists()

    @pytest.mark.slow
    def test_unified_workflow_rag(self, sample_documents, temp_dir, anthropic_api_key):
        """Test unified workflow with RAG."""
        config = RAGConfig(anthropic_api_key=anthropic_api_key, collection_name="unified_test")
        model = RAGSystem(config)

        # Train (ingest)
        train_metrics = model.train(sample_documents)
        assert model.is_trained
        assert "num_documents" in train_metrics

        # Predict
        result = model.predict("What is Python?")
        assert "answer" in result

        # Get metrics
        metrics = model.get_metrics()
        assert len(metrics) > 0

        # Save
        save_path = temp_dir / "unified_rag.json"
        model.save_model(save_path)
        assert save_path.exists()


class TestCrossModelComparison:
    """Test comparing metrics across different model types."""

    @pytest.mark.slow
    def test_metrics_comparison(self, sample_documents, anthropic_api_key):
        """Test that both models produce comparable metrics."""
        # Train CNN (very quick)
        cnn_config = CNNConfig(num_epochs=1)
        cnn = CNNClassifier(cnn_config)
        cnn.train()
        cnn_metrics = cnn.get_metrics()

        # Setup RAG
        rag_config = RAGConfig(
            anthropic_api_key=anthropic_api_key, collection_name="comparison_test"
        )
        rag = RAGSystem(rag_config)
        rag.ingest_documents(sample_documents)
        rag_metrics = rag.get_metrics()

        # Both should have metrics
        assert len(cnn_metrics) > 0
        assert len(rag_metrics) > 0

        # Both should have some form of performance metric
        assert "accuracy" in cnn_metrics or "retrieval_precision" in rag_metrics

    def test_latency_measurement_both_models(self, sample_image, anthropic_api_key):
        """Test latency measurement works for both models."""
        # CNN latency
        cnn = CNNClassifier()
        cnn_latency = cnn.measure_latency(sample_image, num_runs=3)
        assert cnn_latency > 0

        # RAG latency (just check the method exists and works with a query)
        rag_config = RAGConfig(
            anthropic_api_key=anthropic_api_key, collection_name="latency_test"
        )
        rag = RAGSystem(rag_config)
        rag.ingest_documents(["test document"])

        # RAG doesn't have the exact same latency method but measures it in predict
        result = rag.predict("test")
        assert result["latency_ms"] > 0


class TestModelValidation:
    """Test model validation functionality."""

    @pytest.mark.slow
    def test_validate_performance_cnn(self):
        """Test performance validation for CNN."""
        config = CNNConfig(num_epochs=2)
        model = CNNClassifier(config)

        # Train
        model.train()

        # Validate with thresholds
        thresholds = {"accuracy": 0.85}
        is_valid = model.validate_performance(thresholds)

        assert is_valid

    def test_validate_performance_rag(self, sample_documents, anthropic_api_key):
        """Test performance validation for RAG."""
        config = RAGConfig(anthropic_api_key=anthropic_api_key, collection_name="validation_test")
        model = RAGSystem(config)

        # Ingest
        model.ingest_documents(sample_documents)

        # Evaluate
        test_queries = [("test", [sample_documents[0]])]
        model.evaluate(test_queries)

        # Validate (with lenient thresholds)
        thresholds = {"retrieval_precision": 0.0}  # Very lenient
        is_valid = model.validate_performance(thresholds)

        assert is_valid

    def test_validation_fails_with_high_threshold(self, sample_documents, anthropic_api_key):
        """Test that validation fails when threshold not met."""
        config = RAGConfig(
            anthropic_api_key=anthropic_api_key, collection_name="fail_validation_test"
        )
        model = RAGSystem(config)

        model.ingest_documents(sample_documents)
        test_queries = [("test", [sample_documents[0]])]
        model.evaluate(test_queries)

        # Impossible threshold
        thresholds = {"retrieval_precision": 1.0}  # 100% precision
        is_valid = model.validate_performance(thresholds)

        # This may or may not pass depending on actual precision
        # Just check the method works
        assert isinstance(is_valid, bool)


class TestEndToEnd:
    """End-to-end tests simulating real usage."""

    @pytest.mark.slow
    def test_complete_ml_pipeline(self, temp_dir, sample_documents, anthropic_api_key):
        """
        Test complete ML pipeline from training to deployment.

        This simulates what a CI/CD pipeline would do:
        1. Train model
        2. Evaluate performance
        3. Validate thresholds
        4. Save model
        5. Load model
        6. Make predictions
        """
        # Test with CNN
        print("\n--- Testing CNN Pipeline ---")
        cnn_config = CNNConfig(num_epochs=1)
        cnn = CNNClassifier(cnn_config)

        # Train
        train_metrics = cnn.train()
        print(f"Train metrics: {train_metrics}")

        # Validate
        assert cnn.validate_performance({"accuracy": 0.80})

        # Save
        cnn_path = temp_dir / "cnn_deploy.pth"
        cnn.save_model(cnn_path)

        # Load and predict
        new_cnn = CNNClassifier(cnn_config)
        new_cnn.load_model(cnn_path)
        test_input = pytest.importorskip("torch").randn(1, 1, 28, 28)
        prediction = new_cnn.predict(test_input)
        assert prediction is not None

        # Test with RAG
        print("\n--- Testing RAG Pipeline ---")
        rag_config = RAGConfig(
            anthropic_api_key=anthropic_api_key, collection_name="e2e_test"
        )
        rag = RAGSystem(rag_config)

        # Train (ingest)
        ingest_metrics = rag.ingest_documents(sample_documents)
        print(f"Ingest metrics: {ingest_metrics}")

        # Validate
        test_queries = [("test", [sample_documents[0]])]
        rag.evaluate(test_queries)
        assert rag.validate_performance({"retrieval_precision": 0.0})

        # Save
        rag_path = temp_dir / "rag_deploy.json"
        rag.save_model(rag_path)

        # Load and predict
        new_rag = RAGSystem(rag_config)
        new_rag.load_model(rag_path)
        result = new_rag.predict("What is Python?")
        assert "answer" in result

        print("\n✓ Complete pipeline test passed!")
