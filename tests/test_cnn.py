"""Tests for CNN classifier."""
import pytest
import torch

from src.config import CNNConfig
from src.models.cnn_classifier import CNNClassifier, TinyConvNet
from src.utils.metrics import validate_model_threshold


class TestTinyConvNet:
    """Test the TinyConvNet architecture."""

    def test_model_creation(self):
        """Test model can be created."""
        model = TinyConvNet(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, sample_image):
        """Test forward pass with sample input."""
        model = TinyConvNet(num_classes=10)
        output = model(sample_image)

        assert output.shape == (1, 10)
        assert not torch.isnan(output).any()

    def test_batch_forward(self, sample_batch):
        """Test forward pass with batch input."""
        model = TinyConvNet(num_classes=10)
        output = model(sample_batch)

        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_parameter_count(self):
        """Test that model has expected small size."""
        model = TinyConvNet(num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())

        # Should be around 50K parameters
        assert param_count < 100_000, f"Model too large: {param_count} parameters"
        assert param_count > 10_000, f"Model too small: {param_count} parameters"


class TestCNNClassifier:
    """Test the CNN classifier wrapper."""

    def test_initialization(self):
        """Test classifier initialization."""
        config = CNNConfig(num_epochs=1, batch_size=32)
        classifier = CNNClassifier(config)

        assert classifier is not None
        assert classifier.model_name == "TinyConvNet"
        assert not classifier.is_trained

    def test_default_config(self):
        """Test initialization with default config."""
        classifier = CNNClassifier()
        assert classifier.config.num_classes == 10
        assert classifier.config.batch_size == 64

    def test_prediction_shape(self, sample_batch):
        """Test prediction output shape."""
        classifier = CNNClassifier()
        predictions = classifier.predict(sample_batch)

        assert predictions.shape == (4,)
        assert predictions.dtype == torch.long

    @pytest.mark.slow
    def test_quick_training(self, ci_mode):
        """Test training for 1 epoch (quick smoke test)."""
        if ci_mode:
            # Very quick training in CI
            config = CNNConfig(num_epochs=1, batch_size=128)
        else:
            config = CNNConfig(num_epochs=1, batch_size=64)

        classifier = CNNClassifier(config)
        metrics = classifier.train(epochs=1)

        # Check metrics exist
        assert "accuracy" in metrics
        assert "test_loss" in metrics
        assert "parameter_count" in metrics

        # Model should be trained now
        assert classifier.is_trained

    @pytest.mark.slow
    def test_evaluation(self):
        """Test model evaluation."""
        config = CNNConfig(num_epochs=1)
        classifier = CNNClassifier(config)

        # Quick training
        classifier.train(epochs=1)

        # Evaluate
        metrics = classifier.evaluate()

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["parameter_count"] > 0
        assert metrics["model_size_mb"] > 0

    @pytest.mark.slow
    def test_performance_threshold(self):
        """Test that trained model meets minimum performance threshold."""
        config = CNNConfig(num_epochs=2)  # 2 epochs should be enough
        classifier = CNNClassifier(config)

        # Train
        metrics = classifier.train()

        # Validate thresholds
        thresholds = {
            "accuracy": 0.85,  # At least 85% accuracy on MNIST
        }

        is_valid = validate_model_threshold(metrics, thresholds, "CNN")
        assert is_valid

    def test_save_load(self, temp_dir):
        """Test model save and load."""
        config = CNNConfig(num_epochs=1)
        classifier = CNNClassifier(config)

        # Train briefly
        classifier.train(epochs=1)

        # Save
        save_path = temp_dir / "model.pth"
        classifier.save_model(save_path)
        assert save_path.exists()

        # Load into new classifier
        new_classifier = CNNClassifier(config)
        new_classifier.load_model(save_path)

        assert new_classifier.is_trained

    def test_latency_measurement(self, sample_image):
        """Test latency measurement."""
        classifier = CNNClassifier()
        latency = classifier.measure_latency(sample_image, num_runs=5)

        assert latency > 0
        assert latency < 1000  # Should be less than 1 second


class TestCNNIntegration:
    """Integration tests for CNN workflow."""

    @pytest.mark.slow
    def test_full_pipeline(self, temp_dir):
        """Test complete training, evaluation, and save/load pipeline."""
        # Configuration
        config = CNNConfig(num_epochs=1, batch_size=128)

        # Train model
        classifier = CNNClassifier(config)
        train_metrics = classifier.train()

        # Validate performance
        assert train_metrics["accuracy"] > 0.80

        # Save model
        model_path = temp_dir / "cnn_model.pth"
        classifier.save_model(model_path)

        # Load and verify
        new_classifier = CNNClassifier(config)
        new_classifier.load_model(model_path)

        # Evaluate loaded model
        eval_metrics = new_classifier.evaluate()
        assert abs(eval_metrics["accuracy"] - train_metrics["accuracy"]) < 0.01


# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    "not config.getoption('--run-slow')", reason="need --run-slow option to run"
)
