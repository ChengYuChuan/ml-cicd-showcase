"""Abstract base class for all ML models."""
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseMLModel(ABC):
    """
    Abstract base class for ML models.

    This provides a unified interface for different model types,
    allowing consistent CI/CD pipeline handling.
    """

    def __init__(self, config: Any):
        """
        Initialize the model with configuration.

        Args:
            config: Model-specific configuration object
        """
        self.config = config
        self.model_name = getattr(config, "model_name", self.__class__.__name__)
        self.metrics: Dict[str, float] = {}
        self._is_trained = False

    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, float]:
        """
        Train the model.

        Returns:
            Dict[str, float]: Training metrics
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """
        Make predictions on input data.

        Args:
            input_data: Input data for prediction

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dict[str, float]: Evaluation metrics with standardized keys
        """
        pass

    def save_model(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save_implementation(path)
        self._save_metadata(path.parent / f"{path.stem}_metadata.json")

    def load_model(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to the saved model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._load_implementation(path)
        self._is_trained = True

    @abstractmethod
    def _save_implementation(self, path: Path) -> None:
        """Model-specific save implementation."""
        pass

    @abstractmethod
    def _load_implementation(self, path: Path) -> None:
        """Model-specific load implementation."""
        pass

    def get_metrics(self) -> Dict[str, float]:
        """
        Get standardized metrics for the model.

        Returns:
            Dict[str, float]: Metrics including performance and efficiency
        """
        return self.metrics.copy()

    def measure_latency(self, input_data: Any, num_runs: int = 10) -> float:
        """
        Measure prediction latency.

        Args:
            input_data: Sample input for prediction
            num_runs: Number of runs for averaging

        Returns:
            float: Average latency in milliseconds
        """
        import time
        
        latencies = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            self.predict(input_data)
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return round(avg_latency, 4) # 4 decimal places

    def _save_metadata(self, path: Path) -> None:
        """Save model metadata to JSON."""
        metadata = {
            "model_name": self.model_name,
            "config": self._config_to_dict(),
            "metrics": self.metrics,
            "is_trained": self._is_trained,
        }
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _config_to_dict(self) -> dict:
        """Convert config object to dictionary."""
        if hasattr(self.config, "__dict__"):
            return {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}
        return {}

    def validate_performance(self, min_thresholds: Dict[str, float]) -> bool:
        """
        Validate that model meets minimum performance thresholds.

        Args:
            min_thresholds: Dictionary of metric_name -> minimum_value

        Returns:
            bool: True if all thresholds are met
        """
        for metric_name, min_value in min_thresholds.items():
            if metric_name not in self.metrics:
                return False
            if self.metrics[metric_name] < min_value:
                return False
        return True

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
