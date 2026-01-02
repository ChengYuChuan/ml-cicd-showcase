"""Abstract base class for all ML models."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.tracking.mlflow_tracker import MLflowTracker


class BaseMLModel(ABC):
    """
    Abstract base class for ML models.

    This provides a unified interface for different model types,
    allowing consistent CI/CD pipeline handling.
    """

    def __init__(
        self, config: Any, mlflow_tracker: Optional["MLflowTracker"] = None
    ):
        """
        Initialize the model with configuration.

        Args:
            config: Model-specific configuration object
            mlflow_tracker: Optional MLflow tracker for experiment logging
        """
        self.config = config
        self.model_name = getattr(config, "model_name", self.__class__.__name__)
        self.metrics: Dict[str, float] = {}
        self._is_trained = False
        self.tracker = mlflow_tracker

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
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return round(avg_latency, 4)

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

    def train_with_tracking(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        register_model: bool = False,
        registered_model_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train the model with MLflow tracking.

        This method wraps the train() method and automatically logs
        parameters, metrics, and optionally the model to MLflow.

        Args:
            run_name: Name for the MLflow run (defaults to model_name)
            tags: Additional tags to add to the run
            register_model: Whether to register the model in MLflow registry
            registered_model_name: Name for registered model (defaults to model_name)
            *args: Arguments passed to train()
            **kwargs: Keyword arguments passed to train()

        Returns:
            Dict[str, float]: Training metrics
        """
        if self.tracker is None:
            # No tracker configured, just train normally
            return self.train(*args, **kwargs)

        run_name = run_name or f"{self.model_name}-training"
        registered_model_name = registered_model_name or self.model_name

        with self.tracker.start_run(run_name=run_name, tags=tags):
            # Log configuration parameters
            self.tracker.log_params(self._config_to_dict())

            # Train the model
            metrics = self.train(*args, **kwargs)

            # Log metrics
            self.tracker.log_metrics(metrics)

            # Log model if it has a PyTorch model attribute
            if register_model and hasattr(self, "model") and self.model is not None:
                self.tracker.log_model(
                    self.model,
                    artifact_path="model",
                    registered_name=registered_model_name,
                )

            return metrics

    def log_epoch_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log metrics for a specific training epoch.

        Args:
            metrics: Dictionary of metric names and values
            epoch: Current epoch number
        """
        if self.tracker is not None:
            self.tracker.log_metrics(metrics, step=epoch)
