"""MLflow tracking utilities for experiment logging."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLflowConfig


class MLflowTracker:
    """
    MLflow tracking wrapper for logging experiments, metrics, and models.

    This class provides a simplified interface to MLflow's tracking capabilities,
    making it easy to integrate experiment tracking into ML training pipelines.
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow tracker.

        Args:
            config: MLflow configuration. If None, uses default configuration.
        """
        self.config = config or MLflowConfig()
        self._setup_mlflow()
        self.client = MlflowClient()
        self._active_run = None

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        if self.config.enabled:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """
        Start an MLflow run as a context manager.

        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to the run
            nested: Whether this is a nested run

        Yields:
            Active MLflow run
        """
        if not self.config.enabled:
            yield None
            return

        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        self._active_run = run
        try:
            yield run
        finally:
            mlflow.end_run()
            self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.config.enabled:
            return

        # MLflow requires string values, so convert
        clean_params = {}
        for key, value in params.items():
            if value is not None:
                clean_params[key] = str(value)

        mlflow.log_params(clean_params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if not self.config.enabled:
            return

        # Filter out non-numeric values
        clean_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }

        if step is not None:
            mlflow.log_metrics(clean_metrics, step=step)
        else:
            mlflow.log_metrics(clean_metrics)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.config.enabled:
            return

        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within artifacts to store the model
            registered_name: Optional name to register the model under
            **kwargs: Additional arguments passed to mlflow.pytorch.log_model
        """
        if not self.config.enabled:
            return

        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_name,
            **kwargs,
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a local file or directory as an artifact.

        Args:
            local_path: Path to the local file or directory
            artifact_path: Optional subdirectory within artifacts
        """
        if not self.config.enabled:
            return

        mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """
        Log all files in a directory as artifacts.

        Args:
            local_dir: Path to the local directory
            artifact_path: Optional subdirectory within artifacts
        """
        if not self.config.enabled:
            return

        mlflow.log_artifacts(local_dir, artifact_path)

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag on the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        if not self.config.enabled:
            return

        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set multiple tags on the current run.

        Args:
            tags: Dictionary of tag names and values
        """
        if not self.config.enabled:
            return

        mlflow.set_tags(tags)

    def get_run(self, run_id: str) -> Optional[mlflow.entities.Run]:
        """
        Get a run by its ID.

        Args:
            run_id: The run ID

        Returns:
            The run object or None if not found
        """
        if not self.config.enabled:
            return None

        try:
            return self.client.get_run(run_id)
        except Exception:
            return None

    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[list] = None,
    ) -> list:
        """
        Search for runs matching the filter criteria.

        Args:
            filter_string: Filter expression
            max_results: Maximum number of results to return
            order_by: List of columns to order by

        Returns:
            List of matching runs
        """
        if not self.config.enabled:
            return []

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            return []

        return self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["start_time DESC"],
        )

    @property
    def active_run_id(self) -> Optional[str]:
        """Get the ID of the currently active run."""
        if self._active_run:
            return self._active_run.info.run_id
        return None

    @property
    def experiment_id(self) -> Optional[str]:
        """Get the current experiment ID."""
        if not self.config.enabled:
            return None

        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        return experiment.experiment_id if experiment else None
