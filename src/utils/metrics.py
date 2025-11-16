"""Utility functions for ML models."""

from pathlib import Path
from typing import Dict


def validate_model_threshold(
    metrics: Dict[str, float], thresholds: Dict[str, float], model_name: str = "Model"
) -> bool:
    """
    Validate that model metrics meet minimum thresholds.

    Args:
        metrics: Dictionary of metric_name -> value
        thresholds: Dictionary of metric_name -> minimum_value
        model_name: Name of the model for error messages

    Returns:
        bool: True if all thresholds are met

    Raises:
        ValueError: If a threshold is not met
    """
    for metric_name, min_value in thresholds.items():
        if metric_name not in metrics:
            raise ValueError(f"{model_name}: Metric '{metric_name}' not found in results")

        actual_value = metrics[metric_name]
        if actual_value < min_value:
            raise ValueError(
                f"{model_name}: {metric_name} = {actual_value:.4f} "
                f"does not meet threshold {min_value:.4f}"
            )

    print(f"✓ {model_name}: All thresholds met!")
    for metric_name, value in metrics.items():
        if metric_name in thresholds:
            print(f"  - {metric_name}: {value:.4f} (threshold: {thresholds[metric_name]:.4f})")

    return True


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for display.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string
    """
    lines = ["Metrics:"]
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
