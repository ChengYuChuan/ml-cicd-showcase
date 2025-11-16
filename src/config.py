"""Configuration management for ML models."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class CNNConfig:
    """Configuration for CNN model."""

    model_name: str = "TinyConvNet"
    input_size: int = 28
    num_classes: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 3
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class RAGConfig:
    """Configuration for RAG system."""

    model_name: str = "RAGSystem"
    embedding_model: str = "all-MiniLM-L6-v2"
    claude_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: Optional[str] = None
    top_k: int = 3
    max_tokens: int = 1024
    collection_name: str = "knowledge_base"

    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or config")

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}
