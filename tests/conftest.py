"""Pytest configuration and shared fixtures."""
import os
import tempfile
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample MNIST-like image tensor."""
    return torch.randn(1, 1, 28, 28)


@pytest.fixture
def sample_batch():
    """Create a batch of sample images."""
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Deep learning uses neural networks with multiple layers to learn from large amounts of data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
    ]


@pytest.fixture
def sample_queries():
    """Sample queries with expected documents for RAG evaluation."""
    return [
        ("What is Python?", ["Python is a high-level programming language"]),
        ("Explain machine learning", ["Machine learning is a subset of artificial intelligence"]),
        ("What is deep learning?", ["Deep learning uses neural networks"]),
    ]


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def ci_mode():
    """Check if running in CI environment."""
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"