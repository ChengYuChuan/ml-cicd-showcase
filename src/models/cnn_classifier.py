"""Lightweight CNN classifier for MNIST - CI/CD friendly."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.config import CNNConfig
from src.models.base_model import BaseMLModel


class TinyConvNet(nn.Module):
    """
    Ultra-lightweight CNN with ~50K parameters.

    Architecture:
    - Conv1: 1 -> 16 channels (3x3)
    - MaxPool: 28x28 -> 14x14
    - Conv2: 16 -> 32 channels (3x3)
    - MaxPool: 14x14 -> 7x7
    - FC1: 32*7*7 -> 64
    - FC2: 64 -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNClassifier(BaseMLModel):
    """CNN classifier with unified interface for CI/CD."""

    def __init__(self, config: Optional[CNNConfig] = None):
        """
        Initialize CNN classifier.

        Args:
            config: CNNConfig object or None for defaults
        """
        if config is None:
            config = CNNConfig()
        super().__init__(config)

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Initialize model
        self.model = TinyConvNet(num_classes=config.num_classes)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        print(f"Initialized CNN on device: {self.device}")
        print(f"Model parameters: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _get_data_loaders(
        self, train: bool = True
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Get MNIST data loaders.

        Args:
            train: If True, return train and test loaders. If False, only test loader.

        Returns:
            Tuple of (train_loader, test_loader)
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_loader = None
        if train:
            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True
            )

        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train the CNN model.

        Args:
            epochs: Number of epochs (uses config if None)

        Returns:
            Dict with training metrics
        """
        if epochs is None:
            epochs = self.config.num_epochs

        train_loader, test_loader = self._get_data_loaders(train=True)

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
                )

            avg_loss = epoch_loss / len(train_loader) if train_loader else 0.0
            train_acc = correct / total
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={train_acc:.4f}")

        self._is_trained = True

        # Evaluate on test set
        eval_metrics = self.evaluate()
        self.metrics.update(eval_metrics)

        return {
            "train_loss": avg_loss,
            "train_accuracy": train_acc,
            **eval_metrics,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dict with evaluation metrics
        """
        _, test_loader = self._get_data_loaders(train=False)
        if test_loader is None:
            return {}
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total
        avg_loss = test_loss / len(test_loader) if test_loader else 0.0

        metrics = {
            "accuracy": accuracy,
            "test_loss": avg_loss,
            "parameter_count": self.count_parameters(),
            "model_size_mb": self._get_model_size_mb(),
        }

        self.metrics.update(metrics)
        return metrics

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            input_data: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Predicted class labels
        """
        self.model.eval()
        with torch.no_grad():
            if input_data.device != self.device:
                input_data = input_data.to(self.device)
            output = self.model(input_data)
            _, predicted = output.max(1)
        return predicted  # type: ignore[no-any-return]

    def _get_model_size_mb(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)
        return round(size_mb, 2)

    def _save_implementation(self, path: Path) -> None:
        """Save model state dict."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def _load_implementation(self, path: Path) -> None:
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    # Quick test
    config = CNNConfig(num_epochs=1)
    model = CNNClassifier(config)
    metrics = model.train()
    print("\nTraining complete!")
    print(f"Metrics: {metrics}")
