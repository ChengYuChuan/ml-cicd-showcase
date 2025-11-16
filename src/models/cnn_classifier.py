import torch
import torch.nn as nn
from torchvision import datasets, transforms
from .base_model import BaseMLModel

class TinyConvNet(nn.Module):
    """僅 ~50K 參數的CNN - CI友善"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                # -> 14x14 -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class CNNClassifier(BaseMLModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = TinyConvNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, epochs=3):  # CI中只訓練3個epoch
        """快速訓練用於展示"""
        # MNIST載入
        train_loader = self._get_data_loader(train=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return self.evaluate()
    
    def evaluate(self):
        test_loader = self._get_data_loader(train=False)
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'model_size_mb': self._get_model_size(),
            'parameter_count': sum(p.numel() for p in self.model.parameters())
        }