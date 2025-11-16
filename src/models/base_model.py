from abc import ABC, abstractmethod
from typing import Any, Dict
import json
from pathlib import Path

class BaseMLModel(ABC):
    """統一的模型介面 - 讓CI/CD可以一致處理所有模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def train(self, data) -> Dict[str, float]:
        """訓練模型，返回指標"""
        pass
    
    @abstractmethod
    def predict(self, input_data) -> Any:
        """推理"""
        pass
    
    @abstractmethod
    def evaluate(self, test_data) -> Dict[str, float]:
        """評估，返回標準化指標"""
        pass
    
    def save_model(self, path: Path):
        """保存模型"""
        pass
    
    def load_model(self, path: Path):
        """載入模型"""
        pass
    
    def get_metrics(self) -> Dict[str, float]:
        """返回可比較的指標 (accuracy, latency, etc.)"""
        pass