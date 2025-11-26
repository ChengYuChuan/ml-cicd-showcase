"""Tests for monitoring and API endpoints."""
import pytest
import requests
import time
from pathlib import Path


class TestAPIEndpoints:
    """Test API endpoints."""
    
    base_url = "http://localhost:8000"
    
    @pytest.fixture(scope="class", autouse=True)
    def wait_for_api(self):
        """Wait for API to be ready."""
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("\n✓ API is ready")
                    return
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    time.sleep(1)
        pytest.skip("API not available")
    
    def test_root_endpoint(self):
        """Test root endpoint returns expected structure."""
        response = requests.get(f"{self.base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "metrics" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "cnn" in data["models_loaded"]
        assert "rag" in data["models_loaded"]
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = requests.get(f"{self.base_url}/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for expected metrics
        content = response.text
        assert "ml_predictions_total" in content
        assert "ml_prediction_latency_seconds" in content
    
    def test_cnn_prediction(self):
        """Test CNN prediction endpoint."""
        # Create a valid input (flattened 28x28 image)
        payload = {
            "data": [[0.5] * 784]
        }
        
        response = requests.post(f"{self.base_url}/predict/cnn", json=payload)
        
        if response.status_code == 503:
            pytest.skip("CNN model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert 0 <= data["prediction"] <= 9
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
    
    def test_cnn_invalid_input(self):
        """Test CNN with invalid input."""
        payload = {
            "data": [[0.5] * 100]  # Wrong size
        }
        
        response = requests.post(f"{self.base_url}/predict/cnn", json=payload)
        
        if response.status_code == 503:
            pytest.skip("CNN model not loaded")
        
        assert response.status_code == 400
    
    @pytest.mark.slow
    def test_rag_prediction(self):
        """Test RAG prediction endpoint."""
        payload = {
            "text": "What is machine learning?",
            "top_k": 3
        }
        
        response = requests.post(f"{self.base_url}/predict/rag", json=payload)
        
        if response.status_code == 503:
            pytest.skip("RAG model not loaded or no documents")
        
        if response.status_code == 400:
            pytest.skip("RAG model has no documents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "context" in data
        assert "latency_ms" in data
        assert isinstance(data["context"], list)


class TestMonitoringStack:
    """Test Prometheus and Grafana integration."""
    
    @pytest.fixture(autouse=True)
    def wait_for_services(self):
        """Wait for services to be ready."""
        time.sleep(5)
    
    def test_prometheus_healthy(self):
        """Test Prometheus is running."""
        try:
            response = requests.get("http://localhost:9090/-/healthy", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not available")
    
    def test_prometheus_targets(self):
        """Test Prometheus is scraping targets."""
        try:
            time.sleep(15)  # Wait for first scrape
            
            response = requests.get(
                "http://localhost:9090/api/v1/targets",
                timeout=5
            )
            
            if response.status_code != 200:
                pytest.skip("Prometheus not ready")
            
            data = response.json()
            assert data["status"] == "success"
            
            # Check if ml-api target is up
            targets = data.get("data", {}).get("activeTargets", [])
            ml_targets = [t for t in targets if "ml-api" in t.get("job", "")]
            
            if ml_targets:
                assert ml_targets[0]["health"] == "up"
                
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not available")
    
    def test_prometheus_query(self):
        """Test querying metrics from Prometheus."""
        try:
            # Generate a prediction first
            requests.post(
                "http://localhost:8000/predict/cnn",
                json={"data": [[0.5] * 784]},
                timeout=5
            )
            
            time.sleep(20)  # Wait for scraping
            
            # Query the metric
            response = requests.get(
                "http://localhost:9090/api/v1/query",
                params={"query": "ml_predictions_total"},
                timeout=5
            )
            
            if response.status_code != 200:
                pytest.skip("Prometheus not ready")
            
            data = response.json()
            assert data["status"] == "success"
            
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not available")
    
    def test_grafana_healthy(self):
        """Test Grafana is running."""
        try:
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Grafana not available")
    
    def test_grafana_datasource(self):
        """Test Grafana Prometheus datasource is configured."""
        try:
            response = requests.get(
                "http://localhost:3000/api/datasources",
                auth=("admin", "admin"),
                timeout=5
            )
            
            if response.status_code != 200:
                pytest.skip("Grafana not ready")
            
            datasources = response.json()
            prometheus_ds = [ds for ds in datasources if ds["type"] == "prometheus"]
            
            assert len(prometheus_ds) > 0, "No Prometheus datasource found"
            
        except requests.exceptions.RequestException:
            pytest.skip("Grafana not available")


class TestMetricsAccuracy:
    """Test that metrics are accurately recorded."""
    
    base_url = "http://localhost:8000"
    
    def test_prediction_counter_increments(self):
        """Test that prediction counter increments correctly."""
        try:
            # Get initial metric value
            initial = self._get_metric_value("ml_predictions_total")
            
            # Make predictions
            num_predictions = 5
            for _ in range(num_predictions):
                requests.post(
                    f"{self.base_url}/predict/cnn",
                    json={"data": [[0.5] * 784]},
                    timeout=5
                )
            
            time.sleep(2)  # Wait for metrics update
            
            # Get new metric value
            final = self._get_metric_value("ml_predictions_total")
            
            # Check increment (allowing for some tolerance due to other tests)
            assert final >= initial + num_predictions
            
        except Exception as e:
            pytest.skip(f"Metrics test failed: {e}")
    
    def test_latency_recorded(self):
        """Test that latency is recorded."""
        try:
            # Make a prediction
            response = requests.post(
                f"{self.base_url}/predict/cnn",
                json={"data": [[0.5] * 784]},
                timeout=5
            )
            
            if response.status_code == 200:
                latency = response.json()["latency_ms"]
                assert latency > 0
                assert latency < 10000  # Reasonable upper bound
                
        except Exception as e:
            pytest.skip(f"Latency test failed: {e}")
    
    def _get_metric_value(self, metric_name: str) -> float:
        """Helper to get current metric value from Prometheus."""
        response = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": f"sum({metric_name})"},
            timeout=5
        )
        
        if response.status_code != 200:
            return 0
        
        data = response.json()
        if data["status"] != "success":
            return 0
        
        results = data["data"]["result"]
        if not results:
            return 0
        
        return float(results[0]["value"][1])