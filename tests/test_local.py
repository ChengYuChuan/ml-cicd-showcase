"""本地測試 ML API（不需要 Docker）"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_health():
    """測試健康檢查."""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_docs():
    """測試 API 文檔."""
    print_section("2. API Documentation")
    print(f"Swagger UI: {BASE_URL}/docs")
    print(f"ReDoc: {BASE_URL}/redoc")
    print(f"OpenAPI JSON: {BASE_URL}/openapi.json")
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API docs are accessible")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

def test_metrics():
    """測試 Prometheus metrics."""
    print_section("3. Prometheus Metrics Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            
            # 找出關鍵 metrics
            print("\n📊 Available Metrics:")
            metrics = set()
            for line in lines:
                if line and not line.startswith('#'):
                    metric_name = line.split('{')[0] if '{' in line else line.split()[0]
                    metrics.add(metric_name)
            
            for metric in sorted(metrics):
                if metric.startswith('ml_'):
                    print(f"  - {metric}")
            
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

def test_cnn_prediction():
    """測試 CNN 預測."""
    print_section("4. CNN Prediction Test")
    
    try:
        # 創建測試數據（模擬手寫數字圖片）
        payload = {"data": [[0.5] * 784]}
        
        print("Sending prediction request...")
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/predict/cnn",
            json=payload,
            timeout=10
        )
        latency = (time.time() - start) * 1000
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction Result:")
            print(f"   Predicted Digit: {result['prediction']}")
            print(f"   API Latency: {result['latency_ms']:.2f}ms")
            print(f"   Total Latency: {latency:.2f}ms")
            return True
        else:
            print(f"❌ Error: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

def test_multiple_predictions():
    """測試多次預測並收集統計."""
    print_section("5. Performance Test (20 predictions)")
    
    results = {
        'success': 0,
        'failed': 0,
        'latencies': []
    }
    
    print("\nExecuting predictions:")
    for i in range(20):
        try:
            payload = {"data": [[0.5] * 784]}
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/predict/cnn",
                json=payload,
                timeout=10
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                results['success'] += 1
                results['latencies'].append(latency)
                status = "✓"
            else:
                results['failed'] += 1
                status = "✗"
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/20 {status}")
                
        except Exception as e:
            results['failed'] += 1
            print(f"  Request {i + 1}: ✗ ({e})")
        
        time.sleep(0.05)  # 小延遲避免過載
    
    # 統計結果
    print(f"\n📊 Results:")
    print(f"   Total: {results['success'] + results['failed']}")
    print(f"   Success: {results['success']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Success Rate: {results['success']/20*100:.1f}%")
    
    if results['latencies']:
        latencies = sorted(results['latencies'])
        print(f"\n⚡ Latency Statistics:")
        print(f"   Min: {min(latencies):.2f}ms")
        print(f"   Max: {max(latencies):.2f}ms")
        print(f"   Mean: {sum(latencies)/len(latencies):.2f}ms")
        print(f"   Median: {latencies[len(latencies)//2]:.2f}ms")
        print(f"   P95: {latencies[int(len(latencies)*0.95)]:.2f}ms")
    
    return results['success'] > 0

def test_metrics_update():
    """測試 metrics 是否更新."""
    print_section("6. Verify Metrics Update")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            
            print("\n📈 Current Metrics:")
            for line in lines:
                if 'ml_predictions_total' in line and not line.startswith('#'):
                    print(f"  {line}")
                if 'ml_prediction_latency_seconds_count' in line:
                    print(f"  {line}")
            
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
    return False

def main():
    """執行所有測試."""
    print("\n" + "🚀" * 35)
    print("  ML API Local Testing Suite")
    print("🚀" * 35)
    
    print(f"\nTarget: {BASE_URL}")
    print("Make sure the API is running with: python serve.py")
    
    time.sleep(1)
    
    # 執行測試
    tests = [
        ("Health Check", test_health),
        ("API Documentation", test_docs),
        ("Metrics Endpoint", test_metrics),
        ("CNN Prediction", test_cnn_prediction),
        ("Performance Test", test_multiple_predictions),
        ("Metrics Update", test_metrics_update),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
        time.sleep(0.5)
    
    # 總結
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n{'=' * 70}")
    print(f"  Final Score: {passed}/{total} tests passed")
    print(f"{'=' * 70}")
    
    print("\n📚 Next Steps:")
    print("  1. Visit API docs: http://localhost:8000/docs")
    print("  2. View metrics: http://localhost:8000/metrics")
    print("  3. Run traffic generator: python scripts/generate_traffic.py")
    print("  4. Run benchmarks: python scripts/benchmark.py")

if __name__ == "__main__":
    main()