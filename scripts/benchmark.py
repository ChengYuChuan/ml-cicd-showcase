#!/usr/bin/env python3
"""Benchmark ML API performance."""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import requests


def benchmark_endpoint(
    url: str,
    payload: Dict,
    num_requests: int = 100,
    concurrency: int = 1
) -> Dict:
    """
    Benchmark an endpoint.
    
    Args:
        url: Endpoint URL
        payload: Request payload
        num_requests: Total number of requests
        concurrency: Number of concurrent requests
    
    Returns:
        Dict with benchmark results
    """
    latencies: List[float] = []
    errors = 0
    status_codes: Dict[int, int] = {}
    
    def make_request() -> Tuple[bool, float, int]:
        """Make a single request."""
        try:
            start = time.time()
            response = requests.post(url, json=payload, timeout=30)
            latency = (time.time() - start) * 1000
            return response.status_code == 200, latency, response.status_code
        except Exception:
            return False, 0, 500
    
    print(f"\nBenchmarking: {url}")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    
    start_time = time.time()
    
    # Execute requests
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        
        completed = 0
        for future in as_completed(futures):
            success, latency, status_code = future.result()
            
            if success:
                latencies.append(latency)
            else:
                errors += 1
            
            # Track status codes
            status_codes[status_code] = status_codes.get(status_code, 0) + 1
            
            completed += 1
            if completed % 10 == 0 or completed == num_requests:
                print(f"Progress: {completed}/{num_requests}", end="\r")
    
    total_time = time.time() - start_time
    
    print()  # New line after progress
    
    if not latencies:
        return {
            "error": "All requests failed",
            "total_requests": num_requests,
            "errors": errors
        }
    
    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    
    return {
        "total_requests": num_requests,
        "successful": len(latencies),
        "errors": errors,
        "success_rate": len(latencies) / num_requests,
        "total_time_sec": total_time,
        "requests_per_sec": num_requests / total_time,
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)],
        "latency_p99_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)],
        "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "status_codes": status_codes
    }


def print_results(model_name: str, results: Dict):
    """Print benchmark results in a nice format."""
    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} Benchmark Results")
    print(f"{'=' * 60}")
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"\nRequest Statistics:")
    print(f"  Total requests:    {results['total_requests']}")
    print(f"  Successful:        {results['successful']} ({results['success_rate']*100:.1f}%)")
    print(f"  Errors:            {results['errors']}")
    print(f"  Total time:        {results['total_time_sec']:.2f}s")
    print(f"  Throughput:        {results['requests_per_sec']:.2f} req/s")
    
    print(f"\nLatency Statistics (ms):")
    print(f"  Min:               {results['latency_min_ms']:.2f}")
    print(f"  Max:               {results['latency_max_ms']:.2f}")
    print(f"  Mean:              {results['latency_mean_ms']:.2f}")
    print(f"  Median:            {results['latency_median_ms']:.2f}")
    print(f"  P95:               {results['latency_p95_ms']:.2f}")
    print(f"  P99:               {results['latency_p99_ms']:.2f}")
    print(f"  Std Dev:           {results['latency_stdev_ms']:.2f}")
    
    print(f"\nStatus Codes:")
    for code, count in sorted(results['status_codes'].items()):
        print(f"  {code}: {count}")
    
    print(f"\n{'=' * 60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark ML API performance")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "rag", "both"],
        default="cnn",
        help="Which model to benchmark"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests (default: 100)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print("ML API Performance Benchmark")
    print(f"{'=' * 60}")
    print(f"\nConfiguration:")
    print(f"  URL: {args.url}")
    print(f"  Model(s): {args.model}")
    print(f"  Total requests: {args.requests}")
    print(f"  Concurrency: {args.concurrency}")
    
    # Benchmark CNN
    if args.model in ["cnn", "both"]:
        cnn_payload = {"data": [[0.5] * 784]}
        cnn_results = benchmark_endpoint(
            f"{args.url}/predict/cnn",
            cnn_payload,
            args.requests,
            args.concurrency
        )
        print_results("CNN", cnn_results)
    
    # Benchmark RAG
    if args.model in ["rag", "both"]:
        rag_payload = {"text": "What is machine learning?", "top_k": 3}
        rag_results = benchmark_endpoint(
            f"{args.url}/predict/rag",
            rag_payload,
            args.requests,
            args.concurrency
        )
        print_results("RAG", rag_results)
    
    print()


if __name__ == "__main__":
    main()