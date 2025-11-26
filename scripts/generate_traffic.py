#!/usr/bin/env python3
"""Generate test traffic for monitoring the ML API."""
import argparse
import random
import time
from typing import Dict, List

import requests


def generate_cnn_payload() -> Dict:
    """Generate a random CNN prediction payload."""
    # Generate random 28x28 image (normalized 0-1)
    data = [[random.random() for _ in range(784)]]
    return {"data": data}


def generate_rag_payload() -> Dict:
    """Generate a random RAG query payload."""
    queries = [
        "What is machine learning?",
        "Explain deep learning",
        "What is Python?",
        "Tell me about neural networks",
        "What is computer vision?",
        "Explain natural language processing",
        "What is artificial intelligence?",
        "How does supervised learning work?",
    ]
    return {
        "text": random.choice(queries),
        "top_k": random.randint(2, 5)
    }


def make_request(url: str, payload: Dict, timeout: int = 10) -> tuple:
    """
    Make a request and return (success, latency_ms, status_code).
    
    Returns:
        tuple: (success: bool, latency_ms: float, status_code: int)
    """
    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=timeout)
        latency = (time.time() - start) * 1000
        
        success = response.status_code == 200
        return success, latency, response.status_code
        
    except requests.exceptions.Timeout:
        return False, timeout * 1000, 408
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False, 0, 500


def print_stats(stats: Dict):
    """Print statistics."""
    print("\n" + "=" * 60)
    print("Traffic Generation Statistics")
    print("=" * 60)
    
    for model_type in ["cnn", "rag"]:
        if model_type in stats:
            s = stats[model_type]
            print(f"\n{model_type.upper()} Model:")
            print(f"  Total requests: {s['total']}")
            print(f"  Successful: {s['success']} ({s['success']/s['total']*100:.1f}%)")
            print(f"  Failed: {s['failed']} ({s['failed']/s['total']*100:.1f}%)")
            
            if s['latencies']:
                print(f"  Avg latency: {sum(s['latencies'])/len(s['latencies']):.2f}ms")
                print(f"  Min latency: {min(s['latencies']):.2f}ms")
                print(f"  Max latency: {max(s['latencies']):.2f}ms")
    
    print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate test traffic for ML API monitoring"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Requests per second (default: 1.0)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds (default: 60, 0 for infinite)"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "rag", "both"],
        default="both",
        help="Which model to target (default: both)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each request"
    )
    
    args = parser.parse_args()
    
    # Calculate delay between requests
    delay = 1.0 / args.rate
    
    # Initialize stats
    stats = {
        "cnn": {"total": 0, "success": 0, "failed": 0, "latencies": []},
        "rag": {"total": 0, "success": 0, "failed": 0, "latencies": []}
    }
    
    print(f"\nGenerating traffic to {args.url}")
    print(f"Rate: {args.rate} req/s")
    print(f"Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print(f"Target: {args.model}")
    print("\nPress Ctrl+C to stop\n")
    
    start_time = time.time()
    request_count = 0
    
    try:
        while True:
            # Check duration
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break
            
            # Determine which model to call
            if args.model == "both":
                model_type = "cnn" if request_count % 2 == 0 else "rag"
            else:
                model_type = args.model
            
            # Prepare request
            if model_type == "cnn":
                endpoint = f"{args.url}/predict/cnn"
                payload = generate_cnn_payload()
            else:
                endpoint = f"{args.url}/predict/rag"
                payload = generate_rag_payload()
            
            # Make request
            success, latency, status_code = make_request(endpoint, payload)
            
            # Update stats
            stats[model_type]["total"] += 1
            if success:
                stats[model_type]["success"] += 1
                stats[model_type]["latencies"].append(latency)
            else:
                stats[model_type]["failed"] += 1
            
            # Print if verbose
            if args.verbose:
                status_icon = "✓" if success else "✗"
                print(
                    f"{status_icon} [{request_count + 1:4d}] "
                    f"{model_type.upper():3s} | "
                    f"Status: {status_code:3d} | "
                    f"Latency: {latency:7.2f}ms"
                )
            elif request_count % 10 == 0:
                print(f"Processed {request_count} requests...", end="\r")
            
            request_count += 1
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Print final stats
    print_stats(stats)
    print(f"\nTotal time: {time.time() - start_time:.2f}s")
    print(f"Total requests: {request_count}")


if __name__ == "__main__":
    main()