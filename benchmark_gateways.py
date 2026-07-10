#!/usr/bin/env python3
"""
Benchmark script to compare aiohttp vs FastAPI gateway performance.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict


async def benchmark_endpoint(url: str, num_requests: int = 1000) -> Dict[str, float]:
    """Benchmark a single endpoint."""
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create tasks for concurrent requests
        async def make_request():
            req_start = time.time()
            async with session.get(url) as response:
                await response.json()
                return time.time() - req_start
        
        # Run concurrent requests
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        latencies.extend(results)
        
        total_time = time.time() - start_time
    
    return {
        "total_requests": num_requests,
        "total_time": total_time,
        "requests_per_second": num_requests / total_time,
        "avg_latency": statistics.mean(latencies),
        "p50_latency": statistics.quantiles(latencies, n=2)[0],
        "p95_latency": statistics.quantiles(latencies, n=20)[18],
        "p99_latency": statistics.quantiles(latencies, n=100)[98],
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }


async def run_benchmarks():
    """Run benchmarks for both gateway implementations."""
    print("🚀 Gateway Performance Benchmark")
    print("=" * 50)
    
    # Test health endpoint (lightweight)
    print("\n📊 Testing /health endpoint (1000 requests):")
    print("-" * 40)
    
    # aiohttp gateway
    try:
        aiohttp_results = await benchmark_endpoint("http://localhost:8080/health")
        print("aiohttp Gateway:")
        print(f"  Requests/sec: {aiohttp_results['requests_per_second']:.1f}")
        print(f"  Avg Latency:  {aiohttp_results['avg_latency']*1000:.1f}ms")
        print(f"  P95 Latency:  {aiohttp_results['p95_latency']*1000:.1f}ms")
    except Exception as e:
        print(f"  aiohttp Gateway: Error - {e}")
    
    # FastAPI gateway (if running on different port)
    try:
        fastapi_results = await benchmark_endpoint("http://localhost:8081/health")
        print("FastAPI Gateway:")
        print(f"  Requests/sec: {fastapi_results['requests_per_second']:.1f}")
        print(f"  Avg Latency:  {fastapi_results['avg_latency']*1000:.1f}ms")
        print(f"  P95 Latency:  {fastapi_results['p95_latency']*1000:.1f}ms")
    except Exception as e:
        print(f"  FastAPI Gateway: Error - {e}")
    
    # Test models endpoint (heavier)
    print("\n📊 Testing /v1/models endpoint (100 requests):")
    print("-" * 40)
    
    try:
        aiohttp_models = await benchmark_endpoint("http://localhost:8080/v1/models", 100)
        print("aiohttp Gateway:")
        print(f"  Requests/sec: {aiohttp_models['requests_per_second']:.1f}")
        print(f"  Avg Latency:  {aiohttp_models['avg_latency']*1000:.1f}ms")
    except Exception as e:
        print(f"  aiohttp Gateway: Error - {e}")
    
    try:
        fastapi_models = await benchmark_endpoint("http://localhost:8081/v1/models", 100)
        print("FastAPI Gateway:")
        print(f"  Requests/sec: {fastapi_models['requests_per_second']:.1f}")
        print(f"  Avg Latency:  {fastapi_models['avg_latency']*1000:.1f}ms")
    except Exception as e:
        print(f"  FastAPI Gateway: Error - {e}")
    
    print("\n" + "=" * 50)
    print("💡 Expected Results:")
    print("  • FastAPI should be 3-6x faster than aiohttp")
    print("  • uvicorn handles async more efficiently")
    print("  • Better for high-throughput production use")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())










