#!/usr/bin/env python3
"""
Comprehensive benchmark script to compare all gateway implementations:
- aiohttp (original)
- FastAPI + uvicorn
- Starlette + uvicorn
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
            try:
                async with session.get(url) as response:
                    await response.json()
                    return time.time() - req_start
            except Exception as e:
                return -1  # Mark failed requests
        
        # Run concurrent requests
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Filter out failed requests
        latencies = [r for r in results if r >= 0]
        failed_requests = len(results) - len(latencies)
        
        total_time = time.time() - start_time
    
    if not latencies:
        return {
            "total_requests": num_requests,
            "failed_requests": num_requests,
            "success_rate": 0.0,
            "total_time": total_time,
            "requests_per_second": 0.0,
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "p99_latency": 0.0,
            "min_latency": 0.0,
            "max_latency": 0.0
        }
    
    return {
        "total_requests": num_requests,
        "failed_requests": failed_requests,
        "success_rate": len(latencies) / num_requests * 100,
        "total_time": total_time,
        "requests_per_second": len(latencies) / total_time,
        "avg_latency": statistics.mean(latencies),
        "p50_latency": statistics.quantiles(latencies, n=2)[0] if len(latencies) > 1 else latencies[0],
        "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 19 else latencies[-1],
        "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 98 else latencies[-1],
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }


def print_results(name: str, results: Dict[str, float]):
    """Print benchmark results in a formatted way."""
    print(f"{name}:")
    print(f"  Requests/sec: {results['requests_per_second']:.1f}")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Avg Latency:  {results['avg_latency']*1000:.1f}ms")
    print(f"  P95 Latency:  {results['p95_latency']*1000:.1f}ms")
    if results['failed_requests'] > 0:
        print(f"  Failed:       {results['failed_requests']}")


async def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks for all gateway implementations."""
    print("🚀 Comprehensive Gateway Performance Benchmark")
    print("=" * 60)
    
    # Test health endpoint (lightweight)
    print("\n📊 Testing /health endpoint (1000 requests):")
    print("-" * 50)
    
    # aiohttp gateway (port 8080)
    try:
        aiohttp_results = await benchmark_endpoint("http://localhost:8080/health")
        print_results("aiohttp Gateway", aiohttp_results)
    except Exception as e:
        print(f"aiohttp Gateway: Error - {e}")
    
    # FastAPI gateway (port 8081)
    try:
        fastapi_results = await benchmark_endpoint("http://localhost:8081/health")
        print_results("FastAPI Gateway", fastapi_results)
    except Exception as e:
        print(f"FastAPI Gateway: Error - {e}")
    
    # Starlette gateway (port 8082)
    try:
        starlette_results = await benchmark_endpoint("http://localhost:8082/health")
        print_results("Starlette Gateway", starlette_results)
    except Exception as e:
        print(f"Starlette Gateway: Error - {e}")
    
    # Test models endpoint (heavier)
    print("\n📊 Testing /v1/models endpoint (100 requests):")
    print("-" * 50)
    
    try:
        aiohttp_models = await benchmark_endpoint("http://localhost:8080/v1/models", 100)
        print_results("aiohttp Gateway", aiohttp_models)
    except Exception as e:
        print(f"aiohttp Gateway: Error - {e}")
    
    try:
        fastapi_models = await benchmark_endpoint("http://localhost:8081/v1/models", 100)
        print_results("FastAPI Gateway", fastapi_models)
    except Exception as e:
        print(f"FastAPI Gateway: Error - {e}")
    
    try:
        starlette_models = await benchmark_endpoint("http://localhost:8082/v1/models", 100)
        print_results("Starlette Gateway", starlette_models)
    except Exception as e:
        print(f"Starlette Gateway: Error - {e}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("🏆 Performance Summary:")
    print("-" * 30)
    
    # Find best performer for health endpoint
    health_results = []
    if 'aiohttp_results' in locals():
        health_results.append(("aiohttp", aiohttp_results['requests_per_second']))
    if 'fastapi_results' in locals():
        health_results.append(("FastAPI", fastapi_results['requests_per_second']))
    if 'starlette_results' in locals():
        health_results.append(("Starlette", starlette_results['requests_per_second']))
    
    if health_results:
        best_health = max(health_results, key=lambda x: x[1])
        print(f"🏅 Best Health Endpoint: {best_health[0]} ({best_health[1]:.1f} req/s)")
    
    # Find best performer for models endpoint
    models_results = []
    if 'aiohttp_models' in locals():
        models_results.append(("aiohttp", aiohttp_models['requests_per_second']))
    if 'fastapi_models' in locals():
        models_results.append(("FastAPI", fastapi_models['requests_per_second']))
    if 'starlette_models' in locals():
        models_results.append(("Starlette", starlette_models['requests_per_second']))
    
    if models_results:
        best_models = max(models_results, key=lambda x: x[1])
        print(f"🏅 Best Models Endpoint: {best_models[0]} ({best_models[1]:.1f} req/s)")
    
    print("\n💡 Expected Results:")
    print("  • Starlette should be fastest (lightweight ASGI)")
    print("  • FastAPI should be 2nd (built on Starlette)")
    print("  • aiohttp should be slowest (older async framework)")
    print("  • uvicorn provides the fastest ASGI server")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmarks())










