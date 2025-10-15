# LiteRegistry

Lightweight service registry and discovery system for distributed model inference clusters. Built for deployments on HPC environments with load balancing and automatic failover.


## Installation

```bash
pip install literegistry
```

## Components

### Registry (Key-Value Store)
The registry stores service metadata and health information. Choose between:
- **FileSystem**: Simple file-based storage for single-node setups
- **Redis**: Distributed storage for multi-node HPC clusters (recommended for production)

The registry tracks which model servers are available, their endpoints, and performance metrics.

### vLLM Module
Wraps vLLM servers with automatic registry integration. When you launch vLLM through LiteRegistry, it:
- Auto-registers with the registry on startup
- Sends heartbeats to maintain active status
- Reports performance metrics

### Gateway Server
HTTP reverse proxy that routes client requests to model servers. Features:
- OpenAI-compatible API endpoints (`/v1/completions`, `/v1/models`, `/classify`)
- Automatic load balancing based on server latency
- Model routing based on the `model` parameter in requests

### CLI Tool
Command-line interface for monitoring your cluster:
- View registered models and server counts
- Check server health and request statistics
- Monitor latency metrics and request throughput

### Client Library
Python API for programmatic interaction:
- `RegistryClient`: Register servers and query available models
- `RegistryHTTPClient`: Make requests with automatic failover and retry

### How Components Work Together

```
1. vLLM servers register themselves:
   vLLM Instance → Registry (Redis/FS)
   
2. Client sends request to Gateway:
   Client → Gateway Server
   
3. Gateway queries Registry and routes to best server:
   Gateway → Registry (get available servers)
   Gateway → vLLM Instance (send request)
   
4. Gateway reports metrics back:
   Gateway → Registry (update latency/stats)
```

## HPC Cluster Deployment

Complete workflow for deploying distributed model inference:

**1. Start Redis Server**
```bash
literegistry redis --port 6379
```

**2. Launch vLLM Instances** (supports all standard vLLM arguments)
```bash
literegistry vllm \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 4
```

**3. Start Gateway Server**
```bash
literegistry gateway \
  --registry redis://login-node:6379 \
  --host 0.0.0.0 \
  --port 8080
```

**4. Monitor Cluster**
```bash
# Summary view
literegistry summary --registry redis://login-node:6379
```

## Quick Start

### Basic Usage

```python
from literegistry import RegistryClient, get_kvstore
import asyncio

async def main():
    # Auto-detect backend (redis:// or file path)
    store = get_kvstore("redis://localhost:6379")
    client = RegistryClient(store, service_type="model_path")
    
    # Register a server
    await client.register(
        port=8000,
        metadata={"model_path": "meta-llama/Llama-3.1-8B-Instruct"}
    )
    
    # List available models
    models = await client.models()
    print(models)

asyncio.run(main())
```

### HTTP Client with Automatic Failover

```python
from literegistry import RegistryHTTPClient

async with RegistryHTTPClient(client, "meta-llama/Llama-3.1-8B-Instruct") as http_client:
    result, _ = await http_client.request_with_rotation(
        "v1/completions",
        {"prompt": "Hello"},
        timeout=30,
        max_retries=3
    )
```

## Storage Backends

LiteRegistry supports different backends depending on your deployment:

**FileSystem** - For single-node or shared filesystem environments
```python
from literegistry import FileSystemKVStore
store = FileSystemKVStore("registry_data")
```
Use when: Running on a single machine or when all nodes share a filesystem (common in HPC clusters with NFS). Note: Can bottleneck with high concurrency.

**Redis** - For distributed multi-node clusters
```python
from literegistry import RedisKVStore
store = RedisKVStore("redis://localhost:6379")
```
Use when: Running across multiple nodes without shared storage, or need high-concurrency access. Recommended for production HPC deployments.

## Advanced Usage

### Gateway API

The gateway provides OpenAI-compatible HTTP endpoints that work with existing tools:

```bash
# Send completion request
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello"}'

# List all available models
curl http://localhost:8080/v1/models

# Check gateway health
curl http://localhost:8080/health
```

The gateway automatically routes requests to the appropriate model server based on the `model` field.

### Batch Processing with Parallel Requests

Process multiple requests concurrently with automatic load balancing:

```python
async with RegistryHTTPClient(client, model) as http_client:
    # Process 100 requests with max 5 concurrent
    results = await http_client.parallel_requests(
        "v1/completions",
        payloads_list,
        max_parallel_requests=5,
        timeout=30,
        max_retries=3
    )
```

This is useful for batch inference workloads. The client handles retry logic and server rotation automatically.


## Contributing

Contributions welcome! Please submit a Pull Request.

## License

MIT License - see LICENSE file for details
