![LiteRegistry](literegistry.png)

Lightweight service registry and discovery system for distributed model inference clusters. Built for deployments on HPC environments with load balancing and automatic failover.


## Installation

```bash
pip install literegistry
```

## Quick Start

Complete workflow for deploying distributed model inference:

**1. Start Redis Server**
```bash
literegistry redis --port 6379
```

**2. Launch vLLM/SGLang Instances** (supports all standard vLLM/SGLang arguments)
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

**4. Interact with Gateway**

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

**5. Monitor Cluster**
```bash
# Summary view
literegistry summary --registry redis://login-node:6379
```

## Using the Python API

### Writting new servers

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

### Storage Backends

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



## Citation

If you use LiteRegistry in your research, please cite:

```
@software{literegistry2025,
  title={literegistry: Lightweight Service Discovery for Distributed Model Inference},
  author={Faria, Gonçalo and Smith, Noah},
  year={2025},
  url={https://github.com/goncalorafaria/literegistry}
}
```

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

MIT License - see LICENSE file for details
