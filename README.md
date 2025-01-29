# LiteRegistry

LiteRegistry is a liteweight, flexible service registry and discovery system with built-in telemetry and caching. It supports multiple storage backends and provides async-first APIs for modern Python applications.

## Features

- 🌐 HTTP client with automatic retry and server rotation
- 🔄 Parallel request handling with concurrency controls
- 📈 Automatic latency reporting and server health tracking

- 🔄 Flexible storage backends (FileSystem, Consul)
- 📊 Built-in telemetry and latency tracking
- 🚀 Async-first API design
- 💾 Intelligent caching with TTL
- ⚖️ Load balancing with latency-aware routing
- 🔍 Model-aware service discovery
- ❤️ Health checking with automatic inactive server pruning

## Installation

```bash
pip install literegistry
```

## Quick Start

### Basic Server Registration

```python
from literegistry import ServerRegistry, FileSystemKVStore
import asyncio

async def main():
    # Initialize with filesystem backend
    store = FileSystemKVStore("registry_data")
    registry = ServerRegistry(store)
    
    # Register a server
    server_id = await registry.register_server(8000, {"service": "api"})
    
    # Get active servers
    roster = await registry.roster()
    print(f"Active servers: {roster}")

asyncio.run(main())
```

### Model Registry with Load Balancing

```python
from literegistry import ModelRegistry, ConsulKVStore
from aioconsul import Consul

async def main():
    # Initialize with Consul backend
    async with Consul() as consul:
        store = ConsulKVStore(consul, prefix="services/")
        registry = ModelRegistry(store)
        
        # Register a model server
        await registry.register_server(8000, {
            "model_path": "gpt-3",
            "capacity": "high"
        })
        
        # Get best server for model
        best_uri = await registry.get("gpt-3")
        
        # Report latency for load balancing
        registry.report_latency(best_uri, 0.5)
```

## Storage Backends

### FileSystem Backend

```python
from literegistry import FileSystemKVStore

store = FileSystemKVStore("registry_data")
```

### Consul Backend

```python
from literegistry import ConsulKVStore
from aioconsul import Consul

async with Consul() as consul:
    store = ConsulKVStore(consul, prefix="services/")
```

## Advanced Usage

### Error Handling with HTTP Client

The HTTP client provides comprehensive error handling:

```python
async with RegistryHTTPClient(registry, "gpt-3") as client:
    try:
        result, _ = await client.request_with_rotation(
            "v1/completions",
            payload,
            timeout=30,
            max_retries=3
        )
    except ValueError as e:
        # Handle bad request errors
        print(f"Bad request: {e}")
    except RuntimeError as e:
        # Handle retry exhaustion
        print(f"All retries failed: {e}")
```

### Parallel Request Configuration

Control parallel request behavior:

```python
# Configure maximum parallel requests
max_parallel = 5  # Allow up to 5 concurrent requests

results = await client.parallel_requests(
    "v1/completions",
    large_payload_list,
    timeout=30,
    max_retries=3,
    max_parallel_requests=max_parallel
)
```

### Custom Backend Implementation

Create your own storage backend by implementing the `KeyValueStore` interface:

```python
class CustomStore(KeyValueStore):
    async def get(self, key: str) -> Optional[bytes]:
        ...
    
    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        ...
    
    async def delete(self, key: str) -> bool:
        ...
    
    async def exists(self, key: str) -> bool:
        ...
    
    async def keys(self) -> List[str]:
        ...
```

### HTTP Client Usage

The package includes a robust HTTP client that integrates with the registry system:

```python
from literegistry import RegistryHTTPClient, ModelRegistry

# Initialize registry
registry = ModelRegistry(store)

# Use the client with context manager
async with RegistryHTTPClient(registry, "gpt-3") as client:
    # Single request with automatic retries and server rotation
    result, server_idx = await client.request_with_rotation(
        "v1/completions",
        {"prompt": "Hello"},
        timeout=30,
        max_retries=3
    )
    
    # Parallel requests with concurrency control
    payloads = [
        {"prompt": "Hello"},
        {"prompt": "World"}
    ]
    results = await client.parallel_requests(
        "v1/completions",
        payloads,
        timeout=30,
        max_retries=3,
        max_parallel_requests=2
    )
```

The HTTP client provides:
- Automatic server rotation on failures
- Built-in retry mechanism with exponential backoff
- Integrated latency reporting
- Parallel request handling with concurrency limits
- Proper session management via context manager
- Automatic server list refresh on failures

### FastAPI Integration

```python
from fastapi import FastAPI
from literegistry import RequestCounterMiddleware, ServerRegistry

app = FastAPI()
registry = ServerRegistry(store)

# Add request tracking middleware
app.add_middleware(RequestCounterMiddleware, registry=registry)

# Automatically track request counts
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### Telemetry and Load Balancing

The ModelRegistry provides intelligent load balancing based on server latency:

```python
registry = ModelRegistry(store)

# Get multiple servers weighted by performance
uris = await registry.get_all("gpt-3", n=3)  # Get 3 servers

# Report latencies to improve load balancing
for uri in uris:
    response_time = await make_request(uri)
    registry.report_latency(uri, response_time)
```

## Configuration

Key configuration options:

```python
registry = ModelRegistry(
    store,
    max_history=3600,  # Request history window
    cache_ttl=300,     # Cache TTL in seconds
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details