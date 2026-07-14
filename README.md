![LiteRegistry](literegistry.png)

Lightweight service registry and discovery system for distributed model inference clusters. Built for deployments on HPC environments with load balancing and automatic failover.


## Installation

```bash
pip install literegistry
```

## Documentation

Usage guides with argument reference live in [`docs/`](docs/README.md)
(published at [goncalorafaria.github.io/literegistry](https://goncalorafaria.github.io/literegistry/)):

- [CLI reference](docs/cli.md)
- [Registry (Redis & filesystem)](docs/registry.md)
- [Gateway](docs/gateway.md)
- [vLLM & SGLang](docs/vllm-sglang.md)
- [Code & Terminal](docs/code-and-terminal.md)
- [Load balancing](docs/load-balancing.md)
- [Runtimes](docs/runtimes.md)
- [Console](docs/console.md)

## Quick Start

Complete workflow for deploying distributed model inference:

**1. Start Redis Server**
```bash
literegistry redis --port 6379
```

By default this starts Redis inside Apptainer using the official Redis image
`redis_7-alpine.sif`, pulled from `docker://redis:7-alpine`. To use a host
Redis binary instead:

```bash
literegistry redis --runtime local --port 6379
```

To keep Redis attached to the current terminal/process, run it in foreground
mode:

```bash
literegistry redis --runtime local --foreground --port 6379
```

Redis startup prints a machine-readable registry URL that includes the selected
port:

```text
REDIS_URL=redis://hostname:6379
```

**2. Launch vLLM/SGLang Instances** (supports all standard vLLM/SGLang arguments)
```bash
literegistry vllm \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 4
```

To launch vLLM inside Apptainer, choose the Apptainer runtime and provide any
binds or container environment variables. The default vLLM Apptainer image is
`vllm-openai_latest-cu129-ubuntu2404.sif`, pulled from
`docker://vllm/vllm-openai:latest-cu129-ubuntu2404`. Apptainer launches also
bind `$HOME` plus the shell-derived Hugging Face cache paths by default. If
`HF_HOME`, `HF_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`,
`TRANSFORMERS_CACHE`, or `VLLM_CACHE_ROOT` are set in the launching shell, those
values are passed into the container; otherwise LiteRegistry falls back to
cache paths under `$HOME/.cache`.

```bash
literegistry vllm \
  --runtime apptainer \
  --model /mmfs1/gscratch/ark/graf/judges-that-code/thinker/tinker-sft-demo_vllm_model \
  --registry redis://login-node:6379 \
  --port 7248 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --trust-remote-code \
  --language-model-only \
  --safetensors-load-strategy prefetch
```

For SGLang, the default Apptainer image is `sglang_latest.sif`, pulled from
the official `docker://lmsysorg/sglang:latest` image. It uses the same shared
Hugging Face cache defaults.

**3. Start Gateway Server**
```bash
literegistry gateway \
  --registry redis://login-node:6379 \
  --host 0.0.0.0 \
  --port 8080
```

**Start Python Code Executor**

LiteRegistry can also register a stateless Python code execution service. The
service registers itself under `model_path="python"` so the gateway can route
`POST /python` requests to available executor workers.

```bash
literegistry code --registry redis://klone-login01.hyak.local:6379
```

**Start Terminal Pipeline Server**

The terminal server runs restricted, stdin-only log-analysis pipelines. It
accepts `rg`, `grep`, `awk`, `sed`, `jq`, `xsv`, `head`, `tail`, `wc`, and `nl`, joined by
pipes. It does not evaluate shell syntax or permit submitted file paths.

```bash
literegistry terminal --registry redis://klone-login01.hyak.local:6379
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

# Execute Python through the gateway
curl -X POST http://localhost:8080/python \
  -H "Content-Type: application/json" \
  -d '{"code": "print(2 + 2)", "max_runtime": 1.0}'

# Execute Python with a context payload
curl -X POST http://localhost:8080/python \
  -H "Content-Type: application/json" \
  -d '{"code": "data = json.loads(context)\nprint(data[\"name\"])\nprint(data[\"score\"] + 1)", "context_payload": "{\"name\": \"alice\", \"score\": 41}", "max_runtime": 3}'

# Analyze submitted log contents through the gateway
curl -X POST http://localhost:8080/terminal \
  -H "Content-Type: application/json" \
  -d '{"contents": "INFO started\nERROR disk full\nERROR retrying\n", "command": "rg ERROR | head -n 1", "max_runtime": 5}'
```

The gateway automatically routes requests to the appropriate model server based on the `model` field.
For code execution, it routes `/python` requests to services registered as `python`.
For log slicing, it routes `/terminal` requests to services registered as `terminal`.

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
