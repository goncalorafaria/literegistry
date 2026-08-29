![LiteRegistry](literegistry.png)

Lightweight service registry and discovery system for distributed model inference clusters. Built for deployments on HPC environments with load balancing and automatic failover.


## Installation

```bash
pip install literegistry
```

Install optional sibling packages through LiteRegistry extras:

```bash
pip install "literegistry[podman_client]"
pip install "literegistry[podman_beaker]"
pip install "literegistry[all]"
```

These are convenience dependencies; the client and Beaker deployment code
remain separately versioned distributions.

## Documentation

Usage guides with argument reference live in [`docs/`](docs/README.md)
(published at [goncalorafaria.github.io/literegistry](https://goncalorafaria.github.io/literegistry/)):

- [CLI reference](docs/cli.md)
- [Registry (Redis & filesystem)](docs/registry.md)
- [Gateway](docs/gateway.md)
- [Package layout](docs/package-layout.md)
- [vLLM & SGLang](docs/vllm-sglang.md)
- [Code & Terminal](docs/code-and-terminal.md)
- [Podman affinity containers](docs/podman.md)
- [Standalone Podman gateway client](PODMAN_GATEWAY_README.md)
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
accepts `rg`, `grep`, `awk`, `sed`, `jq`, `xsv`, `pandoc`, `sort`, `uniq`, `tr`,
`cut`, `head`, `tail`, `wc`, `cat`, `nl`, and `echo`, joined by
pipes. It does not evaluate shell syntax or permit submitted file paths.

```bash
literegistry terminal --registry redis://klone-login01.hyak.local:6379
```

**Start Rootless Podman Affinity Server**

Podman is a first-class stateful LiteRegistry service. `gateway` exposes
the handshake, terminal, and close lifecycle; `podman` creates and owns the
rootless containers and registers itself under `model_path="podman"`.

```bash
literegistry gateway --registry redis://login-node:6379 --port 8080 --workers 8
literegistry podman \
  --registry redis://login-node:6379 \
  --host 0.0.0.0 \
  --port 8091 \
  --allow-non-loopback=True \
  --advertise-host podman-node \
  --advertise-port 8091 \
  --image python:3.12-slim
```

See [Podman affinity containers](docs/podman.md) for rootless and host-network
deployment details plus the handshake/command/close API.

**Start Search Server**

The search worker combines the live-web providers behind one registered
`model_path="search"`: Serper handles `mode="query"`, while Jina Reader handles
`mode="url"`. Successful responses are cached in a separate logical database
on the registry Redis instance.

```bash
export SERPER_API_KEY=...
export JINA_API_KEY=...
literegistry search \
  --registry redis://login-node:6379 \
  --cache-db 1 \
  --cache-ttl 3600
```

Extra Serper fields such as `gl` and `hl` can be supplied in a query request's
`parameters` object. For URL visits, `parameters` become Jina request headers;
the returned `data` is normalized to `{ "title", "content", "url" }`. To use
another query JSON API, pass `--provider generic` together with
`--search-api-url`; URL visits remain Jina Reader (override its endpoint with
`--fetch-api-url` or `JINA_READER_URL`).

**Start Local Lucene BM25 Server**

The BM25 service is implemented directly in LiteRegistry and uses Pyserini for
Lucene retrieval. Point it at an existing corpus and index, and optionally
register it as a named local-search pool:

```bash
python -m pip install 'literegistry[bm25]'
literegistry bm25 \
  --corpus_jsonl=/data/corpus.jsonl \
  --lucene_index_dir=/data/lucene-index \
  --registry=redis://login-node:6379 \
  --service_name=localsearch:corpus \
  --port=1214
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

# Search through the configured query API
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"mode": "query", "query": "distributed LLM inference", "num_results": 5}'

# Retrieve one URL through the configured fetch API
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"mode": "url", "url": "https://example.com/article"}'
```

The gateway automatically routes requests to the appropriate model server based on the `model` field.
For code execution, it routes `/python` requests to services registered as `python`.
For log slicing, it routes `/terminal` requests to services registered as `terminal`.
For search and URL retrieval, it routes `/search` requests to services registered as `search`.

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
