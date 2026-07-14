# vLLM and SGLang

`literegistry vllm` and `literegistry sglang` start a model server, register it
in the registry under `metadata.model_path=<model>`, and heartbeat while healthy.
Extra CLI flags are forwarded to the underlying server.

## Quick start

```bash
# vLLM (default runtime: apptainer)
literegistry vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 4

# SGLang
literegistry sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://login-node:6379 \
  --tp-size 1
```

Local Python env (no container):

```bash
literegistry vllm \
  --runtime local \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 1
```

## Shared arguments (both wrappers)

| Argument | Default (vLLM / SGLang) | Meaning |
|----------|-------------------------|---------|
| `model` | `meta-llama/Llama-3.1-8B-Instruct` | Model id or local path; becomes `model_path` in the registry |
| `host` | `0.0.0.0` | Bind address for the backend server |
| `registry` | path or Redis URL (differs by wrapper default) | Where to register |
| `port` | random `8000–12000` if omitted | Backend listen port |
| `runtime` | `apptainer` | `local` or `apptainer` — see [Runtimes](runtimes.md) |
| `image` | vLLM / SGLang SIF name | Container image path or basename under `$HOME` |
| `image_source` | official Docker URI | Used by `apptainer pull` when SIF missing |
| `pull_image` | `True` | Pull if image file does not exist |
| `workdir` | `None` | Apptainer `--pwd` |
| `bind` | `None` (+ auto HF/home binds) | Extra `--bind` mounts |
| `env` | `None` (+ auto HF cache env) | Extra `KEY=VALUE` env entries |
| `apptainer_nv` | `True` | Pass `--nv` (GPU) |
| `apptainer_cleanenv` | `True` | Pass `--cleanenv` |
| `apptainer_executable` | `apptainer` | Binary |
| `apptainer_extra_args` | `None` | Extra Apptainer flags |
| `**kwargs` | — | Forwarded to vLLM / SGLang as CLI flags |

### Default images

| Backend | `image` | `image_source` |
|---------|---------|----------------|
| vLLM | `vllm-openai_latest-cu129-ubuntu2404.sif` | `docker://vllm/vllm-openai:latest-cu129-ubuntu2404` |
| SGLang | `sglang_latest.sif` | `docker://lmsysorg/sglang:latest` |

Relative SIF names resolve under `$HOME` (or `LITEREGISTRY_APPTAINER_IMAGE_DIR`).

## Passthrough flags (`**kwargs`)

Any extra Fire argument becomes a backend flag:

- `tensor_parallel_size=4` → `--tensor-parallel-size 4`
- `enable_chunked_prefill=True` → `--enable-chunked-prefill` (boolean flags omit `false`)
- `max_num_seqs=256` → `--max-num-seqs 256`

Examples:

```bash
literegistry vllm \
  --model /path/to/weights \
  --registry redis://login-node:6379 \
  --runtime apptainer \
  --port 7248 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --trust-remote-code

literegistry sglang \
  --model allenai/Llama-3.1-Tulu-3-8B-DPO \
  --registry redis://login-node:6379 \
  --tp_size=1 \
  --mem_fraction_static=0.9
```

SGLang clears proxy env vars (`http_proxy`, `HTTPS_PROXY`, …) before launch so
cluster proxies do not break local model downloads / serving.

## What gets registered

```json
{
  "model_path": "<your --model>",
  "host": "0.0.0.0",
  "port": 8123,
  "backend": "vllm",
  "extra_kwargs": { "...": "passthrough args" },
  "runtime": "apptainer",
  "image": "/home/you/….sif",
  "image_source": "docker://…"
}
```

The gateway’s `model` field in `/v1/completions` must equal this `model_path`
string exactly (including local paths).

## Lifecycle

1. Raise `RLIMIT_NOFILE` (soft up to 65536).
2. `runtime.prepare()` (e.g. pull Apptainer image).
3. Start subprocess: `runtime.build_command(backend_cmd)`.
4. `register_server` with metadata above.
5. Background loop: every `heartbeat_interval` (default 10s), `GET /v1/models`
   locally; if healthy, refresh heartbeat; if not, log unhealthy.
6. On exit: deregister and terminate the process.

## Backend command differences

| | vLLM | SGLang |
|--|------|--------|
| Local command | `python -m vllm.entrypoints.openai.api_server` | `python3 -m sglang.launch_server` |
| Apptainer command | `vllm serve <model>` | same as local (`python3 -m sglang…`) |
| Model flag | `--model` (positional after `serve` in Apptainer) | `--model-path` |

## Verify

```bash
literegistry summary --registry redis://login-node:6379
# expect your model_path with replica count > 0

curl http://GATEWAY:8080/v1/models
curl -X POST http://GATEWAY:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"Hi","max_tokens":8}'
```

Next: [Runtimes](runtimes.md) · [Gateway](gateway.md)
