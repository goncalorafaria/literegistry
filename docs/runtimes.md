# Runtimes

A **runtime** wraps the real backend command (`vllm serve`, `redis-server`, …)
so the same LiteRegistry CLI can run on bare metal or inside Apptainer.

Used by: `literegistry redis`, `vllm`, `sglang` (and anything else calling
`build_runtime`).

## Options

| `runtime` | Class | Behavior |
|-----------|-------|----------|
| `local` | `LocalRuntime` | Run the command in the current environment |
| `apptainer` | `ApptainerRuntime` | `apptainer exec [flags] image.sif <cmd…>` |

```bash
# Container (default for vLLM / SGLang / Redis)
literegistry vllm --runtime apptainer --model … --registry redis://…

# Host Python / host redis-server
literegistry vllm --runtime local --model … --registry redis://…
literegistry redis --runtime local --foreground --port 6379
```

## Building a runtime from CLI values

`build_runtime(...)` accepts:

| Argument | Default | Meaning |
|----------|---------|---------|
| `runtime` | `"local"` in the helper; wrappers often default `"apptainer"` | `local` or `apptainer` |
| `image` | required for apptainer | SIF path or basename |
| `image_source` | `None` | Docker/oras URI for `apptainer pull` |
| `pull_image` | `True` | Pull when image file is missing |
| `workdir` | `None` | Apptainer `--pwd` |
| `bind` | `None` | Bind mounts (`/host:/cont` or list) |
| `env` | `None` | `KEY=VALUE` string, list, or dict |
| `apptainer_nv` | `True` | GPU (`--nv`) |
| `apptainer_cleanenv` | `True` | `--cleanenv` |
| `apptainer_executable` | `"apptainer"` | Binary name |
| `apptainer_extra_args` | `None` | Extra flags before the image |

Unsupported names raise `ValueError`.

## Apptainer details

Effective command shape:

```text
apptainer exec
  [--nv]
  [--cleanenv]
  [--bind …]*
  [--env KEY=VALUE]*
  [--pwd WORKDIR]
  [extra_args…]
  /path/to/image.sif
  <backend command…>
```

### Image resolution

- Absolute `image` paths are used as-is.
- Relative names resolve to
  `$LITEREGISTRY_APPTAINER_IMAGE_DIR` or `$HOME` / `~`,
  e.g. `vllm-….sif` → `$HOME/vllm-….sif`.
- If `pull_image` and `image_source` are set and the file is missing,
  LiteRegistry runs `apptainer pull <image> <source>`.

### Default binds and env (vLLM / SGLang)

Wrappers merge your `--bind` / `--env` with shell-derived defaults so HF caches
work inside `--cleanenv` containers.

**Env (always for those wrappers):**

| Variable | Source |
|----------|--------|
| `HOME` | launching shell |
| `HF_HOME` / hub / transformers caches | shell vars or `$HOME/.cache/…` |
| `VLLM_CACHE_ROOT` | shell or `$HOME/.cache/vllm` (vLLM only) |

**Binds:** each of those absolute paths that already exist is mounted
`path:path`. Your extra `--bind` values are appended.

```bash
literegistry vllm \
  --runtime apptainer \
  --model /datasets/my-model \
  --registry redis://login-node:6379 \
  --bind /datasets/my-model:/datasets/my-model \
  --env HF_TOKEN=hf_xxx \
  --tensor-parallel-size 1
```

Pass multiple binds/env with Fire lists if needed:

```bash
--bind='["/a:/a","/b:/b"]' --env='["FOO=1","BAR=2"]'
```

### Redis runtime notes

Redis defaults to Apptainer with `redis_7-alpine.sif` /
`docker://redis:7-alpine`. With `--runtime local`, point at a host binary via
`redis_server_path` if needed. `--foreground` keeps Redis in the current
process (useful on interactive nodes).

## Metadata in the registry

Container launches attach runtime info to server metadata:

```json
{
  "runtime": "apptainer",
  "image": "/home/you/vllm-….sif",
  "image_source": "docker://vllm/…"
}
```

Local launches only set `"runtime": "local"`.

## Choosing local vs Apptainer

| Prefer `local` when… | Prefer `apptainer` when… |
|----------------------|--------------------------|
| Conda/module stack already has vLLM/SGLang | Login/compute nodes lack the right CUDA stack |
| Debugging the backend binary itself | You want reproducible SIF images across jobs |
| Redis is already installed on the node | You want zero host package installs |

Next: [vLLM & SGLang](vllm-sglang.md) · [Registry](registry.md)
