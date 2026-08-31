# LiteRegistry Base Deployment

`literegistry-base-deployment` is a standalone Beaker deployment package for
the non-Podman LiteRegistry stack previously launched from Datadev:

```text
clients -> one gateway -> Python execution replicas
                       -> restricted terminal replicas
                       -> cached Serper query + Jina URL replicas
                       -> Lucene BM25 local-search replicas
                       -> vLLM generation replicas
                       -> vLLM classification replicas
                                |
                                +-> managed or external Redis registry
```

It depends strongly on `literegistry` and has no Datadev dependency. Shared
coordination lives in `literegistry.coop`; this package contains no copy of the
port allocator, Redis barrier, or artifact-build locks. It copies the proven
mechanics from `literegistry-podman-beaker`: Fire CLI, preview before
submission, host networking, collision-safe dynamic ports, CPU-cluster replica
spreading, managed-or-external Redis, Weka URL files, and direct Beaker specs.

## What is deployed

| Task | Gateway route | Default replicas | Compute |
|---|---|---:|---|
| Gateway | all routes | 1 process group, 8 workers | CPU |
| Python | `POST /python` | 1 | CPU |
| Terminal | `POST /terminal` | 1 | CPU |
| Web search / fetch | `POST /search` | 1 | CPU |
| Local BM25 search | `POST /search` with `model_path=localsearch:...` | 0 | CPU |
| vLLM generation | `/v1/chat/completions`, `/v1/completions` | 0 | GPU |
| vLLM classification | `POST /classify` | 0 | GPU |
| Redis | internal service discovery | 1 when `--registry` is omitted | CPU |

Generation and classification are separate vLLM pools. Generation launches
vLLM with `--task=generate`; classification launches it with
`--task=classify`, which exposes vLLM's sequence-classification endpoint. The
gateway chooses either pool using the request's `model` field.

## End-to-end setup

These commands assume Bash, this repository checkout, Docker, the Beaker CLI,
and access to the target workspace, clusters, budget, and Weka source.

### 1. Install the launcher

```bash
cd /weka/gfaria/literegistry/literegistry_base_deployment
python -m pip install -e '.[test,publish]'
literegistry-base-deployment --help
beaker config test
beaker account whoami
beaker workspace get ai2/oe-agents
```

A published install through the LiteRegistry extra is:

```bash
pip install "literegistry[base_deployment]"
```

The standalone distribution can also be installed directly:

```bash
pip install literegistry-base-deployment
```

### 2. Make LiteRegistry available to Docker

Every runtime image installs this package, which installs
`literegistry>=1.0.36`. Publish that LiteRegistry version to the Python index
used by Docker, or expose its wheel through an HTTP wheelhouse reachable from
inside Docker:

```bash
cd /weka/gfaria/literegistry
python -m build
python -m twine check dist/*
# python -m twine upload dist/literegistry-1.0.36*

export PIP_INDEX_URL=https://python.example/simple
# Or: export PIP_FIND_LINKS=https://python.example/wheels
# Optional for an internal HTTP endpoint:
# export PIP_TRUSTED_HOST=python.example
```

The four default images built by this package need no `PYTHONPATH`, Datadev
checkout, `~/basic_images`, or Weka source-code mount. The optional Lucene image
uses JTC only for its index-building assets; the server is a LiteRegistry service.

### 3. Build the four package images

```bash
cd /weka/gfaria/literegistry/literegistry_base_deployment
export IMAGE_TAG=0.1.0
./scripts/build-images.sh "" "$IMAGE_TAG"
```

The script builds:

```text
literegistry-redis:0.1.0
literegistry-base-services:0.1.0
literegistry-base-terminal:0.1.0
literegistry-base-vllm:0.1.0
```

The services image runs gateway, Python, or web search depending on its Beaker
command. Terminal has every binary allowed by the restricted pipeline server.
Local search is not rebuilt by default: the launcher targets the Beaker image
`goncalof/jtc-local-search-lucene-bm25`, which must contain LiteRegistry 1.0.36
or newer. vLLM uses
`vllm/vllm-openai:latest` by default; pin or replace it when reproducibility
requires a specific vLLM/CUDA combination:

```bash
VLLM_BASE_IMAGE=vllm/vllm-openai:0.11.0 \
  ./scripts/build-images.sh "" "$IMAGE_TAG"
```

The canonical Dockerfile for that JTC image is included at
`docker/Dockerfile.local-search`. It copies JTC's existing `search/` index-building assets. The running
server is
`literegistry.services.bm25_server`; neither this deployment package nor the
image copies `datadev.infra.bm25_server`. To reproduce the image from a JTC checkout:

```bash
cd /weka/gfaria/literegistry/literegistry_base_deployment
BUILD_LOCAL_SEARCH=1 \
JTC_BUILD_CONTEXT=/weka/gfaria/jtc \
  ./scripts/build-images.sh "" "$IMAGE_TAG"
```

The normal build leaves `BUILD_LOCAL_SEARCH=0` and expects the updated Beaker
image to have already been uploaded.

To build and push to an ordinary Docker registry:

```bash
PUSH_IMAGES=1 \
  ./scripts/build-images.sh registry.example/team "$IMAGE_TAG"
```

### 4. Upload the images to Beaker

The launcher image flags take Beaker image names or IDs, not local Docker tags.

```bash
export WORKSPACE=ai2/oe-agents
export BEAKER_TAG="${IMAGE_TAG//./-}"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-redis:$IMAGE_TAG")" \
  --name "literegistry-redis-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-base-services:$IMAGE_TAG")" \
  --name "literegistry-base-services-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-base-terminal:$IMAGE_TAG")" \
  --name "literegistry-base-terminal-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-base-vllm:$IMAGE_TAG")" \
  --name "literegistry-base-vllm-$BEAKER_TAG" --workspace "$WORKSPACE"
```

### 5. Create API-key secrets

Web query search uses Serper and URL fetching uses Jina Reader. The launcher
accepts only Beaker secret names; it never places raw keys in the experiment
spec or command line.

```bash
beaker secret write SERPER_API_KEY
beaker secret write JINA_API_KEY
beaker secret write HF_TOKEN
```

Skip the first two when launching with `--web-search-replicas=0`. Skip the
Hugging Face secret by passing `--hf-token-secret=None` when all model artifacts
are public or already cached.

### 6. Preview a complete stack

This example runs CPU services on Jupiter and GPU model pools on a selected GPU
cluster. Replace the model names, corpus paths, and model cluster with real
values. The corpus and index are Weka paths visible inside every replica.

```bash
literegistry-base-deployment preview \
  --python-replicas=2 \
  --terminal-replicas=2 \
  --web-search-replicas=2 \
  --local-search-replicas=2 \
  --local-search-corpus-jsonl=/weka/gfaria/search/corpus.jsonl \
  --local-search-index-dir=/weka/gfaria/search/lucene-index \
  --generation-model=allenai/example-generation-model \
  --generation-replicas=2 \
  --generation-tp=1 \
  --classification-model=allenai/example-reward-model \
  --classification-replicas=1 \
  --classification-tp=1 \
  --gateway-workers=8 \
  --service-cluster=ai2/jupiter \
  --gateway-cluster=ai2/jupiter \
  --model-cluster=ai2/jupiter \
  --workspace="$WORKSPACE" \
  --budget=ai2/oe-omai \
  --redis-image="literegistry-redis-$BEAKER_TAG" \
  --services-image="literegistry-base-services-$BEAKER_TAG" \
  --terminal-image="literegistry-base-terminal-$BEAKER_TAG" \
  --local-search-image=goncalof/jtc-local-search-lucene-bm25 \
  --vllm-image="literegistry-base-vllm-$BEAKER_TAG"
```

Preview validates and prints the exact Beaker spec without creating anything.
With no `--registry`, it includes one managed Redis task. To reuse Redis, add:

```bash
literegistry-base-deployment preview \
  --registry=redis://jupiter-cs-aus-183.reviz.ai2.in:59936 \
  --python-replicas=1 \
  --terminal-replicas=1 \
  --web-search-replicas=0
```

### 7. Launch and find the gateway

Change `preview` to `launch`. This smaller CPU-only example is useful for a
first deployment test:

```bash
literegistry-base-deployment launch \
  --python-replicas=1 \
  --terminal-replicas=1 \
  --web-search-replicas=1 \
  --service-cluster=ai2/jupiter \
  --gateway-cluster=ai2/jupiter \
  --workspace="$WORKSPACE" \
  --budget=ai2/oe-omai \
  --redis-image="literegistry-redis-$BEAKER_TAG" \
  --services-image="literegistry-base-services-$BEAKER_TAG" \
  --terminal-image="literegistry-base-terminal-$BEAKER_TAG" \
  | tee /tmp/literegistry-base-launch.json

export EXPERIMENT_ID="$(jq -r '.beaker.id' /tmp/literegistry-base-launch.json)"
export EXPERIMENT_NAME="$(jq -r '.experiment_name' /tmp/literegistry-base-launch.json)"
export GATEWAY_FILE="/weka/gfaria/literegistry_base_gateway_${EXPERIMENT_NAME}.url"
until [[ -s "$GATEWAY_FILE" ]]; do sleep 2; done
export GATEWAY_URL="$(tail -n 1 "$GATEWAY_FILE")"
echo "$GATEWAY_URL"

beaker experiment get "$EXPERIMENT_ID"
curl -fsS "$GATEWAY_URL/health"
curl -fsS "$GATEWAY_URL/v1/models"
```

The gateway also prints `GATEWAY_URL=...` in its Beaker logs. The file is an
atomic convenience for other Weka-connected jobs.

### 8. Exercise every gateway route

```bash
curl -fsS -X POST "$GATEWAY_URL/python" \
  -H 'content-type: application/json' \
  -d '{"code":"print(2 + 2)","max_runtime":1}'

curl -fsS -X POST "$GATEWAY_URL/terminal" \
  -H 'content-type: application/json' \
  -d '{"contents":"INFO ok\nERROR ai2 hello\n","command":"rg ERROR","max_runtime":5}'

curl -fsS -X POST "$GATEWAY_URL/search" \
  -H 'content-type: application/json' \
  -d '{"mode":"query","query":"Allen Institute for AI","num_results":3}'

curl -fsS -X POST "$GATEWAY_URL/search" \
  -H 'content-type: application/json' \
  -d '{"mode":"url","url":"https://allenai.org/"}'
```

When local search is enabled, select its registered pool explicitly:

```bash
curl -fsS -X POST "$GATEWAY_URL/search" \
  -H 'content-type: application/json' \
  -d '{"model_path":"localsearch:corpus","mode":"query","query":"ai2 hello","num_results":3}'
```

For a generation model:

```bash
curl -fsS -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"allenai/example-generation-model","messages":[{"role":"user","content":"Say ai2 hello"}],"max_tokens":32}'
```

For a vLLM sequence classifier, the final assistant response is already part
of the conversation, so `add_generation_prompt` stays false:

```bash
curl -fsS -X POST "$GATEWAY_URL/classify" \
  -H 'content-type: application/json' \
  -d '{"model":"allenai/example-reward-model","messages":[{"role":"user","content":"Say hello"},{"role":"assistant","content":"ai2 hello"}],"add_generation_prompt":false}'
```

### 9. Stop the deployment

```bash
literegistry-base-deployment stop "$EXPERIMENT_ID"
```

## Resumption and failure policy

| Task | `context.autoResume` | `propagateFailure` | `propagatePreemption` |
|---|---:|---:|---:|
| Managed Redis | `false` | `true` | `true` |
| Gateway and every worker pool | `true` | `false` | `false` |

Every non-Redis task is resumable and isolated from experiment-wide failure.
Managed Redis is intentionally the only non-resumable task and the only task
whose failure or preemption terminates the complete experiment. With external
`--registry`, no Redis task is created, so every task in this experiment is
resumable.

## Local-search behavior

Local search is implemented by LiteRegistry's first-class BM25 service. The
deployment invokes its `/app/search/build_lucene_index.sh` when no
`segments_*` file exists,
then starts its registered application with
`literegistry.services.bm25_server:create_app`. The Dockerfile uses the JTC
checkout only for the existing Lucene index
builder; there is no Datadev server dependency.

The default service name is `localsearch:<corpus filename stem>`. Override it
with `--local-search-service-name` and pass exactly that value as
`model_path` in gateway `/search` requests.

## Horizontal CPU placement

Comma-separated `--service-cluster` values produce separate Beaker task groups,
with Python, terminal, web-search, and local-search replicas divided as evenly
as possible. This forces placement across the named clusters instead of merely
asking Beaker for many replicas in one cluster:

```bash
literegistry-base-deployment preview \
  --registry=redis://registry.example:6379 \
  --python-replicas=16 \
  --terminal-replicas=16 \
  --web-search-replicas=0 \
  --service-cluster=ai2/neptune,ai2/saturn,ai2/jupiter,ai2/ceres
```

GPU pools currently use the single explicit `--model-cluster`; tensor
parallelism maps directly to Beaker `gpuCount` for each model replica.

## Python API

```python
from literegistry_base_deployment import BaseDeploymentConfig, BaseDeploymentLauncher


config = BaseDeploymentConfig(
    registry="redis://jupiter-cs-aus-183.reviz.ai2.in:59936",
    python_replicas=4,
    terminal_replicas=4,
    web_search_replicas=2,
    service_clusters=("ai2/jupiter", "ai2/ceres"),
    gateway_cluster="ai2/jupiter",
)
launcher = BaseDeploymentLauncher(config)
print(launcher.preview())       # read-only
receipt = launcher.submit()     # creates the Beaker experiment
print(receipt)

# Later:
BaseDeploymentLauncher.stop(receipt["beaker"]["id"])
```

## Development validation

```bash
cd /weka/gfaria/literegistry/literegistry_base_deployment
python -m pip install -e '.[test,publish]'
python -m pytest
python -m build
python -m twine check dist/*
```

The tests verify service composition, shell syntax, managed Redis discovery,
secret references, multi-cluster spreading, generation/classification vLLM
arguments, JTC Lucene image routing, package contents, and Dockerfile
self-containment. Dockerfiles are structurally tested by default; actually
building the four package images requires a Docker daemon and network access.


## Experimental mirror soft affinity

The bundled gateway enables experimental mirror soft affinity by default. If this base deployment is used with external `docker-mirror` services, disable it with
`--docker-mirror-soft-affinity=False` to restore normal load balancing.
