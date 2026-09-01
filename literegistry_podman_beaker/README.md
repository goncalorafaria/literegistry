# LiteRegistry Podman Beaker

literegistry-podman-beaker is a standalone Beaker deployment package for this
stack:

```text
Podman client -> gateway /affinity/* -> rootless Podman replicas
Podman pull   -> gateway /v2/*       -> Docker Hub mirror replicas
                         |
                         +-----------> LiteRegistry -> managed or external Redis
```

It has a strong dependency on literegistry and no dependency on Datadev. The
base LiteRegistry package owns the gateway, including both the Podman affinity
and Docker Registry V2 proxy routes. This package calls that gateway directly
and adds only the Beaker deployment/runtime configuration.

## End-to-end setup

These commands assume Bash, a checkout of this repository, and permission to
create images and experiments in the target Beaker workspace.

### 1. Check the prerequisites

You need Python 3.10 or newer, a running Docker daemon, the Beaker CLI, and
`jq` for the copy-paste shell workflow below.

```bash
python --version
docker version
beaker config test
beaker account whoami
beaker workspace get ai2/oe-agents
```

The Beaker workspace, budget, clusters, and Weka source used below must be
accessible to the active Beaker account. Replace the `ai2/*` values throughout
if you are deploying elsewhere.

### 2. Install the launcher

For development from this checkout:

```bash
cd /weka/gfaria/literegistry/literegistry_podman_beaker
python -m pip install -e '.[test,publish]'
literegistry-podman-beaker --help
```

For a published release, install through the LiteRegistry extra:

```bash
pip install "literegistry[podman_beaker]"
```

The standalone distribution can also be installed directly:

```bash
pip install literegistry-podman-beaker
```

Both forms install LiteRegistry as a required dependency and do not depend on
Datadev.

### 3. Make LiteRegistry available to Docker builds

The gateway image pins `literegistry==1.0.40`; the other images install the
Beaker package, whose dependency requires `literegistry>=1.0.35`. Make version
1.0.40 available on the Python index used by Docker, or expose it through a
wheelhouse URL reachable from the Docker builder:

```bash
cd /weka/gfaria/literegistry
python -m build
python -m twine check dist/*
# Publish to your configured index when appropriate:
# python -m twine upload dist/literegistry-1.0.40*
```

For a private index or HTTP wheelhouse:

```bash
export PIP_INDEX_URL=https://python.example/simple
# Or: export PIP_FIND_LINKS=https://python.example/wheels
# Optional for an internal HTTP host:
# export PIP_TRUSTED_HOST=python.example
```

`PIP_FIND_LINKS` must be reachable from inside the Docker build; a host-only
filesystem path is not sufficient unless it is explicitly exposed to Docker.

### 4. Build the four runtime images

The build script creates Redis infrastructure, gateway, rootless Podman, and
Docker mirror images from the Dockerfiles bundled with this package:

```bash
cd /weka/gfaria/literegistry/literegistry_podman_beaker
export IMAGE_TAG=0.2.11
./scripts/build-images.sh "" "$IMAGE_TAG"
```

This produces:

```text
literegistry-redis:0.2.11
literegistry-podman-gateway:0.2.11
literegistry-podman-server:0.2.11
literegistry-docker-mirror:0.2.11
```

For an ordinary Docker registry, pass its repository prefix and set
`PUSH_IMAGES=1`. Pushing is not required when the images are uploaded directly
from the local Docker daemon to Beaker in the next step.

```bash
PUSH_IMAGES=1 ./scripts/build-images.sh registry.example/team "$IMAGE_TAG"
```

### 5. Upload the images to Beaker

`PodmanStackConfig` expects Beaker image names or IDs, not merely local Docker
tags. Import each local image into the same workspace as the experiment:

```bash
export WORKSPACE=ai2/oe-agents
export BEAKER_TAG="${IMAGE_TAG//./-}"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-redis:$IMAGE_TAG")" \
  --name "literegistry-redis-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-podman-gateway:$IMAGE_TAG")" \
  --name "literegistry-podman-gateway-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-podman-server:$IMAGE_TAG")" \
  --name "literegistry-podman-server-$BEAKER_TAG" --workspace "$WORKSPACE"

beaker image create \
  "$(docker image inspect --format '{{.Id}}' "literegistry-docker-mirror:$IMAGE_TAG")" \
  --name "literegistry-docker-mirror-$BEAKER_TAG" --workspace "$WORKSPACE"
```

Beaker prints each image ID. An ID can be used instead of its name in every
launcher flag below, which avoids ambiguity if a workspace already contains an
image with the same name.

### 6. Preview the generated experiment

Preview validates the configuration and prints the complete Beaker spec without
creating anything:

```bash
literegistry-podman-beaker preview \
  --podman-replicas=4 \
  --docker-mirror-replicas=2 \
  --gateway-workers=8 \
  --service-cluster=ai2/jupiter \
  --gateway-cluster=ai2/jupiter \
  --workspace="$WORKSPACE" \
  --budget=ai2/oe-omai \
  --weka-source=oe-adapt-default \
  --redis-image="literegistry-redis-$BEAKER_TAG" \
  --gateway-image="literegistry-podman-gateway-$BEAKER_TAG" \
  --podman-image="literegistry-podman-server-$BEAKER_TAG" \
  --docker-mirror-image="literegistry-docker-mirror-$BEAKER_TAG"
```

With no `--registry`, the experiment contains exactly one managed Redis task.
To reuse an existing registry, add `--registry=redis://HOST:PORT`; the launcher
then omits the Redis task, although the `--redis-image` value is harmless.

### 7. Launch and monitor the stack

Change `preview` to `launch` and save the JSON launch receipt:

```bash
literegistry-podman-beaker launch \
  --podman-replicas=4 \
  --docker-mirror-replicas=2 \
  --gateway-workers=8 \
  --service-cluster=ai2/jupiter \
  --gateway-cluster=ai2/jupiter \
  --workspace="$WORKSPACE" \
  --budget=ai2/oe-omai \
  --weka-source=oe-adapt-default \
  --redis-image="literegistry-redis-$BEAKER_TAG" \
  --gateway-image="literegistry-podman-gateway-$BEAKER_TAG" \
  --podman-image="literegistry-podman-server-$BEAKER_TAG" \
  --docker-mirror-image="literegistry-docker-mirror-$BEAKER_TAG" \
  | tee /tmp/literegistry-podman-launch.json

export EXPERIMENT_ID="$(jq -r '.beaker.id' /tmp/literegistry-podman-launch.json)"
export EXPERIMENT_NAME="$(jq -r '.experiment_name' /tmp/literegistry-podman-launch.json)"
beaker experiment get "$EXPERIMENT_ID"
beaker experiment logs "$EXPERIMENT_ID" --tail 100
```

### Resumption and failure policy

| Task | `context.autoResume` | `propagateFailure` | `propagatePreemption` |
|---|---:|---:|---:|
| Managed Redis | `false` | `true` | `true` |
| Gateway | `true` | `false` | `false` |
| Docker mirrors | `true` | `false` | `false` |
| Podman replicas | `true` | `false` | `false` |

Gateway, mirror, and Podman tasks are resumable: a task preemption does not
propagate to the complete experiment, and an individual task failure does not
tear down the other services. Managed Redis is intentionally the sole critical
root. Redis is not resumable; if it fails or is preempted, Beaker fails the
complete experiment rather than leaving workers attached to lost registry
state. When `--registry` points to external Redis, no Redis task is created and
all tasks in this experiment remain resumable.

`autoResume` restores the service process, not an active affinity
trajectory. If a Podman replica is preempted, its in-flight containers are
intentionally cleaned up and callers must open a new handshake. A mirror rescheduled onto another
node starts with a cold local cache and warms again on demand.

### 8. Obtain the gateway URL and run a full smoke test

The gateway logs `GATEWAY_URL=...` and atomically writes the same URL to Weka.
Wait for that file, then test gateway health, the Docker Registry V2 route, and
a stateful Podman session:

```bash
export GATEWAY_FILE="/weka/gfaria/literegistry_podman_gateway_${EXPERIMENT_NAME}.url"
until [[ -s "$GATEWAY_FILE" ]]; do sleep 2; done
export GATEWAY_URL="$(tail -n 1 "$GATEWAY_FILE")"
echo "$GATEWAY_URL"

curl -fsS "$GATEWAY_URL/health"
curl -fsS "$GATEWAY_URL/v2/"

HANDSHAKE="$(curl -fsS \
  -H 'content-type: application/json' \
  -d '{"service":"podman","image":"docker.io/library/ubuntu:24.04"}' \
  "$GATEWAY_URL/affinity/handshake")"
AFFINITY_ID="$(jq -r '.affinity_id' <<<"$HANDSHAKE")"
echo "$HANDSHAKE"

curl -fsS \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg id "$AFFINITY_ID" \
    --arg cmd "printf 'ai2 hello\\n' > hello.txt" \
    '{service:"podman",affinity_id:$id,command:$cmd,workdir:"/workspace",timeout:60}')" \
  "$GATEWAY_URL/affinity/podman"

curl -fsS \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg id "$AFFINITY_ID" \
    '{service:"podman",affinity_id:$id,command:"cat hello.txt",workdir:"/workspace",timeout:60}')" \
  "$GATEWAY_URL/affinity/podman"

curl -fsS \
  -H 'content-type: application/json' \
  -d "$(jq -nc --arg id "$AFFINITY_ID" \
    '{service:"podman",affinity_id:$id}')" \
  "$GATEWAY_URL/affinity/close"
```

The second command must return `ai2 hello`. Closing deletes the inner container
and releases its gateway affinity binding.

### 9. Stop the deployment

```bash
literegistry-podman-beaker stop "$EXPERIMENT_ID"
```

Stopping the Beaker experiment removes the running tasks. Mirror cache data is
also lost unless persistent storage was mounted explicitly.

## Development and package validation

```bash
cd /weka/gfaria/literegistry/literegistry_podman_beaker
python -m pip install -e '.[test,publish]'
python -m pytest
python -m build
python -m twine check dist/*
```

All four service images contain the small `literegistry-podman-beaker` runtime
helper used for collision-safe host-port selection. The gateway application and
proxy routes still come directly from `literegistry`. No image imports code from
a Weka checkout or injects `PYTHONPATH`.

## Launch from Python

```python
from literegistry_podman_beaker import PodmanStackConfig, PodmanStackLauncher


config = PodmanStackConfig(
    podman_replicas=4,
    docker_mirror_replicas=2,
    gateway_workers=8,
    service_clusters=("ai2/jupiter",),
    gateway_cluster="ai2/jupiter",
    redis_cluster="ai2/jupiter",
    workspace="ai2/oe-agents",
)

launcher = PodmanStackLauncher(config)
print(launcher.preview())       # no Beaker mutation
receipt = launcher.submit()     # creates the experiment
print(receipt)

# Later:
PodmanStackLauncher.stop(receipt["beaker"]["id"])
```

With `registry` omitted, the experiment launches one Redis task. It publishes
its dynamic host-network URL through Weka; the gateway, mirrors, and Podman
replicas wait for that URL and a successful Redis PING before starting. To use
an existing Redis server instead, set
`registry="redis://jupiter-cs-aus-183.reviz.ai2.in:59936"`; no Redis task is
then created.

To force real placement across several CPU clusters, repeat clusters in the
configuration. The package creates one constrained task group per cluster and
divides each replica count as evenly as possible:

```python
config = PodmanStackConfig(
    podman_replicas=16,
    docker_mirror_replicas=4,
    service_clusters=(
        "ai2/neptune",
        "ai2/saturn",
        "ai2/jupiter",
        "ai2/ceres",
    ),
    gateway_cluster="ai2/phobos",
)
```

## Launch from the CLI

Preview without changing Beaker:

```bash
literegistry-podman-beaker preview \
  --podman-replicas 4 \
  --docker-mirror-replicas 2 \
  --gateway-workers 8 \
  --service-cluster ai2/jupiter \
  --gateway-cluster ai2/jupiter
```

Replace preview with launch to submit. Stop an experiment with:

```bash
literegistry-podman-beaker stop EXPERIMENT_ID
```

Pass `--registry=redis://HOST:PORT` to reuse an external Redis server instead
of launching the managed Redis task.

For multi-cluster placement, pass a comma-separated `--service-cluster`, for example
`--service-cluster=ai2/neptune,ai2/saturn,ai2/jupiter,ai2/ceres`.

## Obtain and test the gateway URL

The gateway prints one machine-readable line at startup:

```text
GATEWAY_URL=http://jupiter-cs-aus-NNN.reviz.ai2.in:PORT
```

The generated stack also atomically writes the same address to:

```text
/weka/gfaria/literegistry_podman_gateway_EXPERIMENT_NAME.url
```

For a managed Redis task, it also writes the internal service address to
`/weka/gfaria/literegistry_podman_registry_EXPERIMENT_NAME.url`. That file is
consumed by the other Beaker tasks; clients should continue to use only the
gateway URL.

Podman replicas read the gateway URL file and configure their native docker.io mirror.
An operator can test both flows:

```bash
GATEWAY_URL=$(cat /weka/gfaria/literegistry_podman_gateway_EXPERIMENT_NAME.url)

curl -fsS "$GATEWAY_URL/health"
curl -fsS "$GATEWAY_URL/v2/"
```

Users only need GATEWAY_URL. They do not need Redis or individual replica URLs.

## Docker Hub credentials and mirror warmup

Warm the cluster through the gateway so every Registry V2 object creates a
Redis soft-affinity binding to the mirror that cached it. The command waits
until all expected mirrors are registered and displays a live `tqdm` progress
bar over the bundled 14,490 unique Open Instruct/Tmax images:

```bash
literegistry-podman-warm-gateway \
  --gateway_url=http://gateway-host:port \
  --expected_mirrors=16 \
  --concurrency=32
```

Use `--images_file=/path/to/images.txt` to replace the bundled list. The
optional `--limit=N` is intended only for smoke tests; without it the command
warms the complete list through the gateway.

For authenticated Docker Hub limits, put an organization access token or
personal access token in a Beaker secret and pass only the secret name:

```python
config = PodmanStackConfig(
    docker_hub_username="docker-hub-user-or-org",
    docker_hub_token_secret="DOCKER_HUB_OAT",
)
```

The Beaker spec references the secret as DOCKER_HUB_TOKEN. This package has no
raw token field, and it never writes the token to Redis or the experiment JSON.

## Execute commands through the deployed stack

Use the separate async client package:

```python
import asyncio

from literegistry_podman_client import PodmanClient


async def main() -> None:
    client = PodmanClient("http://gateway-host:port", workdir="/home/user")
    await client.open()
    session = None
    try:
        session = await client.handshake(
            image="docker.io/library/ubuntu:24.04",
        )
        await session.execute("echo ai2 hello > hello.txt", check=True)
        result = await session.execute("cat hello.txt", check=True)
        print(result.stdout, end="")
    finally:
        if session is not None:
            await session.close()
        await client.aclose()


asyncio.run(main())
```

The handshake returns the affinity/container identity. Subsequent execute and
close requests carry that ID, and a successful close removes both the
container and its affinity binding.

## Operational behavior

- Podman runs rootless in the Podman service image.
- Every task uses Beaker host networking and collision-checked dynamic ports.
- The stack always creates exactly one gateway; it has eight workers by default.
- Omitting `registry` creates exactly one ephemeral Redis task. Supplying a
  Redis URL reuses that server instead.
- Podman requests use the affinity API. Mirror requests are stateless Registry
  V2 GET/HEAD requests under /v2.
- Cache data lives under /var/lib/registry/mirror-N in each mirror task. Without
  an attached persistent volume, it disappears with the task.
- Redis, Podman, and mirror listeners are unauthenticated. Use this only on a trusted
  cluster network, or put an authenticated TLS proxy in front.

## Build self-contained runtime images

The four Dockerfiles build directly from official upstream bases:

- `redis:7-bookworm` for Redis
- `python:3.12-slim-bookworm` for the gateway
- `quay.io/podman/stable` for the rootless Podman server
- `registry:3` plus `python:3.12-slim-bookworm` for the mirror

They do not reference `goncalof/*`, `~/basic_images`, Weka paths, or any
other local image. Every image installs LiteRegistry. Its canonical
`literegistry.coop` helpers provide collision-safe dynamic ports; the gateway itself invokes base LiteRegistry
directly. The mirror also copies the unique 14,490-image warm list from this
directory.

Build all four with one command:

```bash
cd /weka/gfaria/literegistry/literegistry_podman_beaker
./scripts/build-images.sh YOUR_REGISTRY 0.2.11
```

For example, build and push `goncalof/*:0.2.11`:

```bash
PUSH_IMAGES=1 ./scripts/build-images.sh goncalof 0.2.11
```

The script prints:

```text
REDIS_IMAGE=YOUR_REGISTRY/literegistry-redis:0.2.11
GATEWAY_IMAGE=YOUR_REGISTRY/literegistry-podman-gateway:0.2.11
PODMAN_IMAGE=YOUR_REGISTRY/literegistry-podman-server:0.2.11
DOCKER_MIRROR_IMAGE=YOUR_REGISTRY/literegistry-docker-mirror:0.2.11
```

Publish `literegistry==1.0.40` to the
selected Python index before building the runtime images.

If LiteRegistry is served by an internal Python index, export
`PIP_INDEX_URL` and optionally `PIP_TRUSTED_HOST`; the build script passes
them as Docker build arguments. `PIP_FIND_LINKS` can instead point at a local
or internal wheelhouse for pre-publication builds.

Use the resulting installed-package images directly:

```python
config = PodmanStackConfig(
    redis_image="YOUR_REGISTRY/literegistry-redis:0.2.11",
    gateway_image="YOUR_REGISTRY/literegistry-podman-gateway:0.2.11",
    podman_image="YOUR_REGISTRY/literegistry-podman-server:0.2.11",
    docker_mirror_image="YOUR_REGISTRY/literegistry-docker-mirror:0.2.11",
)
```

The final Redis image runs as `USER redis`; Podman runs as `USER podman`; the
mirror runs as `USER mirror`; and the gateway runs as `USER gateway`.


## Experimental mirror soft affinity

Mirror soft affinity is experimental and enabled by default. Disable it with
`--docker-mirror-soft-affinity=False` when previewing or launching. The gateway
otherwise infers repository affinity from `/v2/...`; no mirror handshake is added.
