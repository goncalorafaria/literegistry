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

### Beaker resource contract

Every task in this stack is CPU-only. The launcher deliberately emits exactly
this resource and scheduling shape for Redis, gateway, mirror, and Podman
tasks:

```yaml
resources:
  gpuCount: 0
context:
  minRuntime: "0h"
```

There is intentionally **no `cpuCount` field**. On GPU-shaped Beaker clusters,
adding a CPU request can cause an otherwise CPU-only task to be assigned a GPU
worker. Keep CPU omitted, keep `gpuCount: 0`, and keep the default
`--min-runtime-hours=0`. Do not pass `--omit-resources=True` for normal Beaker
deployments because that also removes the explicit zero-GPU request. Use
`preview` before launch and verify these fields on every generated task.

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

The runtime images pin `literegistry==1.0.47`, including the gateway, Redis,
Podman, mirror, warmup, and live-fire images. Make version 1.0.47 available on
the Python index used by Docker, or expose it through a
wheelhouse URL reachable from the Docker builder:

```bash
cd /weka/gfaria/literegistry
python -m build
python -m twine check dist/*
# Publish to your configured index when appropriate:
# python -m twine upload dist/literegistry-1.0.47*
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
export IMAGE_TAG=0.2.13
./scripts/build-images.sh "" "$IMAGE_TAG"
```

This produces:

```text
literegistry-redis:0.2.13
literegistry-podman-gateway:0.2.13
literegistry-podman-server:0.2.13
literegistry-docker-mirror:0.2.13
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
Every service receives an experiment-scoped `head+file://` Weka URI and reconnects
through it if Redis moves. To reuse an existing registry, add
`--registry=redis://HOST:PORT`. To use an externally managed Redis publisher,
pass `--registry=head+file:///weka/shared/my-stack`. A SQLite or separate
Redis head can use `head+sqlite:///...` or `head+redis://...`. Either option omits the Redis
task, although the `--redis-image` value is harmless. `--head-registry` remains
a convenience spelling for the same head URI.

On `launch`, the managed per-experiment Weka directory is created before Beaker
submission with sticky shared-write permissions, allowing the non-root service
UID to publish endpoints and persist Redis data without opening the parent
directory.

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
| Managed Redis | `true` | `false` | `false` |
| Gateway | `true` | `false` | `false` |
| Docker mirrors | `true` | `false` | `false` |
| Podman replicas | `true` | `false` | `false` |

Gateway, mirror, Podman, and Redis tasks are resumable and non-propagating. If
Redis disappears, workers keep running and registry operations wait on the
shared head registry. When Redis resumes, its current URL is published there
and workers reconnect. Redis AOF state lives under that same shared directory,
so the resumed task loads the prior database. With an external `--registry`,
whether direct or `head+...`, no Redis task is created.

`autoResume` restores the service process, not an active affinity
trajectory. If a Podman replica is preempted, its in-flight containers are
intentionally cleaned up and callers must open a new handshake. A mirror rescheduled onto another
node starts with a cold local cache and warms again on demand.

### 8. Obtain the gateway URL and run a full smoke test

The gateway logs its URL and publishes a TTL-backed `gateway` endpoint through
the configured head backend. Resolve the healthy endpoint,
then test gateway health, the Docker Registry V2 route, and a stateful Podman
session:

```bash
export COOP_ROOT="/weka/gfaria/literegistry/.coop/${EXPERIMENT_NAME}"
export GATEWAY_URL="$(python -m literegistry.coop.endpoints wait \
  --root="file://$COOP_ROOT" \
  --name=gateway \
  --healthcheck=http \
  --timeout=600)"
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
replicas use the Weka directory as a stable head registry, wait for Redis on
startup, and reconnect if its URL changes. To use an existing Redis server, set
`registry="redis://jupiter-cs-aus-183.reviz.ai2.in:59936"`; no Redis task is
then created. To use an externally managed head registry, set
`registry="head+file:///weka/shared/my-stack"`.

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

The gateway prints a machine-readable endpoint line and keeps a TTL-backed
record alive in the deployment's configured endpoint registry:

```text
LITEREGISTRY_ENDPOINT_GATEWAY=http://jupiter-cs-aus-NNN.reviz.ai2.in:PORT
```

For a managed stack, Redis similarly publishes the internal `redis` endpoint.
Every consumer performs a health check before accepting either record. Podman
uses the resolved gateway endpoint as its native `docker.io` mirror.

An operator can resolve and test the gateway directly:

```bash
EXPERIMENT_NAME=YOUR_EXPERIMENT_NAME
COOP_ROOT="/weka/gfaria/literegistry/.coop/${EXPERIMENT_NAME}"
GATEWAY_URL="$(python -m literegistry.coop.endpoints wait \
  --root="file://$COOP_ROOT" --name=gateway --healthcheck=http --timeout=600)"

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

To exercise the same path as a real rollout, use the async Podman client
warmer. Each image follows `handshake -> execute("true") -> close`; successful
images are checkpointed so a resumed Beaker task skips them:

```bash
literegistry-podman-warm-podman \
  --gateway_url=http://gateway-host:port \
  --expected_podman=32 \
  --concurrency=64 \
  --checkpoint_file=/weka/path/podman-warmup-complete.txt
```

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
directory, but does not warm it automatically. Run
`literegistry-podman-warm-gateway` after the stack is ready so warmup is routed
across the mirror pool and does not compete with startup traffic.

Build all four with one command:

```bash
cd /weka/gfaria/literegistry/literegistry_podman_beaker
./scripts/build-images.sh YOUR_REGISTRY 0.2.13
```

For example, build and push `goncalof/*:0.2.13`:

```bash
PUSH_IMAGES=1 ./scripts/build-images.sh goncalof 0.2.13
```

The script prints:

```text
REDIS_IMAGE=YOUR_REGISTRY/literegistry-redis:0.2.13
GATEWAY_IMAGE=YOUR_REGISTRY/literegistry-podman-gateway:0.2.13
PODMAN_IMAGE=YOUR_REGISTRY/literegistry-podman-server:0.2.13
DOCKER_MIRROR_IMAGE=YOUR_REGISTRY/literegistry-docker-mirror:0.2.13
```

Publish `literegistry==1.0.47` to the
selected Python index before building the runtime images.

If LiteRegistry is served by an internal Python index, export
`PIP_INDEX_URL` and optionally `PIP_TRUSTED_HOST`; the build script passes
them as Docker build arguments. `PIP_FIND_LINKS` can instead point at a local
or internal wheelhouse for pre-publication builds.

Use the resulting installed-package images directly:

```python
config = PodmanStackConfig(
    redis_image="YOUR_REGISTRY/literegistry-redis:0.2.13",
    gateway_image="YOUR_REGISTRY/literegistry-podman-gateway:0.2.13",
    podman_image="YOUR_REGISTRY/literegistry-podman-server:0.2.13",
    docker_mirror_image="YOUR_REGISTRY/literegistry-docker-mirror:0.2.13",
)
```

The final Redis image runs as `USER redis`; Podman runs as `USER podman`; the
mirror runs as `USER mirror`; and the gateway runs as `USER gateway`.


## Experimental mirror soft affinity

Mirror soft affinity is experimental and enabled by default. Disable it with
`--docker-mirror-soft-affinity=False` when previewing or launching. The gateway
otherwise infers repository affinity from `/v2/...`; no mirror handshake is added.
### Podman hardening defaults

Beaker Podman replicas enable resource and cleanup controls by default:

| Fire CLI option | `PodmanStackConfig` field | Default |
|---|---|---:|
| `--podman-max-sessions` | `podman_max_sessions` | `None` |
| `--podman-session-memory` | `podman_session_memory` | `4g` |
| `--podman-session-pids-limit` | `podman_session_pids_limit` | `2048` |
| `--podman-session-idle-timeout` | `podman_session_idle_timeout` | `7200` seconds |
| `--podman-janitor-interval` | `podman_janitor_interval` | `300` seconds |
| `--podman-resource-watchdog-interval` | `podman_resource_watchdog_interval` | `5` seconds |
| `--podman-image-prune-until` | `podman_image_prune_until` | `24h` |

Override them on either `preview` or `launch`:

```bash
literegistry-podman-beaker preview \
  --podman-max-sessions=64 \
  --podman-session-memory=8g \
  --podman-session-pids-limit=4096 \
  --podman-session-idle-timeout=14400 \
  --podman-janitor-interval=600 \
  --podman-image-prune-until=48h
```

Pass `None` to disable an optional control explicitly:

```bash
literegistry-podman-beaker preview \
  --podman-session-memory=None \
  --podman-session-pids-limit=None \
  --podman-session-idle-timeout=None \
  --podman-image-prune-until=None
```

Memory and PID limits use native cgroups where the corresponding controllers
are delegated. On nested Beaker hosts, the separate 5-second userspace
watchdog groups host processes by each container's Linux PID namespace and
force-removes confirmed offenders, including reparented `podman exec`
processes. Set `--podman-resource-watchdog-interval=None` to disable that
fallback. Userspace RSS accounting is approximate and native cgroups
remain preferable when available.
Choose the idle timeout above the longest legitimate gap between commands in
one trajectory. Image pruning removes only unused images, but an aggressive
age can increase subsequent cold-pull traffic.
