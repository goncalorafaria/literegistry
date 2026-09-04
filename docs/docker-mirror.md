# Docker Hub mirror

This is an ordinary Docker image cache: the official `registry:3` Docker
Distribution image runs in pull-through mode and caches Docker Hub manifests
and layers on disk under `/var/lib/registry`. LiteRegistry does not implement
the cache and Redis never stores image data. The small LiteRegistry wrapper
only advertises each healthy mirror's host and port and optionally warms it.

The LiteRegistry gateway provides one stable address in front of multiple
`docker-mirror` replicas. There is no handshake, container ID, or
client-supplied affinity key. The gateway proxies registry control requests,
selects a live mirror for each exact registry object, and returns a temporary
`307` redirect for blob `GET` and `HEAD` requests.

```text
manifests/control: Docker / Podman -> gateway -> selected mirror
blob selection:    Docker / Podman -> gateway --307 Location--> client
blob bytes:        Docker / Podman -------------------------> selected mirror
                                                          -> Docker Hub on miss
Redis <- mirror health + URI heartbeats
```

The redirect is deliberately temporary and carries `Cache-Control: no-store`.
Podman returns to the gateway for every new registry request; it follows the
selected mirror only for that individual blob transfer. The gateway therefore
remains the affinity control plane while image bytes do not traverse the
single gateway node.

The cache data itself is persistent when `/var/lib/registry` is mounted.
Manifests, tags, referrers, and root `/v2/` requests remain proxied. Repeated
requests for an exact manifest or blob prefer the same live mirror. When soft
affinity is enabled, the gateway verifies that the binding's server ID and URI
remain in the cached healthy roster. If the owner is absent, another live
mirror is selected and the binding is handed off. The roster cache is five
seconds by default; bindings use a sliding seven-day TTL by default.

## Build

From the repository root:

```bash
docker build \
  -f literegistry_podman_beaker/docker/Dockerfile.mirror \
  -t literegistry-docker-mirror:latest \
  literegistry_podman_beaker
```

The image contains the Distribution registry process, LiteRegistry supervisor,
and cache warmer. It runs as UID 10001, not root.

## Run one mirror

The advertised host must be reachable by both the LiteRegistry gateway and its
Docker/Podman clients because blob downloads are redirected to it. Mount the
cache volume so a restart does not discard already-pulled layers.

```bash
docker volume create docker-mirror-cache

docker run --rm --name docker-mirror \
  -p 5000:5000 \
  -v docker-mirror-cache:/var/lib/registry \
  -e LITEREGISTRY_URL='redis://redis.example:6379' \
  -e DOCKER_MIRROR_ADVERTISE_HOST='mirror.example' \
  -e DOCKER_MIRROR_ADVERTISE_PORT='5000' \
  literegistry-docker-mirror:latest
```

The supervisor starts the registry, requests a real manifest through the
cache, and only then registers `http://mirror.example:5000`. It deregisters
after repeated health failures and on clean shutdown. The Redis metadata never
contains a Docker Hub token.

For authenticated Docker Hub pulls, pass both variables through a secret
manager:

```bash
-e DOCKER_HUB_USERNAME='my-user' \
-e DOCKER_HUB_TOKEN='...'
```

The mirror endpoint is intentionally unauthenticated for trusted cluster
networks. Publishing it on an untrusted network requires an authenticated TLS
proxy or firewall policy.

## Warm the cache

Warm explicit images in the background while the mirror continues serving:

```bash
-e DOCKER_MIRROR_WARM_IMAGES='alpine:3.20,ubuntu:24.04' \
-e DOCKER_MIRROR_WARM_WORKERS='8'
```

The Docker image defaults to the Open Instruct setup:

```text
Image list: allenai/tmax-15k-open-instruct, pinned revision 7b090eca...
Workers:    8
Platform:   linux/amd64
```

The current public dataset's `task-data.tar.gz` contains no `image.txt` files;
its image references are in `env_config.image`. The repository therefore
includes a generated, pinned list of its 14,490 unique images and copies that
small asset into the container. Runtime startup reads only this list: it does
not download or scan the dataset.

A full warm can consume about 4 TB of registry cache, so this default is
intended for the Open Instruct/Tmax mirror. Set
`DOCKER_MIRROR_WARM_IMAGES_FILE` to an empty value to disable the bundled
list, or point it at another newline-delimited image list.
`DOCKER_MIRROR_WARM_IMAGES` may add explicit images. A compatible dataset can
still be selected with `DOCKER_MIRROR_WARM_DATASET` when needed.

## Use the LiteRegistry gateway

Run the LiteRegistry gateway against the same Redis registry. It discovers the
`docker-mirror` roster and exposes the raw Registry V2 API at its own address:

```bash
python -m literegistry.gateway \
  --registry 'redis://redis.example:6379' \
  --port 8080 \
  --workers 8
```

Tentative soft affinity is enabled by default. To disable it:

```bash
python -m literegistry.gateway \
  --registry "redis://redis.example:6379" \
  --port 8080 \
  --workers 8 \
  --docker_mirror_soft_affinity=False
```

With it disabled, the gateway uses the normal mirror sampler and does not read
or write image-affinity keys.

Check the route without a handshake:

```bash
curl -i http://gateway.example:8080/v2/
curl -fsS -H 'Accept: application/vnd.oci.image.manifest.v1+json' http://gateway.example:8080/v2/library/alpine/manifests/3.20

# Blob requests return 307; --location follows the direct mirror transfer.
curl --location --output /dev/null 'http://gateway.example:8080/v2/library/alpine/blobs/sha256:DIGEST'
```

Configure Docker with the gateway as its Docker Hub mirror:

```json
{
  "registry-mirrors": ["http://gateway.example:8080"]
}
```

Because this example uses HTTP, Docker must also trust the endpoint as an
insecure registry, or the gateway should be placed behind TLS. Podman can point
its `docker.io` mirror entry at the same gateway URL with `insecure = true`.
The LiteRegistry Podman server does this automatically when started with
`--registry-mirror http://gateway.example:8080` (or
`PODMAN_REGISTRY_MIRROR`). Datadev's infrastructure launcher supplies the
runtime gateway URL automatically when Podman, mirror, and one gateway are in
the same stack.

## Main settings

| Environment variable | Default | Meaning |
|---|---:|---|
| `LITEREGISTRY_URL` | `redis://127.0.0.1:6379` | Shared LiteRegistry Redis URL |
| `DOCKER_MIRROR_ADVERTISE_HOST` | host FQDN | Host saved in the registry roster |
| `DOCKER_MIRROR_ADVERTISE_PORT` | `5000` | Port saved in the registry roster |
| `DOCKER_MIRROR_HEALTH_IMAGE` | `alpine:3.20` | Real manifest used for health checks |
| `DOCKER_MIRROR_WARM_IMAGES` | empty | Comma-separated explicit warmup images |
| `DOCKER_MIRROR_WARM_IMAGES_FILE` | bundled Tmax list | Newline-delimited default image list |
| `DOCKER_MIRROR_WARM_DATASET` | empty | Optional compatible dataset containing task-data `image.txt` files |
| `DOCKER_MIRROR_WARM_WORKERS` | `8` | Concurrent image warmups |
| `DOCKER_HUB_USERNAME` / `DOCKER_HUB_TOKEN` | empty | Optional upstream credentials |

The equivalent Fire command is `literegistry docker-mirror --help`; the Docker
image runs that server entry point directly.
