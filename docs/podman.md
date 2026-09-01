# Podman affinity containers

LiteRegistry includes a first-class rootless Podman server and the matching
strict-affinity routes in the LiteRegistry gateway. Datadev is not required.

```text
client → Gateway → selected Podman replica → rootless container
                 └── affinity_id → replica URI (Redis, 15-minute TTL)
```

## Start the gateway and server

```bash
literegistry gateway \
  --registry redis://registry:6379 \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 8

literegistry podman \
  --registry redis://registry:6379 \
  --host 0.0.0.0 \
  --port 8091 \
  --allow-non-loopback=True \
  --advertise-host podman-node \
  --advertise-port 8091 \
  --instance-id podman-0 \
  --registry-mirror http://gateway.example:8080 \
  --image python:3.12-slim \
  --network none
```

`--registry-mirror` makes Podman use the LiteRegistry gateway as its native
`docker.io` mirror. The server writes a private rootless `registries.conf` and
sets `CONTAINERS_REGISTRIES_CONF` only for its Podman subprocesses. HTTP is
marked insecure while HTTPS remains verified. Podman tries the gateway first
and retains its normal Docker Hub fallback.

The server binds to loopback by default because it has no application token.
For a managed private cluster using host networking, explicitly pass
`--host=0.0.0.0 --allow-non-loopback=True` and enforce access at the network
boundary. Run the outer service as a non-root user; containers are rootless.

The standalone LiteRegistry Podman Beaker stack can launch `docker_mirror_replicas` and one
gateway itself. It publishes the gateway's actual host and dynamically
allocated port, then supplies that URL to every Podman replica automatically.

> **Optional companions:** Applications can install the async gateway client
> with `pip install "literegistry[podman_client]"`. Operators who want the
> complete rootless Podman, Docker mirror, gateway, and Redis Beaker stack can
> use `pip install "literegistry[podman_beaker]"`. These packages are
> convenience extensions; neither is required to run the core server and HTTP
> API documented below.

## Session lifecycle

Create a container and affinity binding:

```bash
HANDSHAKE=$(curl -fsS http://gateway:8080/affinity/handshake \
  -H 'content-type: application/json' \
  -d '{"service":"podman","image":"python:3.12-slim","client_id":"demo"}')
CONTAINER_ID=$(printf '%s' "$HANDSHAKE" | jq -r .container_id)
```

Run commands. The binding sends every command to the replica that owns the
container:

```bash
curl -fsS http://gateway:8080/affinity/podman \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\",\"command\":\"echo ai2-hello > /workspace/hello.txt\"}"

curl -fsS http://gateway:8080/affinity/podman \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\",\"command\":\"cat /workspace/hello.txt\"}"
```

Delete the container and binding:

```bash
curl -fsS http://gateway:8080/affinity/close \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\"}"
```

The affinity ID is the container ID and is a routing handle, not an
authentication secret. Successful close removes the Redis binding immediately.

## Benchmark Podman throughput

Install the standalone client extra, then run the benchmark tools from the
LiteRegistry checkout:

```bash
python -m pip install "literegistry[podman_client]"

python tools/benchmark_podman_execution_client.py  --gateway-url http://gateway:8080  --replicas 8  --concurrency 16,32,64  --total-sessions 8192  --commands-per-session 4  --output podman-throughput-r8.json
```

`--total-sessions` keeps the workload fixed across concurrency levels. Each
worker completes create, command, and confirmed-delete before taking another
trajectory. Use `--sessions-per-worker` and `--waves` instead for sequential,
phase-isolated waves.

Measure handshake-to-container-ready throughput separately:

```bash
python tools/benchmark_podman_container_creation.py  --gateway-url http://gateway:8080  --replicas 8  --concurrency 1,2,4,8,16,32,64  --output podman-creation-r8.json
```

Compare equivalent throughput results from different replica counts:

```bash
python tools/compare_podman_execution_benchmarks.py  podman-throughput-r1.json podman-throughput-r8.json
```

The comparator accepts both current `literegistry-podman-*` results and legacy
`jtc-podman-*` benchmark JSON.
