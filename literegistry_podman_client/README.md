# literegistry-podman-client

`literegistry-podman-client` is a small, standalone async Python client for
running commands in rootless Podman containers through a LiteRegistry gateway.
The person using it needs only a gateway URL; they do not need Redis, Podman,
Docker, or the full `literegistry` package.

The same gateway may also expose a Docker Hub pull-through mirror. Mirror use
is configured on the Podman servers by the operator, so client code still only
passes a normal image such as `docker.io/library/ubuntu:24.04`.

## Install

From PyPI after publication:

```bash
pip install literegistry-podman-client
```

From this repository:

```bash
pip install ./literegistry_podman_client
```

The distribution name uses hyphens. Python imports use underscores:

```python
from literegistry_podman_client import PodmanClient
```

Its runtime dependencies are `aiohttp` and Python Fire.

## Give a deployment to someone

The operator gives the user one value:

```bash
export PODMAN_GATEWAY_URL=http://gateway.example:8080
```

The user can verify both gateway features:

```bash
curl -fsS "$PODMAN_GATEWAY_URL/health"
curl -fsS "$PODMAN_GATEWAY_URL/v2/"
```

`/health` checks the LiteRegistry gateway. `/v2/` checks its Docker Registry
V2 mirror route. No Redis URL or Podman replica address is exposed to users.

## Small async example

This creates one container, writes a file, reads it in a separate command, and
always deletes the container at the end:

### Without `async with`

```python
import asyncio

from literegistry_podman_client import PodmanClient


async def main() -> None:
    gateway_url = "http://gateway.example:8080"
    client = PodmanClient(gateway_url, workdir="/tmp")
    podman = None
    await client.open()
    try:
        podman = await client.handshake(
            image="docker.io/library/ubuntu:24.04",
            client_id="rollout-17",
        )
        print(podman.container_id)

        await podman.execute("echo ai2 hello > hello.txt", check=True)
        result = await podman.execute("cat hello.txt", check=True)
        print(result.stdout, end="")
    finally:
        try:
            if podman is not None:
                await podman.close()
        finally:
            await client.aclose()


asyncio.run(main())
```

### With `async with`

```python
import asyncio

from literegistry_podman_client import PodmanClient


async def main() -> None:
    async with PodmanClient(
        "http://gateway.example:8080",
        workdir="/tmp",
    ) as client:
        async with client.session(
            image="docker.io/library/ubuntu:24.04",
            client_id="rollout-17",
        ) as podman:
            print(podman.container_id)
            await podman.execute("echo ai2 hello > hello.txt", check=True)
            result = await podman.execute("cat hello.txt", check=True)
            print(result.stdout, end="")


asyncio.run(main())
```

The image pull happens on the selected Podman replica. If the operator wired
those replicas to the gateway's mirror, the pull is transparently cached; the
user does not change the image reference or client configuration.

The installed Fire CLI exposes the same smoke test:

```bash
literegistry-podman-client ai2-hello --gateway="$PODMAN_GATEWAY_URL"
```

The runnable version is [`examples/ai2_hello.py`](examples/ai2_hello.py):

```bash
python literegistry_podman_client/examples/ai2_hello.py \
  --gateway "$PODMAN_GATEWAY_URL"
```

## Explicit lifecycle

Handshake, execute, and close are all async. The affinity ID returned by the
handshake keeps every command on the replica that owns its container.

```python
client = PodmanClient(gateway_url, workdir="/home/user")
await client.open()
session = None
try:
    session = await client.handshake(image=container_image)
    first = await client.execute(
        session.affinity_id,
        "python -c 'print(6 * 7)'",
        timeout=60,
    )
    first.check_returncode()
finally:
    try:
        if session is not None:
            await session.close()
    finally:
        await client.aclose()
```

- `client.close(affinity_id)` or `session.close()` deletes the container and
  its gateway affinity binding.
- `client.aclose()` only closes the local HTTP connection pool. It cannot
  guess which concurrent sessions should be deleted.
- `check=True` or `result.check_returncode()` raises `PodmanCommandError` for a
  non-zero command exit. Without it, stdout, stderr, and the exit code remain
  available on `CommandResult`.

## Concurrent trajectories

One `PodmanClient` is intentionally shareable. It does not store a
private "current container". Each handshake returns a separate session:

```python
async def rollout(client: PodmanClient, number: int) -> str:
    podman = await client.handshake(client_id=f"rollout-{number}")
    try:
        result = await podman.execute("echo $((20 + 22))", check=True)
        return result.stdout.strip()
    finally:
        await podman.close()


client = PodmanClient(gateway_url)
await client.open()
try:
    outputs = await asyncio.gather(
        *(rollout(client, i) for i in range(128))
    )
finally:
    await client.aclose()
```

The HTTP pool is shared, while every request carries its explicit affinity ID.
As soon as one trajectory completes, application code may start another.

## Mirror behavior

There are two distinct flows behind the gateway URL:

```text
Python client -> /affinity/* -> Podman replica -> rootless container
Podman pull   -> /v2/*       -> Docker mirror -> Docker Hub
```

The client exposes `await client.mirror_health()` as a convenience probe, but
it does not configure the mirror. The deployment operator configures each
Podman replica's `containers-registries.conf` to use the gateway URL. This is
what ensures all users of the gateway benefit from the cache.

## Build and publish

```bash
cd literegistry_podman_client
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Publishing requires a PyPI account and token. Verify that the distribution
name `literegistry-podman-client` is available before the first upload.
