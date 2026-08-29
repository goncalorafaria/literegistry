"""Python Fire CLI for exercising a LiteRegistry Podman gateway."""

from __future__ import annotations

import asyncio
import fire
import os

from .client import PodmanClient


async def _run(
    gateway: str,
    image: str,
    workdir: str,
    client_id: str,
) -> None:
    async with PodmanClient(gateway, workdir=workdir) as client:
        async with client.session(image=image, client_id=client_id) as session:
            print(f"container_id={session.container_id}")
            print(f"podman_replica={session.instance_id}")
            await session.execute("echo ai2 hello > hello.txt", check=True)
            result = await session.execute("cat hello.txt", check=True)
            print(result.stdout, end="")
    print("closed=true")


def ai2_hello(
    gateway: str | None = None,
    image: str = "docker.io/library/ubuntu:24.04",
    workdir: str = "/tmp",
    client_id: str = "ai2-hello",
) -> None:
    """Handshake, write, read, and close one Podman session."""
    gateway = gateway or os.getenv("PODMAN_GATEWAY_URL")
    if not gateway:
        raise ValueError("--gateway or PODMAN_GATEWAY_URL is required")
    asyncio.run(_run(gateway, image, workdir, client_id))


def main(argv: list[str] | None = None) -> None:
    fire.Fire({"ai2-hello": ai2_hello}, command=argv)


if __name__ == "__main__":
    main()
