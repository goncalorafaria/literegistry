"""Redis readiness barrier shared by standalone deployment launchers."""

from __future__ import annotations

import socket
import ssl
import time
from urllib.parse import unquote, urlparse

import fire


def redis_ping(registry: str, *, timeout: float = 2.0) -> None:
    """PING Redis using RESP directly, avoiding a redis-py dependency."""
    parsed = urlparse(registry)
    if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
        raise ValueError("registry must be a redis:// or rediss:// URL")
    connection = socket.create_connection((parsed.hostname, parsed.port or 6379), timeout)
    try:
        if parsed.scheme == "rediss":
            connection = ssl.create_default_context().wrap_socket(
                connection, server_hostname=parsed.hostname
            )
        connection.settimeout(timeout)
        stream = connection.makefile("rb")

        def command(*parts: str) -> bytes:
            encoded = [part.encode() for part in parts]
            payload = [f"*{len(encoded)}\r\n".encode()]
            for part in encoded:
                payload.extend((f"${len(part)}\r\n".encode(), part, b"\r\n"))
            connection.sendall(b"".join(payload))
            prefix = stream.read(1)
            line = stream.readline()
            if prefix == b"-":
                raise ConnectionError(line.rstrip().decode(errors="replace"))
            if prefix not in {b"+", b":"} or not line.endswith(b"\r\n"):
                raise ConnectionError("invalid Redis response")
            return line[:-2]

        if parsed.password is not None:
            if parsed.username:
                command("AUTH", unquote(parsed.username), unquote(parsed.password))
            else:
                command("AUTH", unquote(parsed.password))
        database = parsed.path.lstrip("/")
        if database:
            command("SELECT", str(int(database)))
        if command("PING").upper() != b"PONG":
            raise ConnectionError("Redis did not answer PONG")
    finally:
        connection.close()


def wait_for_redis(
    registry: str, *, timeout: float = 600.0, poll_interval: float = 2.0
) -> None:
    """Wait until Redis accepts a PING or raise at the deadline."""
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            redis_ping(registry)
            return
        except (OSError, ValueError, ConnectionError, ssl.SSLError) as error:
            last_error = error
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    raise TimeoutError(f"timed out waiting for Redis {registry}: {last_error}")


def main(argv: list[str] | None = None) -> None:
    fire.Fire({"wait": wait_for_redis}, command=argv)


if __name__ == "__main__":
    main()
