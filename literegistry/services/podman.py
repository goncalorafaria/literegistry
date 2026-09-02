"""HTTP affinity-session primitives backed by Podman containers.

One handshake creates one long-lived inner container. Follow-up requests carry
the returned container ID, so files and processes remain scoped to that same
container for the lifetime of the session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import re
import secrets
import signal
import tempfile
import time
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
_OWNER_LABEL = "io.literegistry.podman-affinity"
_INSTANCE_LABEL = "io.literegistry.podman-affinity.instance"
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")

def build_podman_registry_mirror_config(mirror_url: str) -> str:
    """Build a Podman registries.conf that mirrors docker.io via one gateway."""

    parsed = urlsplit(mirror_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("registry_mirror must be an http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError("registry_mirror must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("registry_mirror must point to the gateway root")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("registry_mirror contains an invalid port") from exc

    location = parsed.netloc
    insecure = "true" if parsed.scheme == "http" else "false"
    return (
        'unqualified-search-registries = ["docker.io"]\n\n'
        '[[registry]]\n'
        'prefix = "docker.io"\n'
        'location = "docker.io"\n\n'
        '[[registry.mirror]]\n'
        f'location = "{location}"\n'
        f'insecure = {insecure}\n'
        'pull-from-mirror = "all"\n'
    )


class PodmanBackendError(RuntimeError):
    pass


class SessionNotFound(KeyError):
    pass


class OutputLimitExceeded(RuntimeError):
    pass


class HandshakeRequest(BaseModel):
    client_id: Optional[str] = Field(default=None, max_length=256)
    image: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=512,
        pattern=r"^[^\s\x00-\x1f\x7f]+$",
        description="OCI image for this session; omitted means the server default",
    )


class SessionRequest(BaseModel):
    container_id: Optional[str] = Field(default=None, min_length=12, max_length=64)
    affinity_id: Optional[str] = Field(default=None, min_length=12, max_length=64)

    def selected_container_id(self) -> str:
        if self.container_id and self.affinity_id and self.container_id != self.affinity_id:
            raise ValueError("container_id and affinity_id must match when both are supplied")
        container_id = self.container_id or self.affinity_id
        if not container_id:
            raise ValueError("container_id or affinity_id is required")
        if not _CONTAINER_ID_RE.fullmatch(container_id):
            raise ValueError("container ID must be 12 to 64 lowercase hexadecimal characters")
        return container_id


class PodmanRequest(SessionRequest):
    command: str = Field(..., min_length=1, max_length=16 * 1024)
    stdin: str = Field(default="", max_length=1024 * 1024)
    timeout: float = Field(default=10.0, ge=0.1, le=60.0)
    workdir: str = Field(default="/workspace", min_length=1, max_length=4096)


class CloseRequest(SessionRequest):
    pass


class HandshakeResponse(BaseModel):
    container_id: str
    affinity_id: str
    instance_id: str
    client_id: Optional[str] = None
    image: str


class PodmanResponse(BaseModel):
    container_id: str
    affinity_id: str
    stdout: str
    stderr: str
    success: bool
    exit_code: int
    execution_time: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True)
class PodmanAffinityConfig:
    podman_binary: str = "podman"
    storage_driver: str = "vfs"
    session_image: str = "docker.io/library/ubuntu:24.04"
    session_network: str = "none"
    session_memory: Optional[str] = None
    session_pids_limit: Optional[int] = None
    session_idle_timeout: Optional[float] = None
    janitor_interval: float = 300.0
    image_prune_until: Optional[str] = None
    instance_id: str = "podman-affinity-1"
    max_stdout_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 256 * 1024
    operation_timeout: float = 300.0
    registry_mirror: Optional[str] = None


@dataclass(frozen=True)
class CompletedPodmanCommand:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class PodmanSessionBackend:
    """Owns Podman containers labelled for exactly one affinity instance."""

    def __init__(self, config: Optional[PodmanAffinityConfig] = None) -> None:
        self.config = config or PodmanAffinityConfig()
        self._locks: dict[str, asyncio.Lock] = {}
        self._owned_container_ids: set[str] = set()
        self._session_last_used: dict[str, float] = {}
        self._locks_guard = asyncio.Lock()
        self._registry_config_file: Optional[Any] = None
        self._podman_env: Optional[dict[str, str]] = None
        if self.config.registry_mirror:
            runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or None
            self._registry_config_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="literegistry-podman-registries-",
                suffix=".conf",
                dir=runtime_dir,
            )
            self._registry_config_file.write(
                build_podman_registry_mirror_config(self.config.registry_mirror)
            )
            self._registry_config_file.flush()
            self._podman_env = dict(os.environ)
            self._podman_env["CONTAINERS_REGISTRIES_CONF"] = (
                self._registry_config_file.name
            )
            logger.info("Configured docker.io mirror %s", self.config.registry_mirror)

    @property
    def _podman(self) -> list[str]:
        return [self.config.podman_binary, f"--storage-driver={self.config.storage_driver}"]

    async def _read_limited(
        self, stream: asyncio.StreamReader, limit: int, stream_name: str
    ) -> tuple[bytes, bool]:
        """Read a stream, keeping at most ``limit`` bytes.

        Excess output is drained and discarded rather than aborting the
        command: aborting kills the exec mid-flight and surfaces an error the
        caller cannot distinguish from a broken session, and any gateway-level
        retry would re-run a command that deterministically overflows again.
        Returns the (possibly truncated) bytes and whether truncation happened.
        """
        chunks: list[bytes] = []
        total = 0
        truncated = False
        while chunk := await stream.read(64 * 1024):
            if truncated:
                continue
            total += len(chunk)
            if total > limit:
                keep = limit - (total - len(chunk))
                if keep > 0:
                    chunks.append(chunk[:keep])
                truncated = True
                logger.warning("%s exceeded %s bytes; truncating", stream_name, limit)
                continue
            chunks.append(chunk)
        return b"".join(chunks), truncated

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()

    async def _write_stdin(self, process: asyncio.subprocess.Process, data: bytes) -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(data)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    async def _run(
        self, args: list[str], *, stdin: bytes = b"", timeout: Optional[float] = None
    ) -> CompletedPodmanCommand:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env=self._podman_env,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_task = asyncio.create_task(
            self._read_limited(process.stdout, self.config.max_stdout_bytes, "stdout")
        )
        stderr_task = asyncio.create_task(
            self._read_limited(process.stderr, self.config.max_stderr_bytes, "stderr")
        )
        stdin_task = asyncio.create_task(self._write_stdin(process, stdin))
        try:
            stdout_read, stderr_read, _, _ = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task, process.wait(), stdin_task),
                timeout=timeout or self.config.operation_timeout,
            )
        except asyncio.TimeoutError:
            await self._terminate(process)
            for task in (stdout_task, stderr_task, stdin_task):
                task.cancel()
            await asyncio.gather(stdout_task, stderr_task, stdin_task, return_exceptions=True)
            raise
        stdout, stdout_truncated = stdout_read
        stderr, stderr_truncated = stderr_read
        return CompletedPodmanCommand(
            tuple(args),
            process.returncode or 0,
            stdout,
            stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def _lock_for(self, container_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(container_id, asyncio.Lock())

    async def create_session(
        self,
        client_id: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        instance_slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.config.instance_id)[:32]
        name = f"literegistry-{instance_slug}-{secrets.token_hex(12)}"
        session_image = image or self.config.session_image
        resource_flags: list[str] = []
        if self.config.session_memory:
            # --memory-swap equal to --memory disables swap growth, so a
            # runaway session is OOM-killed instead of thrashing the replica.
            resource_flags += [
                "--memory",
                self.config.session_memory,
                "--memory-swap",
                self.config.session_memory,
            ]
        if self.config.session_pids_limit:
            resource_flags += ["--pids-limit", str(self.config.session_pids_limit)]
        result = await self._run(
            [
                *self._podman,
                "run",
                "--detach",
                "--name",
                name,
                "--label",
                f"{_OWNER_LABEL}=true",
                "--label",
                f"{_INSTANCE_LABEL}={self.config.instance_id}",
                "--network",
                self.config.session_network,
                *resource_flags,
                "--",
                session_image,
                "/bin/bash",
                "-lc",
                "mkdir -p /workspace && exec sleep infinity",
            ]
        )
        if result.returncode != 0:
            raise PodmanBackendError(
                result.stderr.decode(errors="replace").strip()
                or "Podman failed to create a session container"
            )
        container_id = result.stdout.decode().strip().splitlines()[-1]
        if not _CONTAINER_ID_RE.fullmatch(container_id):
            raise PodmanBackendError("Podman returned an invalid container ID")
        async with self._locks_guard:
            self._owned_container_ids.add(container_id)
            self._session_last_used[container_id] = time.monotonic()
        logger.info("Created container=%s client_id=%r", container_id, client_id)
        return container_id

    async def _require_owned(self, container_id: str) -> None:
        async with self._locks_guard:
            owned = container_id in self._owned_container_ids
        if not owned:
            raise SessionNotFound(container_id)

    async def execute(
        self,
        container_id: str,
        command: str,
        *,
        stdin: str = "",
        timeout: float = 10.0,
        workdir: str = "/workspace",
    ) -> CompletedPodmanCommand:
        lock = await self._lock_for(container_id)
        async with lock:
            await self._require_owned(container_id)
            self._session_last_used[container_id] = time.monotonic()
            # timeout runs inside the inner container, so timed-out children do
            # not remain alive if the outer podman exec process is terminated.
            return await self._run(
                [
                    *self._podman,
                    "exec",
                    "--interactive",
                    "--workdir",
                    workdir,
                    container_id,
                    "/usr/bin/timeout",
                    "--signal=KILL",
                    "--kill-after=1s",
                    f"{timeout}s",
                    "/bin/bash",
                    "-lc",
                    command,
                ],
                stdin=stdin.encode(),
                timeout=timeout + 5.0,
            )

    async def remove_session(self, container_id: str) -> None:
        lock = await self._lock_for(container_id)
        async with lock:
            await self._require_owned(container_id)
            result = await self._run(
                [
                    *self._podman,
                    "rm",
                    "--force",
                    "--time",
                    "0",
                    container_id,
                ]
            )
            if result.returncode != 0:
                raise PodmanBackendError(
                    result.stderr.decode(errors="replace").strip()
                    or "Podman failed to remove the session container"
                )
        async with self._locks_guard:
            self._owned_container_ids.discard(container_id)
            self._session_last_used.pop(container_id, None)
            self._locks.pop(container_id, None)

    async def owned_container_ids(self) -> list[str]:
        result = await self._run(
            [
                *self._podman,
                "ps",
                "--all",
                "--filter",
                f"label={_OWNER_LABEL}=true",
                "--filter",
                f"label={_INSTANCE_LABEL}={self.config.instance_id}",
                "--format",
                "{{.ID}}",
            ]
        )
        if result.returncode != 0:
            raise PodmanBackendError(
                result.stderr.decode(errors="replace").strip()
                or "Podman failed to list session containers"
            )
        return [line for line in result.stdout.decode().splitlines() if line]

    async def cleanup(self) -> None:
        container_ids = await self.owned_container_ids()
        async with self._locks_guard:
            self._owned_container_ids.update(container_ids)
        for container_id in container_ids:
            try:
                await self.remove_session(container_id)
            except SessionNotFound:
                pass

    async def reap_idle_sessions(self) -> list[str]:
        """Remove sessions idle longer than ``session_idle_timeout``.

        Covers clients that died without calling close and containers whose
        handshake response was lost in transit: both keep running otherwise
        until the replica restarts. Containers found in Podman but unknown to
        the activity map are adopted with a fresh timestamp so they get one
        full idle window before being reaped.
        """
        idle_timeout = self.config.session_idle_timeout
        if not idle_timeout:
            return []
        now = time.monotonic()
        listed = await self.owned_container_ids()
        async with self._locks_guard:
            self._owned_container_ids.update(listed)
            for container_id in listed:
                self._session_last_used.setdefault(container_id, now)
            expired = [
                container_id
                for container_id, last_used in self._session_last_used.items()
                if now - last_used > idle_timeout
            ]
        removed: list[str] = []
        for container_id in expired:
            try:
                await self.remove_session(container_id)
            except SessionNotFound:
                pass
            except PodmanBackendError as exc:
                logger.warning("Failed to reap idle container %s: %s", container_id, exc)
            else:
                removed.append(container_id)
                logger.info("Reaped idle container=%s (idle > %.0fs)", container_id, idle_timeout)
        return removed

    async def prune_images(self) -> None:
        """Prune unused images older than ``image_prune_until``.

        Session containers copy their whole image under the vfs storage
        driver, so replicas serving many distinct task images fill their disk
        without periodic pruning.
        """
        until = self.config.image_prune_until
        if not until:
            return
        result = await self._run(
            [
                *self._podman,
                "image",
                "prune",
                "--all",
                "--force",
                "--filter",
                f"until={until}",
            ]
        )
        if result.returncode != 0:
            logger.warning(
                "podman image prune failed: %s",
                result.stderr.decode(errors="replace").strip(),
            )

    async def janitor_loop(self) -> None:
        """Periodically reap idle sessions and prune stale images.

        Run as a background task for the lifetime of the server; a no-op when
        neither ``session_idle_timeout`` nor ``image_prune_until`` is set.
        """
        if not self.config.session_idle_timeout and not self.config.image_prune_until:
            return
        logger.info(
            "Janitor running (interval=%.0fs, session_idle_timeout=%s, image_prune_until=%s)",
            self.config.janitor_interval,
            self.config.session_idle_timeout,
            self.config.image_prune_until,
        )
        while True:
            await asyncio.sleep(self.config.janitor_interval)
            try:
                await self.reap_idle_sessions()
                await self.prune_images()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Janitor iteration failed")


class PodmanAffinityService:
    def __init__(self, backend: PodmanSessionBackend) -> None:
        self.backend = backend

    async def handshake(self, request: HandshakeRequest) -> HandshakeResponse:
        image = request.image or self.backend.config.session_image
        container_id = await self.backend.create_session(request.client_id, image=image)
        return HandshakeResponse(
            container_id=container_id,
            affinity_id=container_id,
            instance_id=self.backend.config.instance_id,
            client_id=request.client_id,
            image=image,
        )

    async def podman(self, request: PodmanRequest) -> PodmanResponse:
        container_id = request.selected_container_id()
        started = time.perf_counter()
        result = await self.backend.execute(
            container_id,
            request.command,
            stdin=request.stdin,
            timeout=request.timeout,
            workdir=request.workdir,
        )
        return PodmanResponse(
            container_id=container_id,
            affinity_id=container_id,
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            success=result.returncode == 0,
            exit_code=result.returncode,
            execution_time=time.perf_counter() - started,
            timed_out=result.returncode in {124, 137},
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def close(self, request: CloseRequest) -> dict[str, Any]:
        container_id = request.selected_container_id()
        await self.backend.remove_session(container_id)
        return {"container_id": container_id, "affinity_id": container_id, "removed": True}


def create_app(
    service: PodmanAffinityService, api_token: Optional[str], *, lifespan=None
) -> FastAPI:
    if api_token is not None and len(api_token) < 32:
        raise ValueError("api_token must be at least 32 characters")

    async def require_bearer_token(
        authorization: Optional[str] = Header(default=None),
    ) -> None:
        expected = f"Bearer {api_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="valid bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    dependencies = [Depends(require_bearer_token)] if api_token is not None else []
    app = FastAPI(
        title="Podman Affinity Server",
        lifespan=lifespan,
        dependencies=dependencies,
    )
    app.state.podman_affinity_service = service

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            containers = await service.backend.owned_container_ids()
        except PodmanBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "healthy",
            "instance_id": service.backend.config.instance_id,
            "sessions": len(containers),
            "image": service.backend.config.session_image,
            "registry_mirror": service.backend.config.registry_mirror,
            "session_memory": service.backend.config.session_memory,
            "session_pids_limit": service.backend.config.session_pids_limit,
            "session_idle_timeout": service.backend.config.session_idle_timeout,
        }

    @app.post("/handshake", response_model=HandshakeResponse)
    async def handshake(request: Optional[HandshakeRequest] = None) -> HandshakeResponse:
        try:
            return await service.handshake(request or HandshakeRequest())
        except (PodmanBackendError, asyncio.TimeoutError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/podman", response_model=PodmanResponse)
    async def podman(request: PodmanRequest) -> PodmanResponse:
        try:
            return await service.podman(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except SessionNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "container_not_found",
                    "container_id": exc.args[0],
                    "message": "container does not exist or belongs to another instance",
                },
            ) from exc
        except OutputLimitExceeded as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=408, detail="command timed out") from exc
        except PodmanBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/close")
    async def close(request: CloseRequest) -> dict[str, Any]:
        try:
            return await service.close(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except SessionNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "container_not_found", "container_id": exc.args[0]},
            ) from exc
        except PodmanBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app
