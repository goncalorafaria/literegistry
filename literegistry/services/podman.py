"""HTTP affinity-session primitives backed by Podman containers.

One handshake creates one long-lived inner container. Follow-up requests carry
the returned container ID, so files and processes remain scoped to that same
container for the lifetime of the session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import math
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
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMORY_LIMIT_RE = re.compile(r"^(\d+(?:\.\d+)?)([bkmg]?)$", re.IGNORECASE)
_MEMORY_SUFFIXES = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}

# Enforce a command deadline inside its session container. Images do not all
# provide coreutils at /usr/bin/timeout, and BusyBox only supports timeout's
# short signal option. Use any timeout found on PATH; otherwise Bash job
# control gives the command its own process group and a watchdog kills that
# whole group at the deadline. Bash and sleep are already session-image
# requirements because create_session uses both for the long-lived entrypoint.
# The deadline and command are positional parameters rather than interpolated
# into this shell program.
_EXEC_WRAPPER = """\
if command -v timeout >/dev/null 2>&1; then
  exec timeout -s KILL "$1" /bin/bash -lc "$2"
fi
exec 3<&0
set -m
/bin/bash -lc "$2" <&3 &
cmd=$!
set +m
exec 3<&-
( sleep "$1"; kill -9 -- "-$cmd" 2>/dev/null ) >/dev/null 2>&1 </dev/null &
watchdog=$!
wait "$cmd" 2>/dev/null
rc=$?
kill "$watchdog" 2>/dev/null
exit "$rc"
"""


def parse_memory_limit(value: str) -> int:
    """Convert a Podman memory value such as ``256m`` or ``4g`` to bytes."""

    match = _MEMORY_LIMIT_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid memory limit: {value!r}")
    amount = float(match.group(1))
    result = int(amount * _MEMORY_SUFFIXES[match.group(2).lower()])
    if result < 1:
        raise ValueError("memory limit must be positive")
    return result

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
    container_id: Optional[str] = Field(default=None, min_length=64, max_length=64)
    affinity_id: Optional[str] = Field(default=None, min_length=64, max_length=64)

    def selected_container_id(self) -> str:
        if self.container_id and self.affinity_id and self.container_id != self.affinity_id:
            raise ValueError("container_id and affinity_id must match when both are supplied")
        container_id = self.container_id or self.affinity_id
        if not container_id:
            raise ValueError("container_id or affinity_id is required")
        if not _CONTAINER_ID_RE.fullmatch(container_id):
            raise ValueError("container ID must be 64 lowercase hexadecimal characters")
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
    max_sessions: Optional[int] = None
    session_memory: Optional[str] = None
    session_pids_limit: Optional[int] = None
    session_idle_timeout: Optional[float] = None
    janitor_interval: float = 300.0
    resource_watchdog_interval: Optional[float] = 5.0
    image_prune_until: Optional[str] = None
    instance_id: str = "podman-affinity-1"
    max_stdout_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 256 * 1024
    operation_timeout: float = 300.0
    registry_mirror: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("session_memory", "image_prune_until"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when supplied")
        if self.session_memory is not None:
            parse_memory_limit(self.session_memory)
        if self.max_sessions is not None and self.max_sessions < 1:
            raise ValueError("max_sessions must be positive when supplied")
        if self.session_pids_limit is not None and self.session_pids_limit < 1:
            raise ValueError("session_pids_limit must be positive when supplied")
        if self.session_idle_timeout is not None and (
            not math.isfinite(self.session_idle_timeout)
            or self.session_idle_timeout <= 0
        ):
            raise ValueError("session_idle_timeout must be positive and finite")
        if not math.isfinite(self.janitor_interval) or self.janitor_interval <= 0:
            raise ValueError("janitor_interval must be positive and finite")
        if self.resource_watchdog_interval is not None and (
            not math.isfinite(self.resource_watchdog_interval)
            or self.resource_watchdog_interval <= 0
        ):
            raise ValueError(
                "resource_watchdog_interval must be positive and finite when supplied"
            )
        if self.max_stdout_bytes < 1 or self.max_stderr_bytes < 1:
            raise ValueError("output limits must be positive")
        if not math.isfinite(self.operation_timeout) or self.operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive and finite")


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
        self._terminating_container_ids: set[str] = set()
        self._session_last_used: dict[str, float] = {}
        self._container_init_pids: dict[str, int] = {}
        self._pending_sessions = 0
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

    def _now(self) -> float:
        return time.monotonic()

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
        async with self._locks_guard:
            capacity = self.config.max_sessions
            sessions_in_use = len(self._owned_container_ids) + self._pending_sessions
            if capacity is not None and sessions_in_use >= capacity:
                raise PodmanBackendError(
                    f"Podman session capacity exhausted ({sessions_in_use}/{capacity})"
                )
            self._pending_sessions += 1
        try:
            return await self._create_reserved_session(client_id, image)
        finally:
            async with self._locks_guard:
                self._pending_sessions -= 1

    async def _create_reserved_session(
        self,
        client_id: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """Create a container after ``create_session`` reserves capacity."""

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
            # `podman run` may create and label a stopped container before the
            # requested entrypoint fails (for example, an image without bash).
            # The command has no usable container ID in this case, so clean up
            # by the unique name before returning the handshake error.
            await self._run(
                [
                    *self._podman,
                    "rm",
                    "--force",
                    "--time",
                    "0",
                    "--ignore",
                    name,
                ]
            )
            raise PodmanBackendError(
                result.stderr.decode(errors="replace").strip()
                or "Podman failed to create a session container"
            )
        container_id = result.stdout.decode().strip().splitlines()[-1]
        if not _CONTAINER_ID_RE.fullmatch(container_id):
            raise PodmanBackendError("Podman returned an invalid container ID")
        async with self._locks_guard:
            self._owned_container_ids.add(container_id)
            self._session_last_used[container_id] = self._now()
        logger.info("Created container=%s client_id=%r", container_id, client_id)
        return container_id

    async def _require_owned(self, container_id: str) -> None:
        async with self._locks_guard:
            owned = (
                container_id in self._owned_container_ids
                and container_id not in self._terminating_container_ids
            )
        if not owned:
            raise SessionNotFound(container_id)

    async def _touch_owned_session(self, container_id: str) -> None:
        async with self._locks_guard:
            if container_id not in self._owned_container_ids:
                raise SessionNotFound(container_id)
            self._session_last_used[container_id] = self._now()

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
            await self._touch_owned_session(container_id)
            try:
                # The deadline runs inside the container, so it can terminate
                # the command's complete process tree before the outer
                # podman-exec failsafe expires.
                return await self._run(
                    [
                        *self._podman,
                        "exec",
                        "--interactive",
                        "--workdir",
                        workdir,
                        container_id,
                        "/bin/bash",
                        "-c",
                        _EXEC_WRAPPER,
                        "literegistry-exec",
                        f"{timeout:g}",
                        command,
                    ],
                    stdin=stdin.encode(),
                    timeout=timeout + 5.0,
                )
            finally:
                # A long command can exceed the idle timeout. Touching on
                # completion makes a janitor that waited on this container's
                # lock re-check against fresh activity and skip deletion.
                try:
                    await self._touch_owned_session(container_id)
                except SessionNotFound:
                    # A resource watchdog is allowed to force-remove an active
                    # command; do not mask that command's Podman result here.
                    pass

    async def _remove_session_locked(self, container_id: str) -> None:
        """Remove a session while its normal lifecycle lock is held."""

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
            self._terminating_container_ids.discard(container_id)
            self._session_last_used.pop(container_id, None)
            self._container_init_pids.pop(container_id, None)
            self._locks.pop(container_id, None)

    async def remove_session(
        self,
        container_id: str,
        *,
        idle_before: Optional[float] = None,
    ) -> bool:
        lock = await self._lock_for(container_id)
        async with lock:
            await self._require_owned(container_id)
            if idle_before is not None:
                async with self._locks_guard:
                    last_used = self._session_last_used.get(container_id)
                    if last_used is None or last_used >= idle_before:
                        return False
            await self._remove_session_locked(container_id)
            # Keep the per-container lock held until ownership is revoked. A
            # queued command can then acquire this same lock only after the
            # in-memory state says the removed container is no longer usable.
        return True

    async def owned_container_ids(self) -> list[str]:
        result = await self._run(
            [
                *self._podman,
                "ps",
                "--all",
                "--no-trunc",
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
        container_ids = [
            line.strip() for line in result.stdout.decode().splitlines() if line.strip()
        ]
        invalid = [
            container_id
            for container_id in container_ids
            if not _CONTAINER_ID_RE.fullmatch(container_id)
        ]
        if invalid:
            raise PodmanBackendError(
                "Podman returned a non-canonical container ID while listing sessions"
            )
        return container_ids

    async def capacity_status(self) -> tuple[int, Optional[int]]:
        """Return reserved/active session count and the configured capacity."""

        async with self._locks_guard:
            return (
                len(self._owned_container_ids) + self._pending_sessions,
                self.config.max_sessions,
            )

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
        now = self._now()
        idle_before = now - idle_timeout
        listed = await self.owned_container_ids()
        async with self._locks_guard:
            self._owned_container_ids.update(listed)
            for container_id in listed:
                self._session_last_used.setdefault(container_id, now)
            expired = [
                container_id
                for container_id, last_used in self._session_last_used.items()
                if last_used < idle_before
            ]
        removed: list[str] = []
        for container_id in expired:
            try:
                removed_now = await self.remove_session(
                    container_id,
                    idle_before=idle_before,
                )
            except SessionNotFound:
                pass
            except PodmanBackendError as exc:
                logger.warning("Failed to reap idle container %s: %s", container_id, exc)
            else:
                if not removed_now:
                    continue
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

    async def _container_init_pid(self, container_id: str) -> Optional[int]:
        """Return and cache the container init PID in the Podman host namespace."""

        async with self._locks_guard:
            cached = self._container_init_pids.get(container_id)
            if cached is not None:
                return cached
            if container_id not in self._owned_container_ids:
                return None
        result = await self._run(
            [*self._podman, "inspect", "--format", "{{.State.Pid}}", container_id],
            timeout=15.0,
        )
        if result.returncode != 0:
            return None
        try:
            pid = int(result.stdout.decode().strip())
        except ValueError:
            return None
        if pid <= 0:
            return None
        async with self._locks_guard:
            if container_id not in self._owned_container_ids:
                return None
            return self._container_init_pids.setdefault(container_id, pid)

    @staticmethod
    def _process_snapshot() -> dict[int, tuple[int, int, int, Optional[int]]]:
        """Read ``/proc`` once as PID -> (PPID, RSS, tasks, PID namespace).

        Podman exec processes are not guaranteed to remain descendants of the
        container init process in the outer PID namespace.  The PID namespace
        inode is the stable membership key shared by every process in one
        container, including exec processes that have been reparented.
        """

        snapshot: dict[int, tuple[int, int, int, Optional[int]]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{entry}/stat", "rb") as handle:
                    fields = handle.read().rsplit(b")", 1)[1].split()
                ppid = int(fields[1])
                rss_bytes = 0
                task_count = 1
                with open(f"/proc/{entry}/status", "rb") as handle:
                    for line in handle:
                        if line.startswith(b"VmRSS:"):
                            rss_bytes = int(line.split()[1]) * 1024
                        elif line.startswith(b"Threads:"):
                            task_count = int(line.split()[1])
                try:
                    pid_namespace = os.stat(f"/proc/{entry}/ns/pid").st_ino
                except OSError:
                    pid_namespace = None
                snapshot[pid] = (ppid, rss_bytes, task_count, pid_namespace)
            except (OSError, IndexError, ValueError):
                continue
        return snapshot

    @staticmethod
    def _subtree_usages(
        snapshot: dict[int, tuple[int, int, int, Optional[int]]], root_pids: set[int]
    ) -> dict[int, tuple[int, int]]:
        """Aggregate all requested subtrees after building the child map once."""

        children: dict[int, list[int]] = {}
        for pid, (ppid, _, _, _) in snapshot.items():
            children.setdefault(ppid, []).append(pid)
        usages: dict[int, tuple[int, int]] = {}
        for root_pid in root_pids:
            if root_pid not in snapshot:
                continue
            rss_bytes = 0
            task_count = 0
            seen: set[int] = set()
            stack = [root_pid]
            while stack:
                pid = stack.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                process = snapshot.get(pid)
                if process is None:
                    continue
                _, process_rss, process_tasks, _ = process
                rss_bytes += process_rss
                task_count += process_tasks
                stack.extend(children.get(pid, ()))
            usages[root_pid] = (rss_bytes, task_count)
        return usages

    @classmethod
    def _subtree_usage(
        cls,
        snapshot: dict[int, tuple[int, int, int, Optional[int]]],
        root_pid: int,
    ) -> Optional[tuple[int, int]]:
        """Return approximate RSS bytes and kernel-task count below ``root_pid``."""

        return cls._subtree_usages(snapshot, {root_pid}).get(root_pid)

    @staticmethod
    def _container_usages(
        snapshot: dict[int, tuple[int, int, int, Optional[int]]], root_pids: set[int]
    ) -> dict[int, tuple[int, int]]:
        """Aggregate usage by the init process's PID namespace.

        Falling back to process ancestry keeps the watchdog useful on systems
        that hide namespace symlinks or while a process is racing with exit.
        """

        namespaces = {
            root_pid: process[3]
            for root_pid in root_pids
            if (process := snapshot.get(root_pid)) is not None
        }
        namespace_totals: dict[int, tuple[int, int]] = {}
        for _, rss_bytes, task_count, pid_namespace in snapshot.values():
            if pid_namespace is None:
                continue
            old_rss, old_tasks = namespace_totals.get(pid_namespace, (0, 0))
            namespace_totals[pid_namespace] = (
                old_rss + rss_bytes,
                old_tasks + task_count,
            )
        ancestry = PodmanSessionBackend._subtree_usages(snapshot, root_pids)
        usages: dict[int, tuple[int, int]] = {}
        for root_pid, pid_namespace in namespaces.items():
            if pid_namespace is not None and pid_namespace in namespace_totals:
                usages[root_pid] = namespace_totals[pid_namespace]
            elif root_pid in ancestry:
                usages[root_pid] = ancestry[root_pid]
        return usages

    def _resource_violation(self, usage: tuple[int, int]) -> Optional[str]:
        rss_bytes, task_count = usage
        if self.config.session_memory is not None:
            limit = parse_memory_limit(self.config.session_memory)
            if rss_bytes > limit:
                return (
                    f"resident memory {rss_bytes} bytes exceeds budget "
                    f"{limit} ({self.config.session_memory})"
                )
        if self.config.session_pids_limit is not None:
            if task_count > self.config.session_pids_limit:
                return (
                    f"task count {task_count} exceeds budget "
                    f"{self.config.session_pids_limit}"
                )
        return None

    async def _force_remove_over_budget(
        self, container_id: str, init_pid: int, reason: str
    ) -> bool:
        """Immediately remove a confirmed offender, including an active command."""

        async with self._locks_guard:
            if (
                container_id not in self._owned_container_ids
                or container_id in self._terminating_container_ids
                or self._container_init_pids.get(container_id) != init_pid
            ):
                return False
            self._terminating_container_ids.add(container_id)
        logger.warning("Session %s %s; force-removing", container_id, reason)
        result = await self._run(
            [*self._podman, "rm", "--force", "--time", "0", container_id]
        )
        async with self._locks_guard:
            self._terminating_container_ids.discard(container_id)
            if result.returncode == 0:
                self._owned_container_ids.discard(container_id)
                self._session_last_used.pop(container_id, None)
                self._container_init_pids.pop(container_id, None)
                self._locks.pop(container_id, None)
        if result.returncode != 0:
            logger.warning(
                "Failed to remove over-budget container %s: %s",
                container_id,
                result.stderr.decode(errors="replace").strip(),
            )
            return False
        return True

    async def enforce_resource_budgets(self) -> list[str]:
        """Remove sessions over budget when native cgroups are unavailable.

        PID discovery is cached, and the host process table is read once per
        sweep. Potential violations must appear in a second fresh snapshot
        before removal. RSS is an approximate fallback rather than cgroup
        accounting; native Podman limits remain enabled where supported.
        """

        if self.config.session_memory is None and self.config.session_pids_limit is None:
            return []
        async with self._locks_guard:
            container_ids = list(self._owned_container_ids)
        pid_values = await asyncio.gather(
            *(self._container_init_pid(container_id) for container_id in container_ids)
        )
        roots = {
            container_id: pid
            for container_id, pid in zip(container_ids, pid_values)
            if pid is not None
        }
        if not roots:
            return []
        first = await asyncio.to_thread(self._process_snapshot)
        first_usages = self._container_usages(first, set(roots.values()))
        candidates = {
            container_id: pid
            for container_id, pid in roots.items()
            if (usage := first_usages.get(pid)) is not None
            and self._resource_violation(usage) is not None
        }
        if not candidates:
            return []
        confirmation = await asyncio.to_thread(self._process_snapshot)
        confirmed_usages = self._container_usages(
            confirmation, set(candidates.values())
        )
        removed: list[str] = []
        for container_id, pid in candidates.items():
            usage = confirmed_usages.get(pid)
            reason = self._resource_violation(usage) if usage is not None else None
            if reason is None:
                continue
            if await self._force_remove_over_budget(container_id, pid, reason):
                removed.append(container_id)
        return removed

    async def resource_watchdog_loop(self) -> None:
        """Continuously enforce resource budgets independently of the janitor."""

        interval = self.config.resource_watchdog_interval
        if interval is None or (
            self.config.session_memory is None
            and self.config.session_pids_limit is None
        ):
            return
        logger.info(
            "Resource watchdog running (interval=%.1fs, session_memory=%s, "
            "session_pids_limit=%s)",
            interval,
            self.config.session_memory,
            self.config.session_pids_limit,
        )
        while True:
            await asyncio.sleep(interval)
            try:
                await self.enforce_resource_budgets()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Resource watchdog iteration failed")

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
            sessions_in_use, max_sessions = await service.backend.capacity_status()
        except PodmanBackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "healthy",
            "instance_id": service.backend.config.instance_id,
            "sessions": len(containers),
            "pending_sessions": max(0, sessions_in_use - len(containers)),
            "max_sessions": max_sessions,
            "available_sessions": (
                None
                if max_sessions is None
                else max(0, max_sessions - sessions_in_use)
            ),
            "image": service.backend.config.session_image,
            "registry_mirror": service.backend.config.registry_mirror,
            "session_memory": service.backend.config.session_memory,
            "session_pids_limit": service.backend.config.session_pids_limit,
            "session_idle_timeout": service.backend.config.session_idle_timeout,
            "janitor_interval": service.backend.config.janitor_interval,
            "resource_watchdog_interval": service.backend.config.resource_watchdog_interval,
            "image_prune_until": service.backend.config.image_prune_until,
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
