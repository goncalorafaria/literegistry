"""Bounded, worker-local command output with acknowledgement cursors."""

from __future__ import annotations

import asyncio
import codecs
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import logging
import secrets
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .podman import PodmanSessionBackend

logger = logging.getLogger(__name__)
_MAX_BUFFER_CHUNKS = 1024


class CommandStreamError(RuntimeError):
    def __init__(self, status: int, code: str, **details: Any) -> None:
        super().__init__(code)
        self.status = status
        self.detail = {"code": code, **details}


@dataclass
class _Command:
    command_id: str
    container_id: str
    request_id: str
    fingerprint: str
    merge_output: bool
    chunks: deque[tuple[int, str, str, int]] = field(default_factory=deque)
    buffered_bytes: int = 0
    next_cursor: int = 0
    done: bool = False
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    lock: asyncio.Lock | None = None
    expiry: asyncio.TimerHandle | None = None

    @property
    def oldest_cursor(self) -> int:
        return self.chunks[0][0] if self.chunks else self.next_cursor


class PodmanCommandStreams:
    """Commands share normal session locks; polling never takes those locks.

    Passing a cursor acknowledges every chunk before it. Output returned by a
    poll remains available until acknowledged, evicted by the byte cap, or
    expired after completion. All state dies with this worker's sessions.
    """

    def __init__(self, backend: PodmanSessionBackend) -> None:
        self.backend = backend
        self.commands: dict[str, _Command] = {}
        self.requests: dict[tuple[str, str], str] = {}
        self._guard = asyncio.Lock()

    async def start(
        self, container_id: str, *, command: str, stdin: str, timeout: float,
        workdir: str, merge_output: bool, request_id: str,
    ) -> dict[str, Any]:
        await self.backend._require_owned(container_id)
        fingerprint = hashlib.sha256(json.dumps(
            [command, stdin, timeout, workdir, merge_output],
        ).encode()).hexdigest()
        async with self._guard:
            key = (container_id, request_id)
            existing = self.commands.get(self.requests.get(key, ""))
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise CommandStreamError(409, "request_id_conflict")
                return self._started(existing)
            if len(self.commands) >= self.backend.config.max_stream_commands:
                raise CommandStreamError(503, "command_stream_capacity")
            lock = await self.backend._lock_for(container_id)
            if lock.locked():
                raise CommandStreamError(409, "session_busy")
            await lock.acquire()
            try:
                await self.backend._require_owned(container_id)
                await self.backend._touch_owned_session(container_id)
                job = _Command(secrets.token_hex(16), container_id, request_id,
                               fingerprint, merge_output)
                job.lock = lock
                self.commands[job.command_id] = job
                self.requests[key] = job.command_id
                job.task = asyncio.create_task(self._execute(
                    job, lock, command=command, stdin=stdin,
                    timeout=timeout, workdir=workdir,
                ))
            except BaseException:
                lock.release()
                raise
        return self._started(job)

    def _started(self, job: _Command) -> dict[str, Any]:
        return {"command_id": job.command_id, "container_id": job.container_id,
                "affinity_id": job.container_id, "cursor": 0,
                "merge_output": job.merge_output,
                "retention_seconds": self.backend.config.stream_retention_seconds}

    def _append(self, job: _Command, stream: str, text: str) -> None:
        if not text:
            return
        size = len(text.encode("utf-8"))
        job.chunks.append((job.next_cursor, stream, text, size))
        job.next_cursor += 1
        job.buffered_bytes += size
        # Bound object overhead too: a command can emit many one-byte writes.
        while (job.buffered_bytes > self.backend.config.stream_buffer_bytes
               or len(job.chunks) > _MAX_BUFFER_CHUNKS):
            job.buffered_bytes -= job.chunks.popleft()[3]
        job.changed.set()

    async def _read(self, job: _Command, pipe: asyncio.StreamReader, stream: str) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        # Replacement characters can expand invalid input by up to three times.
        size = min(16 * 1024, max(1, self.backend.config.stream_buffer_bytes // 3))
        while chunk := await pipe.read(size):
            self._append(job, stream, decoder.decode(chunk))
        self._append(job, stream, decoder.decode(b"", final=True))

    async def _execute(self, job: _Command, lock: asyncio.Lock, **options: Any) -> None:
        process = None
        tasks: list[asyncio.Task] = []
        try:
            process = await asyncio.create_subprocess_exec(
                *self.backend._exec_args(job.container_id, options["command"],
                                        options["timeout"], options["workdir"],
                                        merge_output=job.merge_output),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=(asyncio.subprocess.STDOUT if job.merge_output
                        else asyncio.subprocess.PIPE),
                start_new_session=True, env=self.backend._podman_env,
            )
            assert process.stdout is not None
            tasks = [asyncio.create_task(self._read(job, process.stdout, "stdout")),
                     asyncio.create_task(self.backend._write_stdin(
                         process, options["stdin"].encode())),
                     asyncio.create_task(process.wait())]
            if process.stderr is not None:
                tasks.append(asyncio.create_task(self._read(job, process.stderr, "stderr")))
            await asyncio.wait_for(asyncio.gather(*tasks), options["timeout"] + 5.0)
            job.exit_code = process.returncode
            job.timed_out = job.exit_code in {124, 137}
            await self.backend._require_owned(job.container_id)
        except asyncio.TimeoutError:
            job.timed_out = True
            job.error = "podman_exec_timeout"
        except asyncio.CancelledError:
            job.error = "worker_shutdown"
            raise
        except Exception:
            logger.exception("Streaming command failed: %s", job.command_id)
            job.error = "command_execution_failed"
        finally:
            try:
                if process is not None:
                    await self.backend._terminate(process)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await self.backend._touch_owned_session(job.container_id)
                except KeyError:
                    pass
                finally:
                    lock.release()
                    job.lock = None
                    job.done = True
                    job.changed.set()
                    job.expiry = asyncio.get_running_loop().call_later(
                        self.backend.config.stream_retention_seconds,
                        self._expire, job.command_id,
                    )

    def _expire(self, command_id: str) -> None:
        job = self.commands.pop(command_id, None)
        if job is not None:
            self.requests.pop((job.container_id, job.request_id), None)
            if job.expiry is not None:
                job.expiry.cancel()
            job.chunks.clear()
            job.buffered_bytes = 0
            job.changed.set()

    def _validate_cursor(self, job: _Command, cursor: int) -> None:
        if self.commands.get(job.command_id) is not job:
            raise CommandStreamError(410, "command_expired")
        if cursor < job.oldest_cursor:
            raise CommandStreamError(410, "output_expired", oldest_cursor=job.oldest_cursor)
        if cursor > job.next_cursor:
            raise CommandStreamError(400, "invalid_cursor")

    async def output(
        self, container_id: str, *, command_id: str, cursor: int,
        wait_ms: int, max_chunks: int,
    ) -> dict[str, Any]:
        await self.backend._require_owned(container_id)
        job = self.commands.get(command_id)
        if job is None:
            raise CommandStreamError(410, "command_expired")
        if job.container_id != container_id:
            raise CommandStreamError(404, "command_not_found")
        self._validate_cursor(job, cursor)
        # No awaits while acknowledging/removing chunks: concurrent polls see
        # a consistent cursor, and an older consumer gets an explicit 410.
        while job.chunks and job.chunks[0][0] < cursor:
            job.buffered_bytes -= job.chunks.popleft()[3]
        if cursor == job.next_cursor and not job.done and wait_ms:
            job.changed.clear()
            try:
                await asyncio.wait_for(job.changed.wait(), wait_ms / 1000)
            except asyncio.TimeoutError:
                pass
        self._validate_cursor(job, cursor)
        await self.backend._require_owned(container_id)
        # Revalidate after ownership awaits, which can wait for the watchdog.
        self._validate_cursor(job, cursor)
        selected = []
        for sequence, stream, data, _ in job.chunks:
            if len(selected) >= max_chunks:
                break
            selected.append({"cursor": sequence, "stream": stream, "data": data})
        next_cursor = selected[-1]["cursor"] + 1 if selected else cursor
        return {
            "container_id": container_id, "affinity_id": container_id,
            "command_id": command_id, "chunks": selected, "next_cursor": next_cursor,
            "oldest_cursor": job.oldest_cursor, "has_more": next_cursor < job.next_cursor,
            "running": not job.done, "done": job.done,
            "exit_code": job.exit_code if job.done else None, "timed_out": job.timed_out,
            "success": job.exit_code == 0 and job.error is None if job.done else None,
            "error": job.error, "merge_output": job.merge_output,
        }

    def discard_session(self, container_id: str) -> None:
        """Called after normal removal, while the session lock is held."""
        for job in list(self.commands.values()):
            if job.container_id == container_id and job.done:
                self._expire(job.command_id)

    async def shutdown(self) -> None:
        tasks = [job.task for job in self.commands.values() if job.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # A task cancelled before its first turn never entered _execute's
        # finally block. Release the lock transferred to it by start().
        for job in self.commands.values():
            if job.lock is not None:
                job.lock.release()
                job.lock = None
        for command_id in list(self.commands):
            self._expire(command_id)
