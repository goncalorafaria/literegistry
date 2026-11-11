"""Explicit start/read/wait client for worker-buffered Podman commands.

Uses the native /affinity/podman/start and /output protocol. No shell wrapper,
background polling task, execute fallback, or automatic command resubmission.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from .podman import DEFAULT_PODMAN_GATEWAY_URL, PodmanCommand, PodmanExecutionClient, PodmanStreamError

_STATUS = ('exit_code', 'success', 'timed_out', 'error')


def _validate_wait(wait_ms: int) -> None:
    if type(wait_ms) is not int or not 0 <= wait_ms <= 10000:
        raise ValueError('wait_ms must be an integer between 0 and 10000')


class PodmanStreamingUnavailable(PodmanStreamError):
    """The selected gateway/worker does not expose the native streaming API."""


class BufferedPodmanCommand:
    """Single-consumer command handle; each read consumes only new output.

    The buffer lives on the worker. Slow consumers may receive HTTP 410 when
    its finite retention/byte limit is exceeded; this is never silently skipped.
    """

    def __init__(self, command: PodmanCommand) -> None:
        self._command = command
        self._lock = asyncio.Lock()
        self._final: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        """Whether reads have reached the end; this property does no network I/O."""
        return self._final is not None

    async def _read(self, wait_ms: int) -> dict[str, Any]:
        if self._final is not None:
            return dict(self._final, stdout='', stderr='', running=False, done=True, has_more=False)
        batch = await self._command.read(wait_ms=wait_ms)
        if batch['done'] and not batch['has_more']:
            self._final = {key: batch.get(key) for key in _STATUS}
        return batch

    async def read(self, *, wait_ms: int = 0) -> dict[str, Any]:
        """Return currently buffered stdout/stderr and status. Never wait for exit.

        wait_ms=0 returns immediately; positive values allow bounded long polling.
        Repeated reads after completion return empty output and cached status.
        """
        _validate_wait(wait_ms)
        async with self._lock:
            return await self._read(wait_ms)

    async def batches(self, *, wait_ms: int = 1000) -> AsyncIterator[dict[str, Any]]:
        """Yield new output batches until completion, without accumulating logs."""
        _validate_wait(wait_ms)
        if wait_ms == 0:
            raise ValueError('batches requires positive wait_ms to avoid busy polling')
        while not self.done:
            yield await self.read(wait_ms=wait_ms)
        await self._command.wait(wait_ms=0)

    async def wait(self, *, wait_ms: int = 1000, max_output_chars: int = 1048576) -> dict[str, Any]:
        """Wait for exit and return final status plus bounded *unread* output.

        Output already returned by read()/batches() is not repeated. Continue
        draining after the local collection limit, and report omitted characters.
        """
        _validate_wait(wait_ms)
        if wait_ms == 0:
            raise ValueError('wait requires positive wait_ms to avoid busy polling')
        if type(max_output_chars) is not int or max_output_chars < 0:
            raise ValueError('max_output_chars must be a nonnegative integer')
        async with self._lock:
            output = {'stdout': [], 'stderr': []}
            remaining = max_output_chars
            dropped = 0
            while not self.done:
                batch = await self._read(wait_ms)
                for channel in output:
                    data = batch[channel]
                    keep = data[:remaining]
                    if keep:
                        output[channel].append(keep)
                    remaining -= len(keep)
                    dropped += len(data) - len(keep)
            await self._command.wait(wait_ms=0)  # acknowledge the final cursor
            return dict(self._final, stdout=''.join(output['stdout']),
                        stderr=''.join(output['stderr']), output_truncated=bool(dropped),
                        dropped_output_chars=dropped)


class PodmanAsyncClient:
    """Own a container with a buffered command API, separate from execute().

    open() creates the session; start(command=...) launches a command and returns
    immediately with a reader. Concurrency limits are enforced by the worker:
    a worker that allows only one command returns session_busy for a second.
    """

    def __init__(self, gateway_url: str = DEFAULT_PODMAN_GATEWAY_URL, **session_options: Any) -> None:
        session_options.setdefault('max_retries', 1)
        self._client = PodmanExecutionClient(gateway_url, **session_options)
        self._lifecycle_lock = asyncio.Lock()
        self._commands: set[BufferedPodmanCommand] = set()

    @property
    def container_id(self) -> str | None:
        return self._client.container_id

    @property
    def opened(self) -> bool:
        return self._client.started

    async def open(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            return await self._client.start()

    async def start(self, *, command: str, stdin: str = '', timeout: float = 10,
                    workdir: str | None = None, merge_output: bool = False) -> BufferedPodmanCommand:
        async with self._lifecycle_lock:
            try:
                handle = await self._client.start_command(command=command, stdin=stdin,
                    timeout=timeout, workdir=workdir, merge_output=merge_output)
            except PodmanStreamError as error:
                if error.status_code in (404, 405, 501):
                    raise PodmanStreamingUnavailable(error.status_code, {
                        'detail': {'code': 'streaming_unavailable',
                                   'message': 'The selected gateway/worker does not expose the Podman buffered command API.',
                                   'upstream_response': error.response}}) from error
                raise
            self._commands = {job for job in self._commands if not job.done}
            job = BufferedPodmanCommand(handle)
            self._commands.add(job)
            return job

    async def close(self) -> dict[str, Any] | None:
        """Drain accepted commands, then remove the owned container.

        Preserve read errors (including output expiry), but still attempt session
        cleanup. This does not silently cancel a running command.
        """
        async with self._lifecycle_lock:
            outcomes = await asyncio.gather(*(job.wait(max_output_chars=0) for job in self._commands),
                                            return_exceptions=True)
            result = await self._client.close()
            self._commands.clear()
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            return result

    async def __aenter__(self) -> PodmanAsyncClient:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        try:
            await self.close()
        except Exception:
            if exc is None:
                raise
        return False
