"""Restricted, stdin-only terminal pipelines for local log analysis.

This service deliberately does not implement a shell.  It accepts a sequence
of allowlisted commands joined by ``|`` and executes every stage directly with
stdin from the preceding stage.  Submitted contents are never written to a
user-named path and commands cannot access host paths through this API.
"""

import asyncio
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from literegistry import ServerRegistry, get_kvstore
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_ALLOWED_COMMANDS = frozenset(
    {"rg", "grep", "awk", "sed", "jq", "xsv", "head", "tail", "wc", "cat", "nl", "echo"}
)
_CONTROL_TOKENS = frozenset({";", "&&", "||", "&", "(", ")", "<", ">", ">>", "2>", "2>>"})
_MAX_STAGES = 8
_QUOTED_PIPE_SENTINEL = "\ue000"
_QUOTED_LESS_THAN_SENTINEL = "\ue001"
_QUOTED_GREATER_THAN_SENTINEL = "\ue002"


class PipelineValidationError(ValueError):
    """Raised when a submitted pipeline falls outside the supported language."""


class PipelineLimitError(RuntimeError):
    """Raised when a pipeline exceeds its configured resource limit."""


class TerminalRequest(BaseModel):
    contents: str = Field(..., max_length=2 * 1024 * 1024)
    command: str = Field(..., min_length=1, max_length=16 * 1024)
    max_runtime: float = Field(default=5.0, ge=0.01, le=60)
    truncation: int | None = Field(default=None, ge=1)


class TerminalResponse(BaseModel):
    stdout: str
    stderr: str = ""
    success: bool
    exit_code: int | None = None
    execution_time: float
    truncated: bool = False
    truncated_characters: int = 0


@dataclass
class TerminalServerConfig:
    host: str = "0.0.0.0"
    port: int = 1213
    registry: str = "redis://klone-login01.hyak.local:6379"
    heartbeat_interval: int = 30
    max_output_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 64 * 1024
    max_response_chars: int | None = None
    command_path: str | None = None


def _reject_path(value: str) -> None:
    if value.startswith("/") or value.startswith("~") or ".." in value:
        raise PipelineValidationError("file paths are not permitted; all commands read stdin")


def _validate_pattern_command(command: str, args: list[str]) -> None:
    allowed = {
        "rg": {
            "-i", "--ignore-case", "-v", "--invert-match", "-n", "--line-number",
            "-c", "--count", "-F", "--fixed-strings", "-e", "--regexp",
            "-m", "--max-count", "-A", "--after-context", "-B", "--before-context",
            "-C", "--context", "-w", "--word-regexp", "-x", "--line-regexp",
            "-o", "--only-matching", "-a", "--text", "-I", "--binary",
            "-s", "--no-messages", "--color",
        },
        "grep": {
            "-i", "--ignore-case", "-v", "--invert-match", "-n", "--line-number",
            "-c", "--count", "-F", "--fixed-strings", "-E", "-G", "-e",
            "--regexp", "-m", "--max-count", "-A", "--after-context", "-B",
            "--before-context", "-C", "--context", "-w", "--word-regexp", "-x",
            "--line-regexp", "-o", "--only-matching", "-q", "--quiet", "-s",
            "--no-messages", "-a", "--text", "-I", "-z", "--null-data", "-b",
            "--byte-offset", "-H", "--with-filename", "-h", "--no-filename",
            "--color",
        },
    }[command]
    takes_value = {
        "-e", "--regexp", "-m", "--max-count", "-A", "--after-context", "-B",
        "--before-context", "-C", "--context", "--color",
    }
    attached_value_prefixes = {"-m", "-A", "-B", "-C"}
    pattern_count = 0
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-":
            index += 1
            continue
        if arg.startswith("-"):
            if (
                len(arg) > 2
                and arg[:2] in attached_value_prefixes
                and arg[2:].isdigit()
            ):
                index += 1
                continue
            if any(arg.startswith(f"{option}=") for option in takes_value if option.startswith("--")):
                index += 1
                continue
            if arg not in allowed:
                raise PipelineValidationError(f"{command} option {arg!r} is not permitted")
            if arg in takes_value:
                index += 1
                if index >= len(args):
                    raise PipelineValidationError(f"{command} option {arg!r} needs a value")
        else:
            _reject_path(arg)
            pattern_count += 1
        index += 1
    if pattern_count > 1:
        raise PipelineValidationError(f"{command} accepts a pattern but no file paths")


def _validate_awk(args: list[str]) -> None:
    program: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-F", "-v"}:
            index += 1
            if index >= len(args):
                raise PipelineValidationError(f"awk option {arg!r} needs a value")
        elif arg.startswith("-"):
            raise PipelineValidationError(f"awk option {arg!r} is not permitted")
        elif program is None:
            program = arg
        else:
            raise PipelineValidationError("awk accepts a program but no input file paths")
        index += 1
    if not program:
        raise PipelineValidationError("awk requires a program")
    lowered = program.lower()
    if any(token in lowered for token in ("system(", "getline", "@include")):
        raise PipelineValidationError("awk program uses unsupported process or file I/O")
    if re.search(r"\b(?:print|printf)\b[^;\n]*(?:\||>>?)", program):
        raise PipelineValidationError("awk program uses unsupported process or file I/O")


def _validate_sed(args: list[str]) -> None:
    scripts: list[str] = []
    for arg in args:
        if arg in {"-n", "-E", "-r"}:
            continue
        if arg.startswith("-"):
            raise PipelineValidationError(f"sed option {arg!r} is not permitted")
        scripts.append(arg)
    if len(scripts) != 1:
        raise PipelineValidationError("sed requires one script and does not accept file paths")
    script = scripts[0]
    if re.search(r"(?:^|;|/|[0-9$])\s*[erw](?:\s|$)", script) or re.search(
        r"/[ew](?:\s|$)", script
    ):
        raise PipelineValidationError("sed execution and file read/write commands are not permitted")


def _validate_jq(args: list[str]) -> None:
    flags = {"-r", "-c", "-s", "-e"}
    filters = []
    for arg in args:
        if arg in flags:
            continue
        if arg.startswith("-"):
            raise PipelineValidationError(f"jq option {arg!r} is not permitted")
        filters.append(arg)
    if len(filters) != 1:
        raise PipelineValidationError("jq requires one filter and does not accept file paths")
    if any(token in filters[0] for token in ("include", "import", "module")):
        raise PipelineValidationError("jq module loading is not permitted")


def _validate_xsv(args: list[str]) -> None:
    allowed_subcommands = {"headers", "count", "select", "search", "slice", "stats", "frequency"}
    if not args or args[0] not in allowed_subcommands:
        raise PipelineValidationError(
            "xsv supports only headers, count, select, search, slice, stats, and frequency"
        )
    if any(arg.startswith(("--input", "--output", "-o")) for arg in args[1:]):
        raise PipelineValidationError("xsv file input/output options are not permitted")
    for arg in args[1:]:
        _reject_path(arg)


def _validate_head_tail(command: str, args: list[str]) -> None:
    if not args:
        return
    if len(args) == 1 and args[0].startswith("-") and args[0][1:].isdigit():
        return
    if (
        len(args) == 2
        and args[0] in {"-n", "--lines", "-c", "--bytes"}
        and args[1].lstrip("+-").isdigit()
    ):
        return
    if (
        len(args) == 1
        and args[0].startswith("-c")
        and args[2:].lstrip("+-").isdigit()
    ):
        return
    raise PipelineValidationError(
        f"{command} only supports -n/--lines or -c/--bytes with stdin"
    )


def _validate_wc(args: list[str]) -> None:
    allowed = {
        "-l", "--lines", "-w", "--words", "-c", "--bytes", "-m", "--chars",
        "-L", "--max-line-length",
    }
    if any(arg not in allowed for arg in args):
        raise PipelineValidationError("wc supports only count options with stdin")


def _validate_cat(args: list[str]) -> None:
    if args:
        raise PipelineValidationError("cat accepts no arguments and reads stdin only")


def _validate_echo(args: list[str]) -> None:
    """Allow echo string args and -n/-e/-E; reject path-like operands."""
    for arg in args:
        if arg.startswith("-") and len(arg) > 1 and set(arg[1:]).issubset({"n", "e", "E"}):
            continue
        _reject_path(arg)


def _validate_nl(args: list[str]) -> None:
    """Allow line numbering options without allowing file operands."""
    value_options = {
        "-b", "--body-numbering", "-f", "--footer-numbering", "-h",
        "--header-numbering", "-n", "--number-format", "-w", "--number-width",
        "-s", "--number-separator", "-v", "--starting-line-number", "-i",
        "--line-increment", "-l", "--join-blank-lines",
    }
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "-p":
            index += 1
            continue
        if arg == "-ba":
            index += 1
            continue
        if arg in value_options:
            index += 1
            if index >= len(args):
                raise PipelineValidationError(f"nl option {arg!r} needs a value")
            index += 1
            continue
        raise PipelineValidationError("nl accepts only numbering options and stdin")


def _protect_quoted_metacharacters(pipeline: str) -> str:
    """Hide shell metacharacters inside quoted arguments before tokenization."""
    sentinels = {
        "|": _QUOTED_PIPE_SENTINEL,
        "<": _QUOTED_LESS_THAN_SENTINEL,
        ">": _QUOTED_GREATER_THAN_SENTINEL,
    }
    if any(sentinel in pipeline for sentinel in sentinels.values()):
        raise PipelineValidationError("pipeline contains a reserved character")
    result: list[str] = []
    quote: str | None = None
    escaped = False
    for character in pipeline:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if quote == '"' and character == "\\":
            result.append(character)
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            result.append(character)
        elif character in sentinels and quote is not None:
            result.append(sentinels[character])
        else:
            result.append(character)
    return "".join(result)


def parse_pipeline(pipeline: str) -> list[list[str]]:
    """Parse and validate a limited stdin-only pipeline."""
    if "\n" in pipeline or "\r" in pipeline:
        raise PipelineValidationError("newlines are not permitted in a pipeline")
    try:
        tokens = shlex.split(_protect_quoted_metacharacters(pipeline), posix=True)
    except ValueError as exc:
        raise PipelineValidationError(f"invalid quoting: {exc}") from exc
    # ``rg``/``grep`` return 1 for no matches. The service already treats that
    # as a successful empty result, but accept the common shell idiom without
    # evaluating a shell or accepting arbitrary ``||`` expressions.
    if len(tokens) >= 2 and tokens[-2:] == ["||", "true"]:
        tokens = tokens[:-2]
    if not tokens:
        raise PipelineValidationError("pipeline is empty")
    if any(
        token in _CONTROL_TOKENS
        or "$(" in token
        or "`" in token
        or token.startswith((">", "<"))
        for token in tokens
    ):
        raise PipelineValidationError("shell syntax is not permitted")

    stages: list[list[str]] = [[]]
    for token in tokens:
        if token == "|":
            if not stages[-1]:
                raise PipelineValidationError("pipeline contains an empty stage")
            stages.append([])
        elif "|" in token:
            raise PipelineValidationError("pipes must be separated by spaces")
        else:
            stages[-1].append(
                token.replace(_QUOTED_PIPE_SENTINEL, "|")
                .replace(_QUOTED_LESS_THAN_SENTINEL, "<")
                .replace(_QUOTED_GREATER_THAN_SENTINEL, ">")
            )
    if not stages[-1] or len(stages) > _MAX_STAGES:
        raise PipelineValidationError("pipeline has an empty stage or too many stages")

    for stage in stages:
        command, args = stage[0], stage[1:]
        if command not in _ALLOWED_COMMANDS:
            raise PipelineValidationError(f"command {command!r} is not allowlisted")
        if command == "grep":
            combined_flags = set("ivncFEGwxoqsaIzbHh")
            normalized_args: list[str] = []
            for arg in args:
                if (
                    arg.startswith("-")
                    and len(arg) > 2
                    and arg[1:].isalpha()
                    and set(arg[1:]).issubset(combined_flags)
                ):
                    normalized_args.extend(f"-{flag}" for flag in arg[1:])
                else:
                    normalized_args.append(arg)
            stage[:] = [command, *normalized_args]
            args = normalized_args
        if command in {"rg", "grep"}:
            _validate_pattern_command(command, args)
        elif command == "awk":
            _validate_awk(args)
        elif command == "sed":
            _validate_sed(args)
        elif command == "jq":
            _validate_jq(args)
        elif command == "xsv":
            _validate_xsv(args)
        elif command == "wc":
            _validate_wc(args)
        elif command == "cat":
            _validate_cat(args)
        elif command == "echo":
            _validate_echo(args)
        elif command == "nl":
            _validate_nl(args)
        else:
            _validate_head_tail(command, args)
    return stages


def _request_summary(payload: Any) -> dict[str, Any]:
    """Return debuggable request fields without echoing submitted file contents."""
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    contents = payload.get("contents")
    return {
        "command": payload.get("command"),
        "contents_length": len(contents) if isinstance(contents, str) else None,
        "max_runtime": payload.get("max_runtime"),
        "truncation": payload.get("truncation"),
    }


class TerminalPipelineServer:
    """FastAPI and LiteRegistry wrapper for restricted terminal pipelines."""

    def __init__(self, config: TerminalServerConfig | None = None):
        self.config = config or TerminalServerConfig()
        # Snapshot the launch environment rather than consulting PATH per
        # request. This admits Conda-installed log tools while keeping command
        # resolution fixed for the lifetime of this service.
        self._command_path = self.config.command_path or os.environ.get(
            "PATH", "/usr/local/bin:/usr/bin:/bin"
        )
        self._max_response_chars = (
            self.config.max_response_chars
            if self.config.max_response_chars is not None
            else self.config.max_output_bytes
        )
        self.registry = ServerRegistry(store=get_kvstore(self.config.registry))
        self.url = f"http://{socket.getfqdn()}"
        self.should_run = False
        self.heartbeat_task: asyncio.Task | None = None
        self.app = FastAPI(title="Restricted Terminal Pipeline Server")
        self._install_routes()

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_path": "terminal",
            "host": self.config.host,
            "port": self.config.port,
            "backend": "restricted-pipeline",
            "extra_kwargs": {
                "commands": sorted(_ALLOWED_COMMANDS),
                "max_output_bytes": self.config.max_output_bytes,
                "max_stderr_bytes": self.config.max_stderr_bytes,
                "max_response_chars": self._max_response_chars,
                "command_path": self._command_path,
            },
        }

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()

    async def _read_limited(
        self, stream: asyncio.StreamReader, limit: int, stream_name: str
    ) -> bytes:
        """Read a subprocess stream without allowing unbounded buffering."""
        chunks: list[bytes] = []
        total = 0
        while chunk := await stream.read(64 * 1024):
            total += len(chunk)
            if total > limit:
                raise PipelineLimitError(
                    f"pipeline {stream_name} exceeded the configured limit"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _truncate_stdout(
        self, stdout: str, truncation: int | None
    ) -> tuple[str, bool, int]:
        if truncation is None:
            return stdout, False, 0
        if truncation > self._max_response_chars:
            raise PipelineValidationError(
                f"truncation cannot exceed the server maximum of "
                f"{self._max_response_chars} characters"
            )
        missing = len(stdout) - truncation
        if missing <= 0:
            return stdout, False, 0
        marker = f"\n[ truncated ({missing} characters missing) ]"
        return stdout[:truncation] + marker, True, missing

    async def execute(self, request: TerminalRequest) -> TerminalResponse:
        started = time.perf_counter()
        stages = parse_pipeline(request.command)
        data = request.contents.encode()
        stderr_parts: list[bytes] = []
        deadline = asyncio.get_running_loop().time() + request.max_runtime

        with tempfile.TemporaryDirectory(prefix="literegistry-terminal-") as working_directory:
            for stage in stages:
                executable = shutil.which(stage[0], path=self._command_path)
                if executable is None:
                    stdout, truncated, missing = self._truncate_stdout(
                        data.decode(errors="replace"), request.truncation
                    )
                    return TerminalResponse(
                        stdout=stdout,
                        stderr=f"Required command is unavailable: {stage[0]}",
                        success=False,
                        exit_code=None,
                        execution_time=time.perf_counter() - started,
                        truncated=truncated,
                        truncated_characters=missing,
                    )
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                process = await asyncio.create_subprocess_exec(
                    executable,
                    *stage[1:],
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_directory,
                    env={"PATH": self._command_path, "LC_ALL": "C"},
                    start_new_session=True,
                )
                assert process.stdin is not None
                assert process.stdout is not None
                assert process.stderr is not None
                try:
                    process.stdin.write(data)
                    await asyncio.wait_for(process.stdin.drain(), timeout=remaining)
                except (BrokenPipeError, ConnectionResetError):
                    # Commands such as ``head`` may intentionally exit before
                    # consuming all input.
                    pass
                except asyncio.TimeoutError:
                    await self._terminate(process)
                    raise
                finally:
                    process.stdin.close()
                stdout_task = asyncio.create_task(
                    self._read_limited(
                        process.stdout, self.config.max_output_bytes, "output"
                    )
                )
                stderr_task = asyncio.create_task(
                    self._read_limited(
                        process.stderr, self.config.max_stderr_bytes, "stderr"
                    )
                )
                try:
                    stdout, stderr, _ = await asyncio.wait_for(
                        asyncio.gather(stdout_task, stderr_task, process.wait()),
                        timeout=remaining,
                    )
                except (asyncio.TimeoutError, PipelineLimitError):
                    await self._terminate(process)
                    await asyncio.gather(
                        stdout_task, stderr_task, return_exceptions=True
                    )
                    raise
                if len(stdout) > self.config.max_output_bytes:
                    raise PipelineLimitError("pipeline output exceeded the configured limit")
                if len(stderr) > self.config.max_stderr_bytes:
                    raise PipelineLimitError("pipeline stderr exceeded the configured limit")
                stderr_parts.append(stderr)
                if process.returncode not in ({0, 1} if stage[0] in {"rg", "grep"} else {0}):
                    response_stdout, truncated, missing = self._truncate_stdout(
                        stdout.decode(errors="replace"), request.truncation
                    )
                    return TerminalResponse(
                        stdout=response_stdout,
                        stderr=b"".join(stderr_parts).decode(errors="replace"),
                        success=False,
                        exit_code=process.returncode,
                        execution_time=time.perf_counter() - started,
                        truncated=truncated,
                        truncated_characters=missing,
                    )
                data = stdout

        response_stdout, truncated, missing = self._truncate_stdout(
            data.decode(errors="replace"), request.truncation
        )
        return TerminalResponse(
            stdout=response_stdout,
            stderr=b"".join(stderr_parts).decode(errors="replace"),
            success=True,
            exit_code=0,
            execution_time=time.perf_counter() - started,
            truncated=truncated,
            truncated_characters=missing,
        )

    async def start(self) -> None:
        await self.registry.register_server(self.url, self.config.port, self._metadata())
        self.should_run = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while self.should_run:
            try:
                await self.registry.heartbeat(self.url, self.config.port)
            except Exception:
                pass
            await asyncio.sleep(self.config.heartbeat_interval)

    async def cleanup_async(self) -> None:
        self.should_run = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.registry.deregister()

    def _install_routes(self) -> None:
        @self.app.exception_handler(RequestValidationError)
        async def validation_error_handler(
            request: Request, exc: RequestValidationError
        ) -> JSONResponse:
            summary = _request_summary(getattr(exc, "body", None))
            logger.info("Invalid terminal request: %s", summary)
            return JSONResponse(
                {
                    "error": "invalid terminal request",
                    "details": exc.errors(),
                    "request": summary,
                },
                status_code=400,
            )

        @self.app.post("/terminal", response_model=TerminalResponse)
        async def terminal(request: TerminalRequest) -> TerminalResponse:
            summary = _request_summary(request.dict())
            try:
                response = await self.execute(request)
                logger.info(
                    "Completed terminal request success=%s exit_code=%s execution_time=%.3fs",
                    response.success,
                    response.exit_code,
                    response.execution_time,
                )
                return response
            except PipelineValidationError as exc:
                logger.info("Rejected terminal request: %s; reason=%s", summary, exc)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": str(exc),
                        "request": _request_summary(request.dict()),
                    },
                ) from exc
            except PipelineLimitError as exc:
                logger.info("Limited terminal request: %s; reason=%s", summary, exc)
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            except asyncio.TimeoutError as exc:
                logger.info("Timed out terminal request: %s", summary)
                raise HTTPException(status_code=408, detail="pipeline timed out") from exc

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "message": "POST /terminal with contents and command",
                "commands": sorted(_ALLOWED_COMMANDS),
                "max_output_bytes": self.config.max_output_bytes,
                "max_stderr_bytes": self.config.max_stderr_bytes,
                "max_response_chars": self._max_response_chars,
            }

        @self.app.on_event("startup")
        async def startup() -> None:
            await self.start()

        @self.app.on_event("shutdown")
        async def shutdown() -> None:
            await self.cleanup_async()


def main(
    host: str = "0.0.0.0",
    port: int = 1213,
    registry: str = "redis://klone-login01.hyak.local:6379",
    heartbeat_interval: int = 30,
    max_output_bytes: int = 1024 * 1024,
    max_response_chars: int | None = None,
    command_path: str | None = None,
) -> None:
    """Run the restricted terminal pipeline server with uvicorn."""
    import uvicorn

    config = TerminalServerConfig(
        host=host,
        port=port,
        registry=registry,
        heartbeat_interval=heartbeat_interval,
        max_output_bytes=max_output_bytes,
        max_response_chars=max_response_chars,
        command_path=command_path,
    )
    uvicorn.run(TerminalPipelineServer(config).app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
