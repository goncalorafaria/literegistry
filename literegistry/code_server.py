## this code was mostly taken from RLM and DrTulu.

import ast
import asyncio
import copy
import importlib
import io
import logging
import math
import os
import signal
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from literegistry import get_kvstore, ServerRegistry
from fastapi import FastAPI
from pydantic import BaseModel, Field, root_validator, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Safe Builtins
# =============================================================================

_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "__build_class__": __build_class__,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked: import/open/eval/exec are deliberately unavailable to user code.
    "__import__": None,
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
    "open": None,
}

_RESERVED_TOOL_NAMES = frozenset(
    {
        "FINAL_VAR",
        "SHOW_VARS",
        "context",
        "context_0",
        "history",
        "__builtins__",
    }
)


# Tools are server-side allowlisted because HTTP requests cannot safely send Python functions.
DEFAULT_TOOL_SPECS: dict[str, str] = {
    "_strptime": "_strptime",
    "abc": "abc",
    "argparse": "argparse",
    "ast": "ast",
    "base64": "base64",
    "binascii": "binascii",
    "bisect": "bisect",
    "calendar": "calendar",
    "collections": "collections",
    "copy": "copy",
    "cmath": "cmath",
    "csv": "csv",
    "datetime": "datetime",
    "decimal": "decimal",
    "difflib": "difflib",
    "fnmatch": "fnmatch",
    "functools": "functools",
    "hashlib": "hashlib",
    "heapq": "heapq",
    "hmac": "hmac",
    "html": "html",
    "html.parser": "html.parser",
    "ipaddress": "ipaddress",
    "itertools": "itertools",
    "json": "json",
    "keyword": "keyword",
    "locale": "locale",
    "lxml": "lxml",
    "math": "math",
    "numbers": "numbers",
    "numpy": "numpy",
    "operator": "operator",
    "pandas": "pandas",
    "platform": "platform",
    "pprint": "pprint",
    "pytz": "pytz",
    "random": "random",
    "re": "re",
    "scipy": "scipy",
    "secrets": "secrets",
    "shlex": "shlex",
    "string": "string",
    "struct": "struct",
    "sympy": "sympy",
    "textwrap": "textwrap",
    "time": "time",
    "tqdm": "tqdm",
    "traceback": "traceback",
    "types": "types",
    "typing": "typing",
    "unidecode": "unidecode",
    "unicodedata": "unicodedata",
    "warnings": "warnings",
    "yaml": "yaml",
    "zoneinfo": "zoneinfo",
    "zlib": "zlib",
    "statistics": "statistics",
    "mean": "statistics:mean",
    "median": "statistics:median",
}


def _max_int_str_digits() -> int:
    """Match Python's JSON int string safety limit (see :func:`sys.get_int_max_str_digits`)."""
    try:
        lim = sys.get_int_max_str_digits()
    except AttributeError:
        return 4300
    return lim if lim != 0 else 10**9


def _int_too_large_for_json_serialization(value: int) -> bool:
    if value == 0:
        return False
    lim = _max_int_str_digits()
    if lim >= 10**8:
        return False
    # Upper bound on decimal digit count without converting the int to str.
    digit_upper = int(abs(value).bit_length() * math.log10(2.0)) + 2
    return digit_upper > lim


def _serialize_value(value: Any) -> Any:
    """Convert local variables to JSON-friendly values."""
    if value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return "TOOBIG" if _int_too_large_for_json_serialization(value) else value
    if isinstance(value, float):
        if not math.isfinite(value):
            return "TOOBIG"
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _parse_csv(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_tool(name: str, tool_specs: dict[str, str]) -> Any:
    """Resolve an allowlisted tool name to a module or object."""
    if name in _RESERVED_TOOL_NAMES:
        raise ValueError(f"Tool name '{name}' is reserved")
    if name not in tool_specs:
        raise ValueError(
            f"Tool '{name}' is not allowlisted. Available: {sorted(tool_specs)}"
        )

    spec = tool_specs[name]
    if ":" not in spec:
        return importlib.import_module(spec)

    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _preimport_tools(tool_names: list[str], tool_specs: dict[str, str]) -> None:
    """Import common modules before worker fork so startup cost is paid once."""
    for name in tool_names:
        try:
            _resolve_tool(name, tool_specs)
            logger.info("Pre-imported tool: %s", name)
        except Exception as exc:
            logger.warning("Could not pre-import tool %s: %s", name, exc)


def _worker_init(preimport_tools: list[str], tool_specs: dict[str, str]) -> None:
    """Run in each worker process."""
    for name in preimport_tools:
        try:
            _resolve_tool(name, tool_specs)
        except Exception:
            pass


def _allowed_import_roots(tool_specs: dict[str, str]) -> frozenset[str]:
    """Top-level packages users may ``import`` / ``import from`` (matches tool_specs)."""
    roots: set[str] = set()
    for alias, spec in tool_specs.items():
        roots.add(alias.partition(".")[0])
        roots.add(spec.split(":")[0].partition(".")[0])
    return frozenset(roots)


def _make_restricted_import(tool_specs: dict[str, str]):
    """``builtins.__import__`` that only loads allowlisted top-level packages."""
    allowed = _allowed_import_roots(tool_specs)

    def __import__(name: str, _globals=None, _locals=None, fromlist=(), level=0):
        del _globals, _locals, fromlist
        if level != 0:
            raise ImportError("relative imports are not allowed")
        if not isinstance(name, str) or not name.strip():
            raise ImportError(f"invalid module name: {name!r}")
        stripped = name.strip()
        root = stripped.partition(".")[0]
        if root != "__future__" and root not in allowed:
            raise ImportError(
                f"import of top-level package {root!r} is not allowed; "
                f"allowed: {sorted(allowed)} plus __future__"
            )
        return importlib.import_module(stripped)

    return __import__


def _validate_allowlisted_imports(
    code: str, label: str, tool_specs: dict[str, str]
) -> None:
    """Reject imports from modules outside the server's tool_specs allowlist."""
    allowed = _allowed_import_roots(tool_specs)
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                if top == "__future__":
                    continue
                if top not in allowed:
                    raise ValueError(
                        f"{label} cannot import {alias.name!r}; only server-allowlisted "
                        f"modules are permitted (got top-level {top!r})."
                    )
            continue

        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise ValueError(f"{label} cannot use relative imports.")
            module = node.module
            if module is None:
                raise ValueError(f"{label} has an invalid import-from statement.")
            if module.split(".")[0] == "__future__":
                continue
            top = module.partition(".")[0]
            if top not in allowed:
                raise ValueError(
                    f"{label} cannot import from {module!r}; only server-allowlisted "
                    f"modules are permitted (got top-level {top!r})."
                )


def _final_var_factory(locals_dict: dict[str, Any], final_answer: list[str | None]):
    def _final_var(variable_name: str | Any) -> str:
        if not isinstance(variable_name, str):
            answer = str(variable_name)
            final_answer[0] = answer
            return answer

        variable_name = variable_name.strip().strip("\"'")
        if variable_name in locals_dict:
            answer = str(locals_dict[variable_name])
            final_answer[0] = answer
            return answer

        available = [k for k in locals_dict if not k.startswith("_")]
        return f"Error: Variable '{variable_name}' not found. Available variables: {available}."

    return _final_var


def _show_vars_factory(locals_dict: dict[str, Any]):
    def _show_vars() -> str:
        available = {
            k: type(v).__name__ for k, v in locals_dict.items() if not k.startswith("_")
        }
        if not available:
            return "No variables created yet."
        return f"Available variables: {available}"

    return _show_vars


def _run_user_code(
    code: str,
    max_runtime: float = 5.0,
    context_payload: Any = None,
    setup_code: str | None = None,
    custom_tools: list[str] | None = None,
    return_locals: bool = True,
    default_tools: list[str] | None = None,
    tool_specs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute one stateless request inside a worker process."""

    def _timeout_handler(signum, frame):  # noqa: ARG001
        raise TimeoutError("execution timed out")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max_runtime)

    stdout, stderr = io.StringIO(), io.StringIO()
    start_time = time.perf_counter()
    final_answer: list[str | None] = [None]
    namespace: dict[str, Any] = {}
    keys_before_exec: set[str] = set()

    try:
        resolved_tool_specs = tool_specs or DEFAULT_TOOL_SPECS
        _validate_allowlisted_imports(code, "code", resolved_tool_specs)
        if setup_code:
            _validate_allowlisted_imports(setup_code, "setup_code", resolved_tool_specs)

        tool_names = custom_tools if custom_tools is not None else default_tools or []
        safe_builtins = _SAFE_BUILTINS.copy()
        safe_builtins["__import__"] = _make_restricted_import(resolved_tool_specs)
        # Single namespace: separate globals/locals break cross-function calls
        # (defs bind to locals, lookups inside functions use globals).
        namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }

        namespace["context_0"] = copy.deepcopy(context_payload)
        namespace["context"] = namespace["context_0"]

        for name in tool_names:
            namespace[name] = _resolve_tool(name, resolved_tool_specs)

        namespace["FINAL_VAR"] = _final_var_factory(namespace, final_answer)
        namespace["SHOW_VARS"] = _show_vars_factory(namespace)

        keys_before_exec = set(namespace)

        with redirect_stdout(stdout), redirect_stderr(stderr):
            if setup_code:
                exec(compile(setup_code, "<setup-code>", "exec"), namespace)
            exec(compile(code, "<user-code>", "exec"), namespace)

        success = True
        error = stderr.getvalue() or None
    except Exception as exc:
        success = False
        error = f"{exc}\n{traceback.format_exc()}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)

    result_locals = {}
    if return_locals:
        result_locals = {
            key: _serialize_value(namespace[key])
            for key in namespace
            if key not in keys_before_exec and not key.startswith("_")
        }

    return {
        "stdout": stdout.getvalue(),
        "stderr": error,
        "success": success,
        "locals": result_locals,
        "execution_time": time.perf_counter() - start_time,
        "final_answer": final_answer[0],
    }


class CodeRequest(BaseModel):
    code: str
    max_runtime: float = Field(default=5.0, ge=0.01)
    context_payload: Any = None
    setup_code: str | None = None
    custom_tools: list[str] | None = None
    return_locals: bool = True

    @root_validator(pre=True)
    def accept_legacy_runtime_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "max_runtime" not in values:
            if "code_timeout" in values:
                values["max_runtime"] = values["code_timeout"]
            elif "timeout" in values:
                values["max_runtime"] = values["timeout"]
        return values

    @validator("max_runtime", pre=True)
    def parse_max_runtime(cls, value: Any) -> float:
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped.endswith("ms"):
                return float(stripped[:-2].strip()) / 1000.0
            if stripped.endswith("s"):
                return float(stripped[:-1].strip())
        return value


class CodeResponse(BaseModel):
    stdout: str
    stderr: str | None = None
    success: bool
    locals: dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    final_answer: str | None = None


@dataclass
class StatelessCodeExecutorConfig:
    """Configuration for the stateless FastAPI code executor."""

    title: str = "Stateless Python Code Executor"
    default_tools: list[str] = field(
        default_factory=lambda: [
            "json",
            "math",
            "time",
            "re",
            "pandas",
            "numpy",
            "sympy",
            "datetime",
            "functools",
            "random",
            "typing",
        ]
    )
    preimport_tools: list[str] = field(default_factory=lambda: list(DEFAULT_TOOL_SPECS))
    tool_specs: dict[str, str] = field(
        default_factory=lambda: DEFAULT_TOOL_SPECS.copy()
    )
    pool_size: int = field(default_factory=lambda: os.cpu_count() or 1)
    outer_timeout_grace_seconds: int = 1
    host: str = "0.0.0.0"
    port: int = 1212
    backend: str = "repl"
    registry: str = "redis://klone-login01.hyak.local:6379"
    max_history: int = 3600
    heartbeat_interval: int = 30


class StatelessCodeExecutorServer:
    """FastAPI server wrapper for stateless Python code execution."""

    def __init__(self, config: StatelessCodeExecutorConfig | None = None):
        self.config = config or StatelessCodeExecutorConfig()

        self.registry = ServerRegistry(
            store=get_kvstore(self.config.registry),
            max_history=self.config.max_history,
        )
        self.url = f"http://{socket.getfqdn()}"
        self.should_run = False
        self.heartbeat_task: asyncio.Task | None = None

        _preimport_tools(self.config.preimport_tools, self.config.tool_specs)
        self.process_pool: ProcessPoolExecutor | None = self._new_pool()
        self.app = FastAPI(title=self.config.title)
        self._install_routes()
        logger.info("Process pool started with %s workers", self.config.pool_size)

    def _new_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.config.pool_size,
            initializer=_worker_init,
            initargs=(self.config.preimport_tools, self.config.tool_specs),
        )

    def _registry_metadata(self) -> dict[str, Any]:
        return {
            "model_path": "python",
            "host": self.config.host,
            "port": self.config.port,
            "backend": self.config.backend,
            "extra_kwargs": {
                "available_tools": sorted(self.config.tool_specs),
                "default_tools": self.config.default_tools,
                "preimport_tools": self.config.preimport_tools,
                "pool_size": self.config.pool_size,
                "outer_timeout_grace_seconds": self.config.outer_timeout_grace_seconds,
                "heartbeat_interval": self.config.heartbeat_interval,
            },
        }

    async def register(self) -> None:
        """Register this FastAPI code executor in literegistry."""
        await self.registry.register_server(
            url=self.url,
            port=self.config.port,
            metadata=self._registry_metadata(),
        )
        logger.info("Registered server at %s:%s", self.url, self.config.port)

    def check_health(self) -> bool:
        """Return whether this server is healthy enough to heartbeat."""
        return self.should_run and self.process_pool is not None

    async def heartbeat_loop(self):
        """Run heartbeat in a loop."""
        while self.should_run:
            if self.check_health():
                try:
                    await self.registry.heartbeat(self.url, self.config.port)
                except Exception:
                    logger.exception("Heartbeat failed")
            else:
                print("Server unhealthy!")
            await asyncio.sleep(self.config.heartbeat_interval)

    async def start(self) -> None:
        """Register the server and start the heartbeat task."""
        if self.should_run:
            return

        await self.register()
        self.should_run = True
        self.heartbeat_task = asyncio.create_task(self.heartbeat_loop())

    async def cleanup_async(self) -> None:
        """Clean up resources."""
        self.should_run = False
        if self.heartbeat_task is not None:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None

        try:
            await self.registry.deregister()
        except Exception:
            logger.exception("Failed to deregister server")

        if self.process_pool is not None:
            self.process_pool.shutdown(cancel_futures=True)
            self.process_pool = None
        print("Server stopped and deregistered")

    def cleanup(self) -> None:
        """Sync cleanup wrapper for non-FastAPI callers."""
        asyncio.run(self.cleanup_async())

    async def execute_stateless_code(
        self,
        code: str,
        max_runtime: float = 5.0,
        context_payload: Any = None,
        setup_code: str | None = None,
        custom_tools: list[str] | None = None,
        return_locals: bool = True,
    ) -> CodeResponse:
        """Programmatic API matching the FastAPI endpoint behavior."""
        if self.process_pool is None:
            self.process_pool = self._new_pool()

        start = time.perf_counter()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self.process_pool,
            _run_user_code,
            code,
            max_runtime,
            context_payload,
            setup_code,
            custom_tools,
            return_locals,
            self.config.default_tools,
            self.config.tool_specs,
        )

        try:
            result = await asyncio.wait_for(
                future,
                timeout=max_runtime + self.config.outer_timeout_grace_seconds,
            )
        except asyncio.TimeoutError:
            future.cancel()
            result = {
                "stdout": "",
                "stderr": f"Execution timed out after {max_runtime} seconds (outer)",
                "success": False,
                "locals": {},
                "execution_time": time.perf_counter() - start,
                "final_answer": None,
            }
        except BrokenProcessPool:
            logger.error("Worker pool broken; recreating")
            self.process_pool = self._new_pool()
            result = {
                "stdout": "",
                "stderr": "Worker pool crashed and was restarted. Please retry.",
                "success": False,
                "locals": {},
                "execution_time": time.perf_counter() - start,
                "final_answer": None,
            }

        return CodeResponse(**result)

    def shutdown(self) -> None:
        """Shut down the worker pool."""
        if self.process_pool is not None:
            self.process_pool.shutdown(cancel_futures=True)
            self.process_pool = None

    def _install_routes(self) -> None:
        @self.app.post("/python", response_model=CodeResponse)
        async def execute_code(req: CodeRequest) -> CodeResponse:
            truncated_code = (
                (req.code[:197] + "...") if len(req.code) > 200 else req.code
            )
            logger.info(
                "Received execute request max_runtime=%ss tools=%s code=%r",
                req.max_runtime,
                (
                    req.custom_tools
                    if req.custom_tools is not None
                    else self.config.default_tools
                ),
                truncated_code,
            )

            response = await self.execute_stateless_code(
                code=req.code,
                max_runtime=req.max_runtime,
                context_payload=req.context_payload,
                setup_code=req.setup_code,
                custom_tools=req.custom_tools,
                return_locals=req.return_locals,
            )
            logger.info(
                "Responding success=%s execution_time=%.3fs stdout_len=%s stderr=%s",
                response.success,
                response.execution_time,
                len(response.stdout),
                bool(response.stderr),
            )
            return response

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "message": "POST /python with code, max_runtime, context_payload, setup_code, custom_tools",
                "available_tools": sorted(self.config.tool_specs),
                "default_tools": self.config.default_tools,
                "preimport_tools": self.config.preimport_tools,
                "pool_size": self.config.pool_size,
                "registry_url": self.url,
                "registry_port": self.config.port,
                "heartbeat_interval": self.config.heartbeat_interval,
            }

        @self.app.on_event("startup")
        async def startup_registry() -> None:
            await self.start()

        @self.app.on_event("shutdown")
        async def shutdown_pool() -> None:
            await self.cleanup_async()


def create_app(config: StatelessCodeExecutorConfig | None = None) -> FastAPI:
    """Create a configured FastAPI app."""
    return StatelessCodeExecutorServer(config).app


def main(
    host: str = "0.0.0.0",
    port: int = 1212,
    pool_size: int | None = None,
    default_tools: str | list[str] = (
        "json,math,time,re,pandas,numpy,sympy,datetime,functools,random,typing"
    ),
    preimport_tools: str | list[str] = ",".join(DEFAULT_TOOL_SPECS),
    title: str = "Stateless Python Code Executor",
    outer_timeout_grace_seconds: int = 1,
    log_level: str = "INFO",
    registry: str = "redis://klone-login01.hyak.local:6379",
    heartbeat_interval: int = 4,
) -> None:
    """Run the server with uvicorn."""
    import uvicorn

    logging.getLogger().setLevel(log_level.upper())
    config = StatelessCodeExecutorConfig(
        title=title,
        default_tools=_parse_csv(default_tools),
        preimport_tools=_parse_csv(preimport_tools),
        pool_size=pool_size or os.cpu_count() or 1,
        outer_timeout_grace_seconds=outer_timeout_grace_seconds,
        host=host,
        port=port,
        registry=registry,
        heartbeat_interval=heartbeat_interval,
    )
    server = StatelessCodeExecutorServer(config)
    uvicorn.run(
        server.app,
        host=config.host,
        port=config.port,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
