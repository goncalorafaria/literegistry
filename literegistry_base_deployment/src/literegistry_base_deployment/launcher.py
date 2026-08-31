"""Render and submit a standalone LiteRegistry base stack on Beaker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4


RESULT = {"path": "/tmp/result"}
RESTORE_VENV_PATH = (
    'for LR_VENV in "${VIRTUAL_ENV:-}" /opt/literegistry-services-venv '
    '/opt/literegistry-terminal-venv /opt/literegistry-redis-venv; do '
    'if [[ -n "$LR_VENV" && -x "$LR_VENV/bin/python3" ]]; then '
    'export VIRTUAL_ENV="$LR_VENV" PATH="$LR_VENV/bin:$PATH"; break; fi; done; '
    'unset LR_VENV; '
)
PORT_BLOCK_START = 20000
PORTS_PER_SERVICE = 128
SERVICE_SLOTS = {
    "redis": 0,
    "gateway": 1,
    "python": 2,
    "terminal": 3,
    "search": 4,
    "localsearch": 5,
    "vllm-generate": 6,
    "vllm-classify": 7,
}


def _clusters(value: str | Sequence[str]) -> tuple[str, ...]:
    result = (value,) if isinstance(value, str) else tuple(value)
    if not result or any(not cluster.strip() for cluster in result):
        raise ValueError("service_clusters must contain at least one non-empty cluster")
    return result


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "service"


@dataclass(frozen=True)
class ModelPoolConfig:
    """One vLLM model pool registered under its model path."""

    model: str
    mode: str = "generate"
    replicas: int = 1
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    max_num_batched_tokens: int = 32768
    max_num_seqs: int | None = None

    def validate(self) -> "ModelPoolConfig":
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.mode not in {"generate", "classify"}:
            raise ValueError("model mode must be generate or classify")
        for name in (
            "replicas",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_batched_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_num_seqs is not None and self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive when supplied")
        return self


@dataclass(frozen=True)
class BaseDeploymentConfig:
    """Configuration for gateway, Redis, tools, search, and vLLM."""

    registry: str | None = None
    python_replicas: int = 1
    terminal_replicas: int = 1
    web_search_replicas: int = 1
    local_search_replicas: int = 0
    local_search_corpus_jsonl: str | None = None
    local_search_index_dir: str | None = None
    local_search_service_name: str | None = None
    generation_model: str | None = None
    generation_replicas: int = 0
    generation_tp: int = 1
    classification_model: str | None = None
    classification_replicas: int = 0
    classification_tp: int = 1
    max_model_len: int = 32768
    max_num_batched_tokens: int = 32768
    max_num_seqs: int | None = None
    gateway_workers: int = 8
    docker_mirror_soft_affinity: bool = True
    python_pool_size: int = 32
    service_clusters: tuple[str, ...] | str = ("ai2/jupiter",)
    gateway_cluster: str | None = None
    redis_cluster: str | None = None
    model_cluster: str = "ai2/jupiter"
    workspace: str = "ai2/oe-agents"
    budget: str = "ai2/oe-omai"
    service_priority: str = "normal"
    model_priority: str = "high"
    min_runtime_hours: int = 0
    omit_service_resources: bool = False
    name_prefix: str = "literegistry-base"
    services_image: str = "goncalof/literegistry-base-services"
    terminal_image: str = "goncalof/literegistry-base-terminal"
    local_search_image: str = "goncalof/jtc-local-search-lucene-bm25"
    vllm_image: str = "goncalof/literegistry-base-vllm"
    redis_image: str = "goncalof/literegistry-redis"
    serper_api_key_secret: str | None = "SERPER_API_KEY"
    jina_api_key_secret: str | None = "JINA_API_KEY"
    hf_token_secret: str | None = "HF_TOKEN"
    hf_home: str | None = None
    gateway_timeout: float = 300.0
    registry_cache_ttl_seconds: int = 5
    shared_dir: str = "/weka/gfaria"
    weka_source: str = "oe-adapt-default"

    def resolved_service_clusters(self) -> tuple[str, ...]:
        return _clusters(self.service_clusters)

    def resolved_gateway_cluster(self) -> str:
        return self.gateway_cluster or self.resolved_service_clusters()[0]

    def resolved_redis_cluster(self) -> str:
        return self.redis_cluster or self.resolved_gateway_cluster()

    def model_pools(self) -> tuple[ModelPoolConfig, ...]:
        pools: list[ModelPoolConfig] = []
        if self.generation_replicas:
            pools.append(
                ModelPoolConfig(
                    model=self.generation_model or "",
                    mode="generate",
                    replicas=self.generation_replicas,
                    tensor_parallel_size=self.generation_tp,
                    max_model_len=self.max_model_len,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    max_num_seqs=self.max_num_seqs,
                ).validate()
            )
        if self.classification_replicas:
            pools.append(
                ModelPoolConfig(
                    model=self.classification_model or "",
                    mode="classify",
                    replicas=self.classification_replicas,
                    tensor_parallel_size=self.classification_tp,
                    max_model_len=self.max_model_len,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    max_num_seqs=self.max_num_seqs,
                ).validate()
            )
        return tuple(pools)

    def validate(self) -> "BaseDeploymentConfig":
        if self.registry is not None and not re.fullmatch(r"rediss?://\S+", self.registry):
            raise ValueError("registry must be a redis:// or rediss:// URL when supplied")
        for name in (
            "python_replicas",
            "terminal_replicas",
            "web_search_replicas",
            "local_search_replicas",
            "generation_replicas",
            "classification_replicas",
        ):
            value = getattr(self, name)
            if value < 0 or value > PORTS_PER_SERVICE:
                raise ValueError(f"{name} must be between 0 and {PORTS_PER_SERVICE}")
        if self.gateway_workers < 1 or self.python_pool_size < 1:
            raise ValueError("gateway_workers and python_pool_size must be positive")
        if self.min_runtime_hours < 0:
            raise ValueError("min_runtime_hours must be non-negative")
        if self.gateway_timeout <= 0 or self.registry_cache_ttl_seconds < 1:
            raise ValueError("gateway timeout and registry cache TTL must be positive")
        if self.service_priority not in {"normal", "high", "urgent"}:
            raise ValueError("service_priority must be normal, high, or urgent")
        if self.model_priority not in {"normal", "high", "urgent"}:
            raise ValueError("model_priority must be normal, high, or urgent")
        if self.local_search_replicas and not (
            self.local_search_corpus_jsonl and self.local_search_index_dir
        ):
            raise ValueError(
                "local_search_corpus_jsonl and local_search_index_dir are required "
                "when local_search_replicas is non-zero"
            )
        if self.generation_replicas and not self.generation_model:
            raise ValueError("generation_model is required when generation_replicas is non-zero")
        if self.classification_replicas and not self.classification_model:
            raise ValueError(
                "classification_model is required when classification_replicas is non-zero"
            )
        if self.web_search_replicas and not (
            self.serper_api_key_secret and self.jina_api_key_secret
        ):
            raise ValueError("web search requires Serper and Jina Beaker secret names")
        for name in (
            "services_image",
            "terminal_image",
            "local_search_image",
            "vllm_image",
            "redis_image",
            "model_cluster",
            "workspace",
            "budget",
            "name_prefix",
            "shared_dir",
            "weka_source",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        self.resolved_service_clusters()
        self.model_pools()
        return self


def _service_port(experiment_name: str, service: str) -> int:
    block_size = PORTS_PER_SERVICE * len(SERVICE_SLOTS)
    block_count = (65535 - PORT_BLOCK_START + 1) // block_size
    digest = hashlib.sha256(experiment_name.encode()).hexdigest()[:12]
    bucket = int(digest, 16) % block_count
    return PORT_BLOCK_START + bucket * block_size + SERVICE_SLOTS[service] * PORTS_PER_SERVICE


def _spread(replicas: int, clusters: Sequence[str]) -> list[tuple[str, int, int]]:
    quotient, remainder = divmod(replicas, len(clusters))
    result: list[tuple[str, int, int]] = []
    offset = 0
    for index, cluster in enumerate(clusters):
        count = quotient + int(index < remainder)
        if count:
            result.append((cluster, count, offset))
            offset += count
    return result


def _dynamic_port_command(
    experiment_name: str,
    service: str,
    child_command: str,
    *,
    rank_offset: int = 0,
) -> str:
    preferred = _service_port(experiment_name, service)
    command_json = json.dumps(
        ["bash", "-lc", RESTORE_VENV_PATH + child_command], separators=(",", ":")
    )
    return (
        f"LR_GLOBAL_RANK=$(({rank_offset} + ${{BEAKER_REPLICA_RANK:-0}})); "
        f"LR_PREFERRED_PORT=$(({preferred} + LR_GLOBAL_RANK)); "
        "export LR_GLOBAL_RANK LR_PREFERRED_PORT; "
        "exec python3 -m literegistry.coop.ports run "
        ' --assignment "PORT=${LR_PREFERRED_PORT}"'
        f' --identity "{experiment_name}:{service}:${{LR_GLOBAL_RANK}}"'
        ' --host-id "${BEAKER_NODE_HOSTNAME:-unknown-node}"'
        f" --command-json={shlex.quote(command_json)}"
        " --lock_dir=/tmp/literegistry-port-locks"
    )


class BaseDeploymentLauncher:
    """Build, preview, submit, and stop standalone base-stack experiments."""

    def __init__(self, config: BaseDeploymentConfig) -> None:
        self.config = config.validate()

    def _identity(self) -> str:
        config_hash = hashlib.sha256(repr(self.config).encode()).hexdigest()[:10]
        return (
            f"{_slug(self.config.name_prefix)}"
            f"-py{self.config.python_replicas}-t{self.config.terminal_replicas}"
            f"-s{self.config.web_search_replicas}-ls{self.config.local_search_replicas}"
            f"-g{self.config.generation_replicas}-c{self.config.classification_replicas}"
            f"-{datetime.now(UTC):%Y%m%d-%H%M%S}-{config_hash}-{uuid4().hex[:8]}"
        )

    def _task(
        self,
        name: str,
        image: str,
        command: str,
        *,
        cluster: str,
        replicas: int = 1,
        env_vars: Sequence[Mapping[str, str]] = (),
        gpu_count: int = 0,
        model_task: bool = False,
        critical: bool = False,
    ) -> dict[str, Any]:
        task: dict[str, Any] = {
            "name": name,
            "image": {"beaker": image},
            "command": ["bash", "-lc", RESTORE_VENV_PATH + command],
            "hostNetworking": True,
            "propagateFailure": critical,
            "propagatePreemption": critical,
            "datasets": [
                {"mountPath": "/weka", "source": {"weka": self.config.weka_source}}
            ],
            "result": RESULT,
            "context": {
                "priority": self.config.model_priority if model_task else self.config.service_priority,
                "minRuntime": f"{self.config.min_runtime_hours}h",
                "autoResume": not critical,
            },
            "constraints": {"cluster": [cluster]},
        }
        if replicas != 1:
            task["replicas"] = replicas
        if model_task:
            task["resources"] = {"gpuCount": gpu_count}
        elif not self.config.omit_service_resources:
            # Never set cpuCount: on GPU-shaped Beaker clusters a CPU request
            # can select a GPU worker even when gpuCount is zero.
            task["resources"] = {"gpuCount": 0}
        if env_vars:
            task["envVars"] = list(env_vars)
        return task

    def _registry_commands(self, name: str, tasks: list[dict[str, Any]]) -> str:
        registry_url_file = f"{self.config.shared_dir.rstrip('/')}/literegistry_base_registry_{name}.url"
        if self.config.registry is None:
            redis_child = (
                'REDIS_ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
                f"REDIS_URL_FILE={shlex.quote(registry_url_file)}; "
                'REDIS_URL="redis://${REDIS_ADVERTISE_HOST}:${PORT}"; '
                'REDIS_URL_TMP="${REDIS_URL_FILE}.${BEAKER_REPLICA_RANK:-0}.tmp"; '
                'printf "%s\\n" "$REDIS_URL" > "$REDIS_URL_TMP"; '
                'mv "$REDIS_URL_TMP" "$REDIS_URL_FILE"; '
                'echo "REDIS_URL=$REDIS_URL"; '
                'exec literegistry redis --runtime=local --foreground=True --port="$PORT"'
            )
            tasks.append(
                self._task(
                    "redis",
                    self.config.redis_image,
                    "set -euo pipefail; " + _dynamic_port_command(name, "redis", redis_child),
                    cluster=self.config.resolved_redis_cluster(),
                    critical=True,
                )
            )
            setup = (
                f"REGISTRY_URL_FILE={shlex.quote(registry_url_file)}; "
                'for _ in $(seq 1 120); do [[ -s "$REGISTRY_URL_FILE" ]] && break; sleep 5; done; '
                '[[ -s "$REGISTRY_URL_FILE" ]] || { echo registry-url-timeout >&2; exit 1; }; '
                'REGISTRY=$(tail -n 1 "$REGISTRY_URL_FILE" | tr -d "\\r"); export REGISTRY; '
            )
        else:
            setup = f"REGISTRY={shlex.quote(self.config.registry)}; export REGISTRY; "
        return (
            "set -euo pipefail; "
            + setup
            + "python3 -m literegistry.coop.redis wait "
            + '--registry "$REGISTRY" --timeout 600; '
        )

    def _add_spread_service(
        self,
        tasks: list[dict[str, Any]],
        *,
        experiment_name: str,
        name: str,
        service: str,
        image: str,
        replicas: int,
        wait: str,
        child: str,
        env_vars: Sequence[Mapping[str, str]] = (),
    ) -> None:
        groups = _spread(replicas, self.config.resolved_service_clusters())
        for cluster, count, offset in groups:
            task_name = name if len(groups) == 1 else f"{name}-{_slug(cluster)}"
            command = wait + _dynamic_port_command(
                experiment_name, service, child, rank_offset=offset
            )
            tasks.append(
                self._task(
                    task_name,
                    image,
                    command,
                    cluster=cluster,
                    replicas=count,
                    env_vars=env_vars,
                )
            )

    def build_spec(self, *, experiment_name: str | None = None) -> tuple[str, dict[str, Any]]:
        name = experiment_name or self._identity()
        tasks: list[dict[str, Any]] = []
        wait = self._registry_commands(name, tasks)

        gateway_url_file = f"{self.config.shared_dir.rstrip('/')}/literegistry_base_gateway_{name}.url"
        gateway_child = (
            'GATEWAY_ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
            f"GATEWAY_URL_FILE={shlex.quote(gateway_url_file)}; "
            'GATEWAY_URL="http://${GATEWAY_ADVERTISE_HOST}:${PORT}"; '
            'GATEWAY_URL_TMP="${GATEWAY_URL_FILE}.${BEAKER_REPLICA_RANK:-0}.tmp"; '
            'printf "%s\\n" "$GATEWAY_URL" > "$GATEWAY_URL_TMP"; '
            'mv "$GATEWAY_URL_TMP" "$GATEWAY_URL_FILE"; '
            'echo "GATEWAY_URL=$GATEWAY_URL"; '
            'exec python -m literegistry.gateway --registry="$REGISTRY" --port="$PORT" '
            '--advertise_host="$GATEWAY_ADVERTISE_HOST" '
            f"--workers={self.config.gateway_workers} "
            f"--registry_cache_ttl_seconds={self.config.registry_cache_ttl_seconds} "
            f"--timeout={self.config.gateway_timeout:g} "
            f"--docker_mirror_soft_affinity={self.config.docker_mirror_soft_affinity}"
        )
        tasks.append(
            self._task(
                "gateway",
                self.config.services_image,
                wait + _dynamic_port_command(name, "gateway", gateway_child),
                cluster=self.config.resolved_gateway_cluster(),
            )
        )

        if self.config.python_replicas:
            self._add_spread_service(
                tasks,
                experiment_name=name,
                name="python",
                service="python",
                image=self.config.services_image,
                replicas=self.config.python_replicas,
                wait=wait,
                child=(
                    'exec literegistry code --host=0.0.0.0 --port="$PORT" '
                    f'--pool_size={self.config.python_pool_size} --registry="$REGISTRY"'
                ),
            )
        if self.config.terminal_replicas:
            self._add_spread_service(
                tasks,
                experiment_name=name,
                name="terminal",
                service="terminal",
                image=self.config.terminal_image,
                replicas=self.config.terminal_replicas,
                wait=wait,
                child='exec literegistry terminal --host=0.0.0.0 --port="$PORT" --registry="$REGISTRY"',
            )
        if self.config.web_search_replicas:
            self._add_spread_service(
                tasks,
                experiment_name=name,
                name="search",
                service="search",
                image=self.config.services_image,
                replicas=self.config.web_search_replicas,
                wait=wait,
                child=(
                    'exec literegistry search --host=0.0.0.0 --port="$PORT" '
                    '--registry="$REGISTRY" --cache_db=1 --cache_ttl=3600'
                ),
                env_vars=(
                    {"name": "SERPER_API_KEY", "secret": self.config.serper_api_key_secret or ""},
                    {"name": "JINA_API_KEY", "secret": self.config.jina_api_key_secret or ""},
                ),
            )
        if self.config.local_search_replicas:
            corpus = self.config.local_search_corpus_jsonl or ""
            index = self.config.local_search_index_dir or ""
            service_name = self.config.local_search_service_name or f"localsearch:{Path(corpus).stem}"
            prepare_index = (
                "if ! compgen -G \"$LUCENE_INDEX_DIR/segments_*\" > /dev/null; then "
                "mkdir -p \"$LUCENE_INDEX_DIR\"; "
                "JTC_BM25_CORPUS_FILE=\"$LOCAL_SEARCH_CORPUS_JSONL\" "
                "/app/search/build_lucene_index.sh "
                "\"$(dirname \"$LOCAL_SEARCH_CORPUS_JSONL\")\" \"$LUCENE_INDEX_DIR\"; fi; "
            )
            local_child = (
                f"LOCAL_SEARCH_CORPUS_JSONL={shlex.quote(corpus)}; "
                f"LUCENE_INDEX_DIR={shlex.quote(index)}; "
                f"LITEREGISTRY_SERVICE_NAME={shlex.quote(service_name)}; "
                "export LOCAL_SEARCH_CORPUS_JSONL LUCENE_INDEX_DIR LITEREGISTRY_SERVICE_NAME; "
                + prepare_index
                + "LITEREGISTRY_REGISTRY=\"$REGISTRY\"; "
                "LITEREGISTRY_PORT=\"$PORT\"; "
                "LITEREGISTRY_ADVERTISED_HOST=\"${BEAKER_NODE_HOSTNAME:-$(hostname -f)}\"; "
                "export LITEREGISTRY_REGISTRY LITEREGISTRY_PORT LITEREGISTRY_ADVERTISED_HOST; "
                "exec python -m uvicorn literegistry.services.bm25_server:create_app "
                "--factory --host 0.0.0.0 --port \"$PORT\""
            )
            self._add_spread_service(
                tasks,
                experiment_name=name,
                name="localsearch",
                service="localsearch",
                image=self.config.local_search_image,
                replicas=self.config.local_search_replicas,
                wait=wait,
                child=local_child,
            )

        for pool in self.config.model_pools():
            service = f"vllm-{pool.mode}"
            max_num_seqs = (
                f" --max-num-seqs={pool.max_num_seqs}" if pool.max_num_seqs is not None else ""
            )
            common = (
                f"exec literegistry vllm --runtime=local --model={shlex.quote(pool.model)} "
                '--registry="$REGISTRY" --host=0.0.0.0 --port="$PORT" '
                f"--tensor-parallel-size={pool.tensor_parallel_size} "
                f"--max-model-len={pool.max_model_len} "
                f"--max-num-batched-tokens={pool.max_num_batched_tokens}"
                f"{max_num_seqs} --enable-prefix-caching --trust-remote-code "
                "--safetensors-load-strategy=prefetch "
            )
            child = (
                common + "--task=classify"
                if pool.mode == "classify"
                else common
                + "--task=generate --mamba-cache-mode=align --reasoning-parser=qwen3 "
                "--enable-auto-tool-choice --tool-call-parser=qwen3_coder --language-model-only"
            )
            env: list[dict[str, str]] = []
            if self.config.hf_token_secret:
                env.append({"name": "HF_TOKEN", "secret": self.config.hf_token_secret})
            if self.config.hf_home:
                env.append({"name": "HF_HOME", "value": self.config.hf_home})
            task_name = f"vllm-{pool.mode}-{_slug(Path(pool.model).name)}"
            tasks.append(
                self._task(
                    task_name,
                    self.config.vllm_image,
                    wait + _dynamic_port_command(name, service, child),
                    cluster=self.config.model_cluster,
                    replicas=pool.replicas,
                    env_vars=env,
                    gpu_count=pool.tensor_parallel_size,
                    model_task=True,
                )
            )

        return name, {
            "version": "v2",
            "description": f"LiteRegistry base deployment: {name}",
            "budget": self.config.budget,
            "tasks": tasks,
        }

    def preview(self) -> dict[str, Any]:
        name, spec = self.build_spec()
        return {"experiment_name": name, "config": asdict(self.config), "spec": spec}

    def submit(self) -> dict[str, Any]:
        name, spec = self.build_spec()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{name}-",
            encoding="utf-8",
            delete=False,
        ) as output:
            json.dump(spec, output, indent=2)
            output.write("\n")
            path = Path(output.name)
        try:
            result = subprocess.run(
                [
                    "beaker",
                    "--format",
                    "json",
                    "experiment",
                    "create",
                    "-n",
                    name,
                    "-w",
                    self.config.workspace,
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return {"experiment_name": name, "beaker": json.loads(result.stdout)}
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or error.stdout.strip() or "no Beaker error detail"
            raise RuntimeError(f"Beaker could not create base deployment {name}: {detail}") from error
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def stop(experiment_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        command = ["beaker", "experiment", "stop", experiment_id]
        if not dry_run:
            subprocess.run(command, check=True)
        return {"command": command, "dry_run": dry_run}
