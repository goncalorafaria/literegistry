"""Render and submit the Podman + Docker mirror + gateway + Redis Beaker stack."""

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

from literegistry.coop.endpoints import (
    normalize_endpoint_registry,
    prepare_endpoint_registry_storage,
)
from literegistry.head_registry import (
    head_registry_backend,
    head_registry_uri,
    is_head_registry_uri,
)

WEKA_MOUNT = {"mountPath": "/weka", "source": {"weka": "oe-adapt-default"}}
RESULT = {"path": "/tmp/result"}
PORT_BLOCK_START = 20000
PORTS_PER_SERVICE = 128
SERVICE_SLOTS = {"podman": 0, "gateway": 1, "docker-mirror": 2, "redis": 3}
RESTORE_VENV_PATH = 'if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python3" ]]; then export PATH="${VIRTUAL_ENV}/bin:${PATH}"; fi; '



def _clusters(value: str | Sequence[str]) -> tuple[str, ...]:
    result = (value,) if isinstance(value, str) else tuple(value)
    if not result or any(not cluster.strip() for cluster in result):
        raise ValueError("service_clusters must contain at least one non-empty cluster")
    return result


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return result or "cluster"


def _head_registry_location(value: str | None) -> str | None:
    if value is None or not is_head_registry_uri(value):
        return None
    return head_registry_backend(value)


def _prepare_shared_directory(path: str) -> None:
    """Create a managed Weka directory writable by the service container UID."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o1777)


@dataclass(frozen=True)
class PodmanStackConfig:
    """Configuration for one independently deployable Beaker stack."""

    registry: str | None = None
    head_registry: str | None = None
    podman_replicas: int = 4
    docker_mirror_replicas: int = 2
    gateway_workers: int = 8
    docker_mirror_soft_affinity: bool = True
    service_clusters: tuple[str, ...] | str = ("ai2/jupiter",)
    gateway_cluster: str | None = None
    workspace: str = "ai2/oe-agents"
    budget: str = "ai2/oe-omai"
    priority: str = "normal"
    min_runtime_hours: int = 0
    # Never request cpuCount: on GPU-shaped Beaker clusters that constraint can
    # select a GPU worker. Explicitly request zero GPUs and leave CPU allocation
    # unconstrained.
    omit_resources: bool = False
    name_prefix: str = "literegistry-podman"
    podman_image: str = "goncalof/literegistry-podman-immediate-rm-20260819"
    podman_instance_prefix: str = "podman"
    podman_session_image: str = "docker.io/library/ubuntu:24.04"
    podman_max_sessions: int | None = None
    podman_session_memory: str | None = "4g"
    podman_session_pids_limit: int | None = 2048
    podman_session_idle_timeout: float | None = 7200.0
    podman_janitor_interval: float = 300.0
    podman_resource_watchdog_interval: float | None = 5.0
    podman_image_prune_until: str | None = "24h"
    docker_mirror_image: str = "goncalof/literegistry-docker-mirror"
    gateway_image: str = "goncalof/literegistry-basic"
    redis_image: str = "goncalof/literegistry-redis"
    redis_cluster: str | None = None
    docker_mirror_storage_root: str = "/var/lib/registry"
    docker_mirror_warm_images_file: str | None = None
    docker_hub_username: str | None = None
    docker_hub_username_secret: str | None = None
    docker_hub_token_secret: str | None = None
    podman_registry_mirror: str | None = None
    affinity_ttl_seconds: float = 900.0
    docker_mirror_affinity_ttl_seconds: float = 604800.0
    registry_cache_ttl_seconds: int = 5
    gateway_timeout: float = 300.0
    weka_source: str = "oe-adapt-default"

    def resolved_service_clusters(self) -> tuple[str, ...]:
        return _clusters(self.service_clusters)

    def resolved_gateway_cluster(self) -> str:
        return self.gateway_cluster or self.resolved_service_clusters()[0]

    def resolved_redis_cluster(self) -> str:
        return self.redis_cluster or self.resolved_gateway_cluster()

    def resolved_head_registry(self) -> str | None:
        if self.head_registry is not None:
            return normalize_endpoint_registry(self.head_registry)
        return _head_registry_location(self.registry)

    def resolved_registry(self) -> str | None:
        head = self.resolved_head_registry()
        return head_registry_uri(head) if head is not None else self.registry

    def validate(self) -> "PodmanStackConfig":
        if self.registry is not None and self.head_registry is not None:
            raise ValueError("supply only one of registry or head_registry")
        if self.registry is not None:
            if is_head_registry_uri(self.registry):
                head_registry_backend(self.registry)
            else:
                normalize_endpoint_registry(self.registry)
        if self.head_registry is not None:
            if not self.head_registry.strip():
                raise ValueError("head_registry must be non-empty when supplied")
            if (
                "://" not in self.head_registry
                and not Path(self.head_registry).expanduser().is_absolute()
            ):
                raise ValueError(
                    "head_registry must be file://, sqlite://, redis://, "
                    "or an absolute shared path"
                )
            normalize_endpoint_registry(self.head_registry)
        if self.podman_replicas < 1 or self.podman_replicas > PORTS_PER_SERVICE:
            raise ValueError(
                f"podman_replicas must be between 1 and {PORTS_PER_SERVICE}"
            )
        if (
            self.docker_mirror_replicas < 0
            or self.docker_mirror_replicas > PORTS_PER_SERVICE
        ):
            raise ValueError(
                "docker_mirror_replicas must be between 0 and "
                f"{PORTS_PER_SERVICE}"
            )
        if self.gateway_workers < 1:
            raise ValueError("gateway_workers must be positive")
        if self.podman_session_memory is not None and not self.podman_session_memory.strip():
            raise ValueError("podman_session_memory must be non-empty when supplied")
        if self.podman_max_sessions is not None and self.podman_max_sessions < 1:
            raise ValueError("podman_max_sessions must be positive when supplied")
        if self.podman_session_pids_limit is not None and self.podman_session_pids_limit < 1:
            raise ValueError("podman_session_pids_limit must be positive when supplied")
        if self.podman_session_idle_timeout is not None and self.podman_session_idle_timeout <= 0:
            raise ValueError("podman_session_idle_timeout must be positive when supplied")
        if self.podman_janitor_interval <= 0:
            raise ValueError("podman_janitor_interval must be positive")
        if (
            self.podman_resource_watchdog_interval is not None
            and self.podman_resource_watchdog_interval <= 0
        ):
            raise ValueError(
                "podman_resource_watchdog_interval must be positive when supplied"
            )
        if self.podman_image_prune_until is not None and not self.podman_image_prune_until.strip():
            raise ValueError("podman_image_prune_until must be non-empty when supplied")
        if self.min_runtime_hours < 0:
            raise ValueError("min_runtime_hours must be non-negative")
        if self.priority not in {"normal", "high", "urgent"}:
            raise ValueError("priority must be normal, high, or urgent")
        for name in (
            "podman_image",
            "podman_instance_prefix",
            "podman_session_image",
            "docker_mirror_image",
            "gateway_image",
            "redis_image",
            "docker_mirror_storage_root",
            "workspace",
            "budget",
            "name_prefix",
            "weka_source",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.docker_mirror_affinity_ttl_seconds <= 0:
            raise ValueError("docker mirror affinity TTL must be positive")
        if self.affinity_ttl_seconds <= 0 or self.gateway_timeout <= 0:
            raise ValueError("affinity TTL and gateway timeout must be positive")
        if self.registry_cache_ttl_seconds < 1:
            raise ValueError("registry_cache_ttl_seconds must be at least one")
        username_sources = int(bool(self.docker_hub_username)) + int(
            bool(self.docker_hub_username_secret)
        )
        if username_sources > 1:
            raise ValueError(
                "supply only one of docker_hub_username or docker_hub_username_secret"
            )
        if bool(username_sources) != bool(self.docker_hub_token_secret):
            raise ValueError(
                "a Docker Hub username source and docker_hub_token_secret "
                "must be supplied together"
            )
        if self.podman_registry_mirror is not None and not re.fullmatch(
            r"https?://[^/\s]+/?", self.podman_registry_mirror
        ):
            raise ValueError("podman_registry_mirror must be an HTTP(S) root URL")
        self.resolved_service_clusters()
        if not self.resolved_gateway_cluster().strip():
            raise ValueError("gateway_cluster must be non-empty")
        if not self.resolved_redis_cluster().strip():
            raise ValueError("redis_cluster must be non-empty")
        return self


def _service_port(experiment_name: str, service: str) -> int:
    block_size = PORTS_PER_SERVICE * len(SERVICE_SLOTS)
    block_count = (65535 - PORT_BLOCK_START + 1) // block_size
    digest = hashlib.sha256(experiment_name.encode()).hexdigest()[:12]
    bucket = int(digest, 16) % block_count
    return PORT_BLOCK_START + bucket * block_size + SERVICE_SLOTS[service] * PORTS_PER_SERVICE


def _spread(replicas: int, clusters: Sequence[str]) -> list[tuple[str, int, int]]:
    """Return cluster, replica count, and global rank offset."""

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
) -> str:
    preferred = _service_port(experiment_name, service)
    command_json = json.dumps(["bash", "-lc", RESTORE_VENV_PATH + child_command], separators=(",", ":"))
    return (
        f"LR_PREFERRED_PORT=$(({preferred} + ${{BEAKER_REPLICA_RANK:-0}})); "
        "export LR_PREFERRED_PORT; "
        "exec python3 -m literegistry.coop.ports run "
        ' --assignment "PORT=${LR_PREFERRED_PORT}"'
        f' --identity "{experiment_name}:{service}:${{BEAKER_REPLICA_RANK:-0}}"'
        ' --host-id "${BEAKER_NODE_HOSTNAME:-unknown-node}"'
        " --lock_dir=/tmp/literegistry-port-locks"
        f" --command-json={shlex.quote(command_json)}"
    )


def _managed_endpoint_command(
    root: str,
    name: str,
    uri_variable: str,
    healthcheck: str,
    child_command: str,
) -> str:
    command_json = json.dumps(
        ["bash", "-lc", RESTORE_VENV_PATH + child_command],
        separators=(",", ":"),
    )
    return (
        "exec python3 -m literegistry.coop.endpoints run "
        f"--root={shlex.quote(root)} --name={shlex.quote(name)} "
        f'--uri="${{{uri_variable}}}" --healthcheck={shlex.quote(healthcheck)} '
        f"--command-json={shlex.quote(command_json)}"
    )


def _wait_for_endpoint_command(
    root: str,
    name: str,
    variable: str,
    healthcheck: str,
) -> str:
    return (
        f'{variable}="$(python3 -m literegistry.coop.endpoints wait '
        f"--root={shlex.quote(root)} --name={shlex.quote(name)} "
        f'--healthcheck={shlex.quote(healthcheck)} --timeout=600)"; '
        f"export {variable}; "
    )


class PodmanStackLauncher:
    """Build, preview, submit, and stop standalone Podman stack experiments."""

    def __init__(self, config: PodmanStackConfig) -> None:
        self.config = config.validate()

    def _identity(self) -> str:
        config_hash = hashlib.sha256(repr(self.config).encode()).hexdigest()[:10]
        return (
            f"{_slug(self.config.name_prefix)}-p{self.config.podman_replicas}"
            f"-m{self.config.docker_mirror_replicas}"
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
        critical: bool = False,
        auto_resume: bool | None = None,
        propagate_preemption: bool | None = None,
        env_vars: Sequence[Mapping[str, str]] = (),
    ) -> dict[str, Any]:
        task: dict[str, Any] = {
            "name": name,
            "image": {"beaker": image},
            "command": ["bash", "-lc", RESTORE_VENV_PATH + command],
            "hostNetworking": True,
            "propagateFailure": critical,
            "propagatePreemption": (
                critical
                if propagate_preemption is None
                else propagate_preemption
            ),
            "datasets": [
                {"mountPath": "/weka", "source": {"weka": self.config.weka_source}}
            ],
            "result": RESULT,
            "context": {
                "priority": self.config.priority,
                "minRuntime": f"{self.config.min_runtime_hours}h",
                "autoResume": not critical if auto_resume is None else auto_resume,
            },
            "constraints": {"cluster": [cluster]},
        }
        if replicas != 1:
            task["replicas"] = replicas
        if not self.config.omit_resources:
            task["resources"] = {"gpuCount": 0}
        if env_vars:
            task["envVars"] = list(env_vars)
        return task

    def build_spec(self, *, experiment_name: str | None = None) -> tuple[str, dict[str, Any]]:
        name = experiment_name or self._identity()
        configured_registry = self.config.resolved_registry()
        default_coordination_path = f"/weka/gfaria/literegistry/.coop/{name}"
        endpoint_registry = self.config.resolved_head_registry()
        tasks: list[dict[str, Any]] = []
        clusters = self.config.resolved_service_clusters()

        if self.config.registry is None:
            endpoint_registry = endpoint_registry or normalize_endpoint_registry(
                default_coordination_path
            )
            redis_process = (
                'exec literegistry redis --runtime=local --foreground=True '
                '--port="$PORT" --advertise_host="$REDIS_ADVERTISE_HOST" '
                f"--head_registry={shlex.quote(endpoint_registry)} "
                f"--data_dir={shlex.quote(default_coordination_path + '/redis-data')} "
                "--persistence=True"
            )
            redis_child = (
                'REDIS_ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
                "export REDIS_ADVERTISE_HOST; "
                + redis_process
            )
            tasks.append(
                self._task(
                    "redis",
                    self.config.redis_image,
                    "set -euo pipefail; " + _dynamic_port_command(name, "redis", redis_child),
                    cluster=self.config.resolved_redis_cluster(),
                    critical=False,
                    auto_resume=True,
                    propagate_preemption=False,
                )
            )
            configured_registry = head_registry_uri(endpoint_registry)
        if endpoint_registry is None:
            endpoint_registry = normalize_endpoint_registry(configured_registry)
        if is_head_registry_uri(configured_registry):
            registry_setup = (
                f"REGISTRY={shlex.quote(configured_registry)}; export REGISTRY; "
            )
        elif configured_registry.startswith(("redis://", "rediss://")):
            registry_setup = (
                f"REGISTRY={shlex.quote(configured_registry)}; export REGISTRY; "
                "python3 -m literegistry.coop.redis wait "
                '--registry "$REGISTRY" --timeout 600; '
            )
        else:
            registry_setup = (
                f"REGISTRY={shlex.quote(configured_registry)}; export REGISTRY; "
            )

        wait = "set -euo pipefail; " + registry_setup

        mirror_groups = _spread(self.config.docker_mirror_replicas, clusters)
        for group_index, (cluster, replicas, offset) in enumerate(mirror_groups):
            task_name = "docker-mirror" if len(mirror_groups) == 1 else f"docker-mirror-{_slug(cluster)}"
            child = (
                f"GLOBAL_RANK=$(({offset} + ${{BEAKER_REPLICA_RANK:-0}})); "
                'ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
                'INSTANCE_ID="docker-mirror-${GLOBAL_RANK}"; '
                f"CACHE_ROOT={shlex.quote(self.config.docker_mirror_storage_root)}/mirror-${{GLOBAL_RANK}}; "
                "export ADVERTISE_HOST INSTANCE_ID CACHE_ROOT; "
                "exec python -m literegistry.services.docker_mirror_server "
                '--registry="$REGISTRY" --host=0.0.0.0 --port="$PORT" '
                '--advertise_host="$ADVERTISE_HOST" --advertise_port="$PORT" '
                '--instance_id="$INSTANCE_ID" --storage_root="$CACHE_ROOT" '
                "--allow_non_loopback=True"
            )
            env: list[dict[str, str]] = []
            if self.config.docker_mirror_warm_images_file is not None:
                env.append(
                    {
                        "name": "DOCKER_MIRROR_WARM_IMAGES_FILE",
                        "value": self.config.docker_mirror_warm_images_file,
                    }
                )
            if (
                self.config.docker_hub_username
                or self.config.docker_hub_username_secret
            ):
                username_env = {
                    "name": "DOCKER_HUB_USERNAME",
                    (
                        "secret"
                        if self.config.docker_hub_username_secret
                        else "value"
                    ): (
                        self.config.docker_hub_username_secret
                        or self.config.docker_hub_username
                        or ""
                    ),
                }
                env.extend(
                    [
                        username_env,
                        {
                            "name": "DOCKER_HUB_TOKEN",
                            "secret": self.config.docker_hub_token_secret or "",
                        },
                    ]
                )
            tasks.append(
                self._task(
                    task_name,
                    self.config.docker_mirror_image,
                    wait + _dynamic_port_command(name, "docker-mirror", child),
                    cluster=cluster,
                    replicas=replicas,
                    env_vars=env,
                )
            )

        gateway_process = (
            "exec python -m literegistry.gateway "
            '--registry="$REGISTRY" --port="$PORT" '
            '--advertise_host="$GATEWAY_ADVERTISE_HOST" '
            f"--workers={self.config.gateway_workers} "
            f"--affinity_ttl_seconds={self.config.affinity_ttl_seconds:g} "
            f"--docker_mirror_affinity_ttl_seconds={self.config.docker_mirror_affinity_ttl_seconds:g} "
            f"--registry_cache_ttl_seconds={self.config.registry_cache_ttl_seconds} "
            f"--timeout={self.config.gateway_timeout:g} "
            f"--docker_mirror_soft_affinity={self.config.docker_mirror_soft_affinity}"
        )
        gateway_child = (
            'GATEWAY_ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
            'GATEWAY_URL="http://${GATEWAY_ADVERTISE_HOST}:${PORT}"; '
            "export GATEWAY_URL; "
            + _managed_endpoint_command(
                endpoint_registry,
                "gateway",
                "GATEWAY_URL",
                "http",
                gateway_process,
            )
        )
        tasks.append(
            self._task(
                "gateway",
                self.config.gateway_image,
                wait + _dynamic_port_command(name, "gateway", gateway_child),
                cluster=self.config.resolved_gateway_cluster(),
            )
        )

        podman_hardening_args = [
            f"--janitor_interval={self.config.podman_janitor_interval:g}"
        ]
        if self.config.podman_max_sessions is not None:
            podman_hardening_args.append(
                f"--max_sessions={self.config.podman_max_sessions}"
            )
        if self.config.podman_resource_watchdog_interval is not None:
            podman_hardening_args.append(
                "--resource_watchdog_interval="
                f"{self.config.podman_resource_watchdog_interval:g}"
            )
        if self.config.podman_session_memory is not None:
            podman_hardening_args.append(
                f"--session_memory={shlex.quote(self.config.podman_session_memory)}"
            )
        if self.config.podman_session_pids_limit is not None:
            podman_hardening_args.append(
                f"--session_pids_limit={self.config.podman_session_pids_limit}"
            )
        if self.config.podman_session_idle_timeout is not None:
            podman_hardening_args.append(
                f"--session_idle_timeout={self.config.podman_session_idle_timeout:g}"
            )
        if self.config.podman_image_prune_until is not None:
            podman_hardening_args.append(
                f"--image_prune_until={shlex.quote(self.config.podman_image_prune_until)}"
            )
        podman_hardening_cli = " ".join(podman_hardening_args) + " "
        podman_groups = _spread(self.config.podman_replicas, clusters)
        for group_index, (cluster, replicas, offset) in enumerate(podman_groups):
            task_name = "podman" if len(podman_groups) == 1 else f"podman-{_slug(cluster)}"
            if self.config.podman_registry_mirror:
                mirror_setup = (
                    f"PODMAN_REGISTRY_MIRROR={shlex.quote(self.config.podman_registry_mirror)}; "
                    "export PODMAN_REGISTRY_MIRROR; "
                )
            else:
                mirror_setup = _wait_for_endpoint_command(
                    endpoint_registry,
                    "gateway",
                    "PODMAN_REGISTRY_MIRROR",
                    "http",
                )
            child = (
                mirror_setup
                + f"GLOBAL_RANK=$(({offset} + ${{BEAKER_REPLICA_RANK:-0}})); "
                + 'ADVERTISE_HOST="${BEAKER_NODE_HOSTNAME:-$(hostname -f)}"; '
                + f'INSTANCE_ID={shlex.quote(self.config.podman_instance_prefix)}-"${{GLOBAL_RANK}}"; '
                + "export ADVERTISE_HOST INSTANCE_ID; "
                + 'mkdir -p "${XDG_RUNTIME_DIR:?}"; chmod 700 "$XDG_RUNTIME_DIR"; '
                + "exec literegistry podman "
                + '--host=0.0.0.0 --port="$PORT" '
                + '--advertise_host="$ADVERTISE_HOST" --advertise_port="$PORT" '
                + '--registry="$REGISTRY" '
                + f"--image={shlex.quote(self.config.podman_session_image)} "
                + '--network=none --instance_id="$INSTANCE_ID" --storage_driver=vfs '
                + podman_hardening_cli
                + '--registry_mirror="$PODMAN_REGISTRY_MIRROR" --allow_non_loopback=True'
            )
            tasks.append(
                self._task(
                    task_name,
                    self.config.podman_image,
                    wait + _dynamic_port_command(name, "podman", child),
                    cluster=cluster,
                    replicas=replicas,
                    env_vars=[
                        {"name": "BEAKER_ALLOW_SUBCONTAINERS", "value": "1"},
                        {"name": "BEAKER_SKIP_DOCKER_SOCKET", "value": "1"},
                    ],
                )
            )

        return name, {
            "version": "v2",
            "description": f"LiteRegistry Podman + Docker mirror + Redis stack: {name}",
            "budget": self.config.budget,
            "tasks": tasks,
        }

    def preview(self) -> dict[str, Any]:
        name, spec = self.build_spec()
        redacted = asdict(self.config)
        # Only a Beaker secret name is accepted, never a token value.
        return {"experiment_name": name, "config": redacted, "spec": spec}

    def submit(self) -> dict[str, Any]:
        name, spec = self.build_spec()
        if self.config.registry is None:
            _prepare_shared_directory(f"/weka/gfaria/literegistry/.coop/{name}")
            head = self.config.resolved_head_registry()
            if head is not None:
                prepare_endpoint_registry_storage(head)
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
            raise RuntimeError(f"Beaker could not create stack {name}: {detail}") from error
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def stop(experiment_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        command = ["beaker", "experiment", "stop", experiment_id]
        if not dry_run:
            subprocess.run(command, check=True)
        return {"command": command, "dry_run": dry_run}
