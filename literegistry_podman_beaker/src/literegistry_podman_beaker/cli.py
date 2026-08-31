"""Python Fire CLI for the standalone Beaker launcher."""

from __future__ import annotations

import fire
import json


from .launcher import PodmanStackConfig, PodmanStackLauncher


def _clusters(value: str) -> tuple[str, ...]:
    clusters = tuple(item.strip() for item in value.split(",") if item.strip())
    if not clusters:
        raise ValueError("service_cluster must contain at least one cluster")
    return clusters


class StackCommand:
    """Create one validated stack configuration and preview or launch it."""

    def __init__(self, action: str) -> None:
        self.action = action

    def __call__(
        self,
        registry: str | None = None,
        podman_replicas: int = 4,
        docker_mirror_replicas: int = 2,
        gateway_workers: int = 8,
        docker_mirror_soft_affinity: bool = True,
        service_cluster: str = "ai2/jupiter",
        gateway_cluster: str | None = None,
        workspace: str = "ai2/oe-agents",
        budget: str = "ai2/oe-omai",
        priority: str = "normal",
        min_runtime_hours: int = 0,
        omit_resources: bool = False,
        name_prefix: str = "literegistry-podman",
        podman_image: str = "goncalof/literegistry-podman-immediate-rm-20260819",
        podman_session_image: str = "docker.io/library/ubuntu:24.04",
        docker_mirror_image: str = "goncalof/literegistry-docker-mirror",
        gateway_image: str = "goncalof/literegistry-basic",
        redis_image: str = "goncalof/literegistry-redis",
        redis_cluster: str | None = None,
        docker_mirror_storage_root: str = "/var/lib/registry",
        docker_mirror_warm_images_file: str | None = None,
        docker_hub_username: str | None = None,
        docker_hub_username_secret: str | None = None,
        docker_hub_token_secret: str | None = None,
        podman_registry_mirror: str | None = None,
        affinity_ttl_seconds: float = 900.0,
        docker_mirror_affinity_ttl_seconds: float = 604800.0,
        registry_cache_ttl_seconds: int = 5,
        gateway_timeout: float = 300.0,
        weka_source: str = "oe-adapt-default",
    ) -> None:
        config = PodmanStackConfig(
            registry=registry,
            podman_replicas=podman_replicas,
            docker_mirror_replicas=docker_mirror_replicas,
            gateway_workers=gateway_workers,
            docker_mirror_soft_affinity=docker_mirror_soft_affinity,
            service_clusters=_clusters(service_cluster),
            gateway_cluster=gateway_cluster,
            workspace=workspace,
            budget=budget,
            priority=priority,
            min_runtime_hours=min_runtime_hours,
            omit_resources=omit_resources,
            name_prefix=name_prefix,
            podman_image=podman_image,
            podman_session_image=podman_session_image,
            docker_mirror_image=docker_mirror_image,
            gateway_image=gateway_image,
            redis_image=redis_image,
            redis_cluster=redis_cluster,
            docker_mirror_storage_root=docker_mirror_storage_root,
            docker_mirror_warm_images_file=docker_mirror_warm_images_file,
            docker_hub_username=docker_hub_username,
            docker_hub_username_secret=docker_hub_username_secret,
            docker_hub_token_secret=docker_hub_token_secret,
            podman_registry_mirror=podman_registry_mirror,
            affinity_ttl_seconds=affinity_ttl_seconds,
            docker_mirror_affinity_ttl_seconds=docker_mirror_affinity_ttl_seconds,
            registry_cache_ttl_seconds=registry_cache_ttl_seconds,
            gateway_timeout=gateway_timeout,
            weka_source=weka_source,
        )
        launcher = PodmanStackLauncher(config)
        result = launcher.preview() if self.action == "preview" else launcher.submit()
        print(json.dumps(result, indent=2))


def stop(experiment_id: str, dry_run: bool = False) -> None:
    """Stop one Beaker experiment by ID."""
    result = PodmanStackLauncher.stop(experiment_id, dry_run=dry_run)
    print(json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> None:
    fire.Fire(
        {
            "preview": StackCommand("preview"),
            "launch": StackCommand("launch"),
            "stop": stop,
        },
        command=argv,
    )


if __name__ == "__main__":
    main()
