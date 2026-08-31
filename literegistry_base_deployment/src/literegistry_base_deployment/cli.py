"""Python Fire CLI for the standalone Beaker launcher."""

from __future__ import annotations

import fire
import json

from .launcher import BaseDeploymentConfig, BaseDeploymentLauncher


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
        python_replicas: int = 1,
        terminal_replicas: int = 1,
        web_search_replicas: int = 1,
        local_search_replicas: int = 0,
        local_search_corpus_jsonl: str | None = None,
        local_search_index_dir: str | None = None,
        local_search_service_name: str | None = None,
        generation_model: str | None = None,
        generation_replicas: int = 0,
        generation_tp: int = 1,
        classification_model: str | None = None,
        classification_replicas: int = 0,
        classification_tp: int = 1,
        max_model_len: int = 32768,
        max_num_batched_tokens: int = 32768,
        max_num_seqs: int | None = None,
        gateway_workers: int = 8,
        docker_mirror_soft_affinity: bool = True,
        python_pool_size: int = 32,
        service_cluster: str = "ai2/jupiter",
        gateway_cluster: str | None = None,
        redis_cluster: str | None = None,
        model_cluster: str = "ai2/jupiter",
        workspace: str = "ai2/oe-agents",
        budget: str = "ai2/oe-omai",
        service_priority: str = "normal",
        model_priority: str = "high",
        min_runtime_hours: int = 0,
        omit_service_resources: bool = False,
        name_prefix: str = "literegistry-base",
        services_image: str = "goncalof/literegistry-base-services",
        terminal_image: str = "goncalof/literegistry-base-terminal",
        local_search_image: str = "goncalof/jtc-local-search-lucene-bm25",
        vllm_image: str = "goncalof/literegistry-base-vllm",
        redis_image: str = "goncalof/literegistry-redis",
        serper_api_key_secret: str | None = "SERPER_API_KEY",
        jina_api_key_secret: str | None = "JINA_API_KEY",
        hf_token_secret: str | None = "HF_TOKEN",
        hf_home: str | None = None,
        gateway_timeout: float = 300.0,
        registry_cache_ttl_seconds: int = 5,
        shared_dir: str = "/weka/gfaria",
        weka_source: str = "oe-adapt-default",
    ) -> None:
        config = BaseDeploymentConfig(
            registry=registry,
            python_replicas=python_replicas,
            terminal_replicas=terminal_replicas,
            web_search_replicas=web_search_replicas,
            local_search_replicas=local_search_replicas,
            local_search_corpus_jsonl=local_search_corpus_jsonl,
            local_search_index_dir=local_search_index_dir,
            local_search_service_name=local_search_service_name,
            generation_model=generation_model,
            generation_replicas=generation_replicas,
            generation_tp=generation_tp,
            classification_model=classification_model,
            classification_replicas=classification_replicas,
            classification_tp=classification_tp,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gateway_workers=gateway_workers,
            docker_mirror_soft_affinity=docker_mirror_soft_affinity,
            python_pool_size=python_pool_size,
            service_clusters=_clusters(service_cluster),
            gateway_cluster=gateway_cluster,
            redis_cluster=redis_cluster,
            model_cluster=model_cluster,
            workspace=workspace,
            budget=budget,
            service_priority=service_priority,
            model_priority=model_priority,
            min_runtime_hours=min_runtime_hours,
            omit_service_resources=omit_service_resources,
            name_prefix=name_prefix,
            services_image=services_image,
            terminal_image=terminal_image,
            local_search_image=local_search_image,
            vllm_image=vllm_image,
            redis_image=redis_image,
            serper_api_key_secret=serper_api_key_secret,
            jina_api_key_secret=jina_api_key_secret,
            hf_token_secret=hf_token_secret,
            hf_home=hf_home,
            gateway_timeout=gateway_timeout,
            registry_cache_ttl_seconds=registry_cache_ttl_seconds,
            shared_dir=shared_dir,
            weka_source=weka_source,
        )
        launcher = BaseDeploymentLauncher(config)
        result = launcher.preview() if self.action == "preview" else launcher.submit()
        print(json.dumps(result, indent=2))


def stop(experiment_id: str, dry_run: bool = False) -> None:
    """Stop one Beaker experiment by ID."""
    print(json.dumps(BaseDeploymentLauncher.stop(experiment_id, dry_run=dry_run), indent=2))


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
