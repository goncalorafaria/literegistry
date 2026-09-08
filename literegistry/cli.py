
import pprint
from termcolor import colored
import asyncio
from dataclasses import dataclass
import time
from typing import Any

from literegistry import (
    RegistryClient,
    get_kvstore,
    head_registry_backend,
    head_registry_uri,
    is_head_registry_uri,
)
from literegistry.coop.endpoints import EndpointRecord, get_endpoint_registry
import fire
import literegistry.redis as redis
from literegistry.redis import redact_redis_url
import literegistry.cache_server as cache_server
import literegistry.gateway as gateway
import literegistry.gateway.legacy as old_gateway
import literegistry.services.openai_proxy as openai_proxy
import literegistry.services.vllm_wrapper as vllm
import literegistry.services.sglang_wrapper as sglang
import literegistry.services.code_server as code_server
import literegistry.services.terminal_server as terminal_server
import literegistry.services.search_server as search_server
import literegistry.services.bm25_server as bm25_server
import literegistry.services.podman_server as podman_server
import literegistry.services.docker_mirror_server as docker_mirror_server
import literegistry.console.launcher as console_launcher


DEFAULT_REGISTRY = "redis://klone-login01.hyak.local:6379"


@dataclass(frozen=True)
class RegistryView:
    models: dict[str, list[dict[str, Any]]]
    head_registry: str | None = None
    live_registry: str | None = None
    endpoint: EndpointRecord | None = None


def _redact_registry(value: str) -> str:
    if value.startswith("head+redis://") or value.startswith("head+rediss://"):
        return "head+" + redact_redis_url(value[len("head+") :])
    if value.startswith(("redis://", "rediss://")):
        return redact_redis_url(value)
    return value


def _registry_target(
    registry: str | None,
    head_registry: str | None,
) -> tuple[str, str | None]:
    if registry is not None and head_registry is not None:
        raise ValueError("supply only one of registry or head_registry")
    if head_registry is not None:
        target = head_registry_uri(head_registry)
        return target, head_registry_backend(target)
    target = registry or DEFAULT_REGISTRY
    return (
        target,
        head_registry_backend(target) if is_head_registry_uri(target) else None,
    )


async def _registry_view(
    registry: str | None,
    head_registry: str | None,
    timeout: float,
) -> RegistryView:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    target, head_backend = _registry_target(registry, head_registry)
    store = get_kvstore(target)
    endpoint: EndpointRecord | None = None
    try:
        models = await asyncio.wait_for(
            RegistryClient(store).models(force=True),
            timeout=timeout,
        )
        if head_backend is not None:
            endpoint_registry = get_endpoint_registry(head_backend)
            try:
                endpoint = await endpoint_registry.get("redis")
            finally:
                await endpoint_registry.close()
        live_registry = (
            endpoint.uri
            if endpoint is not None
            else getattr(store, "current_url", None)
        )
        return RegistryView(
            models=models,
            head_registry=(head_registry_uri(head_backend) if head_backend else None),
            live_registry=live_registry,
            endpoint=endpoint,
        )
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            await close()


def _print_registry_resolution(view: RegistryView) -> None:
    if view.head_registry is None:
        return
    print(f"Head registry: {_redact_registry(view.head_registry)}")
    print(
        "Live registry: "
        + (
            _redact_registry(view.live_registry)
            if view.live_registry is not None
            else "unavailable"
        )
    )
    if view.endpoint is not None:
        age = max(0.0, time.time() - view.endpoint.published_at)
        print(f"Redis publisher: {view.endpoint.publisher_id}")
        print(f"Redis publication age: {age:.1f}s")


def check_registry(verbose=False, registry_dir="/gscratch/ark/graf/registry"):
    
    r = RegistryClient(get_kvstore(registry_dir))
    #r = RegistryClient(FileSystemKVStore("/gscratch/ark/graf/registry"))

    pp = pprint.PrettyPrinter(indent=1, compact=True)

    for k, v in asyncio.run(r.models()).items():
        print(f"{colored(k, 'red')}")
        for item in v:
            print(colored("--" * 20, "blue"))
            for key, value in item.items():

                if key == "request_stats":
                    if verbose:
                        print(f"\t{colored(key, 'green')}:{value}")
                    else:
                        if "last_15_minutes_latency" in value:
                            nvalue = value["last_15_minutes"]
                            print(f"\t{colored(key, 'green')}:{colored(nvalue,'red')}")
                        else:
                            print(f"\t{colored(key, 'green')}:NO METRICS YET.")
                else:
                    print(f"\t{colored(key, 'green')}:{value}")

    # pp.pprint(r.get("allenai/Llama-3.1-Tulu-3-8B-SFT"))


def check_summary(
    registry: str | None = None,
    head_registry: str | None = None,
    timeout: float = 10.0,
):
    view = asyncio.run(_registry_view(registry, head_registry, timeout))
    _print_registry_resolution(view)
    for k, v in view.models.items():
        print(f"{colored(k, 'red')} :{colored(len(v),'green')}")

def check_detail(
    registry: str | None = None,
    head_registry: str | None = None,
    timeout: float = 10.0,
):
    view = asyncio.run(_registry_view(registry, head_registry, timeout))
    _print_registry_resolution(view)
    for k, v in view.models.items():
        print(f"{colored(k, 'red')} : {colored(len(v),'green')}")
        for item in v:
            print(colored("--" * 20, "blue"))
            print(f"\t{colored('uri', 'green')}:{item['uri']}")
            print(f"\t{colored('metadata', 'green')}:{item['metadata']}")


def main():
    
    fire.Fire({
        "summary": check_summary,
        "redis": redis.main,
        "cache": cache_server.main,
        "gateway": gateway.main,
        "old-gateway": old_gateway.main,
        "podman": podman_server.main,
        "docker-mirror": docker_mirror_server.main,
        "openai-proxy": openai_proxy.main,
        "vllm": vllm.main,
        "sglang": sglang.main,
        "detail": check_detail,
        "code": code_server.main,
        "terminal": terminal_server.main,
        "search": search_server.main,
        "bm25": bm25_server.main,
        "console": console_launcher.main,
    })


if __name__ == "__main__":

    main()
