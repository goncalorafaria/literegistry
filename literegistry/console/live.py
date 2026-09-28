"""Read monitoring endpoints through the shared head without changing routing."""
import asyncio
import json
import time
from urllib.request import build_opener, ProxyHandler

from literegistry.coop.endpoints import EndpointRecord, get_endpoint_registry
from literegistry.head_registry import head_registry_backend, is_head_registry_uri


def _snapshot(record):
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(record.uri.rstrip('/') + '/gateway-stats', timeout=3) as response:
            data = json.load(response)
        rows = []
        for route, count in data.get('total_requests', {}).items():
            recent = data.get('recent', {}).get(route, {})
            rows.append({'gateway': record.publisher_id, 'url': record.uri, 'route': route,
                         'completed_total': count, 'recent_completed': recent.get('count', 0),
                         'window_seconds': data.get('window_seconds', 5),
                         'average_seconds': recent.get('average_seconds'),
                         'maximum_seconds': recent.get('maximum_seconds')})
        return {'publisher': record.publisher_id, 'url': record.uri, 'rows': rows, 'error': None}
    except (OSError, ValueError) as exc:
        return {'publisher': record.publisher_id, 'url': record.uri, 'rows': [], 'error': str(exc)}


async def _poll_gateways(registry):
    endpoints = get_endpoint_registry(head_registry_backend(registry))
    records = []
    try:
        for name in ('gateway', 'gateway-monitor'):
            for key in await endpoints.store.keys(prefix=endpoints.prefix(name)):
                value = await endpoints.store.get(key)
                if value is not None:
                    record = EndpointRecord.from_bytes(value)
                    if record.name == name:
                        records.append(record)
    finally:
        await endpoints.close()
    unique = {r.uri: r for r in records}
    return await asyncio.gather(*(asyncio.to_thread(_snapshot, r) for r in unique.values()))


def poll_head_gateways(registry):
    if not is_head_registry_uri(registry):
        return []
    return asyncio.run(asyncio.wait_for(_poll_gateways(registry), timeout=8))


def registry_summary(registry, timeout_seconds=4):
    from literegistry.cli import _registry_view
    view = asyncio.run(_registry_view(registry, None, timeout_seconds))
    now = time.time()
    return [{'ts': now, 'model': name, 'count': len(servers)}
            for name, servers in view.models.items()]
