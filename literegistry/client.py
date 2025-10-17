import json
import time
import random
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from literegistry.kvstore import KeyValueStore, FileSystemKVStore
from literegistry.registry import ServerRegistry
from literegistry.telemetry import LatencyMetricAggregator
from literegistry.bandit import Exp3Dynamic, UniformBandit
import asyncio
import numpy as np


class RegistryClient(ServerRegistry):
    """
    Extended ServerRegistry that adds model-specific functionality,
    caching, and telemetry tracking
    """

    def __init__(
        self,
        store: KeyValueStore,
        max_history: int = 3600,
        cache_ttl: int = 60 * 5,
        service_type="model_path",
        penalty_latency=200,
    ):
        """
        Initialize ModelRegistry

        Args:
            store: KeyValueStore implementation to use
            max_history: Maximum number of requests to track
            cache_ttl: Time-to-live for cache entries in seconds
        """
        super().__init__(store, max_history)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = cache_ttl
        self.telemetry = LatencyMetricAggregator()
        self.service_type = service_type
        self.penalty_latency = penalty_latency
        self.bandit = UniformBandit()#Exp3Dynamic(gamma=0.2, L_max=penalty_latency)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < self.cache_ttl

    async def models(self, force: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all models with their servers, optionally bypassing cache

        Args:
            force: If True, bypass cache and force fresh lookup

        Returns:
            Dictionary mapping model paths to lists of server info
        """
        cache_key = "_" + self.service_type

        if not force and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        # Get fresh server list
        roster = await self.roster()
        #logging.info(f"Raw roster data: {roster}")
        
        m: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for server in roster["servers"]:
            metadata = server.get("metadata", {})
            model_path = metadata.get(self.service_type, "default")
            #logging.info(f"Processing server {server.get('uri', 'unknown')} with metadata {metadata}, model_path: {model_path}")
            m[model_path].append(server)

        #logging.info(f"Processed models: {dict(m)}")
        
        # Update cache
        self._cache[cache_key] = dict(m)  # Convert defaultdict to regular dict
        self._cache_timestamps[cache_key] = time.time()

        return self._cache[cache_key]

    async def get_all(
        self, value: str, force: bool = False, n: Optional[int] = None
    ) -> List[str]:
        """
        Get all URIs for a model, using cached values if available

        Args:
            model_path: Path to the model
            force: If True, bypass cache and force fresh lookup
            n: If set, return only n randomly selected URIs weighted by latency

        Returns:
            List of URIs for the requested model
        """
        # Check cache first
        if not force and self._is_cache_valid(value):
            m = self._cache[value]
        else:
            try:
                # Get fresh data and sort by latency
                models_data = await self.models(force=force)
                m = models_data[value]
                m.sort(
                    key=lambda x: (
                        self.telemetry.get_estimated_latency(x["uri"]),
                        random.random(),
                    )
                )

                # Update cache
                self._cache[value] = m
                self._cache_timestamps[value] = time.time()

            except KeyError as e:
                logging.warning(f"Model {value} not found in registry - {e}")
                self._cache[value] = []
                self._cache_timestamps[value] = time.time()
                m = []

        result = [mi["uri"] for mi in m]
        # print(result)

        self.telemetry.prune_inactive()

        return result

    async def sample_servers(self, value: str, n: int):
        servers = await self.get_all(value)
        
        # Handle case when no servers are available
        if not servers:
            return []
            
        result, _ = self.bandit.get_arm(servers, k=n)  # Get URIs for bandit selection
        return result

    async def get(self, value: str, force: bool = False) -> str:
        """
        Get the best URI for a model, using cached values if available

        Args:
            model_path: Path to the model
            force: If True, bypass cache and force fresh lookup

        Returns:
            URI for the best server for this model

        Raises:
            ValueError: If no servers are found for the model
        """
        cached_uris = await self.get_all(value, force=force)
        if not cached_uris:
            raise ValueError(f"Model {value} not found in registry")
        return cached_uris[0]

    def report_latency(self, uri: str, response_time: float, success: bool = True):
        """Report request latency for a URI"""
        self.bandit.update(uri, latency=response_time, success=success)

        if success:
            self.telemetry.report(uri, response_time)
        else:
            self.telemetry.report(uri, self.penalty_latency)

    def invalidate_cache(self, model_path: Optional[str] = None):
        """
        Invalidate cache entries

        Args:
            model_path: If provided, only invalidate this model's cache.
                      If None, invalidate all cache entries.
        """
        if model_path is None:
            self._cache.clear()
            self._cache_timestamps.clear()
        else:
            self._cache.pop(model_path, None)
            self._cache_timestamps.pop(model_path, None)

    async def close(self):
        """Close the registry and clean up resources"""
        try:
            if hasattr(self, 'store') and self.store:
                await self.store.close()
        except Exception as e:
            logging.error(f"Error closing registry: {e}")

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        await self.close()


# Example usage
async def main():
    # Initialize with FileSystem backend
    store = FileSystemKVStore("registry_data")
    registry = RegistryClient(store, service_type="model_path")

    # Register a model server
    await registry.register_server(8000, {"model_path": "gpt-3"})
    await registry.heartbeat(8000)
    # Get all URIs for a model
    uris = await registry.get_all("gpt-3")

    print(f"Model servers: {uris}")

    # Report some latencies
    registry.report_latency(uris[0], 0.5)

    # Get best server for model
    best_uri = await registry.get("gpt-3")
    print(f"Best server: {best_uri}")


if __name__ == "__main__":
    asyncio.run(main())
