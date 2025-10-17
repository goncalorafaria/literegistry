import aiohttp
import asyncio
from typing import Dict, List, Tuple, Optional
import logging
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore
from literegistry.shared_session import get_shared_session
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegistryHTTPClient:
    """
    HTTP client for making requests to model servers via the registry.
    
    Uses shared aiohttp session when available for optimal connection reuse,
    creating a temporary session only as fallback.
    """

    def __init__(
        self,
        registry: RegistryClient,
        value: str,
        max_parallel_requests: int = 512,
        timeout: float = 60,
        max_retries: int = 5,
        use_shared_session: bool = True,
    ):
        """
        Initialize the HTTP client.
        
        Args:
            registry: ModelRegistry instance for service discovery
            value: Model path to use for server selection
            use_shared_session: If True, use shared session (recommended)
        """
        self.registry = registry
        self.value = value
        self._session: Optional[aiohttp.ClientSession] = None
        self._owns_session = False  # Track if we created the session
        self.max_parallel_requests = max_parallel_requests
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_shared_session = use_shared_session
        self._connector = None

    async def __aenter__(self):
        """
        Initialize session - prefer shared session, create temporary if needed.
        
        Production pattern: Try shared session first for connection reuse.
        """
        # Try to get shared session (optimal for connection reuse)
        if self.use_shared_session:
            shared_session = await get_shared_session()
            if shared_session is not None and not shared_session.closed:
                self._session = shared_session
                self._owns_session = False
                logger.debug(f"Using shared session for {self.value}")
                return self
        
        # Fallback: Create temporary session
        logger.warning(
            f"Shared session not available for {self.value}, creating temporary session. "
            "This is less efficient - ensure shared session is initialized at startup."
        )
        
        self._connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=10,
            sock_read=self.timeout
        )
        
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            connector_owner=True
        )
        self._owns_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup - only close session if we created it (not shared).
        
        Important: Never close the shared session here - it's managed at app level.
        """
        try:
            # Only close if we own the session (not shared)
            if self._owns_session and self._session:
                await self._session.close()
                await asyncio.sleep(0.25)  # Graceful close
                self._session = None
                
                if self._connector:
                    await self._connector.close()
                    self._connector = None
            
            # Reset state
            self._session = None
            self._owns_session = False
            
        except Exception as e:
            logger.error(f"Error closing HTTP client session: {e}")

    async def _make_http_request(
        self,
        server: str,
        endpoint: str,
        payload: Dict,
    ) -> Dict:
        """Make a single HTTP POST request to a server."""
        if not self._session:
            raise RuntimeError("Client not initialized - use async with")

        try:
            async with self._session.post(
                f"{server.rstrip('/')}/{endpoint}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"HTTP request failed to {server}/{endpoint}: {e}")
            raise

    async def request_with_rotation(
        self,
        endpoint: str,
        payload: Dict,
        initial_server_idx: int = 0,
    ) -> Tuple[Dict, int]:
        """Make a request with automatic server rotation on failure."""
        servers = await self.registry.sample_servers(
            self.value, n=self.max_parallel_requests
        )
        total_servers = len(servers)

        if not total_servers:
            raise RuntimeError(f"No servers available for model {self.value}")

        attempt = 0
        server_idx = initial_server_idx

        while attempt < self.max_retries:
            start_time = asyncio.get_event_loop().time()

            try:
                server_idx = server_idx % total_servers
                server = servers[server_idx].rstrip("/")
                result = await self._make_http_request(server, endpoint, payload)

                if "status" in result and result["status"] == "failed":
                    raise RuntimeError(f"Error from server: - {result} - {server}")

                # Report successful request latency
                latency = asyncio.get_event_loop().time() - start_time
                self.registry.report_latency(server, latency)

                return result, server_idx

            except Exception as e:
                logging.error(
                    f"Attempt {self.value}:{attempt + 1} failed on server {server}: {str(e)}"
                )
                ## print traceback
                #import traceback
                #traceback.print_exc()
                
                attempt += 1

                # Report failed request latency
                latency = asyncio.get_event_loop().time() - start_time
                self.registry.report_latency(server, latency, success=False)

                # Get fresh server list
                servers = await self.registry.get_all(self.value, force=False)
                total_servers = len(servers)

                # Rotate to next server
                server_idx = server_idx + 1  # % total_servers

                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Failed after {self.max_retries} attempts across {total_servers} servers: {str(e)}"
                    )

                # Exponential backoff with max of 60 seconds
                await asyncio.sleep(min(2**attempt, 60))

        raise RuntimeError("Unexpected end of retry loop")

    async def post(
        self,
        endpoint: str,
        payloads: List[Dict],
        track=True,
    ) -> List[Dict]:
        """Make multiple requests in parallel with server rotation."""
        if not self._session:
            raise RuntimeError("Client not initialized - use async with")

        # Initialize tasks with different starting servers
        tasks = []
        for i, payload in enumerate(payloads):
            tasks.append(
                self.request_with_rotation(
                    endpoint,
                    payload,
                    initial_server_idx=i,
                )
            )

        # Use semaphore to limit concurrent requests
        sem = asyncio.Semaphore(self.max_parallel_requests)

        async def bounded_request(task):
            async with sem:
                return await task

        # Gather results with bounded concurrency
        bounded_tasks = [bounded_request(task) for task in tasks]

        total_requests = len(bounded_tasks)
        progress_bar = tqdm(
            total=total_requests,
            desc="brrr",
            unit="req",
            disable=not track,
        )

        # Function to update progress bar when a task completes
        async def tracked_task(task):
            try:
                result = await task
                progress_bar.update(1)
                return result
            except Exception as e:
                progress_bar.update(1)
                return e

        tracked_tasks = [tracked_task(task) for task in bounded_tasks]

        results = await asyncio.gather(*tracked_tasks, return_exceptions=True)

        # Check for exceptions and extract results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            response, _ = result  # Unpack the response and server_idx
            final_results.append(response)

        return final_results

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make a single HTTP GET request to a server."""
        if not self._session:
            raise RuntimeError("Client not initialized - use async with")

        servers = await self.registry.get_all(self.value)
        if not servers:
            raise RuntimeError(f"No servers available for model {self.value}")

        server = servers[0].rstrip("/")  # Use the first server for simplicity

        async with self._session.get(
            f"{server}/{endpoint}",
            params=params,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            response.raise_for_status()
            return await response.json()


# Example usage
async def main():
    # Initialize registry and client
    store = FileSystemKVStore("registry_data")
    registry = RegistryClient(store, service_type="model_path")

    # Register some test servers
    await registry.register_server(8000, {"model_path": "gpt-3"})
    await registry.register_server(8001, {"model_path": "gpt-3"})

    # Use the client
    async with RegistryHTTPClient(registry, "gpt-3") as client:
        # Single request
        result, server_idx = await client.request_with_rotation(
            "v1/completions", {"prompt": "Hello"}
        )
        print(f"Single request result: {result}")

        # Parallel requests
        payloads = [{"prompt": "Hello"}, {"prompt": "World"}]
        results = await client.parallel_requests(
            "v1/completions",
            payloads,
        )
        print(f"Parallel results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
