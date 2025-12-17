"""
LiteRegistry Gateway Server

This module provides a production-grade gateway server using Starlette and uvicorn,
following best practices from LiteLLM and modern async Python patterns.

FEATURES:
    - Connection pooling with persistent HTTP clients (no file descriptor leaks)
    - Automatic retry and server rotation on failures
    - Starlette lifespan management for proper resource cleanup
    - Connection pool monitoring endpoint
    - Graceful shutdown handling
    - CORS support

RECOMMENDED USAGE (most efficient):
    uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080 --workers 8

ALTERNATIVE USAGE (for development/testing):
    python -m literegistry.gateway

ENDPOINTS:
    GET  /health         - Health check
    GET  /session-stats  - Shared session statistics (monitoring)
    GET  /v1/models      - List available models
    POST /v1/completions - Completion requests
    POST /classify       - Classification requests

Environment variables:
    REGISTRY_PATH: Registry connection string (default: redis://klone-login01.hyak.local:6379)
    PORT: Server port (default: 8080)
    WORKERS: Number of worker processes (default: 1)

MONITORING:
    curl http://localhost:8080/session-stats
    
    Response shows shared session configuration:
    - connector_limit: 0 (unlimited)
    - keepalive_timeout: 120 seconds
    - dns_cache_ttl: 300 seconds
"""

import asyncio
import json
import logging
import signal
import sys
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
import uvicorn
from literegistry.client import RegistryClient
from literegistry import get_kvstore
from literegistry.shared_session import get_session_manager
import pprint
from termcolor import colored
import os
import socket
import fire
import time
# Use shared session via RegistryHTTPClient
from literegistry.http import RegistryHTTPClient

class StarletteGatewayServer:
    """
    Production-grade gateway server using Starlette and uvicorn.
    
    Architecture:
    - Single shared aiohttp session for all requests
    - Connections managed by aiohttp's built-in pooling
    - No manual client pooling needed
    """

    def __init__(
        self,
        registry: RegistryClient,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: float = 15,
        max_retries: int = 20
    ):
        self.registry = registry
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self._shutdown_event = asyncio.Event()
        
        # Request tracking for statistics (sliding window of 5 seconds)
        self._request_history: deque = deque()  # Stores (timestamp, duration, model) tuples
        self._request_lock = asyncio.Lock()
        self._stats_window_seconds = 5.0
        self._stats_log_interval_seconds = 5.0
        self._last_stats_log = 0.0
        
        # Request type counting for periodic reporting
        self._request_type_counts: defaultdict = defaultdict(int)  # Tracks counts by request type
        self._request_type_lock = asyncio.Lock()
        self._request_type_log_interval_seconds = 5.0
        self._last_request_type_log = 0.0
        
        # Probabilities logging for periodic reporting
        self._probabilities_log_interval_seconds = 5.0
        self._last_probabilities_log = 0.0
        self._probabilities_lock = asyncio.Lock()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create app
        self.app = self._create_app()

    async def health_check(self,request: Request):
        """Health check endpoint."""
        try:
            models_data = await self.registry.models()
            return JSONResponse({
                "status": "healthy",
                "service": "registry-gateway", 
                "models_count": len(models_data)
            })
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return JSONResponse({
                "status": "unhealthy",
                "error": str(e)
            }, status_code=503)
    
    async def session_stats(self, request: Request):
        """
        Shared session statistics endpoint for monitoring.
        
        Reports status of the shared aiohttp session.
        """
        session_manager = get_session_manager()
        
        stats = {
            "shared_session_initialized": session_manager.is_initialized,
            "architecture": "single_shared_session",
            "pattern": "production_grade"
        }
        
        if session_manager.is_initialized:
            try:
                session = session_manager.get_session()
                connector = session.connector
                stats.update({
                    "session_closed": session.closed,
                    "connector_limit": connector.limit,
                    "connector_limit_per_host": connector.limit_per_host,
                    "keepalive_timeout": connector.keepalive_timeout,
                    "dns_cache_ttl": connector.ttl_dns_cache
                })
            except Exception as e:
                stats["error"] = str(e)
        
        return JSONResponse({
            "status": "success",
            "session_info": stats
        })
    
    async def list_models(self,request: Request):
        """List available models."""
        try:
            models_data = await self.registry.models()
            models = list(models_data.keys())
            return JSONResponse({
                "models": models,
                "status": "success",
                "data": [{"id": model, "metadata": models_data[model]} for model in models]
            })
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return JSONResponse({
                "error": str(e),
                "status": "failed"
            }, status_code=500)
    
    async def _record_request_and_log_stats(self, model: str, duration: float):
        """
        Record a request and log statistics for requests in the last 5 seconds.
        
        Args:
            model: Model name
            duration: Request duration in seconds
        """
        current_time = time.time()
        
        async with self._request_lock:
            # Add current request
            self._request_history.append((current_time, duration, model))
            
            # Remove old entries (older than stats_window_seconds)
            cutoff_time = current_time - self._stats_window_seconds
            while self._request_history and self._request_history[0][0] < cutoff_time:
                self._request_history.popleft()
            
            # Calculate statistics for recent requests, grouped by model
            if self._request_history:
                # Group durations by model name
                model_durations: defaultdict = defaultdict(list)
                for _, d, m in self._request_history:
                    model_durations[m].append(d)
                
                # Log stats at most once per stats_log_interval_seconds
                if current_time - self._last_stats_log >= self._stats_log_interval_seconds:
                    self._last_stats_log = current_time
                    
                    # Build stats string for each model
                    stats_parts = []
                    for model_name in sorted(model_durations.keys()):
                        durations = model_durations[model_name]
                        count = len(durations)
                        avg_duration = sum(durations) / count
                        max_duration = max(durations)
                        stats_parts.append(
                            f"{model_name}: {count} reqs, avg: {avg_duration:.3f}s, max: {max_duration:.3f}s"
                        )
                    
                    if stats_parts:
                        stats_str = ", ".join(stats_parts)
                        self.logger.info(
                            f"Completion stats (last {self._stats_window_seconds}s): {stats_str}"
                        )
    
    async def _log_probabilities_periodically(self):
        """
        Log bandit algorithm probabilities periodically (every 5 seconds).
        Only logs if the interval has elapsed since last log.
        """
        current_time = time.time()
        
        async with self._probabilities_lock:
            if current_time - self._last_probabilities_log >= self._probabilities_log_interval_seconds:
                self._last_probabilities_log = current_time
                probs = self.registry.bandit._get_probabilities()
                sorted_probs = sorted([(k, round(p, 3)) for k, p in probs.items()], key=lambda x: x[1])
                self.logger.info(f"Probs: {sorted_probs}")
    
    async def _record_request_type(self, request_type: str):
        """
        Record an incoming request by type and periodically log counts.
        
        Args:
            request_type: Type of request (e.g., "completions", "classify", "health", "models", "session-stats")
        """
        current_time = time.time()
        
        async with self._request_type_lock:
            # Increment count for this request type
            self._request_type_counts[request_type] += 1
            
            # Log counts periodically (every 5 seconds)
            if current_time - self._last_request_type_log >= self._request_type_log_interval_seconds:
                self._last_request_type_log = current_time
                
                if self._request_type_counts:
                    counts_str = ", ".join([f"{rtype}: {count}" for rtype, count in sorted(self._request_type_counts.items())])
                    self.logger.info(f"Request counts (last {self._request_type_log_interval_seconds}s): {counts_str}")
                    # Reset counts after logging
                    self._request_type_counts.clear()
            
    async def handle_completions(self,request: Request):
        """
        Handle completion requests.
        
        Uses shared aiohttp session via RegistryHTTPClient for optimal performance.
        """
        start_time = time.time()
        payload = None
        
        try:
            payload = await request.json()
            model = payload.get("model")
            model_name = model if model else "unknown"
            
            # Record request type by model name
            await self._record_request_type(model_name)
            
            if not model:
                duration = time.time() - start_time
                self.logger.info(f"Processing [completions] request for model: {model_name} - duration: {duration:.3f}s")
                await self._record_request_and_log_stats(model_name, duration)
                return JSONResponse({
                    "error": "model parameter required"
                }, status_code=400)
            
            
            async with RegistryHTTPClient(
                self.registry,
                model,
                timeout=self.timeout,
                max_retries=self.max_retries,
                use_shared_session=True  # Use shared session!
            ) as client:
                start_duration = time.time() - start_time
                # self.logger.info(f"Processing [completions] request for model: {model} - duration: {start_duration:.3f}s")
                result, _ = await client.request_with_rotation("v1/completions", payload)
                duration = time.time() - start_time
                #self.logger.info(f"Completed [completions] request for model: {model} - duration: {duration:.3f}s")
                await self._record_request_and_log_stats(model, duration)
                #await self._log_probabilities_periodically()
                
                return JSONResponse(result)
            
            
            
                
        except Exception as e:
            duration = time.time() - start_time
            model_name = payload.get("model", "unknown") if payload else "unknown"
            # Record request type if payload parsing failed (payload is None)
            if payload is None:
                await self._record_request_type("unknown")
            if payload:
                self.logger.error(f"Completion error: {e} : {json.dumps(payload, indent=4)} - duration: {duration:.3f}s")
            else:
                self.logger.error(f"Completion error: {e} - duration: {duration:.3f}s")
            await self._record_request_and_log_stats(model_name, duration)
            return JSONResponse({
                "error": str(e),
                "status": "failed"
            }, status_code=500)
    
    async def handle_classify(self,request: Request):
        """
        Handle classify requests.
        
        Uses shared aiohttp session via RegistryHTTPClient for optimal performance.
        """
        payload = None
        
        try:
            payload = await request.json()
            input = payload.get("input")
            model = payload.get("model")
            
            if not model:
                return JSONResponse({
                    "error": "model parameter required"
                }, status_code=400)
            
            self.logger.info(f"Processing [classify] request for model: {model} - time : {time.time()}")
            # Use shared session via RegistryHTTPClient
            from literegistry.http import RegistryHTTPClient
            
            async with RegistryHTTPClient(
                self.registry,
                model,
                timeout=self.timeout,
                max_retries=self.max_retries,
                use_shared_session=True  # Use shared session!
            ) as client:
                result, _ = await client.request_with_rotation("classify", payload)
                return JSONResponse(result)
                
        except Exception as e:
            if payload:
                self.logger.error(f"Classify error: {e} : {json.dumps(payload, indent=4)}")
            else:
                self.logger.error(f"Classify error: {e}")
            return JSONResponse({
                "error": str(e),
                "status": "failed"
            }, status_code=500)
            
            
    def _create_app(self):
        """Create the Starlette application with lifespan management."""
        
        # Lifespan context manager for proper resource management
        # Initialize shared session at startup for optimal connection reuse
        @asynccontextmanager
        async def lifespan(app: Starlette):
            # STARTUP: Initialize shared aiohttp session
            app.state.gateway_server = self
            
            session_manager = get_session_manager()
            await session_manager.initialize()
            
            self.logger.info("✅ Gateway started with shared aiohttp session")
            self.logger.info(f"   Architecture: Single shared session for all requests")
            self.logger.info(f"   Connection pooling: Managed by aiohttp")
            
            yield
            
            # SHUTDOWN: Cleanup resources
            self.logger.info("Shutting down gateway...")
            await session_manager.shutdown()
            await self.shutdown()
            self.logger.info("✅ Gateway shutdown complete")
        
        # Create app with routes and lifespan
        app = Starlette(
            routes=[
                Route("/health", self.health_check, methods=["GET"]),
                Route("/session-stats", self.session_stats, methods=["GET"]),
                Route("/v1/models", self.list_models, methods=["GET"]),
                Route("/v1/completions", self.handle_completions, methods=["POST"]),
                Route("/classify", self.handle_classify, methods=["POST"])
            ],
            lifespan=lifespan
        )
        
        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Single error handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled exception: {exc}")
            return JSONResponse({
                "error": f"Internal server error - {exc}",
                "status": "failed"
            }, status_code=500)
        
        return app
    
    async def start(self, workers: int = 1):
        """Start the server - no restart logic."""
        if workers > 1:
            # For multiple workers, we need to use uvicorn.run() which is blocking
            # This will be handled differently in main_async
            raise ValueError("Multiple workers must be started via uvicorn CLI or use start_with_workers()")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False  # Reduce noise
        )
        
        server = uvicorn.Server(config)
        self.logger.info(f"Starting server on {self.host}:{self.port} (single worker)")
        
        try:
            await server.serve()
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
    
    def start_with_workers(self, workers: int = 1):
        """Start the server with multiple workers (blocking call)."""
        self.logger.info(f"Starting server on {self.host}:{self.port} with {workers} workers")
        # Note: uvicorn.run() with workers requires an import string, not an app object
        # The registry path should be set via REGISTRY_PATH environment variable
        # or passed to create_app() function
        uvicorn.run(
            "literegistry.gateway:create_app",
            host=self.host,
            port=self.port,
            workers=workers,
            log_level="info",
            access_log=False
        )
    
    async def shutdown(self):
        """
        Graceful shutdown.
        
        Note: Shared session cleanup is handled by session_manager in lifespan.
        """
        self._shutdown_event.set()
        
        # Close registry store
        try:
            if hasattr(self.registry, 'store'):
                await self.registry.store.close()
                self.logger.info("Registry store closed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


async def main_async(registry="redis://klone-login03.hyak.local:6379", port=8080, cache_ttl=20):
    """Simple main function without restart loops."""
    
    store = get_kvstore(registry)
    
    registry = RegistryClient(store=store, service_type="model_path", cache_ttl=cache_ttl)
    server = StarletteGatewayServer(registry, port=port)
    
    gateway_url = f"http://{socket.getfqdn()}:{port}"
    
    # Set up signal handling for single worker async mode
    def signal_handler():
        print("Received shutdown signal")
        asyncio.create_task(server.shutdown())
    
    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, signal_handler)
    
    print(f"Gateway server started at {gateway_url} (single worker)")
    
    try:
        await server.start()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        await server.shutdown()


# For uvicorn direct usage
def create_app():
    """Create app for uvicorn."""
    registry_path = os.getenv("REGISTRY_PATH", "redis://klone-login01.hyak.local:6379")
    
    try:
        
        store=get_kvstore(registry_path)
        
        registry = RegistryClient(store=store, service_type="model_path")
        server = StarletteGatewayServer(registry)
        return server.app
        
    except Exception as e:
        # Fallback error app
        app = Starlette()
        
        @app.route("/{path:path}")
        async def error_handler(request):
            return JSONResponse({
                "error": f"App creation failed: {e}",
                "status": "failed"
            }, status_code=500)
        
        return app


def main(registry="redis://klone-login03.hyak.local:6379", port=8080, workers=1):
    """Main entry point. Use workers > 1 for multi-worker mode."""
    # If multiple workers, use blocking uvicorn.run() with import string
    # Set REGISTRY_PATH env var so create_app() can use it
    if workers > 1:
        # Set environment variable for create_app() to use
        os.environ["REGISTRY_PATH"] = registry
        store = get_kvstore(registry)
        registry_client = RegistryClient(store=store, service_type="model_path")
        server = StarletteGatewayServer(registry_client, port=port)
        gateway_url = f"http://{socket.getfqdn()}:{port}"
        print(f"Gateway server starting at {gateway_url} with {workers} workers")
        server.start_with_workers(workers=workers)
    else:
        # Single worker uses async mode
        asyncio.run(main_async(registry, port))


def run_in_thread(registry="redis://klone-login03.hyak.local:6379", port=8080):
    """
    Thread-safe entry point for running the gateway in a separate thread.
    
    This function creates its own event loop and doesn't register signal handlers,
    making it safe to call from non-main threads.
    
    Args:
        registry: Registry connection string (e.g., "redis://host:port")
        port: Port to run the gateway server on
    """
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize registry components
        store = get_kvstore(registry)
        registry_client = RegistryClient(store=store, service_type="model_path")
        server = StarletteGatewayServer(registry_client, port=port)
        
        # Set up uvicorn access logger to use root logger
        import logging
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = True  # Propagate to root logger
        
        # Create uvicorn config
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,  # Enable access logs to see HTTP requests
            log_config=None  # Use default logging, which will propagate to our handlers
        )
        
        # Run server without signal handlers (since we're in a thread)
        uvicorn_server = uvicorn.Server(config)
        
        gateway_url = f"http://{socket.getfqdn()}:{port}"
        print(f"Gateway server started at {gateway_url}")
        
        # Run the server in this thread's event loop
        loop.run_until_complete(uvicorn_server.serve())
        
    except KeyboardInterrupt:
        print("Gateway interrupted")
    except Exception as e:
        print(f"Gateway error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loop.close()


if __name__ == "__main__":
    fire.Fire(main)
