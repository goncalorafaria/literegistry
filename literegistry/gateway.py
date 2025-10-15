"""
LiteRegistry Gateway Server

This module provides a lightweight gateway server using Starlette and uvicorn.

RECOMMENDED USAGE (most efficient):
    uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080 --workers 8

ALTERNATIVE USAGE (for development/testing):
    python -m literegistry.gateway

Environment variables:
    REGISTRY_PATH: Registry connection string (default: redis://klone-login01.hyak.local:6379)
    PORT: Server port (default: 8080)
    WORKERS: Number of worker processes (default: 1)
"""

import asyncio
import json
import logging
import signal
import sys
from typing import Dict, List, Optional, Any
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
import uvicorn
from literegistry.client import RegistryClient
from literegistry import get_kvstore
import pprint
from termcolor import colored
import os
import socket
import fire


class StarletteGatewayServer:
    """Ultra-lightweight gateway server using Starlette and uvicorn."""

    def __init__(
        self,
        registry: RegistryClient,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: float = 60,
        max_retries: int = 3
    ):
        self.registry = registry
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self._shutdown_event = asyncio.Event()
        
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
            
    async def handle_completions(self,request: Request):
        """Handle completion requests."""
        try:
            payload = await request.json()
            model = payload.get("model")
            
            if not model:
                return JSONResponse({
                    "error": "model parameter required"
                }, status_code=400)
            
            self.logger.info(f"Processing request for model: {model}")
            
            # Use the HTTP client
            from literegistry.http import RegistryHTTPClient
            
            async with RegistryHTTPClient(
                self.registry, 
                model,
                timeout=self.timeout,
                max_retries=self.max_retries
            ) as client:
                result, _ = await client.request_with_rotation("v1/completions", payload)
                return JSONResponse(result)
                
        except Exception as e:
            #self.logger.error(f"Completion error: {e}")
            self.logger.error(f"Completion error: {e} : {json.dumps(payload, indent=4)}")
            return JSONResponse({
                "error": str(e),
                "status": "failed"
            }, status_code=500)
                
    def _create_app(self):
        """Create the Starlette application with minimal error handling."""
        
        
        # Create app with routes
        app = Starlette(
            routes=[
                Route("/health", self.health_check, methods=["GET"]),
                Route("/v1/models", self.list_models, methods=["GET"]),
                Route("/v1/completions", self.handle_completions, methods=["POST"])
            ]
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
    
    async def start(self):
        """Start the server - no restart logic."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False  # Reduce noise
        )
        
        server = uvicorn.Server(config)
        self.logger.info(f"Starting server on {self.host}:{self.port}")
        
        try:
            await server.serve()
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown."""
        self._shutdown_event.set()
        try:
            if hasattr(self.registry, 'store'):
                await self.registry.store.close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


async def main_async(registry="redis://klone-login03.hyak.local:6379", port=8080):
    """Simple main function without restart loops."""
    
    store = get_kvstore(registry)
    
    registry = RegistryClient(store=store, service_type="model_path")
    server = StarletteGatewayServer(registry, port=port)
    
    # Set up signal handling
    def signal_handler():
        print("Received shutdown signal")
        asyncio.create_task(server.shutdown())
    
    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, signal_handler)
    
    gateway_url=f"http://{socket.getfqdn()}:{port}"
    
    print(f"Gateway server started at {gateway_url}")
    
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


def main(registry="redis://klone-login03.hyak.local:6379", port=8080):
    asyncio.run(main_async(registry, port))

if __name__ == "__main__":
    fire.Fire(main)