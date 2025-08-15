import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
import uvicorn
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore
import pprint
from termcolor import colored


class StarletteGatewayServer:
    """Ultra-lightweight gateway server using Starlette and uvicorn."""
    
    def __init__(
        self,
        registry: RegistryClient,
        host: str = "0.0.0.0",
        port: int = 8080,
        max_parallel_requests: int = 8,
        timeout: float = 60,
        max_retries: int = 50,
    ):
        """
        Initialize the Starlette gateway server.
        
        Args:
            registry: RegistryClient instance for service discovery
            host: Host to bind the server to
            port: Port to bind the server to
            max_parallel_requests: Maximum concurrent requests per model
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts per request
        """
        self.registry = registry
        self.host = host
        self.port = port
        self.max_parallel_requests = max_parallel_requests
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create Starlette app directly
        self.app = self._create_app()
        
        # Print roster information
        asyncio.create_task(self._print_roster())
    
    def _create_app(self):
        """Create the Starlette application."""
        
        async def health_check(request: Request):
            """Health check endpoint."""
            return JSONResponse({"status": "healthy", "service": "registry-gateway"})
        
        async def list_models(request: Request):
            """List available models from registry."""
            try:
                # Get models from registry (uses cached values)
                models_data = await self.registry.models()
                models = list(models_data.keys())
                
                return JSONResponse({
                    "models": models,
                    "status": "success",
                    "data": [{"id": model, "metadata": models_data[model]} for model in models]
                })
            except Exception as e:
                self.logger.error(f"Error listing models: {str(e)}")
                return JSONResponse(
                    {"error": str(e), "status": "failed"},
                    status_code=500
                )
        
        async def handle_completions(request: Request):
            """Handle completion requests - the main POST endpoint."""
            try:
                # Get the request body
                payload = await request.json()
                
                # Extract model from payload
                model = payload.get("model")
                if not model:
                    return JSONResponse(
                        {"error": "model parameter is required in request body"},
                        status_code=400
                    )
                
                self.logger.info(f"Routing completion request to model {model}")
                
                # Create a client for this specific model
                from literegistry.http import RegistryHTTPClient
                
                async with RegistryHTTPClient(
                    self.registry,
                    model,
                    max_parallel_requests=self.max_parallel_requests,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                ) as client:
                    # Use request_with_rotation to handle the request
                    result, server_idx = await client.request_with_rotation("v1/completions", payload)
                    
                    self.logger.info(f"Completion request completed successfully from server {server_idx}")
                    return JSONResponse(result)
                    
            except Exception as e:
                self.logger.error(f"Error handling completion request: {str(e)}")
                return JSONResponse(
                    {"error": str(e), "status": "failed"},
                    status_code=500
                )
        
        async def catch_all(request: Request):
            """Handle unsupported endpoints."""
            return JSONResponse(
                {
                    "error": f"Endpoint {request.url.path} not supported",
                    "supported_endpoints": [
                        "GET /health",
                        "GET /v1/models", 
                        "POST /v1/completions"
                    ]
                },
                status_code=404
            )
        
        # Create the app with routes
        app = Starlette(
            routes=[
                Route("/health", health_check, methods=["GET"]),
                Route("/v1/models", list_models, methods=["GET"]),
                Route("/v1/completions", handle_completions, methods=["POST"]),
                Route("/{path:path}", catch_all, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
            ]
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    async def _print_roster(self):
        """Print roster information for debugging."""
        try:
            roster = await self.registry.models()
            
            pp = pprint.PrettyPrinter(indent=1, compact=True)
            
            for k, v in roster.items():
                print(f"{colored(k, 'red')}")
                for item in v:
                    print(colored("--" * 20, "blue"))
                    for key, value in item.items():
                        if key == "request_stats":
                            if "last_15_minutes_latency" in value:
                                nvalue = value["last_15_minutes"]
                                print(f"\t{colored(key, 'green')}:{colored(nvalue,'red')}")
                            else:
                                print(f"\t{colored(key, 'green')}:NO METRICS YET.")
                        else:
                            print(f"\t{colored(key, 'green')}:{value}")
        except Exception as e:
            self.logger.error(f"Error printing roster: {e}")
    
    async def start(self):
        """Start the Starlette server using uvicorn."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            loop="asyncio",
            workers=1,  # Single worker for now, can be increased
            access_log=True,
            log_level="info",
            # Starlette-specific optimizations
            limit_concurrency=1000,
            limit_max_requests=10000,
            timeout_keep_alive=30
        )
        
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    async def stop(self):
        """Stop the server."""
        if hasattr(self, 'server'):
            self.server.should_exit = True
        self.logger.info("Starlette Gateway server stopped")
    
    async def __aenter__(self):
        # Start server in background task
        self.server_task = asyncio.create_task(self.start())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'server_task'):
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        await self.stop()


# Example usage
async def main():
    """Example of how to use the Starlette gateway server."""
    # Initialize registry
    store = FileSystemKVStore("/gscratch/ark/graf/registry")
    registry = RegistryClient(store, service_type="model_path")
    
    # Create and start the gateway server
    async with StarletteGatewayServer(registry, port=8080) as gateway:
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\nShutting down Starlette gateway server...")


if __name__ == "__main__":
    asyncio.run(main())
