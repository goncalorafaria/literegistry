import aiohttp
from aiohttp import web
import asyncio
import json
import logging
from typing import Dict, Optional, Any
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore
import pprint
from termcolor import colored


class RegistryGatewayServer:
    """Gateway server that routes requests to different model servers using the registry."""
    
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
        Initialize the gateway server.
        
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
        self.app = None
        self.runner = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    

    

    
    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response({"status": "healthy", "service": "registry-gateway"})
    
    async def _handle_model_list(self, request: web.Request) -> web.Response:
        """Handle requests to list available models."""
        try:
            # Get models from registry (uses cached values)
            models_data = await self.registry.models()
            models = list(models_data.keys())
            
            return web.json_response({
                "models": models,
                "status": "success",
                "data":[{"id":model, "metadata":models_data[model]} for model in models]
            })
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return web.json_response(
                {"error": str(e), "status": "failed"},
                status=500
            )
    
    async def _handle_completions(self, request: web.Request) -> web.Response:
        """Handle completion requests - the main POST endpoint."""
        try:
            # Get the request body
            try:
                payload = await request.json()
            except json.JSONDecodeError:
                return web.json_response(
                    {"error": "Invalid JSON in request body"},
                    status=400
                )
            
            # Extract model_path from payload
            model_path = payload.get("model")
            if not model_path:
                return web.json_response(
                    {"error": "model parameter is required in request body"},
                    status=400
                )
            
            self.logger.info(f"Routing completion request to model {model_path}")
            
            # Create a client for this specific model
            from literegistry.http import RegistryHTTPClient
            
            async with RegistryHTTPClient(
                self.registry,
                model_path,
                max_parallel_requests=self.max_parallel_requests,
                timeout=self.timeout,
                max_retries=self.max_retries
            ) as client:
                # Use request_with_rotation to handle the request
                # Route to the completions endpoint on the model server
                result, server_idx = await client.request_with_rotation("v1/completions", payload)
                
                self.logger.info(f"Completion request completed successfully from server {server_idx}")
                return web.json_response(result)
                
        except Exception as e:
            self.logger.error(f"Error handling completion request: {str(e)}")
            return web.json_response(
                {"error": str(e), "status": "failed"},
                status=500
            )
    
    async def _handle_unsupported_endpoint(self, request: web.Request) -> web.Response:
        """Handle requests to unsupported endpoints."""
        return web.json_response(
            {
                "error": f"Endpoint {request.path} not supported",
                "supported_endpoints": [
                    "GET /health",
                    "GET /v1/models", 
                    "POST /v1/completions"
                ]
            },
            status=404
        )
    
    def _setup_routes(self):
        """Set up the application routes."""
        self.app = web.Application()
        
        # Health check endpoint
        self.app.router.add_get("/health", self._handle_health_check)
        
        # Model listing endpoint
        self.app.router.add_get("/v1/models", self._handle_model_list)
        
        # Main completion endpoint
        self.app.router.add_post("/v1/completions", self._handle_completions)
        
        # Catch-all for unsupported endpoints
        self.app.router.add_route("*", "/{path:.*}", self._handle_unsupported_endpoint)
        
        # Add middleware for CORS if needed
        async def cors_middleware(app, handler):
            async def middleware(request):
                if request.method == 'OPTIONS':
                    response = web.Response()
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                    return response
                
                response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            
            return middleware
        
        self.app.middlewares.append(cors_middleware)
        
        # Print the full roster information (fixed: no await outside async)
        async def print_roster():
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
            
        asyncio.create_task(print_roster())
    
    async def start(self):
        """Start the gateway server."""
        self._setup_routes()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"Gateway server started on {self.host}:{self.port}")
        self.logger.info("Available endpoints:")
        self.logger.info("  GET  /health - Health check")
        self.logger.info("  GET  /v1/models - List available models")
        self.logger.info("  POST /v1/completions - Main completion endpoint (requires 'model' in body)")
    
    async def stop(self):
        """Stop the gateway server."""
        if self.runner:
            await self.runner.cleanup()
        self.logger.info("Gateway server stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Example usage
async def main():
    """Example of how to use the gateway server."""
    # Initialize registry
    store = FileSystemKVStore("/gscratch/ark/graf/registry")
    registry = RegistryClient(store, service_type="model_path")
    
    # Create and start the gateway server
    async with RegistryGatewayServer(registry, port=8080) as gateway:
        # Keep the server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\nShutting down gateway server...")


if __name__ == "__main__":
    asyncio.run(main())
