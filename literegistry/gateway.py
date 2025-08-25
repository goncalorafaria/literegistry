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
from literegistry.kvstore import FileSystemKVStore, RedisKVStore
import pprint
from termcolor import colored


class StarletteGatewayServer:
    """Ultra-lightweight gateway server using Starlette and uvicorn."""
    
    def __init__(
        self,
        registry: RegistryClient,
        host: str = "0.0.0.0",
        port: int = 8080,
        timeout: float = 60,
        max_retries: int = 5,
        workers: int = 1,
        max_parallel_requests: int = 512,
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
        self.timeout = timeout
        self.max_retries = max_retries
        self.workers = workers
        self.max_parallel_requests = max_parallel_requests
        
        # Track startup time for uptime monitoring
        self._start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create Starlette app directly
        self.app = self._create_app()
        
        # Print roster information only if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._print_roster())
        except RuntimeError:
            # No running event loop, skip roster printing
            pass
    
    def _create_app(self):
        """Create the Starlette application."""
        
        async def health_check(request: Request):
            """Health check endpoint."""
            try:
                # Check if the registry is accessible
                models_data = await self.registry.models()
                
                # Get server status information
                server_status = "running"
                if hasattr(self, 'server') and self.server:
                    if hasattr(self.server, 'should_exit') and self.server.should_exit:
                        server_status = "shutting_down"
                else:
                    server_status = "not_started"
                
                # Get task status
                task_status = "unknown"
                if hasattr(self, 'server_task'):
                    if self.server_task.done():
                        if self.server_task.cancelled():
                            task_status = "cancelled"
                        elif self.server_task.exception():
                            task_status = "failed"
                        else:
                            task_status = "completed"
                    else:
                        task_status = "running"
                
                return JSONResponse({
                    "status": "healthy", 
                    "service": "registry-gateway",
                    "models_count": len(models_data),
                    "server_status": server_status,
                    "task_status": task_status,
                    "timestamp": asyncio.get_event_loop().time(),
                    "uptime": getattr(self, '_start_time', 0),
                    "endpoints": ["/health", "/v1/models", "/v1/completions", "/status", "/metrics", "/debug"]
                })
            except Exception as e:
                # Even if registry check fails, return unhealthy but don't crash
                self.logger.warning(f"Health check failed: {e}")
                return JSONResponse({
                    "status": "unhealthy", 
                    "service": "registry-gateway",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "server_status": getattr(self, 'server', None) is not None,
                    "task_status": getattr(self, 'server_task', None) is not None,
                    "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
                }, status_code=503)
        
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
            payload = None  # Initialize payload variable
            try:
                # Get the request body
                payload = await request.json()
                
                # Log the incoming payload for debugging
                self.logger.info(f"Received completion request with payload: {json.dumps(payload, indent=2)}")
                
                # Extract model from payload
                model = payload.get("model")
                if not model:
                    return JSONResponse(
                        {"error": "model parameter is required in request body"},
                        status_code=400
                    )
                
                self.logger.info(f"Routing completion request to model {model}")
                
                # Debug: Check what servers are available for this model
                try:
                    models_data = await self.registry.models()
                    if model in models_data:
                        servers = models_data[model]
                        self.logger.info(f"Found {len(servers)} servers for model {model}: {[s.get('uri', 'unknown') for s in servers]}")
                    else:
                        self.logger.warning(f"Model {model} not found in registry. Available models: {list(models_data.keys())}")
                except Exception as e:
                    self.logger.warning(f"Could not check model registry: {e}")
                
                # Create a client for this specific model
                from literegistry.http import RegistryHTTPClient
                
                async with RegistryHTTPClient(
                    self.registry,
                    model,
                    max_parallel_requests=self.max_parallel_requests,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                ) as client:
                    try:
                        # Use request_with_rotation to handle the request
                        result, server_idx = await client.request_with_rotation("v1/completions", payload)
                        
                        self.logger.info(f"Completion request completed successfully from server {server_idx}")
                        return JSONResponse(result)
                    except Exception as client_error:
                        # Handle client-specific errors more gracefully
                        self.logger.error(f"Client error for model {model}: {str(client_error)}")
                        self.logger.error(f"Client error type: {type(client_error).__name__}")
                        
                        # Check if it's a 400 Bad Request error
                        if "400" in str(client_error) and "Bad Request" in str(client_error):
                            self.logger.warning(f"Bad request to model {model}, returning 400 response")
                            return JSONResponse(
                                {
                                    "error": "Bad request", 
                                    "status": "failed", 
                                    "message": f"Invalid request to model {model}",
                                    "payload": {
                                        "model": model,
                                        "request_body": payload,
                                        "error_details": str(client_error)
                                    }
                                },
                                status_code=400
                            )
                        
                        # Re-raise the error to be handled by the outer exception handler
                        raise client_error
                    
            except Exception as e:
                # Log the error but don't let it shut down the server
                self.logger.error(f"Error handling completion request: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error details: {str(e)}")
                if payload:
                    self.logger.error(f"Error payload: {json.dumps(payload, indent=2)}")
                else:
                    self.logger.error("Error payload: Could not parse request body")
                
                # Return error response instead of letting exception propagate
                return JSONResponse(
                    {"error": str(e), "status": "failed", "message": "Completion request failed"},
                    status_code=500
                )
        
        async def handle_generic_route(request: Request):
            """Handle any generic endpoint and route to model servers."""
            path = request.url.path.lstrip("/")
            
            # Skip our specific endpoints
            if path in ["health", "v1/models", "v1/completions"]:
                return JSONResponse(
                    {
                        "error": f"Endpoint /{path} is handled by specific handler",
                        "supported_endpoints": [
                            "GET /health",
                            "GET /v1/models", 
                            "POST /v1/completions"
                        ]
                    },
                    status_code=400
                )
            
            # Extract model from query params first
            model = request.query_params.get("model")
            
            # If not in query params, try to get from request body
            if not model and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.json()
                    model = body.get("model")
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            if not model:
                return JSONResponse(
                    {
                        "error": "model parameter is required (in query params or request body)",
                        "example": f"GET /{path}?model=gpt-3 or POST /{path} with {{'model': 'gpt-3'}} in body"
                    },
                    status_code=400
                )
            
            self.logger.info(f"Routing generic request to model {model}, endpoint {path}")
            
            # Debug: Check what servers are available for this model
            try:
                models_data = await self.registry.models()
                if model in models_data:
                    servers = models_data[model]
                    self.logger.info(f"Found {len(servers)} servers for model {model}: {[s.get('uri', 'unknown') for s in servers]}")
                else:
                    self.logger.warning(f"Model {model} not found in registry. Available models: {list(models_data.keys())}")
            except Exception as e:
                self.logger.warning(f"Could not check model registry: {e}")
            
            try:
                # Create a client for this specific model
                from literegistry.http import RegistryHTTPClient
                
                async with RegistryHTTPClient(
                    self.registry,
                    model,
                    max_parallel_requests=self.max_parallel_requests,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                ) as client:
                    # Prepare payload
                    payload = {}
                    
                    # Add query parameters
                    if request.query_params:
                        payload.update(dict(request.query_params))
                    
                    # Add request body for POST/PUT/PATCH
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload.update(body)
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    
                    try:
                        # Use request_with_rotation to handle the request
                        result, server_idx = await client.request_with_rotation(path, payload)
                        
                        self.logger.info(f"Generic request completed successfully from server {server_idx}")
                        return JSONResponse(result)
                    except Exception as client_error:
                        # Handle client-specific errors more gracefully
                        self.logger.error(f"Client error for model {model}, endpoint {path}: {str(client_error)}")
                        self.logger.error(f"Client error type: {type(client_error).__name__}")
                        
                        # Check if it's a 400 Bad Request error
                        if "400" in str(client_error) and "Bad Request" in str(client_error):
                            self.logger.warning(f"Bad request to model {model}, endpoint {path}, returning 400 response")
                            return JSONResponse(
                                {
                                    "error": "Bad request", 
                                    "status": "failed", 
                                    "message": f"Invalid request to {path}",
                                    "payload": {
                                        "model": model,
                                        "endpoint": path,
                                        "request_body": payload,
                                        "error_details": str(client_error)
                                    }
                                },
                                status_code=400
                            )
                        
                        # Re-raise the error to be handled by the outer exception handler
                        raise client_error
                    
            except Exception as e:
                # Log the error but don't let it shut down the server
                self.logger.error(f"Error handling generic request for {path}: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error details: {str(e)}")
                self.logger.error(f"Error payload: {payload}")
                # Print payload for debugging
                
                # Return error response instead of letting exception propagate
                return JSONResponse(
                    {"error": str(e), "status": "failed", "message": f"Request to {path} failed"},
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
                    ],
                    "note": "Use ?model=model_name for generic routing to any endpoint"
                },
                status_code=404
            )
            
         # Define the global exception handler function
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler to prevent server shutdown on errors."""
            self.logger.error(f"Unhandled exception in {request.url.path}: {str(exc)}")
            return JSONResponse(
                {"error": str(exc), "status": "failed", "message": "Internal server error"},
                status_code=500
            )
        
        # Create the app with routes
        app = Starlette(
            routes=[
                Route("/health", health_check, methods=["GET"]),
                Route("/v1/models", list_models, methods=["GET"]),
                Route("/v1/completions", handle_completions, methods=["POST"]),
                # Generic route handler for any other endpoint
                Route("/{path:path}", handle_generic_route, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
            ],
            # Add error handling configuration
            exception_handlers={
                Exception: global_exception_handler
            }
        )
        
       
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add error handling middleware
        @app.middleware("http")
        async def error_handling_middleware(request: Request, call_next):
            """Middleware to catch and handle any errors that might occur."""
            try:
                response = await call_next(request)
                return response
            except asyncio.CancelledError:
                # Handle cancelled requests gracefully
                self.logger.warning(f"Request cancelled for {request.url.path}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                return JSONResponse(
                    {
                        "error": "Request cancelled", 
                        "status": "cancelled", 
                        "message": "Request was cancelled",
                        "payload": payload_info
                    },
                    status_code=499  # Client Closed Request
                )
            except Exception as e:
                # Log the error but don't let it shut down the server
                self.logger.error(f"Error in middleware for {request.url.path}: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error details: {str(e)}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                # Return error response instead of letting exception propagate
                return JSONResponse(
                    {
                        "error": str(e), 
                        "status": "failed", 
                        "message": "Request processing error",
                        "payload": payload_info,
                        "error_type": type(e).__name__
                    },
                    status_code=500
                )
        
        # Add additional error handling for unhandled exceptions
        @app.exception_handler(500)
        async def internal_error_handler(request: Request, exc: Exception):
            """Handle internal server errors."""
            self.logger.error(f"Internal server error in {request.url.path}: {str(exc)}")
            return JSONResponse(
                {"error": "Internal server error", "status": "failed", "message": "An unexpected error occurred"},
                status_code=500
            )
        
        # Add error handling for 404 errors
        @app.exception_handler(404)
        async def not_found_handler(request: Request, exc: Exception):
            """Handle 404 errors."""
            return JSONResponse(
                {"error": "Not found", "status": "failed", "message": f"Endpoint {request.url.path} not found"},
                status_code=404
            )
        
        # Add error handling for 400 errors
        @app.exception_handler(400)
        async def bad_request_handler(request: Request, exc: Exception):
            """Handle 400 errors."""
            self.logger.warning(f"Bad request to {request.url.path}: {str(exc)}")
            
            # Try to extract payload information for debugging
            payload_info = {}
            try:
                # Get query parameters
                if request.query_params:
                    payload_info["query_params"] = dict(request.query_params)
                
                # Get request body for POST/PUT/PATCH requests
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.json()
                        payload_info["body"] = body
                    except (json.JSONDecodeError, AttributeError):
                        payload_info["body"] = "Could not parse request body"
                
                # Get headers (excluding sensitive ones)
                headers = dict(request.headers)
                sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                for header in sensitive_headers:
                    if header in headers:
                        headers[header] = "***REDACTED***"
                payload_info["headers"] = headers
                
            except Exception as e:
                payload_info["error"] = f"Could not extract payload: {str(e)}"
            
            return JSONResponse(
                {
                    "error": "Bad request", 
                    "status": "failed", 
                    "message": f"Invalid request to {request.url.path}",
                    "payload": payload_info,
                    "exception": str(exc)
                },
                status_code=400
            )
        
        # Add error handling for 408 timeout errors
        @app.exception_handler(408)
        async def timeout_handler(request: Request, exc: Exception):
            """Handle 408 timeout errors."""
            self.logger.warning(f"Request timeout to {request.url.path}: {str(exc)}")
            return JSONResponse(
                {"error": "Request timeout", "status": "failed", "message": f"Request to {request.url.path} timed out"},
                status_code=408
            )
        
        # Add error handling for 422 errors (validation errors)
        @app.exception_handler(422)
        async def validation_error_handler(request: Request, exc: Exception):
            """Handle 422 validation errors."""
            self.logger.warning(f"Validation error for {request.url.path}: {str(exc)}")
            return JSONResponse(
                {"error": "Validation error", "status": "failed", "message": f"Request validation failed for {request.url.path}"},
                status_code=422
            )
        
        # Add error handling for 503 errors (service unavailable)
        @app.exception_handler(503)
        async def service_unavailable_handler(request: Request, exc: Exception):
            """Handle 503 service unavailable errors."""
            self.logger.warning(f"Service unavailable for {request.url.path}: {str(exc)}")
            return JSONResponse(
                {"error": "Service unavailable", "status": "failed", "message": f"Service temporarily unavailable for {request.url.path}"},
                status_code=503
            )
        
        # Add timeout handling middleware
        @app.middleware("http")
        async def timeout_middleware(request: Request, call_next):
            """Middleware to handle request timeouts gracefully."""
            try:
                # Set a reasonable timeout for the request
                import asyncio
                response = await asyncio.wait_for(call_next(request), timeout=120.0)  # 2 minute timeout
                return response
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout for {request.url.path}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                return JSONResponse(
                    {
                        "error": "Request timeout", 
                        "status": "failed", 
                        "message": "Request timed out",
                        "payload": payload_info
                    },
                    status_code=408
                )
            except asyncio.CancelledError:
                # Handle cancelled requests gracefully
                self.logger.warning(f"Request cancelled in timeout middleware for {request.url.path}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                return JSONResponse(
                    {
                        "error": "Request cancelled", 
                        "status": "cancelled", 
                        "message": "Request was cancelled",
                        "payload": payload_info
                    },
                    status_code=499  # Client Closed Request
                )
            except Exception as e:
                # Let other middleware handle other errors
                raise e
        
        # Add connection error handling middleware
        @app.middleware("http")
        async def connection_error_middleware(request: Request, call_next):
            """Middleware to handle connection errors gracefully."""
            try:
                response = await call_next(request)
                return response
            except ConnectionError as e:
                self.logger.error(f"Connection error for {request.url.path}: {str(e)}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                return JSONResponse(
                    {
                        "error": "Connection error", 
                        "status": "failed", 
                        "message": "Connection to backend service failed",
                        "payload": payload_info,
                        "connection_error": str(e)
                    },
                    status_code=503
                )
            except Exception as e:
                # Let other middleware handle other errors
                raise e
        
        # Add request cancellation handling middleware
        @app.middleware("http")
        async def cancellation_middleware(request: Request, call_next):
            """Middleware to handle request cancellations gracefully."""
            try:
                response = await call_next(request)
                return response
            except asyncio.CancelledError:
                self.logger.warning(f"Request cancelled in cancellation middleware for {request.url.path}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                return JSONResponse(
                    {
                        "error": "Request cancelled", 
                        "status": "cancelled", 
                        "message": "Request was cancelled",
                        "payload": payload_info
                    },
                    status_code=499  # Client Closed Request
                )
            except Exception as e:
                # Let other middleware handle other errors
                raise e
        
        # Add general error handling middleware as a last resort
        @app.middleware("http")
        async def general_error_middleware(request: Request, call_next):
            """General error handling middleware as a last resort."""
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                # Log the error but don't let it shut down the server
                self.logger.error(f"Unhandled error in general middleware for {request.url.path}: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                
                # Try to extract payload information for debugging
                payload_info = {}
                try:
                    # Get query parameters
                    if request.query_params:
                        payload_info["query_params"] = dict(request.query_params)
                    
                    # Get request body for POST/PUT/PATCH requests
                    if request.method in ["POST", "PUT", "PATCH"]:
                        try:
                            body = await request.json()
                            payload_info["body"] = body
                        except (json.JSONDecodeError, AttributeError):
                            payload_info["body"] = "Could not parse request body"
                    
                    # Get headers (excluding sensitive ones)
                    headers = dict(request.headers)
                    sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                    for header in sensitive_headers:
                        if header in headers:
                            headers[header] = "***REDACTED***"
                    payload_info["headers"] = headers
                    
                except Exception as payload_error:
                    payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
                
                # Return error response instead of letting exception propagate
                return JSONResponse(
                    {
                        "error": "Internal server error", 
                        "status": "failed", 
                        "message": "An unexpected error occurred",
                        "payload": payload_info,
                        "error_type": type(e).__name__,
                        "error_details": str(e)
                    },
                    status_code=500
                )
        
        # Add a final safety net for any unhandled exceptions
        @app.exception_handler(Exception)
        async def final_exception_handler(request: Request, exc: Exception):
            """Final exception handler to catch any unhandled exceptions."""
            self.logger.error(f"Final exception handler caught error for {request.url.path}: {str(exc)}")
            self.logger.error(f"Exception type: {type(exc).__name__}")
            
            # Try to extract payload information for debugging
            payload_info = {}
            try:
                # Get query parameters
                if request.query_params:
                    payload_info["query_params"] = dict(request.query_params)
                
                # Get request body for POST/PUT/PATCH requests
                if request.method in ["POST", "PUT", "PATCH"]:
                    try:
                        body = await request.json()
                        payload_info["body"] = body
                    except (json.JSONDecodeError, AttributeError):
                        payload_info["body"] = "Could not parse request body"
                
                # Get headers (excluding sensitive ones)
                headers = dict(request.headers)
                sensitive_headers = ['authorization', 'cookie', 'x-api-key']
                for header in sensitive_headers:
                    if header in headers:
                        headers[header] = "***REDACTED***"
                payload_info["headers"] = headers
                
            except Exception as payload_error:
                payload_info["error"] = f"Could not extract payload: {str(payload_error)}"
            
            # Return a safe error response with payload information
            return JSONResponse(
                {
                    "error": "Internal server error", 
                    "status": "failed", 
                    "message": "An unexpected error occurred",
                    "payload": payload_info,
                    "exception_type": type(exc).__name__,
                    "exception_details": str(exc)
                },
                status_code=500
            )
        
        # Add a health check endpoint that can help monitor server status
        @app.route("/status", methods=["GET"])
        async def status_check(request: Request):
            """Status check endpoint to monitor server health."""
            try:
                # Check if the registry is accessible
                models_data = await self.registry.models()
                return JSONResponse({
                    "status": "healthy", 
                    "service": "registry-gateway",
                    "models_count": len(models_data),
                    "timestamp": asyncio.get_event_loop().time()
                })
            except Exception as e:
                # Even if registry check fails, return unhealthy but don't crash
                self.logger.warning(f"Status check failed: {e}")
                return JSONResponse({
                    "status": "unhealthy", 
                    "service": "registry-gateway",
                    "error": str(e),
                    "timestamp": asyncio.get_event_loop().time()
                }, status_code=503)
        
        # Add a metrics endpoint for monitoring
        @app.route("/metrics", methods=["GET"])
        async def metrics_endpoint(request: Request):
            """Metrics endpoint for monitoring server performance."""
            try:
                # Return basic metrics
                return JSONResponse({
                    "uptime": asyncio.get_event_loop().time(),
                    "service": "registry-gateway",
                    "endpoints": ["/health", "/v1/models", "/v1/completions", "/status", "/metrics"]
                })
            except Exception as e:
                self.logger.warning(f"Metrics endpoint failed: {e}")
                return JSONResponse({
                    "error": "Metrics unavailable",
                    "service": "registry-gateway"
                }, status_code=503)
        
        # Add a debug endpoint for troubleshooting
        @app.route("/debug", methods=["GET"])
        async def debug_endpoint(request: Request):
            """Debug endpoint for troubleshooting server issues."""
            try:
                # Return debug information
                return JSONResponse({
                    "service": "registry-gateway",
                    "debug_info": {
                        "event_loop_running": asyncio.get_event_loop().is_running(),
                        "current_time": asyncio.get_event_loop().time(),
                        "available_endpoints": ["/health", "/v1/models", "/v1/completions", "/status", "/metrics", "/debug"]
                    }
                })
            except Exception as e:
                self.logger.warning(f"Debug endpoint failed: {e}")
                return JSONResponse({
                    "error": "Debug information unavailable",
                    "service": "registry-gateway"
                }, status_code=503)
        
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
        while True:  # Keep trying to start the server
            try:
                config = uvicorn.Config(
                    app=self.app,
                    host=self.host,
                    port=self.port,
                    loop="asyncio",
                    workers=self.workers,  # Single worker for now, can be increased
                    access_log=True,
                    log_level="info",
                    # Starlette-specific optimizations
                    limit_concurrency=1000,
                    limit_max_requests=10000,
                    timeout_keep_alive=30,
                    # Additional performance optimizations
                    backlog=2048,
                    use_colors=False,  # Disable colors in production
                    date_header=False,  # Disable date header for performance
                    server_header=False,  # Disable server header for security
                    # Error handling and resilience
                    reload=False,  # Disable reload to prevent issues
                    reload_dirs=None,
                    reload_excludes=None,
                    reload_includes=None,
                    # Prevent server shutdown on errors
                    #exit_on_app_factory_error=False,
                    # Additional error handling
                    log_config=None,  # Use default logging
                    # Additional resilience settings
                    timeout_graceful_shutdown=60,  # Increased timeout
                    # Prevent worker process crashes from shutting down the server
                    #worker_exit_on_app_factory_error=False,
                    # Handle connection errors gracefully
                    # Additional error resilience
                    # limit_max_requests_per_worker=10000
                )
                
                self.server = uvicorn.Server(config)
                self.logger.info(f"Starting uvicorn server on {self.host}:{self.port}")
                
                try:
                    await self.server.serve()
                    # If we get here, the server exited normally
                    self.logger.info("Uvicorn server exited normally")
                    break  # Exit the restart loop
                except Exception as e:
                    self.logger.error(f"Uvicorn server error: {e}")
                    self.logger.error(f"Error type: {type(e).__name__}")
                    
                    # Don't let the server shut down on errors
                    self.logger.info("Server encountered error, attempting to restart...")
                    
                    # Check if it's a connection-related error
                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        self.logger.warning("Connection or timeout error detected, server will restart")
                    else:
                        self.logger.error(f"Non-connection error: {e}, but server will restart")
                    
                    # Wait a bit before restarting to prevent rapid restart loops
                    await asyncio.sleep(2)
                    
                    # Continue the while loop to restart
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error creating uvicorn server: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                # Wait a bit and try to continue
                await asyncio.sleep(5)  # Longer wait for configuration errors
                # Continue the while loop to retry
                continue
    
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
        try:
            if hasattr(self, 'server_task'):
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            await self.stop()
            # Ensure registry and store are properly closed
            if hasattr(self, 'registry') and hasattr(self.registry, 'store'):
                await self.registry.store.close()
        except Exception as e:
            self.logger.error(f"Error during gateway cleanup: {e}")


# Example usage
async def main(registry_path = "redis://klone-login01.hyak.local:6379",port=8080, workers=8):
    """Example of how to use the Starlette gateway server."""
    #"/gscratch/ark/graf/registry" # "redis://klone-login01.hyak.local:6379"
    # Initialize registry
    store = None
    registry = None
    gateway = None
    
    try:
        if "redis://" in registry_path:
            store = RedisKVStore(registry_path)
        else:
            store = FileSystemKVStore(registry_path)
            
        registry = RegistryClient(store=store, service_type="model_path")
        
        print(f"🚀 Starting Starlette Gateway Server on port {port}")
        print(f"📊 Workers: {workers}")
        print(f"🗄️  Registry: {registry_path}")
        print("=" * 60)
        
        # Create and start the gateway server
        async with StarletteGatewayServer(registry, port=port, workers=workers) as gateway:
            print("✅ Gateway server started successfully!")
            print("🔄 Server is running and will auto-restart on errors")
            print("⏹️  Press Ctrl+C to stop the server")
            print("=" * 60)
            
            # Keep the server running
            while True:
                try:
                    # Use a more robust way to keep the server running
                    await asyncio.sleep(1)  # Check every second instead of blocking forever
                    
                    # Check if the server task is still running
                    if hasattr(gateway, 'server_task') and not gateway.server_task.done():
                        # Server is running normally
                        continue
                    else:
                        print("⚠️  Server task appears to have stopped, checking status...")
                        
                        # Check if the server is still running
                        if hasattr(gateway, 'server') and gateway.server:
                            print("✅ Server appears to be running, continuing...")
                        else:
                            print("🔄 Server appears to have stopped, it will restart automatically...")
                            # The server should restart automatically via the start() method
                            await asyncio.sleep(2)
                            
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down Starlette gateway server...")
                    break
                except Exception as e:
                    print(f"\n⚠️  Unexpected error in main loop: {e}")
                    print(f"📝 Error type: {type(e).__name__}")
                    # Don't exit, try to keep the server running
                    print("🔄 Attempting to continue server operation...")
                    
                    # Wait a bit and continue
                    await asyncio.sleep(2)
                    
                    # Try to continue running
                    print("🔄 Attempting to continue server operation...")
                    # Continue the loop instead of exiting
                    # Add some additional error handling
                    try:
                        # Check if the server is still running
                        if hasattr(gateway, 'server') and gateway.server:
                            print("✅ Server appears to be running, continuing...")
                        else:
                            print("🔄 Server appears to have stopped, attempting to restart...")
                            # The server should restart automatically via the start() method
                    except Exception as e2:
                        print(f"❌ Error checking server status: {e2}")
                        # Continue anyway
                    
                    # Add a small delay before continuing to prevent rapid error loops
                    await asyncio.sleep(3)
                    
    except Exception as e:
        print(f"❌ Fatal error in main: {e}")
        print(f"📝 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Ensure cleanup happens even on errors
        print("🧹 Cleaning up resources...")
        try:
            if registry and hasattr(registry, 'store'):
                await registry.store.close()
                print("✅ Registry store closed")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
        print("👋 Goodbye!")


# Expose app for direct uvicorn usage
def create_app():
    """Create Starlette app for direct uvicorn usage."""
    store = None
    registry = None
    gateway = None
    
    try:
        # Configuration - can be overridden via environment variables
        import os
        registry_path = os.getenv("REGISTRY_PATH", "redis://klone-login01.hyak.local:6379")
        port = int(os.getenv("PORT", "8080"))
        workers = int(os.getenv("WORKERS", "1"))
        
        print(f"🚀 Creating Starlette app for uvicorn usage")
        print(f"📊 Port: {port}, Workers: {workers}")
        print(f"🗄️  Registry: {registry_path}")
        
        if "redis://" in registry_path:
            store = RedisKVStore(registry_path)
        else:
            store = FileSystemKVStore(registry_path)
        
        registry = RegistryClient(store=store, service_type="model_path")
        gateway = StarletteGatewayServer(registry, port=port, workers=workers)
        
        print("✅ Starlette app created successfully")
        
        # Add cleanup on app shutdown
        @gateway.app.on_event("shutdown")
        async def shutdown_event():
            print("🛑 App shutdown event triggered")
            try:
                if registry and hasattr(registry, 'store'):
                    await registry.store.close()
                    print("✅ Registry store closed during shutdown")
                if gateway:
                    await gateway.stop()
                    print("✅ Gateway stopped during shutdown")
            except Exception as e:
                print(f"❌ Error during app shutdown: {e}")
        
        # Add startup event for better logging
        @gateway.app.on_event("startup")
        async def startup_event():
            print("🚀 App startup event triggered")
            print(f"✅ Gateway server ready on port {port}")
        
        return gateway.app
    except Exception as e:
        # If there's an error creating the app, create a minimal error app
        print(f"❌ Error creating main app: {e}")
        print(f"📝 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Create a minimal error app that won't crash
        error_app = Starlette(
            routes=[
                Route("/{path:path}", lambda request: JSONResponse(
                    {"error": str(e), "status": "failed", "message": "App creation failed"},
                    status_code=500
                ))
            ]
        )
        
        # Add global exception handler
        @error_app.exception_handler(Exception)
        async def error_handler(request: Request, exc: Exception):
            return JSONResponse(
                {"error": str(exc), "status": "failed", "message": "Internal server error"},
                status_code=500
            )
        
        print("⚠️  Created fallback error app - check configuration")
        return error_app

# This allows: uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080
app = create_app()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    try:
        # Try to clean up gracefully
        print("Cleaning up resources...")
        # Add any cleanup code here if needed
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("Attempting to continue...")
        try:
            # Try to run again
            asyncio.run(main())
        except Exception as e2:
            print(f"❌ Second fatal error: {e2}")
            sys.exit(1)
