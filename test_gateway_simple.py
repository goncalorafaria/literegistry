#!/usr/bin/env python3
"""
Simple test script for the Registry Gateway Server.
Tests the fixed models endpoint.
"""

import asyncio
import aiohttp
from literegistry.gateway import RegistryGatewayServer
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore


async def test_gateway():
    """Test the gateway server functionality."""
    
    # Initialize registry and register some test servers
    store = FileSystemKVStore("/gscratch/ark/graf/registry")
    registry = RegistryClient(store, service_type="model_path")
    
    # Register some test servers for different models
    await registry.register_server("http://localhost:8001", {"model_path": "gpt-3"})
    await registry.register_server("http://localhost:8002", {"model_path": "gpt-3"})
    await registry.register_server("http://localhost:8003", {"model_path": "gpt-4"})
    await registry.register_server("http://localhost:8004", {"model_path": "gpt-4"})
    
    print("Registered test servers:")
    print("  gpt-3: localhost:8001, localhost:8002")
    print("  gpt-4: localhost:8003, localhost:8004")
    
    # Start the gateway server
    gateway = RegistryGatewayServer(registry, port=8080)
    await gateway.start()
    
    print(f"\nGateway server started on port 8080")
    print("Testing endpoints...")
    
    try:
        # Test health check
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8080/health") as response:
                result = await response.json()
                print(f"Health check: {result}")
            
            # Test model listing (this should work now!)
            async with session.get("http://localhost:8080/v1/models") as response:
                result = await response.json()
                print(f"Models: {result}")
            
            # Test completion request
            payload = {"model": "gpt-3", "prompt": "Hello, how are you?", "max_tokens": 50}
            async with session.post(
                "http://localhost:8080/v1/completions",
                json=payload
            ) as response:
                if response.status == 500:
                    # Expected since test servers don't exist
                    result = await response.json()
                    print(f"Completion request (expected error): {result}")
                else:
                    result = await response.json()
                    print(f"Completion request: {result}")
    
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        # Stop the gateway server
        await gateway.stop()
        print("\nGateway server stopped")


if __name__ == "__main__":
    asyncio.run(test_gateway())










