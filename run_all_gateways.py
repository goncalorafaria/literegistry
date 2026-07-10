#!/usr/bin/env python3
"""
Script to run all three gateway implementations simultaneously:
- aiohttp gateway on port 8080
- FastAPI gateway on port 8081  
- Starlette gateway on port 8082

This allows for direct performance comparison.
"""

import asyncio
import signal
import sys
from literegistry.gateway import RegistryGatewayServer
from literegistry.gateway_fast import FastAPIGatewayServer
from literegistry.gateway_starlette import StarletteGatewayServer
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore


async def run_gateways():
    """Run all gateway implementations simultaneously."""
    
    # Initialize registry
    store = FileSystemKVStore("/gscratch/ark/graf/registry")
    registry = RegistryClient(store, service_type="model_path")
    
    print("🚀 Starting All Gateway Implementations")
    print("=" * 50)
    print("aiohttp Gateway:    http://localhost:8080")
    print("FastAPI Gateway:    http://localhost:8081")
    print("Starlette Gateway:  http://localhost:8082")
    print("=" * 50)
    
    # Start all gateways
    gateways = []
    
    try:
        # Start aiohttp gateway
        print("Starting aiohttp gateway...")
        aiohttp_gateway = RegistryGatewayServer(registry, port=8080)
        await aiohttp_gateway.start()
        gateways.append(("aiohttp", aiohttp_gateway))
        print("✅ aiohttp gateway started on port 8080")
        
        # Start FastAPI gateway
        print("Starting FastAPI gateway...")
        fastapi_gateway = FastAPIGatewayServer(registry, port=8081)
        await fastapi_gateway.start()
        gateways.append(("FastAPI", fastapi_gateway))
        print("✅ FastAPI gateway started on port 8081")
        
        # Start Starlette gateway
        print("Starting Starlette gateway...")
        starlette_gateway = StarletteGatewayServer(registry, port=8082)
        await starlette_gateway.start()
        gateways.append(("Starlette", starlette_gateway))
        print("✅ Starlette gateway started on port 8082")
        
        print("\n🎯 All gateways are running!")
        print("Run benchmark: python benchmark_all_gateways.py")
        print("Press Ctrl+C to stop all gateways")
        
        # Keep running until interrupted
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\n🛑 Shutting down all gateways...")
    
    except Exception as e:
        print(f"❌ Error starting gateways: {e}")
    
    finally:
        # Stop all gateways
        print("Stopping gateways...")
        for name, gateway in gateways:
            try:
                await gateway.stop()
                print(f"✅ {name} gateway stopped")
            except Exception as e:
                print(f"❌ Error stopping {name} gateway: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n🛑 Received shutdown signal, stopping gateways...")
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        asyncio.run(run_gateways())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)










