#!/usr/bin/env python3
"""
Test script to verify gateway resilience to unexpected shutdowns.
"""

import asyncio
import aiohttp
import time
import signal
import sys
from literegistry.gateway import create_app

async def test_gateway_resilience():
    """Test that the gateway can handle errors without shutting down."""
    
    print("🧪 Testing Gateway Resilience")
    print("=" * 50)
    
    # Create the app
    app = create_app()
    
    # Start the server in the background
    import uvicorn
    import threading
    
    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    await asyncio.sleep(5)
    
    # Test basic functionality
    async with aiohttp.ClientSession() as session:
        try:
            # Test health endpoint
            print("🔍 Testing health endpoint...")
            async with session.get("http://localhost:8080/health") as response:
                result = await response.json()
                print(f"✅ Health check: {result.get('status', 'unknown')}")
            
            # Test models endpoint
            print("🔍 Testing models endpoint...")
            async with session.get("http://localhost:8080/v1/models") as response:
                result = await response.json()
                print(f"✅ Models endpoint: {result.get('status', 'unknown')}")
            
            # Test error handling with invalid request
            print("🔍 Testing error handling...")
            async with session.post("http://localhost:8080/v1/completions", 
                                  json={"invalid": "request"}) as response:
                result = await response.json()
                print(f"✅ Error handling: {response.status} - {result.get('status', 'unknown')}")
            
            print("\n🎉 All tests passed! Gateway is resilient to errors.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    return True

def signal_handler(signum, frame):
    """Handle test interruption."""
    print(f"\n🛑 Test interrupted by signal {signum}")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Set environment variables for testing
        import os
        os.environ["EXIT_ON_ERROR"] = "false"
        os.environ["MAX_RESTART_ATTEMPTS"] = "10"
        os.environ["HEARTBEAT_INTERVAL"] = "5"
        
        print("🚀 Starting Gateway Resilience Test")
        print("Environment variables:")
        print(f"  EXIT_ON_ERROR: {os.getenv('EXIT_ON_ERROR', 'false')}")
        print(f"  MAX_RESTART_ATTEMPTS: {os.getenv('MAX_RESTART_ATTEMPTS', '100')}")
        print(f"  HEARTBEAT_INTERVAL: {os.getenv('HEARTBEAT_INTERVAL', '30')}")
        print()
        
        success = asyncio.run(test_gateway_resilience())
        
        if success:
            print("\n✅ Gateway resilience test completed successfully!")
        else:
            print("\n❌ Gateway resilience test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        sys.exit(1)



