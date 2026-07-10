#!/usr/bin/env python3
"""
Debug script to run the gateway with enhanced error handling and monitoring.
"""

import os
import sys
import asyncio
import signal
from literegistry.gateway import StarletteGatewayServer, RegistryClient
from literegistry.kvstore import RedisKVStore, FileSystemKVStore

async def run_gateway_debug():
    """Run the gateway with debug configuration."""
    
    # Load environment variables
    registry_path = os.getenv("REGISTRY_PATH", "redis://klone-login01.hyak.local:6379")
    port = int(os.getenv("PORT", "8080"))
    workers = int(os.getenv("WORKERS", "1"))
    
    print("🚀 Starting Gateway in Debug Mode")
    print("=" * 50)
    print(f"📊 Port: {port}")
    print(f"👥 Workers: {workers}")
    print(f"🗄️  Registry: {registry_path}")
    print(f"🛡️  EXIT_ON_ERROR: {os.getenv('EXIT_ON_ERROR', 'false')}")
    print(f"🔄 MAX_RESTART_ATTEMPTS: {os.getenv('MAX_RESTART_ATTEMPTS', '100')}")
    print(f"💓 HEARTBEAT_INTERVAL: {os.getenv('HEARTBEAT_INTERVAL', '30')}")
    print("=" * 50)
    
    try:
        # Initialize registry
        if "redis://" in registry_path:
            store = RedisKVStore(registry_path)
        else:
            store = FileSystemKVStore(registry_path)
        
        registry = RegistryClient(store=store, service_type="model_path")
        
        # Create and start gateway
        async with StarletteGatewayServer(registry, port=port, workers=workers) as gateway:
            print("✅ Gateway server started successfully!")
            print("🔄 Server is running and will auto-restart on errors")
            print("⏹️  Press Ctrl+C to stop the server")
            print("=" * 50)
            
            # Keep the server running
            while True:
                try:
                    await asyncio.sleep(1)
                    
                    # Check server status
                    if hasattr(gateway, 'server_task') and not gateway.server_task.done():
                        continue
                    else:
                        print("⚠️  Server task stopped, checking status...")
                        
                        if hasattr(gateway, 'server') and gateway.server:
                            if hasattr(gateway.server, 'should_exit') and gateway.server.should_exit:
                                print("🛑 Server shutdown requested, exiting...")
                                break
                            else:
                                print("✅ Server appears to be running...")
                        else:
                            print("🔄 Server stopped, will restart automatically...")
                            await asyncio.sleep(2)
                            
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down gateway...")
                    break
                except Exception as e:
                    print(f"\n⚠️  Error in main loop: {e}")
                    print("🔄 Continuing...")
                    await asyncio.sleep(2)
                    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}")
    if signum == signal.SIGTERM:
        print("SIGTERM received, exiting...")
        sys.exit(0)
    elif signum == signal.SIGINT:
        print("SIGINT received, attempting graceful shutdown...")

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = asyncio.run(run_gateway_debug())
        if success:
            print("✅ Gateway debug run completed successfully")
        else:
            print("❌ Gateway debug run failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)



