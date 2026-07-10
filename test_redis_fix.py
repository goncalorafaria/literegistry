#!/usr/bin/env python3
"""
Test script to verify Redis KV store fixes
"""
import asyncio
import logging
from literegistry.kvstore import RedisKVStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_redis_connection():
    """Test Redis connection and basic operations"""
    store = None
    try:
        # Test connection
        logger.info("Testing Redis connection...")
        store = RedisKVStore("redis://klone-login01.hyak.local:6379")
        
        # Test ping
        logger.info("Testing ping...")
        ping_result = await store.ping()
        logger.info(f"Ping result: {ping_result}")
        
        # Test basic operations
        logger.info("Testing basic operations...")
        
        # Set a test key
        test_key = "test_connection_fix"
        test_value = "Hello from fixed Redis store!"
        set_result = await store.set(test_key, test_value)
        logger.info(f"Set result: {set_result}")
        
        # Get the test key
        get_result = await store.get(test_key)
        logger.info(f"Get result: {get_result}")
        if get_result:
            logger.info(f"Retrieved value: {get_result.decode('utf-8')}")
        
        # Check if key exists
        exists_result = await store.exists(test_key)
        logger.info(f"Exists result: {exists_result}")
        
        # Get all keys
        keys_result = await store.keys()
        logger.info(f"Keys result: {keys_result}")
        
        # Clean up test key
        delete_result = await store.delete(test_key)
        logger.info(f"Delete result: {delete_result}")
        
        logger.info("✅ All Redis tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if store:
            await store.close()
            logger.info("Redis store closed")

async def test_event_loop_handling():
    """Test that the store handles event loop issues gracefully"""
    logger.info("Testing event loop handling...")
    
    try:
        # Create store
        store = RedisKVStore("redis://klone-login01.hyak.local:6379")
        
        # Test connection
        await store.ping()
        logger.info("✅ Store connected successfully")
        
        # Test basic operation
        await store.set("test_loop", "test_value")
        value = await store.get("test_loop")
        logger.info(f"✅ Basic operation successful: {value}")
        
        # Clean up
        await store.delete("test_loop")
        await store.close()
        
        logger.info("✅ Event loop handling test passed!")
        
    except Exception as e:
        logger.error(f"❌ Event loop handling test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    logger.info("🚀 Starting Redis KV store tests...")
    
    await test_redis_connection()
    await test_event_loop_handling()
    
    logger.info("🏁 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())



