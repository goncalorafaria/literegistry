#!/usr/bin/env python3
"""
Test script to verify event loop fixes
"""
import asyncio
import logging
from literegistry.kvstore import RedisKVStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_same_loop():
    """Test Redis operations in the same event loop"""
    logger.info("Testing Redis operations in same event loop...")
    
    store = RedisKVStore("redis://klone-login01.hyak.local:6379")
    
    try:
        # Test basic operations
        await store.set("test_key", "test_value")
        value = await store.get("test_key")
        logger.info(f"✅ Same loop test passed: {value}")
        
        # Check loop validity
        logger.info(f"Store valid for current loop: {store.is_valid_for_current_loop()}")
        
        # Clean up
        await store.delete("test_key")
        
    except Exception as e:
        logger.error(f"❌ Same loop test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await store.close()

async def test_loop_safety():
    """Test that the store detects loop changes"""
    logger.info("Testing loop safety...")
    
    store = RedisKVStore("redis://klone-login01.hyak.local:6379")
    
    try:
        # Initial connection
        await store.ping()
        initial_loop_id = store._loop_id
        logger.info(f"Initial loop ID: {initial_loop_id}")
        
        # Check validity
        is_valid = store.is_valid_for_current_loop()
        logger.info(f"Store valid for current loop: {is_valid}")
        
        # Simulate a loop change by closing and reconnecting
        await store.close()
        
        # Reconnect (should work fine)
        await store.ping()
        new_loop_id = store._loop_id
        logger.info(f"New loop ID: {new_loop_id}")
        
        # Should still be valid
        is_valid = store.is_valid_for_current_loop()
        logger.info(f"Store valid after reconnect: {is_valid}")
        
        logger.info("✅ Loop safety test passed!")
        
    except Exception as e:
        logger.error(f"❌ Loop safety test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await store.close()

async def main():
    """Run all tests"""
    logger.info("🚀 Starting event loop fix tests...")
    
    await test_same_loop()
    await test_loop_safety()
    
    logger.info("🏁 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())



