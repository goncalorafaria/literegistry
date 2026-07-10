# File Descriptor Leak Fixes

This document outlines the fixes implemented to address the "too many files open" error in the LiteRegistry Gateway.

## Root Causes Identified

1. **HTTP Client Sessions**: `RegistryHTTPClient` was creating new `aiohttp.ClientSession` instances without proper connection pooling limits
2. **Redis Connections**: `RedisKVStore` was creating connections without proper pool management
3. **Missing Resource Cleanup**: Resources weren't being properly closed in error scenarios
4. **No Connection Limits**: Unlimited connection creation could lead to resource exhaustion

## Fixes Implemented

### 1. HTTP Client Improvements (`literegistry/http.py`)

- **Connection Pooling**: Added `TCPConnector` with limits:
  - Total pool size: 100 connections
  - Per-host limit: 30 connections
  - DNS cache TTL: 300 seconds
  - Keepalive timeout: 30 seconds
- **Better Error Handling**: Wrapped session operations in try-catch blocks
- **Proper Cleanup**: Enhanced `__aexit__` method to close both session and connector

### 2. Redis Connection Management (`literegistry/kvstore.py`)

- **Connection Pooling**: Implemented `ConnectionPool` with limits:
  - Maximum connections: 20
  - Retry on timeout: enabled
  - Socket keepalive: enabled
  - Health check interval: 30 seconds
- **Proper Cleanup**: Enhanced `close()` method to close both Redis client and pool
- **Error Logging**: Added comprehensive error logging for debugging

### 3. Gateway Server Cleanup (`literegistry/gateway.py`)

- **Resource Cleanup**: Added cleanup in `__aexit__` method
- **Registry Cleanup**: Ensure registry store is closed on shutdown
- **Error Handling**: Better error handling with proper cleanup in finally blocks
- **Shutdown Events**: Added app shutdown event handlers for cleanup

### 4. Registry Client Context Manager (`literegistry/client.py`)

- **Context Manager**: Added `__aenter__` and `__aexit__` methods
- **Store Cleanup**: Ensure store is closed when registry is closed

## Usage Recommendations

### 1. Always Use Context Managers

```python
# ✅ Correct usage
async with RegistryHTTPClient(registry, model) as client:
    result = await client.request_with_rotation(endpoint, payload)

# ✅ Correct usage
async with StarletteGatewayServer(registry, port=8080) as gateway:
    # Server operations
    pass
```

### 2. Monitor File Descriptors

Use the provided monitoring script to track FD usage:

```bash
# Install monitoring dependencies
pip install -r requirements_monitoring.txt

# Monitor current process
python monitor_fds.py

# Monitor specific PID
python monitor_fds.py --pid 12345

# List open files
python monitor_fds.py --list-files

# List network connections
python monitor_fds.py --list-connections
```

### 3. Environment Variables

Set appropriate limits in your environment:

```bash
# Increase file descriptor limits if needed
ulimit -n 65536

# Set connection pool limits via environment
export MAX_CONNECTIONS=100
export MAX_CONNECTIONS_PER_HOST=30
```

## Monitoring and Debugging

### 1. Check Current FD Usage

```bash
# Check process FD count
lsof -p <PID> | wc -l

# Check system FD limits
cat /proc/sys/fs/file-max
cat /proc/sys/fs/file-nr
```

### 2. Common Symptoms of FD Leaks

- Increasing FD count over time
- "Too many open files" errors
- High system FD usage
- Network connection failures

### 3. Debugging Steps

1. **Monitor FD count** using the provided script
2. **Check for hanging connections** using `netstat` or `ss`
3. **Review logs** for connection errors
4. **Use strace** to track file operations: `strace -e trace=file -p <PID>`

## Performance Impact

- **Connection Pooling**: Reduces connection creation overhead
- **Resource Limits**: Prevents resource exhaustion
- **Better Error Handling**: More robust operation under load
- **Minimal Overhead**: Cleanup operations are lightweight

## Testing the Fixes

1. **Start the gateway server**
2. **Run the monitoring script** in another terminal
3. **Generate load** using your test scripts
4. **Monitor FD count** - it should remain stable
5. **Check for errors** in the gateway logs

## Additional Recommendations

1. **Regular Monitoring**: Run FD monitoring in production
2. **Load Testing**: Test with expected production load
3. **Resource Limits**: Set appropriate system limits
4. **Logging**: Enable debug logging for connection operations
5. **Health Checks**: Implement health checks for connection pools

## Troubleshooting

If you still experience FD leaks:

1. **Check for custom code** that might be opening files/connections
2. **Review third-party libraries** for known FD leak issues
3. **Use strace** to identify which operations are creating FDs
4. **Check for zombie processes** that might be holding connections
5. **Review system logs** for related errors

## Support

For additional help or if issues persist:

1. Check the monitoring script output
2. Review gateway server logs
3. Monitor system resource usage
4. Consider implementing additional connection pooling if needed



