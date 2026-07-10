# Registry Gateway Server

The Registry Gateway Server is an HTTP server that acts as a reverse proxy, routing requests to different model servers based on the `model_path` parameter. It leverages the existing Registry Client's `request_with_rotation` functionality to provide load balancing and failover capabilities.

## Features

- **HTTP Gateway**: Acts as a reverse proxy for model inference requests
- **Model Routing**: Routes requests to different models based on `model_path` parameter
- **Load Balancing**: Uses `request_with_rotation` for automatic server rotation and failover
- **Health Monitoring**: Built-in health check endpoint
- **CORS Support**: Handles cross-origin requests
- **Async Support**: Built with asyncio for high performance

## Architecture

```
Client Request → Gateway Server → Registry Client → Model Servers
     ↓              ↓              ↓              ↓
  HTTP POST    Route by model   Discover      Process
  /endpoint    _path param      servers       request
```

## Usage

### Basic Setup

```python
from literegistry.gateway import RegistryGatewayServer
from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore

# Initialize registry
store = FileSystemKVStore("registry_data")
registry = RegistryClient(store, service_type="model_path")

# Create and start gateway server
async with RegistryGatewayServer(registry, port=8080) as gateway:
    # Server is now running and handling requests
    await asyncio.Future()  # Keep running
```

### Configuration Options

```python
gateway = RegistryGatewayServer(
    registry=registry,
    host="0.0.0.0",           # Bind address
    port=8080,                 # Port number
    max_parallel_requests=8,   # Max concurrent requests per model
    timeout=60,                # Request timeout in seconds
    max_retries=50             # Max retry attempts per request
)
```

## API Endpoints

### Health Check
```
GET /health
```
Returns server health status.

**Response:**
```json
{
    "status": "healthy",
    "service": "registry-gateway"
}
```

### Model Listing
```
GET /v1/models
```
Lists available models from the registry (uses cached values).

**Response:**
```json
{
    "models": ["gpt-3", "gpt-4", "claude-3"],
    "status": "success"
}
```

### Completions Endpoint
```
POST /v1/completions
```
Main endpoint for completion requests. Routes to the appropriate model server based on the `model` parameter in the request body.

**Request Body:**
```json
{
    "model": "gpt-3",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
}
```

**Parameters:**
- `model`: The model to use for completion (required)
- Additional parameters are forwarded to the model server

**Response:** JSON response from the model server

## Request Examples

### 1. Completion Request to GPT-3

```bash
curl -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### 2. Completion Request to GPT-4

```bash
curl -X POST "http://localhost:8080/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "prompt": "Explain quantum computing",
    "max_tokens": 100
  }'
```

### 3. List Available Models

```bash
curl "http://localhost:8080/v1/models"
```

### 4. Health Check

```bash
curl "http://localhost:8080/health"
```

## How It Works

1. **Request Reception**: Gateway receives HTTP requests to specific endpoints
2. **Endpoint Routing**: Routes requests to appropriate handlers based on the path
3. **Model Identification**: For completions, extracts `model` parameter from request body
4. **Registry Lookup**: Uses Registry Client to discover available servers for the model
5. **Request Routing**: Creates a `RegistryHTTPClient` instance for the specific model
6. **Load Balancing**: Calls `request_with_rotation` to handle server rotation and failover
7. **Response Forwarding**: Returns the response from the model server

## Error Handling

- **Missing model parameter**: Returns 400 error if `model` is not specified in request body
- **Invalid JSON**: Returns 400 error if request body is not valid JSON
- **Unsupported endpoints**: Returns 404 error for unsupported paths
- **Registry Errors**: Returns 500 error if registry operations fail
- **Model Server Errors**: Returns 500 error if model server requests fail
- **Network Errors**: Automatic retry with exponential backoff via `request_with_rotation`

## Testing

Run the test script to see the gateway in action:

```bash
python test_gateway.py
```

This will:
1. Start a test registry with mock servers
2. Start the gateway server on port 8080
3. Test various endpoints and routing scenarios
4. Clean up and stop the server

## Dependencies

- `aiohttp`: HTTP server and client functionality
- `asyncio`: Asynchronous programming support
- `literegistry`: Core registry functionality

## Integration

The gateway server integrates seamlessly with your existing Registry Client infrastructure:

- Uses the same `RegistryClient` instance
- Leverages existing `request_with_rotation` logic
- Maintains consistent configuration and timeout settings
- Preserves all retry and failover behavior

## Production Considerations

- **Logging**: Configure appropriate log levels for production
- **Monitoring**: Add metrics collection for request latency and success rates
- **Security**: Implement authentication and rate limiting as needed
- **Load Balancing**: Consider running multiple gateway instances behind a load balancer
- **Health Checks**: Use the `/health` endpoint for load balancer health checks
