from literegistry import RegistryHTTPClient, RegistryClient, FileSystemKVStore
import asyncio


# Example usage
async def main():
    # Initialize registry and client
    store = FileSystemKVStore("registry_data")
    registry = RegistryClient(store, service_type="model_path")

    # Register some test servers
    url = "localhost"
    await registry.register_server(url, 8000, {"model_path": "gpt-3"})
    await registry.register_server(url, 8001, {"model_path": "gpt-3"})

    # Use the client
    async with RegistryHTTPClient(registry, "gpt-3") as client:
        # Single request
        result, server_idx = await client.request_with_rotation(
            "v1/completions", {"prompt": "Hello"}
        )
        print(f"Single request result: {result}")

        # Parallel requests
        payloads = [{"prompt": "Hello"}, {"prompt": "World"}]
        results = await client.parallel_requests(
            "v1/completions",
            payloads,
        )
        print(f"Parallel results: {results}")


if __name__ == "__main__":
    asyncio.run(main())
