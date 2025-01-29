from literegistry.client import RegistryClient
from literegistry.kvstore import FileSystemKVStore
import asyncio


# Example usage
async def main():
    # Initialize with FileSystem backend
    store = FileSystemKVStore("registry_data")
    registry = RegistryClient(store)

    # Register a model server
    await registry.register_server(
        url="localhost", port=8000, metadata={"model_path": "gpt-3"}
    )

    print(await registry.roster())
    # Get all URIs for a model
    uris = await registry.get_all("gpt-3")
    print(f"Model servers: {uris}")

    # Report some latencies
    registry.report_latency(uris[0], 0.5)

    # Get best server for model
    best_uri = await registry.get("gpt-3")
    print(f"Best server: {best_uri}")


if __name__ == "__main__":
    asyncio.run(main())
