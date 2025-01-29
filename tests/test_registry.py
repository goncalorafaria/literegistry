from literegistry.registry import ServerRegistry
from literegistry.kvstore import FileSystemKVStore
import asyncio


# Example usage
async def main():
    # Using FileSystem backend
    store = FileSystemKVStore("registry_data")
    registry = ServerRegistry(store)

    # Register a server
    server_id = await registry.register_server(8000, {"model": "gpt-3"})
    print(f"Registered server: {server_id}")

    # Update heartbeat
    await registry.heartbeat(8000)

    # Get roster
    roster = await registry.roster()
    print("Active servers:", roster)

    # Cleanup
    await registry.deregister()


if __name__ == "__main__":
    asyncio.run(main())
