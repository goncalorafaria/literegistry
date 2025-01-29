from literegistry.consul import ConsulKVStore
from aioconsul import Consul
import asyncio


# Usage Example
async def main():

    # Consul Example (requires running Consul agent)
    async with Consul() as consul:
        consul_store = ConsulKVStore(consul)
        await consul_store.set("app/config1", b"{'timeout': 30}")
        await consul_store.set("app/config2", b"{'retry': 3}")
        print(await consul_store.keys())  # ['app/config1', 'app/config2']


if __name__ == "__main__":
    asyncio.run(main())
