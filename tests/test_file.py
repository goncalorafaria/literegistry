from literegistry.kvstore import FileSystemKVStore
import asyncio


# Usage Example
async def main():
    # FileSystem Example
    fs_store = FileSystemKVStore()
    await fs_store.set("test1.txt", "Hello FS!")
    await fs_store.set("test2.txt", "World FS!")
    print(await fs_store.keys())  # ['test1.txt', 'test2.txt']


if __name__ == "__main__":
    asyncio.run(main())
