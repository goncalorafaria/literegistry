"""Minimal heartbeat-index commands for existing Redis test doubles."""


class HeartbeatIndexMixin:
    @property
    def sorted_sets(self):
        return getattr(self, "state", self.__dict__).setdefault("sorted_sets", {})

    async def zadd(self, name, mapping):
        self.sorted_sets.setdefault(name, {}).update(mapping)
        return len(mapping)

    async def zrem(self, name, key):
        return int(self.sorted_sets.get(name, {}).pop(key, None) is not None)

    async def zrangebyscore(self, name, minimum, maximum):
        if hasattr(self, "_check"):
            self._check()
        return [key.encode() for key, score in self.sorted_sets.get(name, {}).items()
                if float(minimum) <= score <= float(maximum)]

    def pipeline(self, transaction=True):
        assert transaction
        return _Pipeline(self)


class _Pipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __getattr__(self, name):
        def enqueue(*args):
            self.commands.append((name, args))
            return self
        return enqueue

    async def execute(self):
        if hasattr(self.client, "_check"):
            self.client._check()
        return [await getattr(self.client, name)(*args) for name, args in self.commands]
