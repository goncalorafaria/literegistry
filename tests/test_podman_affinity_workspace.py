import asyncio

from literegistry.services.podman import (
    CompletedPodmanCommand,
    PodmanAffinityConfig,
    PodmanSessionBackend,
)


CONTAINER_ID = "c" * 64


def test_session_startup_creates_workspace_before_sleeping():
    class RecordingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.args = None

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.args = args
            return CompletedPodmanCommand(
                tuple(args), 0, (CONTAINER_ID + "\n").encode(), b""
            )

    backend = RecordingBackend()
    container_id = asyncio.run(backend.create_session("agent-1"))

    assert container_id == CONTAINER_ID
    assert backend.args[-3:] == [
        "/bin/bash",
        "-lc",
        "mkdir -p /workspace && exec sleep infinity",
    ]


def test_session_startup_uses_handshake_image():
    class RecordingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(session_image="ubuntu:default"))
            self.args = None

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.args = args
            return CompletedPodmanCommand(
                tuple(args), 0, (CONTAINER_ID + "\n").encode(), b""
            )

    backend = RecordingBackend()
    asyncio.run(backend.create_session(image="quay.io/example/tools:v2"))

    separator = backend.args.index("--")
    assert backend.args[separator + 1] == "quay.io/example/tools:v2"
