import asyncio
import pytest

from literegistry.services.podman import (
    CloseRequest,
    CompletedPodmanCommand,
    HandshakeRequest,
    PodmanAffinityConfig,
    PodmanAffinityService,
    PodmanSessionBackend,
    SessionNotFound,
    SessionRequest,
    PodmanRequest,
    build_podman_registry_mirror_config,
    create_app,
)


CONTAINER_ID = "a" * 64
API_TOKEN = "test-token-which-is-longer-than-32-characters"


class FakeBackend:
    def __init__(self):
        self.config = PodmanAffinityConfig(
            instance_id="replica-a",
            session_image="ubuntu:test",
        )
        self.created_for = []
        self.executions = []
        self.removed = []

    async def create_session(self, client_id=None, image=None):
        self.created_for.append((client_id, image))
        return CONTAINER_ID

    async def execute(self, container_id, command, **kwargs):
        self.executions.append((container_id, command, kwargs))
        return CompletedPodmanCommand(
            args=("podman", "exec"),
            returncode=0,
            stdout=b"hello\n",
            stderr=b"",
        )

    async def remove_session(self, container_id):
        self.removed.append(container_id)

    async def owned_container_ids(self):
        return [CONTAINER_ID]


def test_handshake_returns_same_container_and_affinity_id():
    backend = FakeBackend()
    service = PodmanAffinityService(backend)

    response = asyncio.run(service.handshake(HandshakeRequest(client_id="agent-1")))

    assert response.container_id == CONTAINER_ID
    assert response.affinity_id == CONTAINER_ID
    assert response.instance_id == "replica-a"
    assert response.client_id == "agent-1"
    assert response.image == "ubuntu:test"
    assert backend.created_for == [("agent-1", "ubuntu:test")]


def test_handshake_selects_image_for_new_container():
    backend = FakeBackend()
    service = PodmanAffinityService(backend)

    response = asyncio.run(
        service.handshake(
            HandshakeRequest(client_id="agent-2", image="quay.io/example/tools:v2")
        )
    )

    assert response.image == "quay.io/example/tools:v2"
    assert backend.created_for == [("agent-2", "quay.io/example/tools:v2")]


def test_podman_routes_command_to_selected_container():
    backend = FakeBackend()
    service = PodmanAffinityService(backend)

    response = asyncio.run(
        service.podman(
            PodmanRequest(
                container_id=CONTAINER_ID,
                command="printf hello",
                stdin="input",
                timeout=3,
                workdir="/workspace",
            )
        )
    )

    assert response.stdout == "hello\n"
    assert response.success is True
    assert response.container_id == CONTAINER_ID
    assert backend.executions == [
        (
            CONTAINER_ID,
            "printf hello",
            {"stdin": "input", "timeout": 3.0, "workdir": "/workspace"},
        )
    ]


def test_affinity_id_alias_is_accepted_and_conflicts_are_rejected():
    assert SessionRequest(affinity_id=CONTAINER_ID).selected_container_id() == CONTAINER_ID

    with pytest.raises(ValueError, match="must match"):
        SessionRequest(
            affinity_id=CONTAINER_ID,
            container_id="b" * 64,
        ).selected_container_id()


def test_close_removes_exact_selected_container():
    backend = FakeBackend()
    service = PodmanAffinityService(backend)

    result = asyncio.run(service.close(CloseRequest(container_id=CONTAINER_ID)))

    assert result["removed"] is True
    assert backend.removed == [CONTAINER_ID]


def test_app_exposes_affinity_podman_routes_and_requires_long_token():
    app = create_app(PodmanAffinityService(FakeBackend()), API_TOKEN)
    paths = {route.path for route in app.routes}

    create_app(PodmanAffinityService(FakeBackend()), None)
    assert {"/health", "/handshake", "/podman", "/close"}.issubset(paths)
    with pytest.raises(ValueError, match="at least 32"):
        create_app(PodmanAffinityService(FakeBackend()), "too-short")


def test_backend_uses_owned_container_and_bash_without_outer_shell_interpolation():
    class RecordingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.calls = []
            self._owned_container_ids.add(CONTAINER_ID)

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append((args, stdin, timeout))
            return CompletedPodmanCommand(tuple(args), 0, b"ok\n", b"")

    backend = RecordingBackend()
    command = "echo hello; touch /workspace/persisted"
    result = asyncio.run(
        backend.execute(CONTAINER_ID, command, stdin="abc", timeout=2.5)
    )

    assert result.stdout == b"ok\n"
    exec_args, stdin, outer_timeout = backend.calls[-1]
    assert exec_args[-3:] == ["/bin/bash", "-lc", command]
    assert CONTAINER_ID in exec_args
    assert stdin == b"abc"
    assert outer_timeout == 7.5
    assert all("inspect" not in args for args, _, _ in backend.calls)


def test_backend_forces_immediate_container_removal():
    class RecordingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.calls = []
            self._owned_container_ids.add(CONTAINER_ID)

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append(args)
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

    backend = RecordingBackend()
    asyncio.run(backend.remove_session(CONTAINER_ID))

    assert backend.calls[-1][-5:] == [
        "rm",
        "--force",
        "--time",
        "0",
        CONTAINER_ID,
    ]
    assert all("inspect" not in args for args in backend.calls)
    assert CONTAINER_ID not in backend._owned_container_ids


def test_backend_rejects_container_not_retained_in_memory():
    backend = PodmanSessionBackend(PodmanAffinityConfig(instance_id="replica-a"))

    with pytest.raises(SessionNotFound):
        asyncio.run(backend.execute(CONTAINER_ID, "true"))

def test_registry_mirror_config_uses_gateway_and_keeps_docker_hub_fallback():
    contents = build_podman_registry_mirror_config(
        "http://gateway.example:8080"
    )

    assert 'prefix = "docker.io"' in contents
    assert 'location = "docker.io"' in contents
    assert 'location = "gateway.example:8080"' in contents
    assert "insecure = true" in contents
    assert 'pull-from-mirror = "all"' in contents


@pytest.mark.parametrize(
    "url",
    [
        "gateway.example:8080",
        "ftp://gateway.example:8080",
        "http://user:secret@gateway.example:8080",
        "http://gateway.example:8080/not-root",
    ],
)
def test_registry_mirror_config_rejects_unsafe_or_non_root_urls(url):
    with pytest.raises(ValueError, match="registry_mirror"):
        build_podman_registry_mirror_config(url)


def test_backend_points_only_podman_subprocesses_at_generated_mirror_config():
    backend = PodmanSessionBackend(
        PodmanAffinityConfig(
            instance_id="replica-a",
            registry_mirror="https://gateway.example:8443",
        )
    )

    assert backend._podman_env is not None
    config_path = backend._podman_env["CONTAINERS_REGISTRIES_CONF"]
    with open(config_path, encoding="utf-8") as stream:
        contents = stream.read()
    assert 'location = "gateway.example:8443"' in contents
    assert "insecure = false" in contents
