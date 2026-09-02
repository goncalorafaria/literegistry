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


def _recording_backend(config=None):
    class RecordingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(config or PodmanAffinityConfig(instance_id="replica-a"))
            self.calls = []

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append((list(args), stdin, timeout))
            if args[1:3] == ["run", "--detach"] or "run" in args[:4]:
                return CompletedPodmanCommand(tuple(args), 0, (CONTAINER_ID + "\n").encode(), b"")
            if "ps" in args:
                return CompletedPodmanCommand(tuple(args), 0, b"", b"")
            return CompletedPodmanCommand(tuple(args), 0, b"ok\n", b"")

    return RecordingBackend()


def test_create_session_applies_memory_and_pids_limits():
    backend = _recording_backend(
        PodmanAffinityConfig(
            instance_id="replica-a",
            session_memory="4g",
            session_pids_limit=2048,
        )
    )

    container_id = asyncio.run(backend.create_session(client_id="agent-1"))

    assert container_id == CONTAINER_ID
    run_args = backend.calls[0][0]
    memory_idx = run_args.index("--memory")
    assert run_args[memory_idx + 1] == "4g"
    swap_idx = run_args.index("--memory-swap")
    assert run_args[swap_idx + 1] == "4g"
    pids_idx = run_args.index("--pids-limit")
    assert run_args[pids_idx + 1] == "2048"
    # Resource flags must come before the image separator.
    assert memory_idx < run_args.index("--")


def test_create_session_omits_resource_flags_by_default():
    backend = _recording_backend()

    asyncio.run(backend.create_session())

    run_args = backend.calls[0][0]
    assert "--memory" not in run_args
    assert "--pids-limit" not in run_args


def test_read_limited_truncates_and_drains_instead_of_raising():
    backend = PodmanSessionBackend(PodmanAffinityConfig(instance_id="replica-a"))

    async def scenario():
        stream = asyncio.StreamReader()
        stream.feed_data(b"x" * (200 * 1024))
        stream.feed_eof()
        return await backend._read_limited(stream, 64 * 1024, "stdout")

    data, truncated = asyncio.run(scenario())
    assert truncated is True
    assert data == b"x" * (64 * 1024)

    async def small():
        stream = asyncio.StreamReader()
        stream.feed_data(b"tiny")
        stream.feed_eof()
        return await backend._read_limited(stream, 64 * 1024, "stdout")

    data, truncated = asyncio.run(small())
    assert (data, truncated) == (b"tiny", False)


def test_podman_response_carries_truncation_flags():
    class TruncatingBackend(FakeBackend):
        async def execute(self, container_id, command, **kwargs):
            return CompletedPodmanCommand(
                args=("podman", "exec"),
                returncode=0,
                stdout=b"partial",
                stderr=b"",
                stdout_truncated=True,
            )

    service = PodmanAffinityService(TruncatingBackend())
    response = asyncio.run(
        service.podman(PodmanRequest(affinity_id=CONTAINER_ID, command="yes"))
    )

    assert response.stdout == "partial"
    assert response.stdout_truncated is True
    assert response.stderr_truncated is False
    assert response.success is True


def test_reap_idle_sessions_removes_only_expired_containers():
    fresh_id = "b" * 64
    backend = _recording_backend(
        PodmanAffinityConfig(instance_id="replica-a", session_idle_timeout=600)
    )
    backend._owned_container_ids.update({CONTAINER_ID, fresh_id})
    import time as _time

    now = _time.monotonic()
    backend._session_last_used[CONTAINER_ID] = now - 3600
    backend._session_last_used[fresh_id] = now

    removed = asyncio.run(backend.reap_idle_sessions())

    assert removed == [CONTAINER_ID]
    assert CONTAINER_ID not in backend._owned_container_ids
    assert fresh_id in backend._owned_container_ids


def test_reap_idle_sessions_adopts_unknown_containers_before_reaping():
    backend = _recording_backend(
        PodmanAffinityConfig(instance_id="replica-a", session_idle_timeout=600)
    )

    async def listed(self):
        return [CONTAINER_ID]

    backend.owned_container_ids = listed.__get__(backend)

    removed = asyncio.run(backend.reap_idle_sessions())

    # First sighting: adopted with a fresh timestamp, not reaped yet.
    assert removed == []
    assert CONTAINER_ID in backend._session_last_used

    backend._session_last_used[CONTAINER_ID] -= 3600
    removed = asyncio.run(backend.reap_idle_sessions())
    assert removed == [CONTAINER_ID]


def test_reap_idle_sessions_disabled_without_timeout():
    backend = _recording_backend()
    backend._owned_container_ids.add(CONTAINER_ID)
    backend._session_last_used[CONTAINER_ID] = 0.0

    assert asyncio.run(backend.reap_idle_sessions()) == []
    assert CONTAINER_ID in backend._owned_container_ids


def test_prune_images_issues_filtered_prune():
    backend = _recording_backend(
        PodmanAffinityConfig(instance_id="replica-a", image_prune_until="24h")
    )

    asyncio.run(backend.prune_images())

    prune_args = backend.calls[-1][0]
    assert prune_args[-5:] == ["prune", "--all", "--force", "--filter", "until=24h"]

    quiet = _recording_backend()
    asyncio.run(quiet.prune_images())
    assert quiet.calls == []


def test_parse_memory_limit():
    from literegistry.services.podman import parse_memory_limit

    assert parse_memory_limit("4g") == 4 * 1024**3
    assert parse_memory_limit("256m") == 256 * 1024**2
    assert parse_memory_limit("512K") == 512 * 1024
    assert parse_memory_limit("1073741824") == 1073741824
    with pytest.raises(ValueError):
        parse_memory_limit("")


def test_memory_watchdog_removes_over_budget_sessions():
    over_id = CONTAINER_ID
    under_id = "b" * 64
    pid_of = {over_id: 111, under_id: 222}
    rss_of = {111: 400 * 1024**2, 222: 10 * 1024**2}

    class WatchdogBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(instance_id="replica-a", session_memory="256m")
            )
            self._owned_container_ids.update({over_id, under_id})
            self.calls = []

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append(list(args))
            if "inspect" in args:
                container_id = args[-1]
                return CompletedPodmanCommand(tuple(args), 0, f"{pid_of[container_id]}\n".encode(), b"")
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

    backend = WatchdogBackend()
    # Measurement reads the Podman *host* /proc subtree, not cgroups or the
    # container's own /proc; stub it deterministically as (rss_bytes, procs).
    backend._subtree_stats = staticmethod(lambda pid: (rss_of.get(pid, 0), 1))

    removed = asyncio.run(backend.enforce_resource_budgets())

    assert removed == [over_id]
    assert over_id not in backend._owned_container_ids
    assert under_id in backend._owned_container_ids
    assert any("inspect" in args for args in backend.calls)


def test_memory_watchdog_never_kills_on_unmeasurable_or_absent_budget():
    class Backend(PodmanSessionBackend):
        def __init__(self, memory, inspect_rc):
            super().__init__(
                PodmanAffinityConfig(instance_id="replica-a", session_memory=memory)
            )
            self._owned_container_ids.add(CONTAINER_ID)
            self._inspect_rc = inspect_rc

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "inspect" in args:
                return CompletedPodmanCommand(tuple(args), self._inspect_rc, b"123\n", b"")
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

    # inspect fails -> no PID -> never killed
    failing = Backend("256m", inspect_rc=1)
    assert asyncio.run(failing.enforce_resource_budgets()) == []
    assert CONTAINER_ID in failing._owned_container_ids

    # PID exited between inspect and measurement (subtree returns None) -> never killed
    exited = Backend("256m", inspect_rc=0)
    exited._subtree_stats = staticmethod(lambda pid: None)
    assert asyncio.run(exited.enforce_resource_budgets()) == []
    assert CONTAINER_ID in exited._owned_container_ids

    # No budget configured -> no-op
    assert asyncio.run(Backend(None, inspect_rc=0).enforce_resource_budgets()) == []


def test_pids_watchdog_removes_fork_bombed_sessions():
    over_id = CONTAINER_ID
    under_id = "b" * 64
    pid_of = {over_id: 111, under_id: 222}
    # (rss_bytes, proc_count): over_id has too many processes, both under memory.
    stats_of = {111: (5 * 1024**2, 5000), 222: (5 * 1024**2, 12)}

    class Backend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(
                    instance_id="replica-a", session_memory="256m", session_pids_limit=2048
                )
            )
            self._owned_container_ids.update({over_id, under_id})

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "inspect" in args:
                return CompletedPodmanCommand(tuple(args), 0, f"{pid_of[args[-1]]}\n".encode(), b"")
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

    backend = Backend()
    backend._subtree_stats = staticmethod(lambda pid: stats_of[pid])

    removed = asyncio.run(backend.enforce_resource_budgets())

    assert removed == [over_id]
    assert over_id not in backend._owned_container_ids
    assert under_id in backend._owned_container_ids
