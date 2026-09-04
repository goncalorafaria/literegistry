import asyncio
import os
import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from literegistry.services import podman as podman_module
from literegistry.services.podman import (
    CloseRequest,
    CompletedPodmanCommand,
    HandshakeRequest,
    PodmanAffinityConfig,
    PodmanAffinityService,
    PodmanBackendError,
    PodmanSessionBackend,
    SessionNotFound,
    SessionRequest,
    PodmanRequest,
    build_podman_registry_mirror_config,
    create_app,
    parse_memory_limit,
)


CONTAINER_ID = "a" * 64
API_TOKEN = "test-token-which-is-longer-than-32-characters"


class FakeBackend:
    def __init__(self):
        self.config = PodmanAffinityConfig(
            instance_id="replica-a",
            session_image="ubuntu:test",
            max_sessions=64,
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

    async def capacity_status(self):
        return (1, 64)


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


def test_health_reports_session_capacity():
    client = TestClient(create_app(PodmanAffinityService(FakeBackend()), None))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["sessions"] == 1
    assert response.json()["pending_sessions"] == 0
    assert response.json()["max_sessions"] == 64
    assert response.json()["available_sessions"] == 63


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
    assert exec_args[-6:-3] == ["/bin/bash", "-c", podman_module._EXEC_WRAPPER]
    assert exec_args[-3] == "literegistry-exec"
    assert exec_args[-2:] == ["2.5", command]
    assert "/usr/bin/timeout" not in exec_args
    assert CONTAINER_ID in exec_args
    assert stdin == b"abc"
    assert outer_timeout == 7.5
    assert all("inspect" not in args for args, _, _ in backend.calls)


def test_exec_wrapper_uses_available_timeout_and_bash_fallback(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")

    def run(command, deadline, *, with_timeout, stdin=b""):
        env = dict(os.environ)
        if not with_timeout:
            # Hide timeout while preserving the tools the existing session
            # contract already requires.
            shim = tmp_path / "bin"
            shim.mkdir(exist_ok=True)
            for tool in ("bash", "cat", "sleep"):
                target = shutil.which(tool)
                if target is not None:
                    link = shim / tool
                    if not link.exists():
                        link.symlink_to(target)
            env["PATH"] = str(shim)
        started = time.monotonic()
        process = subprocess.run(
            [
                bash,
                "-c",
                podman_module._EXEC_WRAPPER,
                "literegistry-exec",
                str(deadline),
                command,
            ],
            input=stdin,
            capture_output=True,
            env=env,
            timeout=15,
        )
        return process, time.monotonic() - started

    for with_timeout in (True, False):
        if with_timeout and shutil.which("timeout") is None:
            continue

        process, _ = run(
            "cat; printf done; exit 3",
            10,
            with_timeout=with_timeout,
            stdin=b"input\n",
        )
        assert process.returncode == 3
        assert process.stdout == b"input\ndone"

        process, elapsed = run(
            "sleep 30 & child=$!; echo $child; wait $child",
            1,
            with_timeout=with_timeout,
        )
        assert process.returncode in {137, -9}
        assert elapsed < 5
        child_pid = int(process.stdout.strip())
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            os.kill(child_pid, 9)
            pytest.fail(f"timed-out child process survived: {child_pid}")


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


def test_create_session_removes_container_by_name_when_run_fails():
    class FailingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.calls = []

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append(list(args))
            if "run" in args:
                return CompletedPodmanCommand(
                    tuple(args), 126, b"", b"/bin/bash not found"
                )
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

    backend = FailingBackend()

    with pytest.raises(PodmanBackendError, match="bash not found"):
        asyncio.run(backend.create_session(image="alpine:3.20"))

    run_args, remove_args = backend.calls
    name = run_args[run_args.index("--name") + 1]
    assert remove_args[-6:] == [
        "rm",
        "--force",
        "--time",
        "0",
        "--ignore",
        name,
    ]
    assert backend._owned_container_ids == set()


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


def test_hardening_config_rejects_invalid_values():
    invalid = (
        {"session_memory": ""},
        {"session_memory": "four gigs"},
        {"max_sessions": 0},
        {"session_pids_limit": 0},
        {"session_idle_timeout": 0},
        {"janitor_interval": 0},
        {"resource_watchdog_interval": 0},
        {"image_prune_until": ""},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            PodmanAffinityConfig(**kwargs)


def test_parse_memory_limit():
    assert parse_memory_limit("4g") == 4 * 1024**3
    assert parse_memory_limit("256m") == 256 * 1024**2
    assert parse_memory_limit("0.5g") == 512 * 1024**2
    assert parse_memory_limit("1073741824") == 1073741824


def test_session_capacity_reserves_slots_before_starting_podman():
    class CapacityBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(instance_id="replica-a", max_sessions=1)
            )
            self.run_started = asyncio.Event()
            self.release_run = asyncio.Event()

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "run" in args:
                self.run_started.set()
                await self.release_run.wait()
                return CompletedPodmanCommand(tuple(args), 0, f"{CONTAINER_ID}\n".encode(), b"")
            raise AssertionError(args)

    async def scenario():
        backend = CapacityBackend()
        first = asyncio.create_task(backend.create_session("first"))
        await backend.run_started.wait()
        with pytest.raises(PodmanBackendError, match=r"capacity exhausted \(1/1\)"):
            await backend.create_session("second")
        backend.release_run.set()
        assert await first == CONTAINER_ID
        assert await backend.capacity_status() == (1, 1)

    asyncio.run(scenario())


def test_resource_watchdog_falls_back_to_ancestry_without_pid_namespace():
    snapshot = {
        101: (1, 10 * 1024**2, 1, None),
        102: (101, 20 * 1024**2, 2, None),
    }

    assert PodmanSessionBackend._container_usages(snapshot, {101}) == {
        101: (30 * 1024**2, 3)
    }


def test_resource_watchdog_confirms_and_removes_memory_and_task_violations():
    memory_id = CONTAINER_ID
    tasks_id = "b" * 64

    class WatchdogBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(
                    instance_id="replica-a",
                    session_memory="256m",
                    session_pids_limit=32,
                )
            )
            self._owned_container_ids.update({memory_id, tasks_id})
            self._container_init_pids.update({memory_id: 101, tasks_id: 201})
            self.removed = []
            self.snapshot_calls = 0

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "rm" in args:
                self.removed.append(args[-1])
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")

        def _process_snapshot(self):
            self.snapshot_calls += 1
            return {
                101: (1, 10 * 1024**2, 1, 1001),
                102: (999, 300 * 1024**2, 1, 1001),
                201: (1, 10 * 1024**2, 1, 2001),
                202: (999, 10 * 1024**2, 64, 2001),
            }

    backend = WatchdogBackend()
    removed = asyncio.run(backend.enforce_resource_budgets())

    assert set(removed) == {memory_id, tasks_id}
    assert set(backend.removed) == {memory_id, tasks_id}
    assert backend.snapshot_calls == 2
    assert backend._owned_container_ids == set()


def test_resource_watchdog_caches_init_pid_and_keeps_under_budget_session():
    class WatchdogBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(instance_id="replica-a", session_memory="256m")
            )
            self._owned_container_ids.add(CONTAINER_ID)
            self.inspect_calls = 0

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "inspect" in args:
                self.inspect_calls += 1
                return CompletedPodmanCommand(tuple(args), 0, b"101\n", b"")
            raise AssertionError(args)

        @staticmethod
        def _process_snapshot():
            return {101: (1, 10 * 1024**2, 1, 1001)}

    backend = WatchdogBackend()
    assert asyncio.run(backend.enforce_resource_budgets()) == []
    assert asyncio.run(backend.enforce_resource_budgets()) == []
    assert backend.inspect_calls == 1
    assert CONTAINER_ID in backend._owned_container_ids


def test_resource_watchdog_can_terminate_while_command_lock_is_held():
    class WatchdogBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(instance_id="replica-a", session_memory="1m")
            )
            self._owned_container_ids.add(CONTAINER_ID)
            self._container_init_pids[CONTAINER_ID] = 101
            self.command_started = asyncio.Event()
            self.release_command = asyncio.Event()

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "exec" in args:
                self.command_started.set()
                await self.release_command.wait()
                return CompletedPodmanCommand(tuple(args), 137, b"", b"killed")
            if "rm" in args:
                return CompletedPodmanCommand(tuple(args), 0, b"", b"")
            raise AssertionError(args)

        @staticmethod
        def _process_snapshot():
            return {101: (1, 2 * 1024**2, 1, 1001)}

    async def scenario():
        backend = WatchdogBackend()
        command = asyncio.create_task(backend.execute(CONTAINER_ID, "allocate"))
        await backend.command_started.wait()
        assert await backend.enforce_resource_budgets() == [CONTAINER_ID]
        assert not command.done()
        backend.release_command.set()
        result = await command
        assert result.returncode == 137
        assert CONTAINER_ID not in backend._owned_container_ids

    asyncio.run(scenario())


def test_conditional_remove_preserves_recently_active_session():
    backend = _recording_backend(
        PodmanAffinityConfig(instance_id="replica-a", session_idle_timeout=600)
    )
    backend._owned_container_ids.add(CONTAINER_ID)
    backend._session_last_used[CONTAINER_ID] = 2.0

    removed = asyncio.run(
        backend.remove_session(
            CONTAINER_ID,
            idle_before=1.0,
        )
    )

    assert removed is False



def test_session_requests_require_canonical_full_container_ids():
    with pytest.raises(ValueError, match="at least 64"):
        SessionRequest(affinity_id=CONTAINER_ID[:12])


def test_owned_container_ids_requests_and_validates_untruncated_ids():
    class ListingBackend(PodmanSessionBackend):
        def __init__(self, listed_id):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.listed_id = listed_id
            self.calls = []

        async def _run(self, args, *, stdin=b"", timeout=None):
            self.calls.append(list(args))
            return CompletedPodmanCommand(
                tuple(args),
                0,
                f"{self.listed_id}\n".encode(),
                b"",
            )

    backend = ListingBackend(CONTAINER_ID)
    assert asyncio.run(backend.owned_container_ids()) == [CONTAINER_ID]
    assert "--no-trunc" in backend.calls[0]

    truncated = ListingBackend(CONTAINER_ID[:12])
    with pytest.raises(PodmanBackendError, match="non-canonical"):
        asyncio.run(truncated.owned_container_ids())


def test_janitor_waiting_on_long_command_observes_completion_activity():
    class InterleavingBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(
                PodmanAffinityConfig(
                    instance_id="replica-a",
                    session_idle_timeout=0.05,
                )
            )
            self.command_started = asyncio.Event()
            self.release_command = asyncio.Event()
            self.sessions_listed = asyncio.Event()
            self.remove_calls = []
            self.now = 100.0

        def _now(self):
            return self.now

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "ps" in args:
                self.sessions_listed.set()
                return CompletedPodmanCommand(
                    tuple(args), 0, f"{CONTAINER_ID}\n".encode(), b""
                )
            if "exec" in args:
                self.command_started.set()
                await self.release_command.wait()
                return CompletedPodmanCommand(tuple(args), 0, b"done\n", b"")
            if "rm" in args:
                self.remove_calls.append(list(args))
                return CompletedPodmanCommand(tuple(args), 0, b"", b"")
            raise AssertionError(args)

    async def scenario():
        backend = InterleavingBackend()
        backend._owned_container_ids.add(CONTAINER_ID)
        backend._session_last_used[CONTAINER_ID] = 0.0

        command = asyncio.create_task(
            backend.execute(CONTAINER_ID, "sleep 10", timeout=10)
        )
        await backend.command_started.wait()
        backend.now = 200.0

        janitor = asyncio.create_task(backend.reap_idle_sessions())
        await backend.sessions_listed.wait()
        backend.release_command.set()

        result = await command
        removed = await janitor
        assert result.stdout == b"done\n"
        assert removed == []
        assert backend.remove_calls == []
        assert CONTAINER_ID in backend._owned_container_ids

    asyncio.run(scenario())


def test_command_queued_behind_remove_cannot_enter_deleted_container():
    class RemovalBackend(PodmanSessionBackend):
        def __init__(self):
            super().__init__(PodmanAffinityConfig(instance_id="replica-a"))
            self.remove_started = asyncio.Event()
            self.release_remove = asyncio.Event()
            self.exec_calls = []

        async def _run(self, args, *, stdin=b"", timeout=None):
            if "rm" in args:
                self.remove_started.set()
                await self.release_remove.wait()
                return CompletedPodmanCommand(tuple(args), 0, b"", b"")
            if "exec" in args:
                self.exec_calls.append(list(args))
                return CompletedPodmanCommand(tuple(args), 0, b"unexpected", b"")
            raise AssertionError(args)

    async def scenario():
        backend = RemovalBackend()
        backend._owned_container_ids.add(CONTAINER_ID)
        backend._session_last_used[CONTAINER_ID] = 0.0

        removal = asyncio.create_task(backend.remove_session(CONTAINER_ID))
        await backend.remove_started.wait()
        command = asyncio.create_task(backend.execute(CONTAINER_ID, "true"))
        await asyncio.sleep(0)
        backend.release_remove.set()

        assert await removal is True
        with pytest.raises(SessionNotFound):
            await command
        assert backend.exec_calls == []

    asyncio.run(scenario())
