"""Confirmed watchdog loss is terminal, bounded, and distinct from rm failure."""

import asyncio

from fastapi.testclient import TestClient
import pytest

from literegistry.services.podman import (
    CompletedPodmanCommand, PodmanAffinityConfig, PodmanAffinityService,
    PodmanSessionBackend, ResourceViolation, SessionLost, SessionNotFound, create_app,
)

CID = "a" * 64


class WatchdogBackend(PodmanSessionBackend):
    def __init__(self, *, memory="1m", pids=None):
        super().__init__(PodmanAffinityConfig(session_memory=memory, session_pids_limit=pids))
        self._owned_container_ids.add(CID)
        self._container_init_pids[CID] = 101
        self.calls = []
        self.removal_result = 0
        self.clock = 100.0

    def _now(self):
        return self.clock

    def _process_snapshot(self):
        return {101: (1, 2 * 1024**2, 5, 1001)}

    async def _run(self, args, **kwargs):
        self.calls.append(args)
        if "rm" in args and isinstance(self.removal_result, BaseException):
            raise self.removal_result
        return CompletedPodmanCommand(tuple(args), self.removal_result if "rm" in args else 0, b"", b"")


@pytest.mark.parametrize(
    "memory,pids,reason,limit,observed,unit",
    [("1m", None, "memory_limit", 1024**2, 2*1024**2, "rss_bytes"),
     (None, 4, "pids_limit", 4, 5, "tasks")],
)
def test_watchdog_loss_survives_as_410_for_commands_and_close(memory, pids, reason, limit, observed, unit):
    backend = WatchdogBackend(memory=memory, pids=pids)
    assert asyncio.run(backend.enforce_resource_budgets()) == [CID]
    with TestClient(create_app(PodmanAffinityService(backend), None)) as client:
        for endpoint in ("podman", "close"):
            response = client.post(f"/{endpoint}", json={"affinity_id": CID, "command": "echo hello"})
            assert response.status_code == 410
            assert response.json()["detail"] == {
                "code": "sandbox_lost", "container_id": CID, "reason": reason,
                "recoverable": False, "limit": limit, "observed": observed,
                "unit": unit, "enforcement": "userspace_watchdog",
            }
        # Unknown IDs retain 404; never invent a resource violation.
        response = client.post("/podman", json={"affinity_id": "f"*64, "command": "hello"})
        assert response.status_code == 404
        assert response.json()["detail"]["error"] == "container_not_found"
    assert len(backend.calls) == 1  # Only rm; no command was sent to Podman.


def test_termination_records_expire_and_evict_oldest():
    async def scenario():
        backend = WatchdogBackend()
        backend._MAX_TERMINATION_RECORDS = 2
        backend._TERMINATION_TTL_SECONDS = 10
        violation = ResourceViolation("memory_limit", 1, 2, "rss_bytes")
        for digit in "abc":
            cid = digit*64
            backend._owned_container_ids.add(cid)
            backend._container_init_pids[cid] = 101
            assert await backend._force_remove_over_budget(cid, 101, violation)
            backend.clock += 1
        assert list(backend._termination_records) == ["b"*64, "c"*64]
        with pytest.raises(SessionNotFound) as error:
            await backend.execute(CID, "echo hello")
        assert not isinstance(error.value, SessionLost)
        with pytest.raises(SessionLost):
            await backend.execute("b"*64, "echo hello")
        backend.clock = 112
        with pytest.raises(SessionNotFound) as error:
            await backend.execute("c"*64, "echo hello")
        assert not isinstance(error.value, SessionLost)
        assert not backend._termination_records
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [1, RuntimeError("rm failed"), asyncio.CancelledError()])
def test_failed_removal_does_not_report_terminal_loss(failure):
    async def scenario():
        backend = WatchdogBackend()
        backend.removal_result = failure
        if isinstance(failure, BaseException):
            with pytest.raises(type(failure)):
                await backend.enforce_resource_budgets()
        else:
            assert await backend.enforce_resource_budgets() == []
        assert not backend._pending_terminations
        assert not backend._terminating_container_ids
        assert not backend._termination_records
        assert (await backend.execute(CID, "echo hello")).returncode == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("returncode", [0, 1])
def test_inflight_command_waits_for_removal_confirmation(returncode):
    async def scenario():
        backend = WatchdogBackend()
        started = asyncio.Event()
        removing = asyncio.Event()
        finish_remove = asyncio.Event()
        command_finished = asyncio.Event()

        async def run(args, **kwargs):
            if "exec" in args:
                started.set()
                await removing.wait()
                command_finished.set()
                return CompletedPodmanCommand(tuple(args), 137, b"", b"killed")
            removing.set()
            await finish_remove.wait()
            return CompletedPodmanCommand(tuple(args), returncode, b"", b"")

        backend._run = run
        command = asyncio.create_task(backend.execute(CID, "allocate"))
        await started.wait()
        removal = asyncio.create_task(backend.enforce_resource_budgets())
        await command_finished.wait()
        await asyncio.sleep(0)
        assert not command.done()
        finish_remove.set()
        await removal
        if returncode == 0:
            with pytest.raises(SessionLost):
                await command
        else:
            assert (await command).returncode == 137
    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_removal():
    async def scenario():
        backend = WatchdogBackend()
        removing = asyncio.Event()
        release = asyncio.Event()
        async def run(args, **kwargs):
            removing.set()
            await release.wait()
            return CompletedPodmanCommand(tuple(args), 0, b"", b"")
        backend._run = run
        removal = asyncio.create_task(backend.enforce_resource_budgets())
        await removing.wait()
        waiter = asyncio.create_task(backend.execute(CID, "echo hello"))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        assert await removal == [CID]
        with pytest.raises(SessionLost):
            await backend.execute(CID, "echo hello")
    asyncio.run(scenario())


@pytest.mark.parametrize("exists_code,state_code,state,status", [
    (0, 0, b"true\n", 200), (1, 0, b"", 404), (125, 0, b"", 503),
    (0, 0, b"false\n", 410), (0, 125, b"", 503), (0, 0, b"invalid", 503),
])
def test_session_probe_is_read_only_and_distinguishes_uncertainty(exists_code, state_code, state, status):
    backend = WatchdogBackend()
    calls = []
    async def run(args, **kwargs):
        calls.append(args)
        if "exists" in args:
            return CompletedPodmanCommand(tuple(args), exists_code, b"", b"")
        assert "inspect" in args
        return CompletedPodmanCommand(tuple(args), state_code, state, b"")
    backend._run = run
    with TestClient(create_app(PodmanAffinityService(backend), None)) as client:
        response = client.get(f"/sessions/{CID}")
        assert response.status_code == status
        if status == 410:
            assert response.json()["detail"]["reason"] == "container_stopped"
            assert response.json()["detail"]["recoverable"] is False
        if status == 200:
            assert response.json() == {"container_id": CID, "status": "active"}
        count = len(calls)
        assert client.get("/sessions/not-a-container-id").status_code == 400
        assert len(calls) == count
    assert all("rm" not in args and "exec" not in args and "run" not in args for args in calls)


def test_session_probe_preserves_watchdog_reason_and_requires_auth():
    backend = WatchdogBackend()
    asyncio.run(backend.enforce_resource_budgets())
    token = "a"*32
    with TestClient(create_app(PodmanAffinityService(backend), token)) as client:
        assert client.get(f"/sessions/{CID}").status_code == 401
        response = client.get(f"/sessions/{CID}", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 410
        assert response.json()["detail"]["reason"] == "memory_limit"
    assert len(backend.calls) == 1
