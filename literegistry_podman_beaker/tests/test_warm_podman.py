from __future__ import annotations

import asyncio
from pathlib import Path

from literegistry_podman_beaker import warm_podman


class _FakeSession:
    def __init__(self, image: str, calls: list[tuple[str, str]]) -> None:
        self.image = image
        self.calls = calls

    async def execute(self, command: str, **kwargs) -> None:
        self.calls.append(("execute", f"{self.image}:{command}"))

    async def close(self) -> None:
        self.calls.append(("close", self.image))


class _FakeClient:
    calls: list[tuple[str, str]] = []

    def __init__(self, gateway_url: str, **kwargs) -> None:
        self.calls.append(("init", gateway_url))

    async def open(self) -> None:
        self.calls.append(("open", ""))

    async def aclose(self) -> None:
        self.calls.append(("aclose", ""))

    async def handshake(self, *, image: str, client_id: str) -> _FakeSession:
        self.calls.append(("handshake", image))
        return _FakeSession(image, self.calls)


def test_podman_warmer_uses_client_lifecycle_and_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    images_file = tmp_path / "images.txt"
    images_file.write_text("demo:one\ndemo:two\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.txt"
    checkpoint.write_text("demo:one\n", encoding="utf-8")
    _FakeClient.calls = []
    monkeypatch.setattr(warm_podman, "PodmanClient", _FakeClient)
    monkeypatch.setattr(warm_podman, "_active_service_count", lambda *args: 32)

    result = asyncio.run(
        warm_podman.warm_podman(
            "http://gateway.example:8080",
            images_file=str(images_file),
            concurrency=2,
            expected_podman=32,
            checkpoint_file=str(checkpoint),
            log_every=1,
        )
    )

    assert ("handshake", "demo:one") not in _FakeClient.calls
    assert ("handshake", "demo:two") in _FakeClient.calls
    assert ("execute", "demo:two:true") in _FakeClient.calls
    assert ("close", "demo:two") in _FakeClient.calls
    assert result["resumed"] == 1
    assert result["successes"] == 1
    assert result["failures"] == 0
    assert set(checkpoint.read_text(encoding="utf-8").splitlines()) == {
        "demo:one",
        "demo:two",
    }


def test_podman_warmer_can_skip_execute_probe(monkeypatch, tmp_path: Path) -> None:
    images_file = tmp_path / "images.txt"
    images_file.write_text("demo:one\n", encoding="utf-8")
    _FakeClient.calls = []
    monkeypatch.setattr(warm_podman, "PodmanClient", _FakeClient)
    monkeypatch.setattr(warm_podman, "_active_service_count", lambda *args: 1)

    result = asyncio.run(
        warm_podman.warm_podman(
            "http://gateway.example:8080",
            images_file=str(images_file),
            execute_probe=False,
            log_every=1,
        )
    )

    assert not any(call[0] == "execute" for call in _FakeClient.calls)
    assert result["successes"] == 1
