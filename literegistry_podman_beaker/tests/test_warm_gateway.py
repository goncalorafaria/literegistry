from __future__ import annotations

import asyncio
import json
from pathlib import Path

from literegistry_podman_beaker import warm_gateway


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None



def test_bundled_gateway_warm_list_is_unique() -> None:
    asset = (
        Path(warm_gateway.__file__).resolve().parent
        / "assets"
        / warm_gateway.ASSET_NAME
    )
    images = warm_gateway.load_images_file(str(asset))

    assert len(images) == 14_490
    assert len(images) == len(set(images))


def test_gateway_warmer_waits_for_mirrors_and_reports_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    images_file = tmp_path / "images.txt"
    images_file.write_text("demo:one\ndemo:two\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(warm_gateway, "_active_mirror_count", lambda url: 4)

    def fake_warm(image, gateway, *, platform):
        calls.append(image)
        return image, True, "ok"

    monkeypatch.setattr(warm_gateway, "warm_image", fake_warm)

    result = asyncio.run(
        warm_gateway.warm_gateway(
            "http://gateway.example:8080",
            images_file=str(images_file),
            concurrency=2,
            expected_mirrors=4,
        )
    )

    assert set(calls) == {"demo:one", "demo:two"}
    assert result["mirrors"] == 4
    assert result["successes"] == 2
    assert result["failures"] == 0


def test_active_mirror_count_reads_gateway_models(monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "docker-mirror",
                "metadata": [
                    {"status": "active"},
                    {"status": "inactive"},
                    {"status": "active"},
                ],
            }
        ]
    }
    response = _Response()
    monkeypatch.setattr(warm_gateway, "urlopen", lambda *args, **kwargs: response)
    monkeypatch.setattr(json, "load", lambda stream: payload)

    assert warm_gateway._active_mirror_count("http://gateway.example") == 2
