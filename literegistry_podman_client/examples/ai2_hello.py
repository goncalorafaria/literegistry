"""Write and read a file in separate commands, then remove the container."""

from __future__ import annotations

import fire

from literegistry_podman_client.cli import ai2_hello


if __name__ == "__main__":
    fire.Fire(ai2_hello)
