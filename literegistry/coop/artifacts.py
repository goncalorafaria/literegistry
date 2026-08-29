"""Single-writer coordination for warmed or built directory artifacts."""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import os
from pathlib import Path
import shutil
import tempfile


class IncompleteArtifactError(RuntimeError):
    """A target exists but does not satisfy its caller-defined ready check."""


def ensure_directory_artifact(
    target: str | Path,
    *,
    ready: Callable[[Path], bool],
    build: Callable[[Path], None],
    lock_path: str | Path | None = None,
) -> Path:
    """Build or warm a directory once and atomically publish it.

    Cooperation is scoped by a sibling advisory lock. The readiness predicate
    is checked only while holding that lock. A builder writes into a temporary
    sibling directory, and the completed artifact becomes visible through one
    atomic rename. Other processes either see the previous complete artifact
    or the new complete artifact, never the staging directory.

    An incomplete non-empty target is preserved and reported instead of being
    deleted implicitly. Empty targets are safe to replace.
    """
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_lock = Path(lock_path) if lock_path else (
        destination.parent / f".{destination.name}.materialize.lock"
    )
    resolved_lock.parent.mkdir(parents=True, exist_ok=True)

    with resolved_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_dir() and ready(destination):
            return destination
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise IncompleteArtifactError(
                    f"artifact target exists but is incomplete: {destination}"
                )
            destination.rmdir()

        build_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.materialize-",
                dir=destination.parent,
            )
        )
        staging = build_root / "artifact"
        staging.mkdir()
        try:
            build(staging)
            if not ready(staging):
                raise IncompleteArtifactError(
                    f"builder did not produce a complete artifact: {destination}"
                )
            os.replace(staging, destination)
        finally:
            shutil.rmtree(build_root, ignore_errors=True)
    return destination
