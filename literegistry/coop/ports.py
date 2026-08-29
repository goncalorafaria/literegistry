"""Collision-safe host-port selection and child-process supervision."""

from __future__ import annotations

import fcntl
import fire
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence


PORT_STARTUP_TIMEOUT = 91


def _port_socket(port: int, bind_host: str) -> socket.socket:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.bind((bind_host, port))
    except OSError:
        candidate.close()
        raise
    return candidate


def _port_available(port: int, bind_host: str) -> bool:
    try:
        candidate = _port_socket(port, bind_host)
    except OSError:
        return False
    candidate.close()
    return True


def port_candidates(
    identity: str,
    variable: str,
    preferred: int,
    *,
    minimum: int,
    maximum: int,
    attempts: int,
) -> list[int]:
    """Return the preference followed by stable identity-derived fallbacks."""
    result = [preferred]
    width = maximum - minimum + 1
    for index in range(attempts - 1):
        digest = hashlib.sha256(f"{identity}\0{variable}\0{index}".encode()).digest()
        result.append(minimum + int.from_bytes(digest[:8], "big") % width)
    return result


def _terminate(process: subprocess.Popen[Any], grace: float = 10.0) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.poll() or 0)
    try:
        return process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_with_dynamic_ports(
    command: Sequence[str],
    *,
    assignments: Mapping[str, int],
    identity: str,
    host_id: str,
    lock_dir: str | Path = "/weka/gfaria/.literegistry/port-locks",
    minimum: int = 1024,
    maximum: int = 65000,
    candidate_attempts: int = 512,
    collision_retries: int = 8,
    startup_timeout: float = 120.0,
    poll_interval: float = 0.1,
    bind_host: str = "0.0.0.0",
) -> int:
    """Choose free ports, start a child, and hold a host lock until it binds."""
    if not command or not assignments:
        raise ValueError("a command and at least one port assignment are required")
    if not identity or not host_id:
        raise ValueError("identity and host_id must be non-empty")
    if minimum < 1 or maximum > 65535 or minimum > maximum:
        raise ValueError("invalid port range")
    for variable, preferred in assignments.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError(f"invalid environment variable: {variable!r}")
        if not minimum <= preferred <= maximum:
            raise ValueError(f"preferred port for {variable} is outside the port range")

    lock_root = Path(lock_dir)
    lock_root.mkdir(parents=True, exist_ok=True)
    host_key = hashlib.sha256(host_id.encode()).hexdigest()[:24]
    lock_path = lock_root / f"host-{host_key}.lock"
    excluded: set[int] = set()
    caught_signal: int | None = None

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal caught_signal
        caught_signal = signum

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, handle_signal)

    process: subprocess.Popen[Any] | None = None
    try:
        for launch_attempt in range(collision_retries):
            selected: dict[str, int] = {}
            with lock_path.open("a+", encoding="utf-8") as lock:
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if caught_signal is not None:
                            return 128 + caught_signal
                        time.sleep(poll_interval)

                reservations: list[socket.socket] = []
                try:
                    for variable, preferred in assignments.items():
                        candidates = port_candidates(
                            f"{identity}:{launch_attempt}",
                            variable,
                            preferred,
                            minimum=minimum,
                            maximum=maximum,
                            attempts=candidate_attempts,
                        )
                        for port in candidates:
                            if port in excluded or port in selected.values():
                                continue
                            try:
                                reservation = _port_socket(port, bind_host)
                            except OSError:
                                continue
                            reservations.append(reservation)
                            selected[variable] = port
                            break
                        else:
                            raise RuntimeError(f"could not find a free port for {variable}")
                    environment = os.environ.copy()
                    environment.update({name: str(port) for name, port in selected.items()})
                    print(
                        "LITEREGISTRY_COOP_PORTS "
                        + " ".join(f"{name}={port}" for name, port in selected.items()),
                        flush=True,
                    )
                finally:
                    for reservation in reservations:
                        reservation.close()

                process = subprocess.Popen(list(command), env=environment, start_new_session=True)
                deadline = time.monotonic() + startup_timeout
                while True:
                    if caught_signal is not None:
                        _terminate(process)
                        return 128 + caught_signal
                    child_code = process.poll()
                    unavailable = {
                        port for port in selected.values() if not _port_available(port, bind_host)
                    }
                    if child_code is not None:
                        if unavailable:
                            excluded.update(selected.values())
                            process = None
                            break
                        return child_code
                    if len(unavailable) == len(selected):
                        break
                    if time.monotonic() >= deadline:
                        _terminate(process)
                        return PORT_STARTUP_TIMEOUT
                    time.sleep(poll_interval)
                if process is not None:
                    break
        else:
            raise RuntimeError("competing processes repeatedly claimed selected ports")

        assert process is not None
        while process.poll() is None:
            if caught_signal is not None:
                _terminate(process)
                return 128 + caught_signal
            time.sleep(poll_interval)
        return int(process.returncode or 0)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def parse_assignments(value: str | Sequence[str]) -> dict[str, int]:
    """Parse one or more ``ENV=PORT`` assignments."""
    raw_assignments = [value] if isinstance(value, str) else list(value)
    assignments: dict[str, int] = {}
    for assignment in raw_assignments:
        variable, separator, raw_port = assignment.partition("=")
        if not separator or not variable or not raw_port:
            raise ValueError("port assignments must use ENV=PORT")
        if variable in assignments:
            raise ValueError(f"duplicate port assignment: {variable}")
        assignments[variable] = int(raw_port)
    return assignments


def run_command(
    assignment: str | Sequence[str],
    identity: str,
    host_id: str,
    command_json: str | Sequence[str],
    lock_dir: str = "/weka/gfaria/.literegistry/port-locks",
    startup_timeout: float = 120.0,
    poll_interval: float = 0.1,
) -> None:
    """Fire command for collision-safe port selection and child supervision."""
    command = json.loads(command_json) if isinstance(command_json, str) else list(command_json)
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("command_json must encode a list of strings")
    return_code = run_with_dynamic_ports(
        command,
        assignments=parse_assignments(assignment),
        identity=identity,
        host_id=host_id,
        lock_dir=lock_dir,
        startup_timeout=startup_timeout,
        poll_interval=poll_interval,
    )
    if return_code:
        raise SystemExit(return_code)


def main(argv: list[str] | None = None) -> None:
    fire.Fire({"run": run_command}, command=argv)


if __name__ == "__main__":
    main()
