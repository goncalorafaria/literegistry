import subprocess
import sys
from pathlib import Path


def _add_arg(command, name, value):
    if value is None:
        return
    command.extend([name, str(value)])


def main(
    logs=None,
    logs_dir=None,
    registry=None,
    slurm_logs_dir=None,
    vllm_logs_dir=None,
    seed_recent=None,
    poll_seconds=None,
    window=None,
    refresh=None,
    refresh_seconds=None,
    poll_registry=None,
    registry_poll_seconds=None,
    show_vllm=None,
    vllm_newest_files=None,
    vllm_tail_lines=None,
    vllm_window=None,
    address=None,
    port=None,
    server_address="127.0.0.1",
    server_port=8765,
):
    streamlit_address = address or server_address
    streamlit_port = port or server_port
    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        str(streamlit_address),
        "--server.port",
        str(streamlit_port),
        "--",
    ]

    _add_arg(command, "--logs", logs)
    _add_arg(command, "--logs-dir", logs_dir)
    _add_arg(command, "--registry", registry)
    _add_arg(command, "--slurm-logs-dir", slurm_logs_dir)
    _add_arg(command, "--vllm-logs-dir", vllm_logs_dir)
    _add_arg(command, "--seed-recent", seed_recent)
    _add_arg(command, "--poll-seconds", poll_seconds)
    _add_arg(command, "--window", window)
    _add_arg(command, "--refresh", refresh)
    _add_arg(command, "--refresh-seconds", refresh_seconds)
    _add_arg(command, "--poll-registry", poll_registry)
    _add_arg(command, "--registry-poll-seconds", registry_poll_seconds)
    _add_arg(command, "--show-vllm", show_vllm)
    _add_arg(command, "--vllm-newest-files", vllm_newest_files)
    _add_arg(command, "--vllm-tail-lines", vllm_tail_lines)
    _add_arg(command, "--vllm-window", vllm_window)

    raise SystemExit(subprocess.call(command))
