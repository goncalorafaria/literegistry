import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_NGROK_URL_RE = re.compile(r"url=(https://[^\s]+)")


def _start_ngrok(port, startup_timeout=30):
    """Start an ngrok tunnel to ``port`` and return (process, public_url).

    The public URL is parsed directly from ngrok's logfmt stdout rather than the
    local inspection API, because an HTTP proxy on this host intercepts requests
    to 127.0.0.1:4040. A daemon thread keeps draining ngrok's output for the
    lifetime of the process so the pipe never blocks.

    Returns (None, None) if ngrok is not installed or fails to come up, so the
    console still launches without a tunnel.
    """
    ngrok_bin = shutil.which("ngrok")
    if not ngrok_bin:
        print(
            "[ngrok] 'ngrok' not found on PATH; skipping tunnel. "
            "Install it (https://ngrok.com/download) to expose the console.",
            file=sys.stderr,
        )
        return None, None

    proc = subprocess.Popen(
        [ngrok_bin, "http", str(port), "--log", "stdout", "--log-format", "logfmt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    result = {"url": None, "err": None}

    def _drain():
        for line in proc.stdout:
            if result["url"] is None:
                m = _NGROK_URL_RE.search(line)
                if m:
                    result["url"] = m.group(1)
                elif "err=" in line and "err=nil" not in line:
                    result["err"] = line.strip()

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if result["url"]:
            break
        if proc.poll() is not None:
            print("[ngrok] tunnel process exited early.", file=sys.stderr)
            return None, None
        time.sleep(0.5)

    public_url = result["url"]
    if public_url:
        print(f"[ngrok] Console public URL: {public_url}")
    else:
        msg = "[ngrok] Tunnel did not report a public URL within "
        msg += f"{startup_timeout}s."
        if result["err"]:
            msg += f" Last error: {result['err']}"
        print(msg, file=sys.stderr)
    return proc, public_url


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
    ngrok=True,
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

    ngrok_proc = None
    if ngrok:
        ngrok_proc, _ = _start_ngrok(streamlit_port)

    try:
        returncode = subprocess.call(command)
    finally:
        if ngrok_proc is not None and ngrok_proc.poll() is None:
            ngrok_proc.terminate()
            try:
                ngrok_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ngrok_proc.kill()

    raise SystemExit(returncode)
