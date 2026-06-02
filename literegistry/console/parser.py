from pathlib import Path
import re
import subprocess
import time


REQUEST_RE = re.compile(r"Request counts \(last (?P<window>[\d.]+)s\): (?P<body>.*)")
COMPLETION_RE = re.compile(r"Completion stats \(last (?P<window>[\d.]+)s\): (?P<body>.*)")
REQUEST_ITEM_RE = re.compile(r"(?P<model>[^,]+?):\s*(?P<count>\d+)(?=,\s*|$)")
COMPLETION_ITEM_RE = re.compile(
    r"(?P<model>.*?):\s*"
    r"(?P<count>\d+)\s+reqs,\s*"
    r"avg:\s*(?P<avg>[\d.]+)s,\s*"
    r"max:\s*(?P<max>[\d.]+)s"
    r"(?=,\s+.*?:\s*\d+\s+reqs,\s*avg:|$)"
)
REGISTRY_ITEM_RE = re.compile(r"^\s*(?P<model>[^:]+?)\s*:\s*(?P<count>\d+)\s*$")
REGISTRY_STATUS_PREFIXES = (
    "successfully connected",
    "connected",
    "connecting",
    "redis",
)
TAIL_BLOCK_SIZE = 8192


class ParsedEvent:
    def __init__(self, ts, file, kind, window, model, count, avg, max, raw):
        self.ts = ts
        self.file = file
        self.kind = kind
        self.window = window
        self.model = model
        self.count = count
        self.avg = avg
        self.max = max
        self.raw = raw

    def as_dict(self):
        return {
            "ts": self.ts,
            "time": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "file": self.file,
            "kind": self.kind,
            "window": self.window,
            "model": self.model,
            "count": self.count,
            "avg": self.avg,
            "max": self.max,
            "raw": self.raw,
        }


def clean_model_name(model):
    return model.strip().strip(",").strip()


def log_roots(logs):
    if isinstance(logs, (str, Path)):
        logs = [logs]
    return [Path(log) for log in logs]


def list_log_files(logs):
    files = []
    seen = set()
    for root in log_roots(logs):
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = sorted(path for path in root.rglob("*.log") if path.is_file())
        else:
            candidates = []

        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def tail_text(path, max_lines):
    if max_lines <= 0:
        return []

    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks = []
        newline_count = 0

        while position > 0 and newline_count <= max_lines:
            read_size = min(TAIL_BLOCK_SIZE, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")

    data = b"".join(reversed(chunks))
    raw_lines = data.splitlines()[-max_lines:]
    return [
        raw_line.decode("utf-8", errors="replace").replace("\x00", "")
        for raw_line in raw_lines
    ]


def parse_metric_line(line, file_name, ts):
    request = REQUEST_RE.search(line)
    if request:
        window = float(request.group("window"))
        body = request.group("body").strip()
        return [
            ParsedEvent(
                ts=ts,
                file=file_name,
                kind="requests",
                window=window,
                model=clean_model_name(item.group("model")),
                count=int(item.group("count")),
                avg=None,
                max=None,
                raw=line,
            )
            for item in REQUEST_ITEM_RE.finditer(body)
        ]

    completion = COMPLETION_RE.search(line)
    if completion:
        window = float(completion.group("window"))
        body = completion.group("body").strip()
        return [
            ParsedEvent(
                ts=ts,
                file=file_name,
                kind="completions",
                window=window,
                model=clean_model_name(item.group("model")),
                count=int(item.group("count")),
                avg=float(item.group("avg")),
                max=float(item.group("max")),
                raw=line,
            )
            for item in COMPLETION_ITEM_RE.finditer(body)
        ]

    return []


def parse_logs(logs_dir, max_lines_per_file=1000):
    events = []
    for path in list_log_files(logs_dir):
        lines = tail_text(path, max_lines=max_lines_per_file)
        metric_lines = [line for line in lines if "Request counts" in line or "Completion stats" in line]
        if not metric_lines:
            continue

        mtime = path.stat().st_mtime
        fallback_window = 5.0
        ts = mtime - (len(metric_lines) - 1) * fallback_window
        for line in metric_lines:
            parsed = parse_metric_line(line, path.name, ts)
            events.extend(parsed)
            if parsed:
                ts += parsed[0].window
            else:
                ts += fallback_window

    return [event.as_dict() for event in sorted(events, key=lambda event: event.ts)]


def parse_registry_summary(output, ts=None):
    snapshot_ts = time.time() if ts is None else ts
    rows = []
    for line in output.splitlines():
        cleaned = line.strip()
        lowered = cleaned.lower()
        if not cleaned or "://" in cleaned or lowered.startswith(REGISTRY_STATUS_PREFIXES):
            continue
        match = REGISTRY_ITEM_RE.match(line)
        if not match:
            continue
        model = match.group("model").strip()
        if not model or model.lower().startswith(REGISTRY_STATUS_PREFIXES):
            continue
        rows.append(
            {
                "ts": snapshot_ts,
                "time": time.strftime("%H:%M:%S", time.localtime(snapshot_ts)),
                "model": model,
                "count": int(match.group("count")),
            }
        )
    return rows


def poll_registry_summary(registry_url, timeout_seconds=4.0):
    if not registry_url:
        return []

    try:
        result = subprocess.run(
            ["literegistry", "summary", "--registry", registry_url],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    return parse_registry_summary(result.stdout)


def poll_registry_summary_with_status(registry_url, timeout_seconds=4.0):
    if not registry_url:
        return [], "No registry URL provided."

    try:
        result = subprocess.run(
            ["literegistry", "summary", "--registry", registry_url],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return [], "Registry summary timed out."
    except OSError as exc:
        return [], "Could not run literegistry: {}".format(exc)

    rows = parse_registry_summary(result.stdout)
    if rows:
        return rows, ""

    detail = result.stderr.strip() or result.stdout.strip() or "No parseable registry rows."
    return [], detail


def unique_values(rows, key):
    return sorted({str(row[key]) for row in rows if row.get(key)})
