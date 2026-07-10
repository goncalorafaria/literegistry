from datetime import datetime
from pathlib import Path
import re
import time

try:
    from .parser import DEFAULT_LOG_MAX_AGE_SECONDS, find_recent_files, log_roots, tail_text
except ImportError:
    from parser import DEFAULT_LOG_MAX_AGE_SECONDS, find_recent_files, log_roots, tail_text


METRIC_RE = re.compile(
    r"INFO (?P<stamp>\d\d-\d\d \d\d:\d\d:\d\d).*?"
    r"Engine (?P<engine>\d+): "
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, "
    r"Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[\d.]+)%"
)
HTTP_RE = re.compile(
    r'INFO:\s+(?P<client>[\d.]+):\d+\s+-\s+"(?P<method>[A-Z]+)\s+'
    r'(?P<path>\S+)\s+HTTP/\S+"\s+(?P<status>\d+)'
)
MODEL_RE = re.compile(r"\bmodel\s+(?P<model>\S+)\s*$")
ARGS_MODEL_RE = re.compile(r"'model(?:_tag)?':\s*'(?P<model>[^']+)'")
PORT_RE = re.compile(r"Starting vLLM server on http://[^:]+:(?P<port>\d+)")
LOAD_TIME_RE = re.compile(r"Model loading took (?P<gb>[\d.]+) GiB and (?P<seconds>[\d.]+) seconds")
KV_MEMORY_RE = re.compile(r"Available KV cache memory: (?P<gb>[\d.]+) GiB")
KV_TOKENS_RE = re.compile(r"GPU KV cache size: (?P<tokens>[\d,]+) tokens")
CONCURRENCY_RE = re.compile(r"Maximum concurrency for (?P<context>[\d,]+) tokens per request: (?P<concurrency>[\d.]+)x")
GRAPH_RE = re.compile(r"Graph capturing finished in (?P<seconds>[\d.]+) secs, took (?P<gb>[\d.]+) GiB")
ENGINE_INIT_RE = re.compile(r"init engine .* took (?P<seconds>[\d.]+) s \(compilation: (?P<compile>[\d.]+) s\)")

STAMP_RE = re.compile(r"(?P<stamp>\d\d-\d\d \d\d:\d\d:\d\d)")
ERROR_LEVEL_RE = re.compile(r"\b(?P<level>ERROR|CRITICAL|FATAL)\b")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
EXCEPTION_RE = re.compile(
    r"^\s*(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Killed|Timeout|OOM))\b"
)
OOM_RE = re.compile(r"out of memory|CUDA out of memory|OutOfMemoryError", re.IGNORECASE)
# Lines that contain the ERROR token but are not actually failures.
ERROR_FALSE_POSITIVE_RE = re.compile(
    r"error[_-]?(rate|count|handler|callback)|0 errors|no errors", re.IGNORECASE
)

# vLLM v1 re-logs subprocess tracebacks with a per-line prefix such as
# "(EngineCore_0 pid=123) ERROR 06-04 12:00:01 [core.py:586] ". Strip it so the
# underlying Python traceback structure can be recognised on every line.
# Each optional segment consumes only its own single trailing separator space so
# the traceback's own indentation (e.g. the 2 spaces before "File") is preserved.
PREFIX_STRIP_RE = re.compile(
    r"^(?:\(.*?\)[ \t]?)?"
    r"(?:\x1b\[[0-9;]*m)?"
    r"(?:(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)[ \t]?)?"
    r"(?:\d\d-\d\d \d\d:\d\d:\d\d(?:[.,]\d+)?[ \t]?)?"
    r"(?:\[[^\]]*\][ \t]?)?"
)
FRAME_RE = re.compile(r'File "[^"]+", line \d+')
EXC_SUMMARY_RE = re.compile(
    r"[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Killed|Timeout|OOM|Abort|Fault)"
    r"(?::|\s*$)"
)
TRACEBACK_CHAIN_PHRASES = (
    "During handling of the above exception",
    "The above exception was the direct cause",
)
# A fresh, unrelated telemetry record marks the end of a traceback block.
TRACEBACK_BOUNDARY_PHRASES = (
    "Avg prompt throughput",
    "Request counts",
    "Completion stats",
)


def _strip_log_prefix(line):
    return PREFIX_STRIP_RE.sub("", line, count=1)


def list_recent_slurm_files(
    logs,
    limit=80,
    suffixes=(".log", ".err"),
    max_age_seconds=DEFAULT_LOG_MAX_AGE_SECONDS,
):
    files = []
    seen = set()
    name_patterns = ["*{}".format(suffix) for suffix in suffixes]
    for root in log_roots(logs):
        if root.is_file():
            candidates = [root] if root.suffix in suffixes else []
        elif root.exists():
            candidates = find_recent_files(
                root,
                name_patterns=name_patterns,
                max_age_seconds=max_age_seconds,
            )
        else:
            candidates = []

        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)

    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def slurm_job_id(path):
    match = re.search(r"slurm_(\d+)", path.name)
    return match.group(1) if match else path.stem


def parse_stamp(stamp, file_mtime):
    year = datetime.fromtimestamp(file_mtime).year
    parsed = datetime.strptime("{} {}".format(year, stamp), "%Y %m-%d %H:%M:%S")
    return time.mktime(parsed.timetuple())


def parse_vllm_line(line, path, fallback_ts):
    metric = METRIC_RE.search(line)
    if metric:
        ts = parse_stamp(metric.group("stamp"), path.stat().st_mtime)
        return {
            "kind": "engine_metric",
            "ts": ts,
            "time": time.strftime("%H:%M:%S", time.localtime(ts)),
            "file": path.name,
            "job_id": slurm_job_id(path),
            "engine": metric.group("engine"),
            "prompt_tps": float(metric.group("prompt")),
            "generation_tps": float(metric.group("generation")),
            "running_reqs": int(metric.group("running")),
            "waiting_reqs": int(metric.group("waiting")),
            "kv_cache_pct": float(metric.group("kv")),
            "prefix_hit_pct": float(metric.group("prefix")),
            "raw": line,
        }

    http = HTTP_RE.search(line)
    if http:
        return {
            "kind": "http",
            "ts": fallback_ts,
            "time": time.strftime("%H:%M:%S", time.localtime(fallback_ts)),
            "file": path.name,
            "job_id": slurm_job_id(path),
            "method": http.group("method"),
            "path": http.group("path"),
            "status": int(http.group("status")),
            "client": http.group("client"),
            "raw": line,
        }

    return None


def classify_error_line(line):
    """Return an error level string for a failure line, or None if it is benign."""
    if ERROR_FALSE_POSITIVE_RE.search(line):
        return None
    if OOM_RE.search(line):
        return "OOM"
    if TRACEBACK_RE.search(line):
        return "TRACEBACK"
    level_match = ERROR_LEVEL_RE.search(line)
    if level_match:
        return level_match.group("level")
    content = _strip_log_prefix(line)
    if EXCEPTION_RE.match(content) or EXC_SUMMARY_RE.match(content):
        return "EXCEPTION"
    return None


def _error_ts(line, path, fallback_ts):
    stamp_match = STAMP_RE.search(line)
    if stamp_match:
        try:
            return parse_stamp(stamp_match.group("stamp"), path.stat().st_mtime)
        except (ValueError, OSError):
            return fallback_ts
    return fallback_ts


def _make_error_event(path, ts, level, headline, detail_lines):
    detail = "\n".join(detail_lines).strip()
    return {
        "kind": "error",
        "ts": ts,
        "time": time.strftime("%H:%M:%S", time.localtime(ts)),
        "file": path.name,
        "job_id": slurm_job_id(path),
        "level": level,
        "message": headline.strip()[:300],
        "detail": detail,
        "line_count": len(detail_lines),
        "raw": detail,
    }


def _looks_like_traceback_line(line):
    """True if ``line`` is plausibly part of a Python traceback (prefix-aware)."""
    if FRAME_RE.search(line):
        return True
    content = _strip_log_prefix(line)
    if content[:1] in (" ", "\t"):
        return True
    if content.lstrip().startswith('File "'):
        return True
    if "Traceback (most recent call last)" in line:
        return True
    return any(phrase in line for phrase in TRACEBACK_CHAIN_PHRASES)


def _is_region_boundary(line):
    """A fresh, unrelated telemetry record ends an error region."""
    if any(p in line for p in TRACEBACK_BOUNDARY_PHRASES):
        return True
    return bool(HTTP_RE.search(line) or METRIC_RE.search(line))


def _line_continues_region(line):
    """True if ``line`` belongs to the current error/traceback region."""
    content = _strip_log_prefix(line)
    if content.strip() == "":
        # Blank (or prefix-only) line: keep the region glued together.
        return True
    if classify_error_line(line):
        return True
    if EXC_SUMMARY_RE.search(content):
        return True
    return _looks_like_traceback_line(line)


def collect_error_region(lines, start_index, max_block_lines=400):
    """Group a contiguous run of error/traceback lines into one block.

    vLLM v1 re-emits an entire traceback as many ``ERROR``-prefixed lines, so a
    single failure spans many lines (engine message + full traceback + chained
    exceptions). This collects them all into one entry, stopping at the next
    unrelated log record. Returns (block_lines, next_index).
    """
    block = [lines[start_index]]
    index = start_index + 1
    n = len(lines)
    prev_was_frame = bool(FRAME_RE.search(lines[start_index]))
    while index < n and len(block) < max_block_lines:
        line = lines[index]
        if _is_region_boundary(line):
            break
        # The line directly under a "File ...", line N frame is the source
        # snippet (and may be unindented after prefix stripping); always keep it.
        if not prev_was_frame and not _line_continues_region(line):
            break
        prev_was_frame = bool(FRAME_RE.search(line))
        block.append(line)
        index += 1

    # Drop trailing blank/prefix-only lines from the captured block.
    while block and _strip_log_prefix(block[-1]).strip() == "":
        block.pop()
    return block, index


def is_error_region_start(line):
    return classify_error_line(line) is not None and _strip_log_prefix(line).strip() != ""


def _region_level(block):
    has_traceback = any("Traceback (most recent call last)" in line for line in block)
    if has_traceback:
        return "TRACEBACK"
    for line in block:
        if OOM_RE.search(line):
            return "OOM"
    for line in block:
        level = classify_error_line(line)
        if level:
            return level
    return "ERROR"


def _region_headline(block):
    for line in reversed(block):
        content = _strip_log_prefix(line).strip()
        if content and EXC_SUMMARY_RE.search(content):
            return content
    for line in block:
        content = _strip_log_prefix(line).strip()
        if content and "Traceback (most recent call last)" not in content:
            return content
    return _strip_log_prefix(block[0]).strip() or block[0].strip()


def parse_metadata_line(line, path):
    base = {
        "file": path.name,
        "job_id": slurm_job_id(path),
    }

    for regex in (MODEL_RE, ARGS_MODEL_RE):
        match = regex.search(line)
        if match:
            row = dict(base)
            row.update({"field": "model", "value": match.group("model")})
            return row

    checks = [
        (PORT_RE, "port", "port"),
        (LOAD_TIME_RE, "load_seconds", "seconds"),
        (KV_MEMORY_RE, "kv_memory_gib", "gb"),
        (KV_TOKENS_RE, "kv_cache_tokens", "tokens"),
        (CONCURRENCY_RE, "max_concurrency", "concurrency"),
        (GRAPH_RE, "graph_capture_seconds", "seconds"),
        (ENGINE_INIT_RE, "engine_init_seconds", "seconds"),
    ]
    for regex, field, group in checks:
        match = regex.search(line)
        if not match:
            continue
        value = match.group(group).replace(",", "")
        row = dict(base)
        row.update({"field": field, "value": value})
        return row

    return None


def parse_vllm_logs(logs_dir, newest_files=80, tail_lines=1000):
    events = []
    metadata = []
    errors = []
    for path in list_recent_slurm_files(logs_dir, limit=newest_files):
        lines = tail_text(path, max_lines=tail_lines)
        fallback_ts = path.stat().st_mtime - max(0, len(lines) - 1)
        n = len(lines)
        index = 0
        while index < n:
            line = lines[index]
            ts = fallback_ts + index

            event = parse_vllm_line(line, path, ts)
            if event:
                events.append(event)

            meta = parse_metadata_line(line, path)
            if meta:
                metadata.append(meta)

            if is_error_region_start(line):
                block, next_index = collect_error_region(lines, index)
                if block:
                    errors.append(
                        _make_error_event(
                            path,
                            _error_ts(line, path, ts),
                            _region_level(block),
                            _region_headline(block),
                            block,
                        )
                    )
                index = max(next_index, index + 1)
                continue

            index += 1

    events.sort(key=lambda row: row["ts"])
    errors.sort(key=lambda row: row["ts"])
    return events, metadata, errors


def metadata_wide(metadata):
    rows = {}
    for item in metadata:
        key = (item["job_id"], item["file"])
        if key not in rows:
            rows[key] = {"job_id": item["job_id"], "file": item["file"]}
        rows[key][item["field"]] = item["value"]
    return list(rows.values())
