from datetime import datetime
from pathlib import Path
import re
import time

try:
    from .parser import log_roots, tail_text
except ImportError:
    from parser import log_roots, tail_text


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


def list_recent_slurm_files(logs, limit=80, suffixes=(".log", ".err")):
    files = []
    seen = set()
    for root in log_roots(logs):
        if root.is_file():
            candidates = [root] if root.suffix in suffixes else []
        elif root.exists():
            candidates = [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in suffixes
            ]
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
    for path in list_recent_slurm_files(logs_dir, limit=newest_files):
        lines = tail_text(path, max_lines=tail_lines)
        fallback_ts = path.stat().st_mtime - max(0, len(lines) - 1)
        for index, line in enumerate(lines):
            ts = fallback_ts + index
            event = parse_vllm_line(line, path, ts)
            if event:
                events.append(event)

            meta = parse_metadata_line(line, path)
            if meta:
                metadata.append(meta)

    events.sort(key=lambda row: row["ts"])
    return events, metadata


def metadata_wide(metadata):
    rows = {}
    for item in metadata:
        key = (item["job_id"], item["file"])
        if key not in rows:
            rows[key] = {"job_id": item["job_id"], "file": item["file"]}
        rows[key][item["field"]] = item["value"]
    return list(rows.values())
