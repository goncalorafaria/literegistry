import html
import ast
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue
import time

import fire
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from .parser import parse_logs, poll_registry_summary_with_status, unique_values
    from .streaming import LogStream
    from .vllm_parser import parse_vllm_logs, metadata_wide
except ImportError:
    from parser import parse_logs, poll_registry_summary_with_status, unique_values
    from streaming import LogStream
    from vllm_parser import parse_vllm_logs, metadata_wide


DEFAULT_LOGS = Path(__file__).resolve().parents[1] / "logs" / "gateway"
DEFAULT_SLURM_LOGS = Path(__file__).resolve().parents[1] / "logs" / "slurmcompose"
DEFAULT_LOG_LOCATIONS = [DEFAULT_LOGS, DEFAULT_SLURM_LOGS]
DEFAULT_REGISTRY = "redis://klone-login03.hyak.local:6379"
STARTUP_SEED_LINES = 50
REGISTRY_HISTORY_LIMIT = 500
WINDOW_OPTIONS = {
    "2 min": 120,
    "5 min": 300,
    "15 min": 900,
    "30 min": 1800,
    "1 hour": 3600,
    "2 hours": 7200,
    "6 hours": 21600,
    "12 hours": 43200,
    "all parsed": None,
}
VLLM_WINDOW_OPTIONS = {
    "10 min": 600,
    "30 min": 1800,
    "1 hour": 3600,
    "2 hours": 7200,
    "6 hours": 21600,
    "12 hours": 43200,
    "all parsed": None,
}


def _collect_app_args(
    logs=None,
    logs_dir=str(DEFAULT_LOGS),
    logs_path=None,
    registry=None,
    slurm_logs_dir=str(DEFAULT_SLURM_LOGS),
    vllm_logs_dir=None,
    seed_recent=True,
    poll_seconds=0.5,
    window="1 hour",
    refresh=True,
    refresh_seconds=5,
    poll_registry=True,
    registry_poll_seconds=5,
    show_vllm=True,
    vllm_newest_files=80,
    vllm_tail_lines=1000,
    vllm_window="2 hours",
):
    log_locations = logs or [logs_path or logs_dir, vllm_logs_dir or slurm_logs_dir]
    return {
        "logs": log_locations,
        "logs_dir": logs_path or logs_dir,
        "registry": registry or DEFAULT_REGISTRY,
        "slurm_logs_dir": vllm_logs_dir or slurm_logs_dir,
        "seed_recent": seed_recent,
        "poll_seconds": poll_seconds,
        "window": window,
        "refresh": refresh,
        "refresh_seconds": refresh_seconds,
        "poll_registry": poll_registry,
        "registry_poll_seconds": registry_poll_seconds,
        "show_vllm": show_vllm,
        "vllm_newest_files": vllm_newest_files,
        "vllm_tail_lines": vllm_tail_lines,
        "vllm_window": vllm_window,
    }


def parse_app_args(argv):
    if not argv:
        return {}

    return fire.Fire(
        _collect_app_args,
        command=argv,
        name="literegistry-console",
        serialize=lambda value: None,
    )


APP_ARGS = parse_app_args(sys.argv[1:])


def app_arg(name, default):
    return APP_ARGS.get(name, default)


def normalize_log_locations(value):
    if value is None:
        return [str(path) for path in DEFAULT_LOG_LOCATIONS]
    if isinstance(value, Path):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        raw_items = []
        for item in value:
            raw_items.extend(normalize_log_locations(item))
    else:
        text = str(value).strip()
        if not text:
            return []
        if text[0] in "[(":
            try:
                parsed = ast.literal_eval(text)
                return normalize_log_locations(parsed)
            except (SyntaxError, ValueError):
                pass
        raw_items = []
        for line in text.splitlines():
            raw_items.extend(part.strip() for part in line.split(","))

    paths = []
    seen = set()
    for item in raw_items:
        path = str(item).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def log_locations_text(locations):
    return "\n".join(str(location) for location in locations)


def bool_arg(name, default):
    value = app_arg(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def int_arg(name, default):
    try:
        return int(app_arg(name, default))
    except (TypeError, ValueError):
        return default


def float_arg(name, default):
    try:
        return float(app_arg(name, default))
    except (TypeError, ValueError):
        return default


def option_index(options, selected, default_index):
    try:
        return list(options).index(selected)
    except ValueError:
        return default_index


st.set_page_config(
    page_title="literegistry console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    [data-testid="stMetricValue"] { font-size: 1.7rem; }
    .small-note { color: #667085; font-size: 0.9rem; }

    /* Keep content fully readable during auto-refresh reruns: Streamlit dims
       "stale" elements (and shows a fade/overlay) while it recomputes. */
    [data-stale="true"],
    div[data-stale="true"],
    .element-container[data-stale="true"],
    [data-testid="stAppViewContainer"] [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
    .stApp [data-stale="true"] * { opacity: 1 !important; }
    /* Hide the top "running" indicator bar so it doesn't flash on each refresh. */
    [data-testid="stStatusWidget"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_seed_events(logs_path, max_lines_per_file):
    rows = parse_logs(logs_path, max_lines_per_file=max_lines_per_file)
    for row in rows:
        row["source"] = "seed"
    return rows


def rows_to_frame(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime([datetime.fromtimestamp(ts) for ts in df["ts"]])
    df["rate"] = df["count"] / df["window"].clip(lower=0.001)
    return df


@st.cache_data(ttl=3, show_spinner=False)
def load_vllm_events(logs_path, newest_files, tail_lines):
    rows, metadata, errors = parse_vllm_logs(
        logs_path,
        newest_files=newest_files,
        tail_lines=tail_lines,
    )
    event_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metadata_wide(metadata))
    error_df = pd.DataFrame(errors)
    if not event_df.empty:
        event_df["datetime"] = pd.to_datetime(
            [datetime.fromtimestamp(ts) for ts in event_df["ts"]]
        )
    if not error_df.empty:
        error_df["datetime"] = pd.to_datetime(
            [datetime.fromtimestamp(ts) for ts in error_df["ts"]]
        )
    return event_df, meta_df, error_df


def drain_stream_queue():
    drained = 0
    while not st.session_state.event_queue.empty():
        st.session_state.events.append(st.session_state.event_queue.get())
        drained += 1
    return drained


def restart_stream(logs_path, poll_seconds, seed_recent, max_lines_per_file):
    stream = st.session_state.get("stream")
    if stream is not None:
        stream.stop()

    st.session_state.event_queue = Queue()
    st.session_state.events = []
    if seed_recent:
        st.session_state.events.extend(load_seed_events(logs_path, max_lines_per_file))

    stream = LogStream(
        logs_path,
        st.session_state.event_queue,
        poll_seconds=poll_seconds,
        start_at_end=True,
    )
    stream.start()
    st.session_state.stream = stream
    st.session_state.stream_config = (
        logs_path,
        poll_seconds,
        seed_recent,
        max_lines_per_file,
    )
    st.session_state.stream_started_at = time.time()


def filter_recent(df, seconds):
    if df.empty or seconds is None:
        return df
    cutoff = time.time() - seconds
    return df[df["ts"] >= cutoff].copy()


def render_vllm_panel(events, metadata, selected_jobs, window_seconds):
    if events.empty:
        st.warning("No vLLM telemetry found in the selected Slurm log tails.")
        return

    metric_events = events[events["kind"] == "engine_metric"].copy()
    http_events = events[events["kind"] == "http"].copy()
    metric_events = filter_recent(metric_events, window_seconds)
    http_events = filter_recent(http_events, window_seconds)

    if selected_jobs:
        metric_events = metric_events[
            metric_events["job_id"].isin(selected_jobs)
        ].copy()
        http_events = http_events[http_events["job_id"].isin(selected_jobs)].copy()

    avg_prompt = metric_events["prompt_tps"].mean() if not metric_events.empty else 0.0
    avg_generation = (
        metric_events["generation_tps"].mean() if not metric_events.empty else 0.0
    )
    max_running = (
        int(metric_events["running_reqs"].max()) if not metric_events.empty else 0
    )
    max_waiting = (
        int(metric_events["waiting_reqs"].max()) if not metric_events.empty else 0
    )
    max_kv = metric_events["kv_cache_pct"].max() if not metric_events.empty else 0.0
    post_count = (
        int((http_events["method"] == "POST").sum()) if not http_events.empty else 0
    )

    cols = st.columns(6)
    cols[0].metric("Avg prompt tok/s", f"{avg_prompt:,.1f}")
    cols[1].metric("Avg gen tok/s", f"{avg_generation:,.1f}")
    cols[2].metric("Max running", f"{max_running:,}")
    cols[3].metric("Max waiting", f"{max_waiting:,}")
    cols[4].metric("Max KV cache", f"{max_kv:.1f}%")
    cols[5].metric("POSTs seen", f"{post_count:,}")

    engine_tab, jobs_tab, http_tab = st.tabs(["Engine metrics", "Backend jobs", "HTTP"])

    with engine_tab:
        if metric_events.empty:
            st.info("No vLLM engine heartbeat samples in this wall-clock window.")
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Throughput")
                throughput = metric_events.melt(
                    id_vars=["datetime", "job_id", "file"],
                    value_vars=["prompt_tps", "generation_tps"],
                    var_name="metric",
                    value_name="tokens_per_s",
                )
                st.line_chart(
                    throughput,
                    x="datetime",
                    y="tokens_per_s",
                    color="metric",
                    height=310,
                )
            with right:
                st.subheader("Queue pressure")
                pressure = metric_events.melt(
                    id_vars=["datetime", "job_id", "file"],
                    value_vars=["running_reqs", "waiting_reqs"],
                    var_name="metric",
                    value_name="requests",
                )
                st.line_chart(
                    pressure, x="datetime", y="requests", color="metric", height=310
                )

            left, right = st.columns(2)
            with left:
                st.subheader("GPU KV cache usage")
                st.area_chart(
                    metric_events,
                    x="datetime",
                    y="kv_cache_pct",
                    color="job_id",
                    height=300,
                )
            with right:
                st.subheader("Prefix cache hit rate")
                st.line_chart(
                    metric_events,
                    x="datetime",
                    y="prefix_hit_pct",
                    color="job_id",
                    height=300,
                )

    with jobs_tab:
        st.subheader("Startup metadata")
        if metadata.empty:
            st.info("No startup metadata found in the selected Slurm tails.")
        else:
            st.dataframe(
                metadata.sort_values("job_id"),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Job summary")
        if metric_events.empty:
            st.info("No engine samples available for the selected jobs/window.")
        else:
            summary = (
                metric_events.groupby("job_id")
                .agg(
                    file=("file", "last"),
                    samples=("job_id", "size"),
                    avg_prompt_tps=("prompt_tps", "mean"),
                    avg_generation_tps=("generation_tps", "mean"),
                    max_running=("running_reqs", "max"),
                    max_waiting=("waiting_reqs", "max"),
                    max_kv_cache_pct=("kv_cache_pct", "max"),
                    latest_prefix_hit_pct=("prefix_hit_pct", "last"),
                )
                .reset_index()
            )
            st.dataframe(
                summary.sort_values("avg_generation_tps", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with http_tab:
        if http_events.empty:
            st.info("No HTTP request lines found in the selected window.")
        else:
            by_path = (
                http_events.groupby(["method", "path", "status"])
                .size()
                .reset_index(name="count")
            )
            st.subheader("Endpoint counts")
            st.dataframe(
                by_path.sort_values("count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                by_path.sort_values("count", ascending=False).head(20),
                x="path",
                y="count",
                color="method",
            )


def render_vllm_errors_panel(errors, selected_jobs, window_seconds):
    st.subheader("vLLM machine errors")
    if errors.empty:
        st.success("No error/traceback lines found in the selected vLLM log tails.")
        return

    errors = filter_recent(errors, window_seconds)
    if selected_jobs and "job_id" in errors.columns:
        errors = errors[errors["job_id"].isin(selected_jobs)].copy()

    if errors.empty:
        st.success("No vLLM errors in the selected jobs/time window.")
        return

    total_errors = len(errors)
    jobs_affected = errors["job_id"].nunique() if "job_id" in errors.columns else 0
    oom_count = int((errors["level"] == "OOM").sum()) if "level" in errors.columns else 0

    cols = st.columns(3)
    cols[0].metric("Error lines", f"{total_errors:,}")
    cols[1].metric("Jobs affected", f"{jobs_affected:,}")
    cols[2].metric("OOM events", f"{oom_count:,}")

    by_level = (
        errors.groupby(["job_id", "level"]).size().reset_index(name="count")
        if {"job_id", "level"}.issubset(errors.columns)
        else pd.DataFrame()
    )
    if not by_level.empty:
        st.caption("Error counts by job and level")
        st.dataframe(
            by_level.sort_values("count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    recent = errors.sort_values("ts", ascending=False)
    max_show = st.slider(
        "Errors to show",
        5,
        200,
        max(5, min(40, len(recent))),
        step=5,
        key="vllm_error_show",
    )
    show_full = st.toggle(
        "Expand full tracebacks",
        value=True,
        key="vllm_error_expand",
        help="Show the complete traceback/error text for each entry.",
    )

    st.caption("Most recent errors first")
    for _, row in recent.head(max_show).iterrows():
        level = row.get("level", "ERROR")
        job_id = row.get("job_id", "?")
        when = row.get("time", "")
        file_name = row.get("file", "")
        headline = str(row.get("message", "")).splitlines()[0] if row.get("message") else ""
        line_count = int(row.get("line_count", 1) or 1)
        suffix = f" · {line_count} lines" if line_count > 1 else ""
        label = f"[{level}] {when} · job {job_id} · {file_name}{suffix} — {headline}"
        detail = str(row.get("detail") or row.get("raw") or headline)
        with st.expander(label, expanded=show_full and line_count > 1):
            st.code(detail, language="text")

    with st.expander("Error table (compact)", expanded=False):
        error_cols = [
            c
            for c in ["time", "job_id", "file", "level", "line_count", "message"]
            if c in errors.columns
        ]
        st.dataframe(
            recent[error_cols],
            use_container_width=True,
            hide_index=True,
        )


def weighted_average(values, weights):
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return 0.0
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def poll_registry_if_due(registry, enabled, poll_interval):
    if "registry_rows" not in st.session_state:
        st.session_state.registry_rows = []
    if "registry_history" not in st.session_state:
        st.session_state.registry_history = []

    if not enabled or not registry:
        return 0

    now = time.time()
    last_poll = st.session_state.get("last_registry_poll", 0.0)
    if now - last_poll < poll_interval:
        return 0

    rows, error = poll_registry_summary_with_status(registry)
    st.session_state.last_registry_poll = now
    if not rows:
        st.session_state.registry_error = error
        return 0

    st.session_state.registry_error = ""
    rows = clean_registry_rows(rows)
    if not rows:
        st.session_state.registry_error = (
            "Registry summary did not include model/service count rows."
        )
        return 0

    st.session_state.registry_rows = rows
    st.session_state.registry_history.extend(rows)
    st.session_state.registry_history = st.session_state.registry_history[
        -REGISTRY_HISTORY_LIMIT:
    ]
    return len(rows)


def clean_registry_rows(rows):
    clean = []
    for row in rows:
        model = str(row.get("model", "")).strip()
        lowered = model.lower()
        if (
            not model
            or "://" in model
            or "redis" in lowered
            or lowered.startswith("successfully connected")
        ):
            continue
        clean.append(row)
    return clean


def split_rectangles(items, x, y, width, height):
    if not items:
        return []
    if len(items) == 1:
        item = dict(items[0])
        item.update({"x": x, "y": y, "w": width, "h": height})
        return [item]

    total = sum(item["count"] for item in items)
    half = total / 2.0
    running = 0
    split_at = 0
    best_gap = total
    for index, item in enumerate(items):
        running += item["count"]
        gap = abs(half - running)
        if gap < best_gap:
            best_gap = gap
            split_at = index + 1

    first = items[:split_at]
    second = items[split_at:]
    first_total = sum(item["count"] for item in first)

    if width >= height:
        first_width = width * first_total / total
        return split_rectangles(first, x, y, first_width, height) + split_rectangles(
            second, x + first_width, y, width - first_width, height
        )

    first_height = height * first_total / total
    return split_rectangles(first, x, y, width, first_height) + split_rectangles(
        second, x, y + first_height, width, height - first_height
    )


def registry_treemap_html(rows):
    active_rows = [
        {"model": str(row["model"]), "count": int(row["count"])}
        for row in rows
        if int(row.get("count", 0)) > 0
    ]
    if not active_rows:
        return ""

    colors = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#0891b2",
        "#ca8a04",
        "#be185d",
        "#475569",
        "#ea580c",
        "#0f766e",
    ]
    active_rows = sorted(active_rows, key=lambda row: row["count"], reverse=True)
    rects = split_rectangles(active_rows, 0.0, 0.0, 100.0, 100.0)
    tiles = []
    legend_items = []
    for index, rect in enumerate(rects):
        color = colors[index % len(colors)]
        label = html.escape(rect["model"])
        count = int(rect["count"])
        area = rect["w"] * rect["h"]
        tile_number = index + 1
        if area >= 700:
            label_class = "large-label"
            label_html = """
                <div class="registry-index">#{tile_number}</div>
                <div class="registry-name">{label}</div>
                <div class="registry-count">{count}</div>
            """
        elif area >= 260:
            label_class = "medium-label"
            label_html = """
                <div class="registry-index">#{tile_number}</div>
                <div class="registry-name">{label}</div>
                <div class="registry-count">{count}</div>
            """
        elif area >= 95:
            label_class = "small-label"
            label_html = """
                <div class="registry-index">#{tile_number}</div>
                <div class="registry-count">{count}</div>
            """
        else:
            label_class = "tiny-label"
            label_html = """<div class="registry-index">#{tile_number}</div>"""
        tiles.append(
            """
            <div class="registry-tile" style="left:{x:.3f}%; top:{y:.3f}%; width:{w:.3f}%; height:{h:.3f}%; background:{color};">
                <div class="registry-label {label_class}">
                    {label_html}
                </div>
            </div>
            """.format(
                x=rect["x"],
                y=rect["y"],
                w=rect["w"],
                h=rect["h"],
                color=color,
                label=label,
                count=count,
                tile_number=tile_number,
                label_class=label_class,
                label_html=label_html.format(
                    label=label,
                    count=count,
                    tile_number=tile_number,
                ),
            )
        )
        legend_items.append(
            """
            <div class="legend-row">
                <div class="legend-swatch" style="background:{color};"></div>
                <div class="legend-text">
                    <div class="legend-name">#{tile_number} {label}</div>
                    <div class="legend-count">{count} replicas</div>
                </div>
            </div>
            """.format(
                color=color,
                tile_number=tile_number,
                label=label,
                count=count,
            )
        )
    return """
    <!doctype html>
    <html>
    <head>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .registry-wrap {{
                display: flex;
                gap: 14px;
                align-items: stretch;
                width: 100%;
                min-height: 640px;
            }}
            .registry-square {{
                flex: 0 0 min(64vw, 640px);
                height: min(64vw, 640px);
                min-width: 420px;
                min-height: 420px;
                position: relative;
                border: 1px solid #d0d5dd;
                background: #f8fafc;
                overflow: hidden;
            }}
            .registry-tile {{
                position: absolute;
                box-sizing: border-box;
                border: 2px solid #ffffff;
                color: #ffffff;
                overflow: hidden;
                animation: tile-pop 360ms ease-out both;
                transition: left 450ms ease, top 450ms ease, width 450ms ease, height 450ms ease;
            }}
            .registry-label {{
                padding: 10px;
                line-height: 1.15;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
            }}
            .registry-index {{
                display: inline-block;
                font-weight: 800;
                font-size: 12px;
                line-height: 1;
                margin-bottom: 5px;
                padding: 3px 5px;
                border-radius: 4px;
                background: rgba(0, 0, 0, 0.24);
            }}
            .registry-name {{
                font-weight: 700;
                font-size: 14px;
                overflow-wrap: anywhere;
            }}
            .registry-count {{
                font-size: 28px;
                font-weight: 800;
                margin-top: 3px;
            }}
            .medium-label {{
                padding: 8px;
            }}
            .medium-label .registry-name {{
                font-size: 12px;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }}
            .medium-label .registry-count {{
                font-size: 21px;
            }}
            .small-label {{
                padding: 6px;
            }}
            .small-label .registry-count {{
                font-size: 18px;
                line-height: 1;
            }}
            .tiny-label {{
                padding: 4px;
            }}
            .tiny-label .registry-index {{
                font-size: 10px;
                padding: 2px 4px;
                margin: 0;
            }}
            .legend {{
                flex: 1 1 260px;
                min-width: 220px;
                max-height: 640px;
                overflow: auto;
                border: 1px solid #d0d5dd;
                background: #ffffff;
                padding: 10px;
                box-sizing: border-box;
            }}
            .legend-title {{
                font-size: 13px;
                font-weight: 800;
                color: #344054;
                margin-bottom: 8px;
            }}
            .legend-row {{
                display: flex;
                gap: 8px;
                align-items: flex-start;
                padding: 7px 0;
                border-top: 1px solid #eef2f6;
            }}
            .legend-row:first-of-type {{
                border-top: 0;
            }}
            .legend-swatch {{
                width: 12px;
                height: 12px;
                border-radius: 3px;
                margin-top: 3px;
                flex: 0 0 auto;
            }}
            .legend-name {{
                font-size: 12px;
                font-weight: 700;
                color: #101828;
                overflow-wrap: anywhere;
            }}
            .legend-count {{
                font-size: 12px;
                color: #667085;
                margin-top: 1px;
            }}
            @media (max-width: 760px) {{
                .registry-wrap {{
                    display: block;
                    min-height: 0;
                }}
                .registry-square {{
                    width: 100vw;
                    height: 100vw;
                    min-width: 0;
                    min-height: 360px;
                    max-height: 620px;
                }}
                .legend {{
                    margin-top: 10px;
                    max-height: 260px;
                }}
            }}
            @keyframes tile-pop {{
                from {{ transform: scale(0.985); opacity: 0.70; }}
                to {{ transform: scale(1); opacity: 1; }}
            }}
        </style>
    </head>
    <body>
        <div class="registry-wrap">
            <div class="registry-square">{tiles}</div>
            <div class="legend">
                <div class="legend-title">Model index</div>
                {legend_items}
            </div>
        </div>
    </body>
    </html>
    """.format(tiles="".join(tiles), legend_items="".join(legend_items))


def render_registry_treemap(rows):
    tree_html = registry_treemap_html(rows)
    if not tree_html:
        st.info("Registry returned no positive replica counts.")
        return
    components.html(tree_html, height=690, scrolling=False)


def render_registry_panel():
    registry_rows = clean_registry_rows(st.session_state.get("registry_rows", []))
    history_rows = clean_registry_rows(st.session_state.get("registry_history", []))
    st.session_state.registry_rows = registry_rows
    st.session_state.registry_history = history_rows
    registry_df = pd.DataFrame(registry_rows)
    history_df = pd.DataFrame(history_rows)

    if registry_df.empty:
        st.info("Polling is on, but no registry counts have arrived yet.")
        error = st.session_state.get("registry_error", "")
        if error:
            st.caption(error)
        st.code("literegistry summary --registry redis://host:6379", language="bash")
        return

    total_replicas = int(registry_df["count"].sum())
    latest_ts = float(registry_df["ts"].max())
    st.metric("Total replicas", f"{total_replicas:,}")
    st.markdown(
        "<div class='small-note'>Latest registry snapshot: {}</div>".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_ts))
        ),
        unsafe_allow_html=True,
    )
    render_registry_treemap(registry_df.to_dict("records"))

    if not history_df.empty:
        history_df["datetime"] = pd.to_datetime(
            [datetime.fromtimestamp(ts) for ts in history_df["ts"]]
        )
        st.subheader("Registry counts over time")
        st.line_chart(history_df, x="datetime", y="count", color="model", height=260)

    st.subheader("Current registry counts")
    st.dataframe(
        registry_df.sort_values("count", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


with st.sidebar:
    st.header("literegistry console")
    log_locations = normalize_log_locations(app_arg("logs", DEFAULT_LOG_LOCATIONS))
    logs_text = st.text_area(
        "Log locations",
        value=log_locations_text(log_locations),
        help="One path per line. The console scans these locations for gateway and vLLM log lines.",
        height=88,
    )
    log_locations = normalize_log_locations(logs_text)
    seed_recent = st.toggle(
        "Seed recent log tail",
        value=bool_arg("seed_recent", True),
    )
    st.caption(
        f"Startup seed is capped at the latest {STARTUP_SEED_LINES:,} lines per file."
    )
    poll_seconds = st.slider(
        "Tail poll seconds",
        0.2,
        5.0,
        float_arg("poll_seconds", 0.5),
        step=0.1,
    )
    window_label = st.selectbox(
        "Time window",
        list(WINDOW_OPTIONS),
        index=option_index(WINDOW_OPTIONS, app_arg("window", "1 hour"), 4),
    )
    refresh = st.toggle("Live refresh", value=bool_arg("refresh", True))
    refresh_seconds = st.slider(
        "Refresh seconds",
        2,
        30,
        int_arg("refresh_seconds", 5),
    )
    restart_clicked = st.button("Restart listeners", use_container_width=True)

    st.divider()
    registry = st.text_input(
        "Registry",
        value=str(app_arg("registry", DEFAULT_REGISTRY)),
        placeholder="redis://klone-login03.hyak.local:6379",
    )
    poll_registry = st.toggle(
        "Poll registry summary",
        value=bool_arg("poll_registry", True),
    )
    registry_poll_seconds = st.slider(
        "Registry poll seconds",
        1,
        30,
        int_arg("registry_poll_seconds", 5),
    )

    st.divider()
    show_vllm = st.toggle(
        "Show vLLM Slurm telemetry",
        value=bool_arg("show_vllm", True),
    )
    vllm_newest_files = st.slider(
        "vLLM newest files",
        10,
        300,
        int_arg("vllm_newest_files", 80),
        step=10,
    )
    vllm_tail_lines = st.slider(
        "vLLM tail lines per file",
        200,
        5000,
        int_arg("vllm_tail_lines", 1000),
        step=100,
    )
    vllm_window_label = st.selectbox(
        "vLLM time window",
        list(VLLM_WINDOW_OPTIONS),
        index=option_index(VLLM_WINDOW_OPTIONS, app_arg("vllm_window", "2 hours"), 3),
    )


if "event_queue" not in st.session_state:
    st.session_state.event_queue = Queue()
if "events" not in st.session_state:
    st.session_state.events = []

stream_config = (tuple(log_locations), poll_seconds, seed_recent, STARTUP_SEED_LINES)
if restart_clicked or st.session_state.get("stream_config") != stream_config:
    restart_stream(log_locations, poll_seconds, seed_recent, STARTUP_SEED_LINES)

drained_count = drain_stream_queue()
registry_polled_count = poll_registry_if_due(
    registry,
    poll_registry,
    registry_poll_seconds,
)
events = rows_to_frame(st.session_state.events)
if show_vllm:
    vllm_events, vllm_metadata, vllm_errors = load_vllm_events(
        log_locations,
        vllm_newest_files,
        vllm_tail_lines,
    )
else:
    vllm_events = pd.DataFrame()
    vllm_metadata = pd.DataFrame()
    vllm_errors = pd.DataFrame()

job_id_set = set()
if not vllm_events.empty:
    vllm_metric_events = vllm_events[vllm_events["kind"] == "engine_metric"].copy()
    if not vllm_metric_events.empty:
        job_id_set.update(vllm_metric_events["job_id"].unique())
if not vllm_errors.empty and "job_id" in vllm_errors.columns:
    job_id_set.update(vllm_errors["job_id"].unique())
vllm_jobs = sorted(job_id_set)

with st.sidebar:
    if show_vllm:
        selected_vllm_jobs = st.multiselect("vLLM jobs", vllm_jobs, default=vllm_jobs)
    else:
        selected_vllm_jobs = []

if events.empty:
    st.title("literegistry console")
    st.warning("Listening for gateway metric lines. No events have arrived yet.")
    tab_registry_empty, tab_vllm_empty, tab_vllm_errors_empty = st.tabs(
        ["Registry", "vLLM", "vLLM errors"]
    )
    with tab_registry_empty:
        if poll_registry:
            render_registry_panel()
            if registry_polled_count:
                st.caption(
                    f"Polled {registry_polled_count} registry rows this refresh."
                )
        else:
            st.info(
                "Enable registry polling and provide a Redis URL to show live registry counts."
            )
    with tab_vllm_empty:
        if show_vllm:
            render_vllm_panel(
                vllm_events,
                vllm_metadata,
                selected_vllm_jobs,
                VLLM_WINDOW_OPTIONS[vllm_window_label],
            )
        else:
            st.info("Enable vLLM Slurm telemetry in the sidebar.")
    with tab_vllm_errors_empty:
        if show_vllm:
            render_vllm_errors_panel(
                vllm_errors,
                selected_vllm_jobs,
                VLLM_WINDOW_OPTIONS[vllm_window_label],
            )
        else:
            st.info("Enable vLLM Slurm telemetry in the sidebar.")
    st.code(
        "literegistry console --logs /path/to/logs --registry redis://host:6379",
        language="bash",
    )
    if refresh:
        time.sleep(refresh_seconds)
        st.rerun()
    st.stop()

all_models = unique_values(events.to_dict("records"), "model")
all_files = unique_values(events.to_dict("records"), "file")

with st.sidebar:
    selected_models = st.multiselect("Models/tools", all_models, default=all_models)
    selected_files = st.multiselect("Log files", all_files, default=all_files)

filtered = events[
    events["model"].isin(selected_models) & events["file"].isin(selected_files)
].copy()
filtered = filter_recent(filtered, WINDOW_OPTIONS[window_label])

if filtered.empty:
    st.title("literegistry console")
    latest_event = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(events["ts"].max())
    )
    now_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    st.warning(
        "No current rows match the model, file, and wall-clock time-window filters. "
        "Latest parsed gateway event is {}; current time is {}.".format(
            latest_event, now_label
        )
    )
    tab_registry_empty_filter, tab_vllm_empty_filter, tab_vllm_errors_empty_filter = (
        st.tabs(["Registry", "vLLM", "vLLM errors"])
    )
    with tab_registry_empty_filter:
        if poll_registry:
            render_registry_panel()
        else:
            st.info(
                "Enable registry polling and provide a Redis URL to show live registry counts."
            )
    with tab_vllm_empty_filter:
        if show_vllm:
            render_vllm_panel(
                vllm_events,
                vllm_metadata,
                selected_vllm_jobs,
                VLLM_WINDOW_OPTIONS[vllm_window_label],
            )
        else:
            st.info("Enable vLLM Slurm telemetry in the sidebar.")
    with tab_vllm_errors_empty_filter:
        if show_vllm:
            render_vllm_errors_panel(
                vllm_errors,
                selected_vllm_jobs,
                VLLM_WINDOW_OPTIONS[vllm_window_label],
            )
        else:
            st.info("Enable vLLM Slurm telemetry in the sidebar.")
    if refresh:
        time.sleep(refresh_seconds)
        st.rerun()
    st.stop()

requests = filtered[filtered["kind"] == "requests"].copy()
completions = filtered[filtered["kind"] == "completions"].copy()

st.title("literegistry console")
latest_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(events["ts"].max()))
latest_age = max(0, int(time.time() - float(events["ts"].max())))
queue_depth = st.session_state.event_queue.qsize()
stream_age = int(time.time() - st.session_state.get("stream_started_at", time.time()))
st.markdown(
    f"<div class='small-note'>Queued event stream: {len(events):,} processed rows, {queue_depth} waiting, {drained_count} drained this refresh. Latest event: {latest_time} ({latest_age}s ago). Listener age: {stream_age}s.</div>",
    unsafe_allow_html=True,
)
if (
    WINDOW_OPTIONS[window_label] is not None
    and latest_age > WINDOW_OPTIONS[window_label]
):
    st.warning(
        "Gateway metric summaries are stale for the selected window. "
        "The rate charts are based on wall-clock time, so old 6pm samples are now excluded."
    )

total_requests = int(requests["count"].sum()) if not requests.empty else 0
total_completions = int(completions["count"].sum()) if not completions.empty else 0
avg_latency = (
    weighted_average(completions["avg"], completions["count"])
    if not completions.empty
    else 0.0
)
max_latency = float(completions["max"].max()) if not completions.empty else 0.0
pressure = total_requests - total_completions

kpi_cols = st.columns(5)
kpi_cols[0].metric("Requests", f"{total_requests:,}")
kpi_cols[1].metric("Completions", f"{total_completions:,}")
kpi_cols[2].metric("Backlog pressure", f"{pressure:,}")
kpi_cols[3].metric("Avg latency", f"{avg_latency:.2f}s")
kpi_cols[4].metric("Max latency", f"{max_latency:.2f}s")

rate_rows = filtered.pivot_table(
    index=["datetime", "model", "file"],
    columns="kind",
    values="rate",
    aggfunc="sum",
).reset_index()
for column in ("requests", "completions"):
    if column not in rate_rows:
        rate_rows[column] = 0.0
rate_rows["backlog_pressure_per_s"] = rate_rows["requests"] - rate_rows["completions"]

vllm_errors_label = "vLLM errors"
if show_vllm and not vllm_errors.empty:
    vllm_errors_label = f"vLLM errors ({len(vllm_errors):,})"

tab_live, tab_breakdown, tab_registry, tab_vllm, tab_vllm_errors, tab_raw = st.tabs(
    ["Live pressure", "Breakdowns", "Registry", "vLLM", vllm_errors_label, "Raw events"]
)

with tab_live:
    left, right = st.columns(2)
    with left:
        st.subheader("Request and completion rate")
        long_rates = rate_rows.melt(
            id_vars=["datetime", "model", "file"],
            value_vars=["requests", "completions"],
            var_name="metric",
            value_name="rate_per_s",
        )
        st.line_chart(
            long_rates,
            x="datetime",
            y="rate_per_s",
            color="metric",
            height=310,
        )
    with right:
        st.subheader("Backlog pressure")
        st.area_chart(
            rate_rows,
            x="datetime",
            y="backlog_pressure_per_s",
            color="model",
            height=310,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Average latency")
        st.line_chart(
            completions,
            x="datetime",
            y="avg",
            color="model",
            height=300,
        )
    with right:
        st.subheader("Max latency")
        st.line_chart(
            completions,
            x="datetime",
            y="max",
            color="model",
            height=300,
        )

with tab_breakdown:
    by_model = filtered.pivot_table(
        index="model",
        columns="kind",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for column in ("requests", "completions"):
        if column not in by_model:
            by_model[column] = 0
    by_model["backlog_pressure"] = by_model["requests"] - by_model["completions"]
    if completions.empty:
        latency_by_model = pd.DataFrame(columns=["model", "avg_latency", "max_latency"])
    else:
        latency_by_model = (
            completions.groupby("model")
            .apply(
                lambda group: pd.Series(
                    {
                        "avg_latency": weighted_average(group["avg"], group["count"]),
                        "max_latency": group["max"].max(),
                    }
                )
            )
            .reset_index()
        )
    by_model = by_model.merge(latency_by_model, how="left", on="model").fillna(0)

    left, right = st.columns(2)
    with left:
        st.subheader("Busy models/tools")
        st.bar_chart(
            by_model.sort_values("requests", ascending=False), x="model", y="requests"
        )
    with right:
        st.subheader("Slow models/tools")
        st.bar_chart(
            by_model.sort_values("max_latency", ascending=False),
            x="model",
            y="max_latency",
        )

    st.subheader("Model/tool summary")
    st.dataframe(
        by_model.sort_values(["backlog_pressure", "requests"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Workflow source summary")
    by_file = filtered.pivot_table(
        index="file",
        columns="kind",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    st.dataframe(by_file, use_container_width=True, hide_index=True)

with tab_registry:
    if not poll_registry:
        st.info(
            "Enable registry polling and provide a Redis URL to show live registry counts."
        )
        st.code("literegistry summary --registry redis://host:6379", language="bash")
    else:
        render_registry_panel()
        if registry_polled_count:
            st.caption(f"Polled {registry_polled_count} registry rows this refresh.")

with tab_vllm:
    if not show_vllm:
        st.info("Enable vLLM Slurm telemetry in the sidebar.")
    else:
        render_vllm_panel(
            vllm_events,
            vllm_metadata,
            selected_vllm_jobs,
            VLLM_WINDOW_OPTIONS[vllm_window_label],
        )

with tab_vllm_errors:
    if not show_vllm:
        st.info("Enable vLLM Slurm telemetry in the sidebar.")
    else:
        render_vllm_errors_panel(
            vllm_errors,
            selected_vllm_jobs,
            VLLM_WINDOW_OPTIONS[vllm_window_label],
        )

with tab_raw:
    show_rows = st.slider("Raw rows", 25, 500, 100, step=25)
    raw_cols = ["time", "file", "kind", "model", "count", "avg", "max", "raw"]
    st.dataframe(
        filtered.sort_values("ts", ascending=False)[raw_cols].head(show_rows),
        use_container_width=True,
        hide_index=True,
    )

with st.sidebar:
    st.divider()
    st.caption("Run with:")
    st.code(
        "literegistry console --logs /path/to/logs --registry redis://host:6379",
        language="bash",
    )

if refresh:
    time.sleep(refresh_seconds)
    st.rerun()
