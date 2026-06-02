from __future__ import print_function

import threading
import time
from pathlib import Path

try:
    from .parser import list_log_files, parse_metric_line
except ImportError:
    from parser import list_log_files, parse_metric_line


class LogStream(object):
    def __init__(self, logs_dir, event_queue, poll_seconds=0.5, start_at_end=True):
        self.logs_dir = logs_dir
        self.event_queue = event_queue
        self.poll_seconds = poll_seconds
        self.start_at_end = start_at_end
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="gateway-log-stream", daemon=True)
        self.files = {}
        self.initial_scan = True

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        for handle in list(self.files.values()):
            try:
                handle.close()
            except OSError:
                pass
        self.files = {}

    def run(self):
        while not self.stop_event.is_set():
            self._discover_files()
            self._read_new_lines()
            self.stop_event.wait(self.poll_seconds)

    def _discover_files(self):
        for path in list_log_files(self.logs_dir):
            key = str(path)
            if key in self.files:
                continue
            try:
                handle = path.open("rb")
                if self.start_at_end and self.initial_scan:
                    handle.seek(0, 2)
                self.files[key] = handle
            except OSError:
                continue
        self.initial_scan = False

    def _read_new_lines(self):
        for key, handle in list(self.files.items()):
            path = Path(key)
            if not path.exists():
                self._close_file(key)
                continue

            current_size = path.stat().st_size
            if current_size < handle.tell():
                handle.seek(0, 0)

            while not self.stop_event.is_set():
                raw_line = handle.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").replace("\x00", "").rstrip("\n")
                self._queue_line(path.name, line)

    def _queue_line(self, file_name, line):
        if "Request counts" not in line and "Completion stats" not in line:
            return
        ts = time.time()
        for event in parse_metric_line(line, file_name, ts):
            row = event.as_dict()
            row["source"] = "stream"
            self.event_queue.put(row)

    def _close_file(self, key):
        handle = self.files.pop(key, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
