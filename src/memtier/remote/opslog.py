"""Structured JSONL logging for every remote-memory operation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class OpsLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1)

    def log(self, **fields):
        record = {"ts": time.time(), **fields}
        line = json.dumps(record)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self):
        self._fh.close()
