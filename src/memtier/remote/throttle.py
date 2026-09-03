"""Bandwidth pacing for the simulated remote-memory link.

A real network's bandwidth limit shows up as backpressure spread across the
whole transfer, not one lump delay — so this paces send() in fixed-size
chunks against a wall-clock budget, correcting for drift each chunk rather
than sleeping a fixed amount per chunk (which would drift under scheduler
jitter on Windows).
"""

from __future__ import annotations

import socket
import time

CHUNK_SIZE = 64 * 1024


def make_bandwidth_limited_sender(bandwidth_mb_per_s: float | None):
    """Returns a callable(sock, payload) that sends `payload` paced to
    `bandwidth_mb_per_s` megabytes/sec. None disables pacing (send as fast
    as the OS/network allows)."""

    if bandwidth_mb_per_s is None or bandwidth_mb_per_s <= 0:
        def _unthrottled(sock: socket.socket, payload: bytes):
            sock.sendall(payload)

        return _unthrottled

    bytes_per_s = bandwidth_mb_per_s * 1_000_000

    def _throttled(sock: socket.socket, payload: bytes):
        start = time.perf_counter()
        sent = 0
        n = len(payload)
        while sent < n:
            end = min(sent + CHUNK_SIZE, n)
            sock.sendall(payload[sent:end])
            sent = end
            target_elapsed = sent / bytes_per_s
            actual_elapsed = time.perf_counter() - start
            sleep_for = target_elapsed - actual_elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    return _throttled
