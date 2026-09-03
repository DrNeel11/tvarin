"""Client for the remote memory server.

Latency/bandwidth on the download path (GET) is simulated server-side (see
server.py); on the upload path (PUT) it's simulated here, since that's the
leg the client actually controls the pacing of. A single persistent TCP
connection is reused across requests, matching how a real prefetcher would
hold one connection to the remote tier rather than reconnecting per tensor.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from memtier.remote import protocol
from memtier.remote.throttle import make_bandwidth_limited_sender


@dataclass
class TransferResult:
    found: bool
    size_bytes: int
    total_s: float
    payload: bytes | None = None


class RemoteMemoryClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5959,
        upload_bandwidth_mb_per_s: float | None = None,
        timeout_s: float = 30.0,
    ):
        self.host = host
        self.port = port
        self._sender = make_bandwidth_limited_sender(upload_bandwidth_mb_per_s)
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout_s)
        self._sock.connect((self.host, self.port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def get(self, key: str) -> TransferResult:
        t0 = time.perf_counter()
        protocol.send_get(self._sock, key)
        op, payload = protocol.recv_response(self._sock)
        total_s = time.perf_counter() - t0
        if op == protocol.OP_RESP_NOTFOUND:
            return TransferResult(found=False, size_bytes=0, total_s=total_s)
        return TransferResult(found=True, size_bytes=len(payload), total_s=total_s, payload=payload)

    def put(self, key: str, payload: bytes) -> TransferResult:
        t0 = time.perf_counter()
        protocol.send_put(self._sock, key, payload, chunked_sender=self._sender)
        op, _ = protocol.recv_response(self._sock)
        total_s = time.perf_counter() - t0
        return TransferResult(found=(op == protocol.OP_RESP_OK), size_bytes=len(payload), total_s=total_s)

    def delete(self, key: str) -> TransferResult:
        t0 = time.perf_counter()
        protocol.send_delete(self._sock, key)
        op, _ = protocol.recv_response(self._sock)
        total_s = time.perf_counter() - t0
        return TransferResult(found=(op == protocol.OP_RESP_OK), size_bytes=0, total_s=total_s)
