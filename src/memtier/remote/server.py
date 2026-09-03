"""Remote memory server: an in-RAM key/value block store reachable over TCP,
with injectable per-request latency and a bandwidth cap on responses.

Latency/bandwidth simulation is intentionally simple and stated as such:
- `latency_ms` is applied once per request (a sleep before the response is
  sent), modeling one-way propagation delay on the response leg only, not a
  true bidirectional per-packet RTT. Windows has no OS-level traffic-shaping
  equivalent to Linux `tc netem`, so this is software-injected delay inside
  our own server, not a claim about real WAN jitter.
- `bandwidth_mb_per_s` paces the response body in fixed chunks (see
  throttle.py); it does not model TCP slow-start, congestion, or jitter.

This is Phase 2: a general block store, tested standalone with synthetic
blocks. Phase 3 wires actual model tensors through the same client.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path

from memtier.remote import protocol
from memtier.remote.opslog import OpsLogger
from memtier.remote.throttle import make_bandwidth_limited_sender


class RemoteMemoryServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5959,
        latency_ms: float = 0.0,
        bandwidth_mb_per_s: float | None = None,
        log_path: Path | None = None,
    ):
        self.host = host
        self.port = port
        self.latency_s = latency_ms / 1000.0
        self.bandwidth_mb_per_s = bandwidth_mb_per_s
        self._sender = make_bandwidth_limited_sender(bandwidth_mb_per_s)
        self._store: dict[str, bytes] = {}
        self._store_lock = threading.Lock()
        self._logger = OpsLogger(log_path) if log_path else None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()

    def _log(self, **fields):
        if self._logger:
            self._logger.log(
                latency_ms=self.latency_s * 1000,
                bandwidth_mb_per_s=self.bandwidth_mb_per_s,
                **fields,
            )

    def _handle_client(self, conn: socket.socket, addr):
        conn.settimeout(30.0)
        try:
            while not self._stop.is_set():
                try:
                    op, key, payload = protocol.recv_request(conn)
                except (ConnectionError, socket.timeout, OSError):
                    break

                t_req = time.perf_counter()

                if op == protocol.OP_GET:
                    with self._store_lock:
                        value = self._store.get(key)
                    if self.latency_s > 0:
                        time.sleep(self.latency_s)
                    t_before_send = time.perf_counter()
                    if value is None:
                        protocol.send_resp_notfound(conn)
                        self._log(
                            op="GET", key=key, found=False, size_bytes=0,
                            wait_s=t_before_send - t_req, transfer_s=0.0,
                        )
                    else:
                        protocol.send_resp_ok(conn, value, chunked_sender=self._sender)
                        t_done = time.perf_counter()
                        self._log(
                            op="GET", key=key, found=True, size_bytes=len(value),
                            wait_s=t_before_send - t_req,
                            transfer_s=t_done - t_before_send,
                        )

                elif op == protocol.OP_PUT:
                    with self._store_lock:
                        self._store[key] = payload
                    if self.latency_s > 0:
                        time.sleep(self.latency_s)
                    protocol.send_resp_ok(conn, None)
                    self._log(
                        op="PUT", key=key, found=True, size_bytes=len(payload),
                        wait_s=time.perf_counter() - t_req, transfer_s=0.0,
                    )

                elif op == protocol.OP_DELETE:
                    with self._store_lock:
                        existed = self._store.pop(key, None) is not None
                    protocol.send_resp_ok(conn, None) if existed else protocol.send_resp_notfound(conn)
                    self._log(op="DELETE", key=key, found=existed, size_bytes=0,
                              wait_s=time.perf_counter() - t_req, transfer_s=0.0)
        finally:
            conn.close()

    def serve_forever(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(16)
        print(
            f"RemoteMemoryServer listening on {self.host}:{self.port} "
            f"(latency={self.latency_s*1000:.1f}ms, "
            f"bandwidth={self.bandwidth_mb_per_s or 'unlimited'} MB/s)"
        )
        try:
            while not self._stop.is_set():
                try:
                    self._sock.settimeout(1.0)
                    conn, addr = self._sock.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
        finally:
            self._sock.close()
            if self._logger:
                self._logger.close()

    def stop(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5959)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--bandwidth-mb-per-s", type=float, default=None)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    log_path = Path(args.log_path) if args.log_path else None
    server = RemoteMemoryServer(
        host=args.host,
        port=args.port,
        latency_ms=args.latency_ms,
        bandwidth_mb_per_s=args.bandwidth_mb_per_s,
        log_path=log_path,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
