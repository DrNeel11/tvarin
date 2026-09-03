"""Wire protocol for the remote memory server.

Deliberately a tiny hand-rolled binary framing (not HTTP/JSON) so that
latency and bandwidth simulation happen at a level where we control exactly
what's being timed, and framing overhead doesn't dominate small-block
transfers the way HTTP headers would.

Frame layout, all integers big-endian:
    1 byte    op
    4 bytes   key_len
    key_len bytes  key (utf-8)
    [PUT/RESP_OK only] 8 bytes payload_len, then payload_len bytes payload
"""

from __future__ import annotations

import socket
import struct

OP_PUT = 1
OP_GET = 2
OP_RESP_OK = 3
OP_RESP_NOTFOUND = 4
OP_DELETE = 5

_HEADER = struct.Struct(">BI")  # op, key_len
_PAYLOAD_LEN = struct.Struct(">Q")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("socket closed while expecting more data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_get(sock: socket.socket, key: str):
    key_b = key.encode("utf-8")
    sock.sendall(_HEADER.pack(OP_GET, len(key_b)) + key_b)


def send_put(sock: socket.socket, key: str, payload: bytes, chunked_sender=None):
    key_b = key.encode("utf-8")
    header = _HEADER.pack(OP_PUT, len(key_b)) + key_b + _PAYLOAD_LEN.pack(len(payload))
    sock.sendall(header)
    if chunked_sender is not None:
        chunked_sender(sock, payload)
    else:
        sock.sendall(payload)


def send_delete(sock: socket.socket, key: str):
    key_b = key.encode("utf-8")
    sock.sendall(_HEADER.pack(OP_DELETE, len(key_b)) + key_b)


def recv_request(sock: socket.socket):
    """Server side: read one request. Returns (op, key, payload_or_None)."""
    header = recv_exact(sock, _HEADER.size)
    op, key_len = _HEADER.unpack(header)
    key = recv_exact(sock, key_len).decode("utf-8")
    if op == OP_PUT:
        (payload_len,) = _PAYLOAD_LEN.unpack(recv_exact(sock, _PAYLOAD_LEN.size))
        payload = recv_exact(sock, payload_len)
        return op, key, payload
    return op, key, None


def send_resp_ok(sock: socket.socket, payload: bytes | None, chunked_sender=None):
    if payload is None:
        sock.sendall(_HEADER.pack(OP_RESP_OK, 0) + _PAYLOAD_LEN.pack(0))
        return
    header = _HEADER.pack(OP_RESP_OK, 0) + _PAYLOAD_LEN.pack(len(payload))
    sock.sendall(header)
    if chunked_sender is not None:
        chunked_sender(sock, payload)
    else:
        sock.sendall(payload)


def send_resp_notfound(sock: socket.socket):
    sock.sendall(_HEADER.pack(OP_RESP_NOTFOUND, 0))


def recv_response(sock: socket.socket):
    """Client side: read one response. Returns (op, payload_or_None)."""
    header = recv_exact(sock, _HEADER.size)
    op, _zero = _HEADER.unpack(header)
    if op == OP_RESP_OK:
        (payload_len,) = _PAYLOAD_LEN.unpack(recv_exact(sock, _PAYLOAD_LEN.size))
        payload = recv_exact(sock, payload_len) if payload_len else b""
        return op, payload
    return op, None
