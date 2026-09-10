"""Real, end-to-end integration test of the Ramer proxy server -- against
the live Ollama on this machine, no mocks, per this project's own testing
discipline (see app/tests/test_scheduler.py for the pure-logic tests that
DO use a fake chat_impl; this file is deliberately the complementary real
half). Uses qwen2.5:3b-instruct, the smallest installed model, to keep the
suite fast.
"""
import json
import urllib.error
import urllib.request

from app.core.proxy_server import RamerProxyServer

MODEL = "qwen2.5:3b-instruct"


def _start_server() -> RamerProxyServer:
    server = RamerProxyServer(port=0, on_log=lambda msg: None)
    server.start_background()
    return server


def test_tags_passthrough_matches_real_ollama():
    server = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data["models"]]
        assert MODEL in names
    finally:
        server.stop()


def test_chat_streaming_relay_returns_real_content():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": True,
            "options": {"num_ctx": 2048, "num_predict": 10},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/api/chat", data=body, headers={"Content-Type": "application/json"},
        )
        lines = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            for raw in resp:
                raw = raw.strip()
                if raw:
                    lines.append(json.loads(raw))
        assert len(lines) > 1, "expected multiple streamed NDJSON chunks, not one buffered blob"
        assert lines[-1].get("done") is True
        full_content = "".join(c.get("message", {}).get("content", "") for c in lines)
        assert len(full_content.strip()) > 0

        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{server.port}/ramer/status", timeout=10).read())
        assert status["requests_served"] == 1
        assert status["resident_model"] == MODEL
    finally:
        server.stop()


def test_chat_non_streaming_returns_single_json_object():
    server = _start_server()
    try:
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            "stream": False,
            "options": {"num_ctx": 2048, "num_predict": 10},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/api/chat", data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            assert resp.status == 200
            result = json.loads(resp.read())
        assert result["done"] is True
        assert len(result["message"]["content"].strip()) > 0
    finally:
        server.stop()


def test_unknown_path_returns_404():
    server = _start_server()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{server.port}/not/a/real/path", timeout=10)
            assert False, "expected an HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()
