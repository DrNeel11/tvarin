"""Per-token timestamp capture for TTFT / decode-throughput measurement.

Relies on transformers' generate() loop calling `streamer.put(next_tokens.cpu())`
once per generated token — the `.cpu()` copy is a real device-to-host sync,
so the timestamp taken inside `put()` reflects when that token was actually
ready, not just when it was enqueued.
"""

from __future__ import annotations

import time

from transformers.generation.streamers import BaseStreamer


class TokenTimingStreamer(BaseStreamer):
    """Records one timestamp per *generated* token.

    generate() calls `streamer.put(input_ids.cpu())` once with the full
    prompt before the decode loop starts (see transformers
    generation/utils.py), then once per generated token thereafter. The
    first call is the prompt echo, not a generated token, and must be
    dropped or TTFT/decode-throughput measurements are contaminated by it.
    """

    def __init__(self):
        self.timestamps: list[float] = []
        self._seen_prompt_echo = False

    def put(self, value):
        if not self._seen_prompt_echo:
            self._seen_prompt_echo = True
            return
        self.timestamps.append(time.perf_counter())

    def end(self):
        pass
