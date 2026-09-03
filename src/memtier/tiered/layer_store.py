"""Serialize/deserialize a decoder layer's parameters for the RAM/remote tiers.

Plain torch.save/load (not safetensors) here: this is our own in-memory
round-trip for offloaded layers, a different concern from the safetensors
mmap loading used to read the base model checkpoint off disk (see
docs/ARCHITECTURE.md's interception-points section) -- there's no file to
mmap for a blob we generated ourselves and are shipping over a socket.
"""

from __future__ import annotations

import io

import torch


def serialize_layer(layer: torch.nn.Module) -> bytes:
    state = {name: p.detach().to("cpu") for name, p in layer.named_parameters()}
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def deserialize_layer(data: bytes) -> dict[str, torch.Tensor]:
    buf = io.BytesIO(data)
    return torch.load(buf, map_location="cpu", weights_only=True)
