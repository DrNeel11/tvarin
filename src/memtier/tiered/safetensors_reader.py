"""Direct safetensors checkpoint reads, bypassing whatever a live model's
in-memory representation currently is.

Needed because accelerate's own multi-device `device_map="auto"` dispatch
represents "offloaded" (CPU-assigned, in a mixed CPU+GPU dispatch) layers as
meta-tensor placeholders backed by its own internal hooks -- extracting real
data via `layer.named_parameters()` fails on them ("Cannot copy out of meta
tensor"). Reading the same tensors straight from the checkpoint files on
disk sidesteps that entirely, independent of accelerate's live state.
"""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from memtier.config import HF_CACHE_DIR


def resolve_snapshot_dir(repo_id: str) -> Path:
    # Not huggingface_hub.snapshot_download(local_files_only=True): that
    # requires the *entire* repo (README, LICENSE, .gitattributes) to be
    # present, which our model loads never fetched since they're irrelevant
    # to running the model. We already know the cache layout and already
    # have everything we actually need (config, index, shards).
    org_name = repo_id.replace("/", "--")
    cache_root = HF_CACHE_DIR / "hub" / f"models--{org_name}" / "snapshots"
    candidates = [d for d in cache_root.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no cached snapshot found for {repo_id!r} under {cache_root}")
    if len(candidates) > 1:
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


class SafetensorsMultiReader:
    def __init__(self, repo_id: str):
        self.snapshot_dir = resolve_snapshot_dir(repo_id)
        index_path = self.snapshot_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def _handle(self, shard_filename: str):
        if shard_filename not in self._handles:
            self._handles[shard_filename] = safe_open(
                str(self.snapshot_dir / shard_filename), framework="pt", device="cpu"
            )
        return self._handles[shard_filename]

    def get_tensor(self, full_name: str):
        shard = self.weight_map[full_name]
        return self._handle(shard).get_tensor(full_name)

    def read_layer_state(self, layer_idx: int, prefix: str = "model.layers") -> dict:
        want_prefix = f"{prefix}.{layer_idx}."
        names = [k for k in self.weight_map if k.startswith(want_prefix)]
        return {name[len(want_prefix):]: self.get_tensor(name) for name in names}
