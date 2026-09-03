"""Minimal model-agnostic layer discovery.

Empirically motivated, not speculative: `model.model.layers` (Llama-style
-- covers Llama, Mistral, Qwen2, Gemma, Phi and most modern HF causal LMs,
since their modeling code is largely adapted from Llama's) was the only
path this project's code assumed, and it fails outright on GPT-NeoX-style
architectures (`model.gpt_neox.layers`) -- confirmed directly with
EleutherAI/pythia-160m, not assumed. This covers the handful of known
naming conventions rather than building a general-purpose model-graph
inspector, which nothing measured here has shown a need for yet.
"""

from __future__ import annotations

import torch.nn as nn

# (path to the ModuleList, dotted-name prefix used in the checkpoint's
#  state dict for entries in it)
_KNOWN_PATHS = [
    ("model.layers", "model.layers"),          # Llama, Mistral, Qwen2, Gemma, Phi, ...
    ("gpt_neox.layers", "gpt_neox.layers"),     # GPT-NeoX, Pythia
    ("transformer.h", "transformer.h"),         # GPT-2, BLOOM, Falcon (older)
    ("transformer.blocks", "transformer.blocks"),  # MPT
]


def find_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    """Returns (layer_list, checkpoint_key_prefix) for the first known
    naming convention that matches. Raises with the actual module tree
    (not a generic message) if none do, so extending _KNOWN_PATHS for a
    new architecture is a one-line addition once actually needed."""
    for attr_path, key_prefix in _KNOWN_PATHS:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj, key_prefix

    raise ValueError(
        f"No known decoder-layer path matched for {type(model).__name__}. "
        f"Top-level submodules: {[n for n, _ in model.named_children()]}. "
        f"Add its layer-list path to _KNOWN_PATHS in {__file__}."
    )
