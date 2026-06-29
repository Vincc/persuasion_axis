"""Generation + activation extraction, adapted to the real assistant_axis internals API.

The README advertises `load_model`/`extract_response_activations` top-level functions that
do not actually exist in assistant_axis. The real building blocks are ProbingModel,
ConversationEncoder, ActivationExtractor, and SpanMapper (assistant_axis.internals), and the
batch extraction here intentionally mirrors pipeline/2_activations.py -- the literal script
used to build the released axis -- so our activations are apples-to-apples comparable:
batch_conversations + build_batch_turn_spans + SpanMapper.map_spans, then take the assistant
turn(s) (index 1, 3, ... in a single-system+user+assistant conversation).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from assistant_axis import generate_response
from assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel

# Hidden size used by the synthetic stand-in model in --smoke mode. Arbitrary but must stay
# consistent between the synthetic activations produced here and the synthetic axis in run.py.
SYNTHETIC_HIDDEN_SIZE = 16
SYNTHETIC_TOTAL_LAYERS = 32


class SyntheticTokenizer:
    name_or_path = "synthetic"
    pad_token_id = 0
    eos_token_id = 0


class SyntheticProbingModel:
    """Stand-in for ProbingModel so --smoke mode can exercise the pipeline without a GPU or weights."""

    def __init__(self, model_id: str = "synthetic") -> None:
        self.model_name = model_id
        self.hidden_size = SYNTHETIC_HIDDEN_SIZE
        self.tokenizer = SyntheticTokenizer()

    def get_layers(self):
        return list(range(SYNTHETIC_TOTAL_LAYERS))


def load_model(model_id: str, smoke: bool = False, device: Optional[str] = None) -> object:
    """Load a ProbingModel (or its synthetic stand-in for --smoke mode)."""
    if smoke or model_id == "synthetic":
        return SyntheticProbingModel(model_id)
    return ProbingModel(model_id, device=device)


def load_axis(path: str) -> torch.Tensor:
    from assistant_axis import load_axis as base_load_axis

    return base_load_axis(path)


# Known pre-computed axis filenames in the lu-christina/assistant-axis-vectors HF dataset repo.
PRETRAINED_AXIS_FILENAMES = {
    "google/gemma-2-27b-it": "gemma-2-27b/assistant_axis.pt",
}


def download_pretrained_axis(model_id: str) -> str:
    """Download the pre-computed axis for a known model and return its local path."""
    if model_id not in PRETRAINED_AXIS_FILENAMES:
        raise ValueError(
            f"No known pre-computed axis filename for {model_id!r}; pass --axis-path explicitly. "
            f"Known models: {sorted(PRETRAINED_AXIS_FILENAMES)}"
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id="lu-christina/assistant-axis-vectors",
        repo_type="dataset",
        filename=PRETRAINED_AXIS_FILENAMES[model_id],
    )


def get_config(model_id: str) -> Dict[str, object]:
    from assistant_axis import get_config as base_get_config

    if model_id == "synthetic":
        return {"target_layer": 22, "total_layers": SYNTHETIC_TOTAL_LAYERS, "short_name": "Synthetic"}

    return base_get_config(model_id)


def _synthetic_activation(conversation: List[Dict[str, str]], layers: List[int]) -> Dict[int, torch.Tensor]:
    """Deterministic pseudo-activation keyed by the persona's system prompt, for plumbing tests only."""
    system_msgs = [m["content"] for m in conversation if m["role"] == "system"]
    persona_text = system_msgs[0] if system_msgs else "assistant"
    basis = torch.arange(SYNTHETIC_HIDDEN_SIZE, dtype=torch.float32)
    persona_bytes = sum(ord(ch) for ch in persona_text)
    vector = torch.sin((basis + persona_bytes) / 7.0)
    return {layer: vector * (1.0 + 0.01 * layer) for layer in layers}


def extract_response_activations(
    pm: object,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 0,
    batch_size: int = 16,
    max_length: int = 2048,
) -> List[Dict[int, torch.Tensor]]:
    """Generate a response for each (system, user) conversation and extract mean response
    activations at `layers`, returning one {layer_idx: Tensor[hidden_size]} dict per conversation.
    """
    if isinstance(pm, SyntheticProbingModel):
        return [_synthetic_activation(conv, layers) for conv in conversations]

    torch.manual_seed(seed)
    do_sample = temperature > 0.0

    full_conversations = []
    for conv in conversations:
        response = generate_response(
            pm.model,
            pm.tokenizer,
            conv,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
        )
        full_conversations.append(conv + [{"role": "assistant", "content": response}])

    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)

    from assistant_axis.internals import SpanMapper

    span_mapper = SpanMapper(pm.tokenizer)

    results: List[Dict[int, torch.Tensor]] = []
    for start in range(0, len(full_conversations), batch_size):
        batch = full_conversations[start : start + batch_size]

        batch_activations, batch_metadata = extractor.batch_conversations(
            batch, layer=layers, max_length=max_length
        )
        _, batch_spans, _ = encoder.build_batch_turn_spans(batch)
        conv_activations_list = span_mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for conv_acts in conv_activations_list:
            if conv_acts.numel() == 0:
                results.append({layer: torch.zeros(pm.hidden_size) for layer in layers})
                continue
            # conv_acts: (num_turns, num_layers, hidden_size). Turn 0 = system+user, turn 1 = assistant.
            assistant_acts = conv_acts[1::2]
            mean_act = assistant_acts.mean(dim=0).float().cpu()
            results.append({layer: mean_act[i] for i, layer in enumerate(layers)})

    return results
