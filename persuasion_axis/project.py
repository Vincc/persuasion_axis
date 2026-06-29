import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from assistant_axis import project as base_project


def project(activations: torch.Tensor, axis: torch.Tensor, layer: int, normalize: bool = True) -> float:
    return base_project(activations, axis, layer=layer, normalize=normalize)


def compute_persona_vectors(persona_activations: Dict[str, torch.Tensor], assistant_default_id: str, layer: int) -> Dict[str, torch.Tensor]:
    assistant_default = persona_activations[assistant_default_id]
    vectors: Dict[str, torch.Tensor] = {}
    for persona_id, activation in persona_activations.items():
        vectors[persona_id] = activation - assistant_default
    return vectors


def compute_axis_explained_fraction(persona_vector: torch.Tensor, axis_vector: torch.Tensor) -> float:
    persona_vector = persona_vector.float()
    axis_vector = axis_vector.float()
    denom = persona_vector.norm().item()
    if denom < 1e-12:
        return 0.0
    proj = float(torch.dot(persona_vector, axis_vector))
    return (proj * proj) / (denom * denom)


def compute_cross_cosine_matrix(vectors: Dict[str, torch.Tensor]) -> np.ndarray:
    persona_ids = list(vectors.keys())
    matrix = np.zeros((len(persona_ids), len(persona_ids)), dtype=float)
    norms = {k: float(v.norm().item()) for k, v in vectors.items()}
    for i, left in enumerate(persona_ids):
        for j, right in enumerate(persona_ids):
            if i == j:
                matrix[i, j] = 1.0
                continue
            if norms[left] < 1e-12 or norms[right] < 1e-12:
                matrix[i, j] = 0.0
                continue
            dot = float(torch.dot(vectors[left].float(), vectors[right].float()))
            matrix[i, j] = dot / (norms[left] * norms[right])
    return matrix


def compute_spearman_correlation(observed_ranks: Sequence[int], expected_ranks: Sequence[int]) -> Tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except Exception:
        return float("nan"), float("nan")

    corr = spearmanr(observed_ranks, expected_ranks)
    return float(corr.statistic), float(corr.pvalue)
