from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_projection_plot(output_dir: Path, persona_ids: Sequence[str], means: Sequence[float], stds: Sequence[float], baseline_id: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(persona_ids))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(persona_ids, rotation=30, ha="right")
    ax.set_ylabel("Projection")
    ax.set_title("Persona projections onto the Assistant Axis")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = output_dir / "projection_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_cross_cosine_plot(output_dir: Path, persona_ids: Sequence[str], matrix: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(persona_ids)))
    ax.set_xticklabels(persona_ids, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(persona_ids)))
    ax.set_yticklabels(persona_ids)
    ax.set_title("Cross-cosine similarity of persona vectors")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    path = output_dir / "cross_cosine.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_layer_sweep_plot(output_dir: Path, persona_ids: Sequence[str], layer_values: Sequence[int], projections: Dict[str, List[float]]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for persona_id in persona_ids:
        ax.plot(layer_values, projections[persona_id], marker="o", label=persona_id)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Projection")
    ax.set_title("Layer sweep")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = output_dir / "layer_sweep.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
