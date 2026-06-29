from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from .data import build_conversation, load_personas, load_queries
from .extract import (
    SYNTHETIC_HIDDEN_SIZE,
    download_pretrained_axis,
    extract_response_activations,
    get_config,
    load_axis,
    load_model,
)
from .plots import save_cross_cosine_plot, save_layer_sweep_plot, save_projection_plot
from .project import (
    compute_axis_explained_fraction,
    compute_cross_cosine_matrix,
    compute_persona_vectors,
    compute_spearman_correlation,
    project,
)


def run_experiment(
    model_id: str = "google/gemma-2-27b-it",
    smoke: bool = False,
    output_dir: str | None = None,
    seed: int = 7,
    max_new_tokens: int = 256,
    layer: int | None = None,
    layers_to_sweep: List[int] | None = None,
    personas_path: str | None = None,
    queries_path: str | None = None,
    axis_path: str | None = None,
    device: str | None = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    output_dir_p = Path(output_dir or "results")
    output_dir_p.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir_p / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    personas = load_personas(personas_path)
    queries = load_queries(queries_path)
    if smoke:
        # Milestone-1 plumbing smoke test: 2 personas x 2 queries is enough to validate wiring.
        personas = personas[:2]
        queries = queries[:2]

    if layer is None:
        config = get_config(model_id)
        layer = int(config["target_layer"])

    if layers_to_sweep is None:
        layers_to_sweep = [max(0, layer - 8), max(0, layer - 4), layer, layer + 4, layer + 8]

    all_layers = sorted(set(layers_to_sweep) | {layer})

    if axis_path is None and smoke:
        # Synthetic placeholder axis, correctly sized so every swept layer (including `layer`)
        # is indexable -- axis[layer] must not go out of bounds.
        total_layers = max(SYNTHETIC_HIDDEN_SIZE, max(all_layers) + 1)
        axis = torch.randn(total_layers, SYNTHETIC_HIDDEN_SIZE, dtype=torch.float32)
    else:
        if axis_path is None:
            axis_path = download_pretrained_axis(model_id)
        axis = load_axis(axis_path)

    pm = load_model(model_id, smoke=smoke, device=device)

    # Build the full persona x query grid up front so extraction can batch across it.
    grid_index: List[Dict[str, Any]] = []
    conversations: List[List[Dict[str, str]]] = []
    for persona in personas:
        for query in queries:
            grid_index.append({"persona_id": persona["id"], "query_id": query["id"]})
            conversations.append(build_conversation(persona, query))

    activations = extract_response_activations(
        pm,
        conversations,
        layers=all_layers,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        seed=seed,
        batch_size=batch_size,
    )

    projection_rows: List[Dict[str, Any]] = []
    persona_target_acts: Dict[str, List[torch.Tensor]] = {p["id"]: [] for p in personas}
    persona_target_projections: Dict[str, List[float]] = {p["id"]: [] for p in personas}

    for entry, act_by_layer in zip(grid_index, activations):
        persona_id = entry["persona_id"]
        row = {"persona_id": persona_id, "query_id": entry["query_id"]}
        target_act = act_by_layer[layer]
        target_projection = project(target_act, axis, layer=layer)
        row["projection"] = float(target_projection)
        for sweep_layer in layers_to_sweep:
            row[f"projection_layer_{sweep_layer}"] = float(project(act_by_layer[sweep_layer], axis, layer=sweep_layer))
        projection_rows.append(row)

        persona_target_acts[persona_id].append(target_act)
        persona_target_projections[persona_id].append(float(target_projection))

    persona_ids = [p["id"] for p in personas]
    expected_rank_by_id = {p["id"]: p["expected_rank"] for p in personas}

    persona_means = [float(np.mean(persona_target_projections[pid])) for pid in persona_ids]
    persona_stds = [float(np.std(persona_target_projections[pid], ddof=0)) for pid in persona_ids]
    persona_cis = [
        1.96 * std / math.sqrt(len(persona_target_projections[pid])) if len(persona_target_projections[pid]) > 1 else 0.0
        for std, pid in zip(persona_stds, persona_ids)
    ]

    # Rank 1 = highest mean projection = most Assistant-like, matching expected_rank's convention.
    order = sorted(range(len(persona_ids)), key=lambda i: -persona_means[i])
    observed_rank_by_id = {persona_ids[idx]: rank + 1 for rank, idx in enumerate(order)}
    observed_ranks = [observed_rank_by_id[pid] for pid in persona_ids]
    expected_ranks = [expected_rank_by_id[pid] for pid in persona_ids]
    rho, p_value = compute_spearman_correlation(observed_ranks, expected_ranks)

    persona_mean_acts = {pid: torch.stack(persona_target_acts[pid]).mean(dim=0) for pid in persona_ids}
    persona_vectors = compute_persona_vectors(persona_mean_acts, assistant_default_id="assistant_default", layer=layer)
    axis_vector = axis[layer].float()
    axis_explained = {pid: compute_axis_explained_fraction(v, axis_vector) for pid, v in persona_vectors.items()}
    cosine_matrix = compute_cross_cosine_matrix(persona_vectors)

    projection_df = pd.DataFrame(projection_rows)
    projection_df.to_csv(output_dir_p / "projections.csv", index=False)

    summary_rows = []
    for pid, mean, std, ci in zip(persona_ids, persona_means, persona_stds, persona_cis):
        summary_rows.append(
            {
                "persona_id": pid,
                "mean_projection": mean,
                "std_projection": std,
                "ci95": ci,
                "expected_rank": expected_rank_by_id[pid],
                "observed_rank": observed_rank_by_id[pid],
                "axis_explained_fraction": axis_explained.get(pid, 0.0),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir_p / "persona_summary.csv", index=False)

    np.savetxt(output_dir_p / "cross_cosine.csv", cosine_matrix, delimiter=",", header=",".join(persona_ids), comments="")

    axis_filename = "synthetic" if smoke else "gemma-2-27b/assistant_axis.pt"
    assistant_default_gap = None
    if "assistant_default" in persona_ids and "propagandist" in persona_ids:
        assistant_default_gap = float(
            persona_means[persona_ids.index("assistant_default")] - persona_means[persona_ids.index("propagandist")]
        )

    summary = {
        "model_id": model_id,
        "smoke": smoke,
        "layer": layer,
        "layers_to_sweep": layers_to_sweep,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "persona_count": len(personas),
        "query_count": len(queries),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "assistant_default_vs_propagandist_gap": assistant_default_gap,
        "axis_filename": axis_filename,
    }
    with (output_dir_p / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with (output_dir_p / "run_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "model_id": model_id,
                "smoke": smoke,
                "layer": layer,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "layers_to_sweep": layers_to_sweep,
                "personas_path": personas_path,
                "queries_path": queries_path,
                "axis_filename": axis_filename,
            },
            fh,
        )

    save_projection_plot(figs_dir, persona_ids, persona_means, persona_stds, "assistant_default")
    save_cross_cosine_plot(figs_dir, persona_ids, cosine_matrix)
    sweep_projections = {
        pid: [
            float(np.mean([row[f"projection_layer_{sl}"] for row in projection_rows if row["persona_id"] == pid]))
            for sl in layers_to_sweep
        ]
        for pid in persona_ids
    }
    save_layer_sweep_plot(figs_dir, persona_ids, layers_to_sweep, sweep_projections)

    direction = "the predicted direction" if rho is not None and not math.isnan(rho) and rho > 0 else "an undetermined or unpredicted direction"
    readme_lines = [
        "# Persuasion Axis Experiment",
        "",
        f"Model: `{model_id}` | layer {layer} | {len(personas)} personas x {len(queries)} queries"
        + (" (SMOKE RUN -- synthetic model and axis, plumbing check only, not a real result)" if smoke else ""),
        "",
        f"Spearman rho between observed and expected persona rank: {rho:.3f} (p={p_value:.3g})."
        if rho is not None and not math.isnan(rho)
        else "Spearman rho could not be computed (scipy unavailable).",
        f"Persuasive personas separated from the Assistant baseline in {direction}.",
        "",
        "Assistant-default vs propagandist gap (mean projection): "
        + (f"{assistant_default_gap:.4f}" if assistant_default_gap is not None else "n/a"),
        "",
        "## Caveat",
        "The persona vectors here are role-rollout diffs against the Assistant baseline, built from "
        "response activations under a system-prompted persona. This is NOT necessarily the same object "
        "as a behavioral persuasion direction built from contrast pairs over actual persuasion attempts "
        "(the planned causal follow-up). Before assuming they coincide, check their cosine similarity.",
    ]
    (output_dir_p / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persuasion-axis measurement experiment")
    parser.add_argument("--model", dest="model_id", default="google/gemma-2-27b-it")
    parser.add_argument("--smoke", action="store_true", help="Run plumbing smoke test with a synthetic model and axis")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--layers-to-sweep", type=int, nargs="*", default=None)
    parser.add_argument("--personas-path", default=None)
    parser.add_argument("--queries-path", default=None)
    parser.add_argument("--axis-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_id = "synthetic" if (args.smoke and args.model_id == "google/gemma-2-27b-it") else args.model_id
    summary = run_experiment(
        model_id=model_id,
        smoke=args.smoke,
        output_dir=args.output_dir,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        layer=args.layer,
        layers_to_sweep=args.layers_to_sweep,
        personas_path=args.personas_path,
        queries_path=args.queries_path,
        axis_path=args.axis_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
