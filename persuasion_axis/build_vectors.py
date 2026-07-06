"""Extract per-persona role vectors averaged across all (system_prompt x question) rollouts.

For each persona in configs/personas/*.json (5 system prompts each), runs the full
extraction_questions.jsonl (240 questions) to produce a tensor of shape
[total_layers, hidden_dim] saved as {output_dir}/{persona_name}.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .data import build_conversation
from .extract import extract_response_activations, get_config, load_model

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSONAS_DIR = PACKAGE_ROOT / "configs" / "personas"
DEFAULT_QUESTIONS_PATH = PACKAGE_ROOT / "data" / "extraction_questions.jsonl"


def _load_persona_files(personas_dir: Path) -> List[Tuple[str, List[str]]]:
    result = []
    for path in sorted(personas_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        prompts = [entry["pos"] for entry in data["instruction"]]
        result.append((path.stem, prompts))
    if not result:
        raise FileNotFoundError(f"No persona JSON files found in {personas_dir}")
    return result


def _load_questions(questions_path: Path) -> List[str]:
    questions = []
    with questions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line)["question"])
    return questions


def _average_activations(
    activation_dicts: List[Dict[int, torch.Tensor]],
    layers: List[int],
) -> torch.Tensor:
    """Stack and mean per-conversation activations into [total_layers, hidden_dim]."""
    layer_means: List[torch.Tensor] = []
    for layer_idx in layers:
        vecs = torch.stack([d[layer_idx] for d in activation_dicts])  # [n_convs, hidden_dim]
        layer_means.append(vecs.mean(dim=0))
    return torch.stack(layer_means)  # [total_layers, hidden_dim]


def build_persona_vectors(
    model_id: str = "google/gemma-2-27b-it",
    smoke: bool = False,
    output_dir: str = "results/persona_vectors",
    batch_size: int = 16,
    max_new_tokens: int = 256,
    device: Optional[str] = None,
    personas_dir: str = str(DEFAULT_PERSONAS_DIR),
    questions_path: str = str(DEFAULT_QUESTIONS_PATH),
) -> Dict[str, torch.Tensor]:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    config = get_config(model_id)
    total_layers = int(config["total_layers"])
    layers = list(range(total_layers))

    pm = load_model(model_id, smoke=smoke, device=device)
    conversation_tokenizer = None if smoke else pm.tokenizer

    personas = _load_persona_files(Path(personas_dir))
    questions = _load_questions(Path(questions_path))

    if smoke:
        personas = [(name, prompts[:2]) for name, prompts in personas]
        questions = questions[:2]

    results: Dict[str, torch.Tensor] = {}
    persona_bar = tqdm(personas, desc="personas", unit="persona")
    for persona_name, system_prompts in persona_bar:
        persona_bar.set_postfix({"current": persona_name})
        conversations = [
            build_conversation({"system_prompt": p}, {"text": q}, conversation_tokenizer)
            for p in system_prompts
            for q in questions
        ]
        activation_dicts: List[Dict[int, torch.Tensor]] = []
        conv_bar = tqdm(total=len(conversations), desc=f"  {persona_name}", unit="conv", leave=False)
        for i in range(0, len(conversations), batch_size):
            chunk = conversations[i : i + batch_size]
            activation_dicts.extend(
                extract_response_activations(
                    pm,
                    chunk,
                    layers=layers,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    seed=0,
                    batch_size=batch_size,
                )
            )
            conv_bar.update(len(chunk))
        conv_bar.close()

        vector = _average_activations(activation_dicts, layers)
        out_path = output_dir_p / f"{persona_name}.pt"
        torch.save(vector, out_path)
        results[persona_name] = vector
        tqdm.write(f"  {persona_name}: saved {out_path}  shape={tuple(vector.shape)}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-persona role vectors averaged across all system-prompt x question rollouts"
    )
    parser.add_argument("--model", dest="model_id", default="google/gemma-2-27b-it")
    parser.add_argument("--smoke", action="store_true", help="Run with synthetic model for plumbing test")
    parser.add_argument("--output-dir", default="results/persona_vectors")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--personas-dir", default=str(DEFAULT_PERSONAS_DIR))
    parser.add_argument("--questions", dest="questions_path", default=str(DEFAULT_QUESTIONS_PATH))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_id = "synthetic" if args.smoke and args.model_id == "google/gemma-2-27b-it" else args.model_id
    build_persona_vectors(
        model_id=model_id,
        smoke=args.smoke,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        personas_dir=args.personas_dir,
        questions_path=args.questions_path,
    )


if __name__ == "__main__":
    main()
