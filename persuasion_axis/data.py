import json
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSONAS_PATH = PACKAGE_ROOT / "configs" / "personas.json"
DEFAULT_QUERIES_PATH = PACKAGE_ROOT / "configs" / "queries.json"


def load_personas(path: str | None = None) -> List[Dict[str, Any]]:
    personas_path = Path(path) if path is not None else DEFAULT_PERSONAS_PATH
    with personas_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_queries(path: str | None = None) -> List[Dict[str, Any]]:
    queries_path = Path(path) if path is not None else DEFAULT_QUERIES_PATH
    with queries_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_conversation(persona: Dict[str, Any], query: Dict[str, Any], tokenizer: Any = None) -> List[Dict[str, str]]:
    """Build a (system, user) conversation, folding the persona prompt into the user turn
    for models whose chat template doesn't support a system role (e.g. Gemma 2).

    Matches assistant_axis.generation.format_conversation, the same function the canonical
    pipeline (1_generate.py / VLLMGenerator.generate_for_role) uses to build the real axis --
    so persona prompts are encoded the same way here as they were when the axis was computed.
    If no tokenizer is given (e.g. the synthetic smoke-test model), a system message is used
    directly since no real chat template is ever applied to it.
    """
    if tokenizer is None:
        return [
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user", "content": query["text"]},
        ]

    from assistant_axis import format_conversation

    return format_conversation(persona["system_prompt"], query["text"], tokenizer)
