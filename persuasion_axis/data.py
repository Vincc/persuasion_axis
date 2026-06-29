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


def build_conversation(persona: Dict[str, Any], query: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": persona["system_prompt"]},
        {"role": "user", "content": query["text"]},
    ]
