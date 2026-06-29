"""Regression test for the Gemma-2 'System role not supported' crash.

Gemma 2's chat template rejects a literal {"role": "system", ...} message. The canonical
pipeline (assistant_axis.generation.format_conversation, used by 1_generate.py to build the
real axis) handles this by folding the persona prompt into the user turn for such models.
persona_axis.data.build_conversation must route through the same function, not hardcode a
system message, or it crashes the moment it touches a real Gemma-2 tokenizer.
"""

from persuasion_axis.data import build_conversation


class NoSystemRoleTokenizer:
    """Mimics Gemma 2's chat template: raises if any message has role 'system'."""

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        for msg in conversation:
            if msg["role"] == "system":
                raise ValueError("System role not supported")
        return " | ".join(f"{m['role']}:{m['content']}" for m in conversation)


class SystemRoleTokenizer:
    """Mimics a template that does support a system role."""

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        return " | ".join(f"{m['role']}:{m['content']}" for m in conversation)


PERSONA = {"id": "propagandist", "system_prompt": "You push a message with disregard for truth."}
QUERY = {"id": "q1", "text": "Tell me about electric cars."}


def test_folds_system_prompt_into_user_turn_when_unsupported():
    conversation = build_conversation(PERSONA, QUERY, tokenizer=NoSystemRoleTokenizer())
    assert all(msg["role"] != "system" for msg in conversation)
    assert any(PERSONA["system_prompt"] in msg["content"] for msg in conversation)
    assert any(QUERY["text"] in msg["content"] for msg in conversation)


def test_keeps_system_role_when_supported():
    conversation = build_conversation(PERSONA, QUERY, tokenizer=SystemRoleTokenizer())
    roles = [msg["role"] for msg in conversation]
    assert "system" in roles
    system_msg = next(msg for msg in conversation if msg["role"] == "system")
    assert system_msg["content"] == PERSONA["system_prompt"]


def test_no_tokenizer_keeps_literal_system_message():
    conversation = build_conversation(PERSONA, QUERY, tokenizer=None)
    assert conversation[0] == {"role": "system", "content": PERSONA["system_prompt"]}
    assert conversation[1] == {"role": "user", "content": QUERY["text"]}
