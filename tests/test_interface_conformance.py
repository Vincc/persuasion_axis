"""Catch API drift in assistant_axis early.

The repo's README advertises a `load_model`/`extract_response_activations` top-level API that
does not actually exist (confirmed by reading the source). This test pins down the real
building blocks our experiment layer (persuasion_axis.extract) depends on, so a future
assistant_axis refactor fails this test loudly instead of silently breaking our extraction.
"""

import inspect

import assistant_axis
from assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel, SpanMapper


def test_assistant_axis_exports_used_functions():
    assert hasattr(assistant_axis, "load_axis")
    assert hasattr(assistant_axis, "project")
    assert hasattr(assistant_axis, "get_config")
    assert hasattr(assistant_axis, "generate_response")


def test_load_axis_signature():
    params = inspect.signature(assistant_axis.load_axis).parameters
    assert "path" in params


def test_project_signature():
    params = inspect.signature(assistant_axis.project).parameters
    assert {"activations", "axis", "layer"}.issubset(params)


def test_get_config_signature():
    params = inspect.signature(assistant_axis.get_config).parameters
    assert "model_name" in params


def test_probing_model_constructible_signature():
    params = inspect.signature(ProbingModel.__init__).parameters
    assert {"model_name", "device"}.issubset(params)
    assert hasattr(ProbingModel, "get_layers")
    assert hasattr(ProbingModel, "hidden_size")


def test_activation_extractor_batch_conversations_signature():
    params = inspect.signature(ActivationExtractor.batch_conversations).parameters
    assert {"conversations", "layer", "max_length"}.issubset(params)


def test_conversation_encoder_build_batch_turn_spans_signature():
    params = inspect.signature(ConversationEncoder.build_batch_turn_spans).parameters
    assert "conversations" in params


def test_span_mapper_map_spans_signature():
    params = inspect.signature(SpanMapper.map_spans).parameters
    assert {"batch_activations", "batch_spans", "batch_metadata"}.issubset(params)
