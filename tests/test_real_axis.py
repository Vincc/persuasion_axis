"""Milestone 3: confirm the pre-computed Gemma-2-27B axis downloads, loads, and is layer-indexable.

No GPU needed -- this only exercises huggingface_hub + assistant_axis.load_axis/project.
Requires network access; skipped if the download fails for any reason (offline CI, etc.).
"""

import pytest
import torch


@pytest.fixture(scope="module")
def real_axis_path():
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id="lu-christina/assistant-axis-vectors",
            repo_type="dataset",
            filename="gemma-2-27b/assistant_axis.pt",
        )
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not download pre-computed axis: {exc}")


def test_axis_loads_and_is_layer_indexable(real_axis_path):
    from assistant_axis import get_config, load_axis

    axis = load_axis(real_axis_path)
    config = get_config("google/gemma-2-27b-it")

    assert isinstance(axis, torch.Tensor)
    assert axis.ndim == 2
    assert axis.shape[0] == config["total_layers"]

    target_layer = config["target_layer"]
    assert 0 <= target_layer < axis.shape[0]
    layer_vector = axis[target_layer]
    assert layer_vector.shape == (axis.shape[1],)
    assert torch.isfinite(layer_vector.float()).all()


def test_project_returns_finite_scalar_for_dummy_activation(real_axis_path):
    from assistant_axis import get_config, load_axis, project

    axis = load_axis(real_axis_path)
    config = get_config("google/gemma-2-27b-it")
    hidden_size = axis.shape[1]

    dummy_activation = torch.randn(hidden_size)
    projection = project(dummy_activation, axis, layer=config["target_layer"])

    assert isinstance(projection, float)
    assert projection == projection  # not NaN
    assert abs(projection) < float("inf")
