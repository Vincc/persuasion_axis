from pathlib import Path

from persuasion_axis.run import run_experiment


def test_smoke_run_creates_expected_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"

    summary = run_experiment(
        model_id="synthetic",
        smoke=True,
        output_dir=str(output_dir),
        seed=7,
        max_new_tokens=16,
        layer=22,
    )

    assert summary["persona_count"] == 2
    assert summary["query_count"] == 2
    assert (output_dir / "projections.csv").exists()
    assert (output_dir / "persona_summary.csv").exists()
    assert (output_dir / "cross_cosine.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "run_config.yaml").exists()
    assert (output_dir / "figs" / "projection_bar.png").exists()
    assert (output_dir / "figs" / "cross_cosine.png").exists()
    assert (output_dir / "README.md").exists()
