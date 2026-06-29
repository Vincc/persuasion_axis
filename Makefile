.PHONY: smoke test

smoke:
	uv run python -m persuasion_axis.run --smoke --output-dir results/smoke

test:
	uv run pytest tests/ -v
