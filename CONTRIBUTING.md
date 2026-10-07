# Contributing

The supported architecture is the shared evidence MIL head in
`src/dinov3_global/head.py`, with `configs/default.yaml` as the canonical protocol.
Keep changes to image transforms consistent across training, inference, and
attribute target generation. Changing class/state/attribute order is a checkpoint
format change and must be documented.

Install development dependencies with `python -m pip install -e ".[dev]"`.
Before submitting changes, run:

```sh
python -m pytest
ruff check src scripts tests
ruff format --check src scripts tests
```

CPU tests use synthetic data and mock encoder weights. For a licensed local
CUDA setup, `python scripts/smoke.py` verifies two-view training and exact resume.

Use a fresh output directory for a new experiment. Record the source revision,
effective configuration, split membership, checkpoint selection, and weight
hashes. Report single heads separately from ensembles. Preserve failed selection
decisions and uncertainty; never overwrite historical measurements to match a
new development choice.

Research records in `docs/experiments/` are frozen. Add a clearly identified new
record for future experiments rather than changing old results. Dataset pixels,
checkpoints, access tokens, and generated run directories must stay out of Git.
