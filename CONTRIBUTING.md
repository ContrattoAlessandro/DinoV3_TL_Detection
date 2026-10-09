# Contributing

The released architecture is the shared evidence MIL head with frozen
DINOv3 ViT-B/16. `configs/default.yaml` is the canonical training protocol.

Keep image transforms consistent across training, inference, and target
generation. Class/state/attribute order and state-dict keys form part of the
checkpoint contract. A format change requires explicit documentation and a
loading migration.

Install development dependencies and run the checks before submitting changes:

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check src scripts tests
ruff format --check src scripts tests
```

CPU tests use synthetic data and mock encoders. With licensed local data and
CUDA, `python scripts/smoke.py` checks full-resolution training and exact resume.

Use a fresh output directory for each run. Record configuration, source identity,
membership, checkpoint selection, and weight hashes. Test and transfer scores
must not fit thresholds, calibration, or epochs. Keep new measurements separate
from the published records in `docs/results/`.

Dataset pixels, encoder weights, access tokens, environment folders, and generated
runs stay out of Git. The only bundled weight file is the small head checkpoint
declared in `metadata/checkpoints.json`. Preserve upstream parser attribution
and the repository license.
