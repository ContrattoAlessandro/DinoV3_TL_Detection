# DinoV3 Traffic Light Research

This codebase focuses on DinoV3 research experiments for image-level traffic
light relevance classification on the DriveU Traffic Light Dataset (DTLD).
The task predicts **RR** (relevant red/yellow), **RG** (relevant green), or
**NoR** (no relevant stop/go signal).

The model combines a frozen DINOv3 backbone with a token-level multiple-instance
learning head. All experiment code, configurations, tests, reports, and run
artifacts live in [`dinov3_global/`](dinov3_global/).

## Setup

Run commands from the repository root:

```sh
python -m pip install -r dinov3_global/requirements.txt
```

Accept the model license on Hugging Face and authenticate with `hf auth login`
for `facebook/dinov3-vits16plus-pretrain-lvd1689m`, or set
`backbone.local_ckpt` in the experiment configuration to a local snapshot.

## Data

The shared datasets remain at the repository root so existing configurations
and saved checkpoint paths continue to work:

- `datasets/DTLD/v2.0/`: native DTLD train/test annotations.
- `datasets/DTLD_jpg/`: full-resolution JPEG images in `train/` and `test/`.
- `datasets/DTLD_1280/`: cropped and resized images used by the experiments.

To convert raw DTLD TIFF images, install the additional preprocessing dependency
and use the conversion tools now located inside the experiment folder:

```sh
python -m pip install -r dinov3_global/requirements-preprocessing.txt
python dinov3_global/scripts/convert_tif_sequential.py --help
python dinov3_global/scripts/convert_tif.py --help
```

Create the 1280x720 image dump with:

```sh
python dinov3_global/scripts/make_1280.py --src datasets/DTLD_jpg --dst datasets/DTLD_1280
```

DTLD annotations are parsed within `dinov3_global/dinov3_global/dtld_native.py`.

## Experiments

```sh
python dinov3_global/tests/test_correctness.py
python dinov3_global/scripts/train.py --config dinov3_global/configs/v4.yaml --out dinov3_global/runs/exp5
python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp5
```

Use a new output directory for each experiment. Configurations `base.yaml`,
`v3.yaml`, and `v4.yaml` preserve earlier experiment settings. Select models and
settings on validation data; use `scripts/evaluate.py` on the official test
split only for a final report after selection.

For model details, inference, benchmarking, and city-held-out validation, see
the [experiment README](dinov3_global/README.md). Research context and findings
are recorded in [CONTEXT_v3.md](dinov3_global/CONTEXT_v3.md),
[AUDIT_v3.md](dinov3_global/AUDIT_v3.md),
[DECISIONS_v4.md](dinov3_global/DECISIONS_v4.md),
[RESULTS.md](dinov3_global/RESULTS.md), and
[RESULTS_v4.md](dinov3_global/RESULTS_v4.md).

The original codebase's [AGPL-3.0 license](LICENSE) is retained.
