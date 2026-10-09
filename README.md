# DinoGlobal-MIL

**Traffic-light relevance classification with frozen DINOv3 features and attribute-supervised multiple-instance learning.**

DinoGlobal-MIL predicts the relevant traffic-light signal from a complete RGB
driving image. A compact shared head estimates lamp presence, state, housing
direction, pictogram, and relevance, then pools this evidence into three logits.
Lamp annotations are needed for training; inference uses only the image.

This repository provides one model: **frozen DINOv3 ViT-B/16 with a 412,830-parameter
evidence head**. The released predictor is the fold-0, seed-0, epoch-7 EMA
checkpoint. Its head weights are included in [checkpoints/](checkpoints/README.md);
the licensed encoder and dataset images are acquired separately.

![Architecture](docs/assets/architecture.svg)

## Task

The output order is **RR, RG, NoR**:

| Class | Image-level annotation rule |
|:--|:--|
| RR | At least one relevant red, yellow, or red-yellow lamp. |
| RG | At least one relevant green lamp and no relevant RR lamp. |
| NoR | No relevant RR/RG lamp, including relevant off/unknown-only scenes. |

RR takes precedence when annotations conflict. Reported mAP is the mean
one-vs-rest image-classification AP; it is not object-detection AP.

## Installation

Use Python **3.11+** and install a PyTorch build for your device, then:

```sh
python -m pip install -e ".[dev,preprocessing]"
```

Accept the DINOv3 model license and authenticate with `hf auth login` before
downloading its weights. The exact encoder revision is pinned in
[backbone.py](src/dinov3_global/backbone.py). The
[tested environment](metadata/environment.json) records package and CUDA versions.

For offline use, place the licensed snapshot at `runs/backbones/dinov3_vitb16/`.
The included checkpoint uses that cache when present; otherwise its loader
resolves the same pinned Hugging Face model. New training configurations can
set `backbone.local_ckpt` to another licensed snapshot directory.

## Inference

```sh
python scripts/infer.py --ckpt checkpoints/dinoglobal_vitb_fold0.pt --images path/to/images --out runs/inference --overlays
```

The model letterboxes the full image to **720 × 1280**, fuses layer-6 and final
encoder features, and returns RGB-to-`[RR, RG, NoR]` logits plus auxiliary evidence
maps. Inference uses EMA head weights and requires no lamp bounding boxes.
See the [method](docs/method.md) and [model card](docs/model_card.md).

## Data and training

Obtain DTLD under its source terms. Place native annotations in
`datasets/DTLD/v2.0/`, then prepare full-resolution RGB JPEGs:

```sh
python scripts/preprocessing/convert_dtld.py --raw datasets/DTLD --labels datasets/DTLD/v2.0 --out datasets/DTLD_jpg
```

[Data documentation](docs/data.md) describes coordinates, labels, masks, and
the ATLAS/VZC transfer benchmarks. Local dataset images are excluded from Git.

[configs/default.yaml](configs/default.yaml) defines the canonical model,
augmentation, loss, and optimizer settings. Ordinary training uses a
session-disjoint validation partition within official DTLD train:

```sh
python scripts/train.py --out runs/train --seed 0
python scripts/train.py --out runs/train --resume runs/train/last.pt
```

To reproduce the released model's held-city membership, audit and train fold 0:

```sh
python scripts/cross_validate.py --fold-manifest metadata/dtld_city_folds.json --runtime-loader configs/runtime.json --out runs/city_cv --audit-only
python scripts/cross_validate.py --fold-manifest metadata/dtld_city_folds.json --runtime-loader configs/runtime.json --out runs/city_cv --fold-indices 0
```

The encoder stays frozen. Image-classification losses average both augmented
views; attribute losses supervise the first view. Training runs for 12 epochs
with AdamW and EMA. Highest validation mAP at T=1 selects `best.pt`, retaining the
earlier epoch on ties. `last.pt` also saves optimizer, scaler, and RNG state.
The [reproduction record](docs/reproduction/README.md) contains the released
run's configuration, history, and validation report.

## Results and evaluation

One EMA head, raw logits at **T=1**. Values below are percentages.

| Dataset | Images | mAP | Balanced accuracy | Macro F1 | RR recall | RG recall | NoR recall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| DTLD official test | 12,453 | 72.80 | 71.56 | 72.97 | 87.24 | 95.63 | 31.81 |
| ATLAS manual benchmark | 528 | 90.25 | 82.93 | 80.53 | 83.27 | 89.47 | 76.06 |
| VZC-TLD published test | 598 | 79.46 | 70.22 | 71.53 | 86.71 | 76.70 | 47.25 |

Training used 21,493 DTLD frames with 7,032 held-city validation frames from
Dortmund, Kassel, and Fulda. Selected validation mAP was **70.13%**. Only this
city fold and training seed are represented by the released checkpoint.
ATLAS/VZC influenced development and remain exploratory transfer benchmarks.
Low NoR recall is a material limitation.

```sh
python scripts/evaluate.py --ckpt checkpoints/dinoglobal_vitb_fold0.pt --runtime-loader configs/runtime.json --out runs/evaluations/dtld
python scripts/evaluate_ood.py --labels metadata/atlas_labels.json --images datasets/atlas_relevance_expanded --ckpt checkpoints/dinoglobal_vitb_fold0.pt --include-uncertain --out runs/evaluations/atlas
python scripts/download_vzc.py
python scripts/evaluate_vzc.py --ckpt checkpoints/dinoglobal_vitb_fold0.pt --out runs/evaluations/vzc
python scripts/summarize_results.py --dtld-test runs/evaluations/dtld --out runs/result_exports
```

[Full results](docs/results/README.md) include raw predictions, class metrics,
confusion matrices, slices, timing samples, and hashes. The
[evaluation protocol](docs/evaluation.md) defines membership audits, calibration,
and the limits of the claims.

## Repository structure

```text
checkpoints/            Released head-only checkpoint and its identity
configs/                Canonical model and runtime settings
src/dinov3_global/      Encoder, evidence head, data, losses, training, metrics
scripts/                Data preparation, training, inference, evaluation
tests/                  CPU regression tests; no encoder download required
metadata/               Checkpoint identity, city folds, labels, environment
docs/                   Method, data, model card, reproduction, results
datasets/               Local data; excluded from Git
runs/                   Local encoder cache and generated artifacts; excluded from Git
```

## Verification and license

```sh
python -m pytest
ruff check src scripts tests
ruff format --check src scripts tests
```

With licensed local data and CUDA, `python scripts/smoke.py` checks full-resolution
two-view training, EMA loading, and exact epoch-boundary resume.
[Publication verification](metadata/publication_verification.json) records that
cleanup preserved the released checkpoint and predictions.

Code is licensed under [AGPL-3.0](LICENSE). Backbone and dataset assets retain
their source terms; DINOv3 encoder weights are not bundled. DTLD parser helpers
retain their upstream attribution. Cite the accompanying paper, DINOv3, and the
source datasets when reporting results. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development conventions.
