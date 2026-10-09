# Model card

## Model and intended use

DinoGlobal-MIL classifies the relevant traffic-light state from a complete RGB
driving frame. It is a research model for image-level traffic-light relevance
classification and analysis of attribute evidence. It has not been validated
as a complete driving-control system.

The encoder is frozen DINOv3 ViT-B/16, pinned to revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. A shared 412,830-parameter MIL head
predicts lamp presence, state, direction, pictogram, and relevance. Inputs use
720 × 1280 full-frame letterboxing. Outputs are logits ordered RR/RG/NoR plus
auxiliary evidence maps; those maps are not calibrated bounding-box detections.

## Training and evaluation

The released head trained on 21,493 official DTLD training frames. Validation
held out Dortmund, Kassel, and Fulda (7,032 frames). Seed 0 and highest raw
validation mAP selected epoch-7 EMA. Attribute supervision applies to the first
augmented view; global CE supervises both views. No test/transfer calibration,
threshold fitting, or epoch fitting is used in the primary metrics.

| Benchmark | mAP | Balanced accuracy | NoR recall |
|:--|--:|--:|--:|
| DTLD official test | 72.80% | 71.56% | 31.81% |
| ATLAS manual relevance labels | 90.25% | 82.93% | 76.06% |
| VZC-TLD published test | 79.46% | 70.22% | 47.25% |

The [full results](results/README.md) include class counts, precision/recall,
confusion matrices, calibration, and data sensitivities.

## Limitations

- NoR rejection is weak on DTLD and VZC: irrelevant or off/unknown lamps can
  produce false signal evidence.
- Relevant red/yellow/red-yellow takes priority over green when labels conflict.
  This annotation policy is not a substitute for route or lane reasoning.
- Predictions depend on dataset-defined relevance and visual context; camera,
  lighting, geography, and annotation shifts may change performance.
- ATLAS uses manually assigned relevance labels, including uncertain labels.
  ATLAS/VZC informed development, so transfer scores remain exploratory.
- Only one city fold and seed are represented. There is no repeated-seed
  confirmation or full-training-split refit.

## Distribution and reproducibility

The included file contains only trained head weights and metadata. Licensed
encoder weights and dataset images are not bundled. Code uses the root
[AGPL-3.0 license](../LICENSE); external assets retain their source terms.

[Checkpoint identity](../metadata/checkpoints.json) ·
[Method](method.md) · [Data](data.md) ·
[Evaluation](evaluation.md) · [Reproduction](reproduction/README.md)
