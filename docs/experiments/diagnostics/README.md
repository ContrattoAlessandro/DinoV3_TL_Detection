# A–D fixed-checkpoint diagnostic evaluations

A–D are single fold-0 EMA models fixed at the best validation-mAP epoch. All failed the original eligibility rule. These tests do not change the v5 retention decision.

DTLD: all 12,453 official test frames. VZC_TLD: all 598 published test images; source training images excluded. ATLAS: the confirmed 528-image manual benchmark, not official test ground truth. All metrics use raw logits at T=1.

The matched reference is v5_fold0. Other v5 folds, the four-head reference ensemble, and the prior axial pilot are shown separately. Cached references use exactly aligned image membership and labels.

## DTLD

| Model | mAP | Macro F1 | Mean RR/RG recall | NoR recall | Δ mAP vs v5 fold 0 (pp) |
|---|---:|---:|---:|---:|---:|
| v5_fold0 | 70.00% | 68.80% | 84.25% | 57.90% | +0.00 |
| A | 70.24% | 68.82% | 84.37% | 55.05% | +0.24 |
| B | 72.11% | 72.30% | 89.37% | 38.67% | +2.12 |
| C | 70.11% | 70.04% | 90.38% | 25.52% | +0.11 |
| D | 69.96% | 65.51% | 91.69% | 9.52% | -0.04 |
| v6_axial_fold0 | 71.58% | 72.85% | 88.55% | 52.00% | +1.58 |
| v5_fold1 | 66.28% | 63.09% | 77.87% | 57.14% | -3.72 |
| v5_fold2 | 69.69% | 67.94% | 83.50% | 56.19% | -0.31 |
| v5_fold3 | 68.27% | 67.98% | 83.43% | 55.62% | -1.73 |
| v5_ensemble4 | 70.46% | 69.58% | 84.98% | 57.71% | +0.46 |

## ATLAS

| Model | mAP | Macro F1 | Mean RR/RG recall | NoR recall | Δ mAP vs v5 fold 0 (pp) |
|---|---:|---:|---:|---:|---:|
| v5_fold0 | 81.22% | 73.73% | 62.42% | 94.68% | +0.00 |
| A | 79.00% | 73.09% | 64.66% | 88.30% | -2.22 |
| B | 81.47% | 77.16% | 80.17% | 74.47% | +0.26 |
| C | 86.84% | 71.97% | 88.12% | 52.13% | +5.63 |
| D | 86.93% | 64.06% | 87.59% | 35.11% | +5.71 |
| v6_axial_fold0 | 68.09% | 65.46% | 51.56% | 91.49% | -13.13 |
| v5_fold1 | 78.84% | 61.04% | 45.67% | 94.68% | -2.37 |
| v5_fold2 | 78.89% | 73.90% | 63.93% | 92.55% | -2.32 |
| v5_fold3 | 79.25% | 70.72% | 57.74% | 94.68% | -1.96 |
| v5_ensemble4 | 81.94% | 74.06% | 62.42% | 95.21% | +0.73 |

## VZC_TLD

| Model | mAP | Macro F1 | Mean RR/RG recall | NoR recall | Δ mAP vs v5 fold 0 (pp) |
|---|---:|---:|---:|---:|---:|
| v5_fold0 | 68.83% | 61.13% | 57.25% | 86.81% | +0.00 |
| A | 67.49% | 62.06% | 60.29% | 79.12% | -1.35 |
| B | 69.74% | 66.32% | 68.62% | 70.33% | +0.90 |
| C | 72.17% | 66.18% | 81.82% | 32.97% | +3.34 |
| D | 75.63% | 62.12% | 79.89% | 25.27% | +6.79 |
| v6_axial_fold0 | 58.29% | 46.51% | 38.69% | 84.62% | -10.54 |
| v5_fold1 | 69.28% | 48.41% | 39.03% | 94.51% | +0.45 |
| v5_fold2 | 63.39% | 48.93% | 39.79% | 94.51% | -5.44 |
| v5_fold3 | 66.78% | 55.77% | 49.73% | 90.11% | -2.06 |
| v5_ensemble4 | 69.98% | 56.25% | 48.77% | 94.51% | +1.14 |

## Checkpoints

- A: epoch 11, SHA-256 41de1b1d150b8f564c138bccdf205cb0b8ebfefddfd8284337477f5f41318423.
- B: epoch 9, SHA-256 ffb74ecc1ff6bb32ff440430364ee3871ea9bc6f19dc180f5211314932aee729.
- C: epoch 5, SHA-256 7be4e19ae4ac03a1422983a15aa561f4d2f91a324be8e0e055dad56536c985da.
- D: epoch 3, SHA-256 ae850983488e5e5e27b0e255f0f215f2d7e138693c59d3ec9406cab289c017e8.

Per-image raw predictions, per-class precision/recall/F1, confusion matrices, raw calibration metrics and available slices are retained alongside the frozen plan. External results are exploratory, as these datasets have influenced architecture development.
