# Axial spatial relevance pilot

DTLD fold 0, seed 0, twelve epochs. Each model uses its best validation-selected EMA checkpoint.
All metrics use raw logits at T=1. Deltas below are v6 minus v5 in percentage points.

| Dataset | v5 mAP | v6 mAP | Δ mAP | v5 AP NoR | v6 AP NoR | Δ AP NoR | Δ balanced accuracy | Δ macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DTLD fold-0 validation | 68.72 | 70.67 | +1.95 | 20.50 | 24.62 | +4.12 | +0.94 | +2.75 |
| DTLD official test | 70.00 | 71.58 | +1.58 | 24.36 | 28.16 | +3.80 | +0.90 | +4.05 |
| ATLAS transfer | 81.22 | 68.09 | -13.13 | 77.05 | 59.29 | -17.76 | -8.30 | -8.27 |
| VZC-TLD published test | 68.83 | 58.29 | -10.54 | 43.87 | 32.61 | -11.27 | -13.10 | -14.61 |

## Full primary metrics

AP, accuracy, F1, and ECE values are percentages; NLL and Brier are unitless. Lower ECE, NLL, and Brier are better.

| Dataset | Model | AP RR | AP RG | AP NoR | mAP |
|---|---|---:|---:|---:|---:|
| DTLD fold-0 validation | v5_fold0 | 89.92 | 95.73 | 20.50 | 68.72 |
| DTLD fold-0 validation | v6_axial_fold0 | 91.54 | 95.84 | 24.62 | 70.67 |
| DTLD official test | v5_fold0 | 90.87 | 94.76 | 24.36 | 70.00 |
| DTLD official test | v6_axial_fold0 | 91.63 | 94.95 | 28.16 | 71.58 |
| ATLAS transfer | v5_fold0 | 89.58 | 77.01 | 77.05 | 81.22 |
| ATLAS transfer | v6_axial_fold0 | 82.11 | 62.86 | 59.29 | 68.09 |
| VZC-TLD published test | v5_fold0 | 89.49 | 73.14 | 43.87 | 68.83 |
| VZC-TLD published test | v6_axial_fold0 | 79.95 | 62.32 | 32.61 | 58.29 |

| Dataset | Model | Bal. acc. | Accuracy | Macro F1 | Weighted F1 | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DTLD fold-0 validation | v5_fold0 | 72.27 | 83.15 | 66.89 | 85.49 | 5.42 | 0.4980 | 0.2638 |
| DTLD fold-0 validation | v6_axial_fold0 | 73.21 | 86.45 | 69.64 | 87.70 | 3.65 | 0.4413 | 0.2222 |
| DTLD official test | v5_fold0 | 75.47 | 83.75 | 68.80 | 85.82 | 6.88 | 0.4880 | 0.2581 |
| DTLD official test | v6_axial_fold0 | 76.37 | 87.83 | 72.85 | 88.64 | 4.50 | 0.4088 | 0.2042 |
| ATLAS transfer | v5_fold0 | 73.17 | 75.19 | 73.73 | 75.00 | 5.80 | 0.6504 | 0.3645 |
| ATLAS transfer | v6_axial_fold0 | 64.87 | 66.67 | 65.46 | 66.39 | 17.98 | 1.0070 | 0.5379 |
| VZC-TLD published test | v5_fold0 | 67.10 | 64.21 | 61.13 | 67.13 | 7.27 | 0.8562 | 0.4998 |
| VZC-TLD published test | v6_axial_fold0 | 54.00 | 45.99 | 46.51 | 50.52 | 31.86 | 1.4317 | 0.8133 |

## NoR operating performance

| Dataset | v5 precision | v6 precision | v5 recall | v6 recall | v5 F1 | v6 F1 |
|---|---:|---:|---:|---:|---:|---:|
| DTLD fold-0 validation | 17.68 | 22.64 | 48.18 | 44.22 | 25.86 | 29.94 |
| DTLD official test | 21.30 | 29.90 | 57.90 | 52.00 | 31.15 | 37.97 |
| ATLAS transfer | 63.80 | 53.75 | 94.68 | 91.49 | 76.23 | 67.72 |
| VZC-TLD published test | 31.85 | 21.10 | 86.81 | 84.62 | 46.61 | 33.77 |

## Compute

| Model | Head parameters | Batch-4 latency (ms) | Throughput (images/s) | Peak inference memory (MiB) |
|---|---:|---:|---:|---:|
| v5_fold0 | 489,475 | 65.68 | 60.90 | 463.4 |
| v6_axial_fold0 | 1,380,739 | 67.78 | 59.02 | 466.5 |

Candidate training duration: 3.94 hours (wall time from pipeline training start to checkpoint freeze, including final validation). Peak allocated training GPU memory: 1262.7 MiB.
Historical v5 training time and memory were not recorded; they are unavailable.

## Interpretation

- DTLD fold-0 validation: mAP +1.95 pp; AP NoR +4.12 pp.
- DTLD official test: mAP +1.58 pp; AP NoR +3.80 pp.
- ATLAS transfer: mAP -13.13 pp; AP NoR -17.76 pp.
- VZC-TLD published test: mAP -10.54 pp; AP NoR -11.27 pp.

This is one fold and one seed. Validation selected the checkpoints; ATLAS and VZC are exploratory transfer evaluations. The comparison does not establish attention to lane markers or a causal lane-association mechanism.

[Complete metrics and provenance](comparison.json) · [Metrics CSV](metrics.csv) · [Per-class metrics](per_class_metrics.csv) · [Confusion matrices](confusion_matrices.csv) · [Efficiency](efficiency.csv)

The live history was truncated during training. Missing epoch records were recovered with explicit provenance; epoch timings remain unavailable for [6]. Duration therefore uses the recorded pipeline start and checkpoint-freeze timestamps. Model weights and final evaluation predictions were unaffected.
