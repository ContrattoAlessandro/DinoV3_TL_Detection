# Retained frozen ViT-B results

The retained DinoGlobal-MIL predictor is one fold-0 **epoch-7 EMA** head,
with **412,830 trainable parameters** and a frozen DINOv3 ViT-B/16 encoder.
Checkpoint SHA-256: `edb2bb7bfd4a40cf470cd62f1dbd14e25b14d9c8feb107207be03922c528042e`.
The canonical local path is `runs/pretrained/dinoglobal_vitb_fold0.pt`.

Training used 21,493 DTLD frames and 7,032 held-city validation frames from
Dortmund, Kassel and Fulda. Highest validation EMA mAP selected epoch 7:
**70.13195734%**. Test and transfer data did not fit weights or select the epoch.
These are the original measured scores; publication cleanup did not rerun,
recalibrate, or modify them. Dataset identities and label policies are frozen.

## Primary metrics

All scores in the table are percentages.

| Dataset | Images | AP RR | AP RG | AP NoR | mAP | Balanced accuracy | Accuracy | Macro F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DTLD | 12,453 | 93.39 | 94.87 | 30.12 | 72.80 | 71.56 | 90.00 | 72.97 |
| ATLAS | 528 | 95.00 | 88.34 | 87.42 | 90.25 | 82.93 | 81.82 | 80.53 |
| VZC_TLD | 598 | 92.74 | 80.57 | 65.07 | 79.46 | 70.22 | 77.76 | 71.53 |

## Per-class metrics

AP, precision, recall, and F1 are percentages.

| Dataset | Class | Images | AP | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|
| DTLD | RR | 4,359 | 93.39 | 91.46 | 87.24 | 89.30 |
| DTLD | RG | 7,569 | 94.87 | 91.71 | 95.63 | 93.63 |
| DTLD | NoR | 525 | 30.12 | 41.44 | 31.81 | 35.99 |
| ATLAS | RR | 245 | 95.00 | 92.31 | 83.27 | 87.55 |
| ATLAS | RG | 95 | 88.34 | 65.89 | 89.47 | 75.89 |
| ATLAS | NoR | 188 | 87.42 | 80.34 | 76.06 | 78.14 |
| VZC_TLD | RR | 331 | 92.74 | 82.95 | 86.71 | 84.79 |
| VZC_TLD | RG | 176 | 80.57 | 71.43 | 76.70 | 73.97 |
| VZC_TLD | NoR | 91 | 65.07 | 68.25 | 47.25 | 55.84 |

## Other aggregate and probability metrics

Precision, recall, F1, and ECE are percentages. NLL and Brier use their natural scales; lower is better for ECE/NLL/Brier.

| Dataset | Macro precision | Macro recall | Weighted precision | Weighted recall | Weighted F1 | Mean RR/RG recall | ECE ↓ | NLL ↓ | Brier ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DTLD | 74.87 | 71.56 | 89.51 | 90.00 | 89.69 | 91.44 | 7.55 | 0.3578 | 0.1718 |
| ATLAS | 79.51 | 82.93 | 83.29 | 81.82 | 82.10 | 86.37 | 19.77 | 0.6731 | 0.3748 |
| VZC_TLD | 74.21 | 70.22 | 77.32 | 77.76 | 77.20 | 81.71 | 9.76 | 0.6424 | 0.3488 |

## Confusion matrices

Rows are true classes; columns are argmax predictions. Every test image is included.

| Dataset | Actual | Predicted RR | Predicted RG | Predicted NoR |
|:--|:--|--:|--:|--:|
| DTLD | RR | 3,803 | 462 | 94 |
| DTLD | RG | 189 | 7,238 | 142 |
| DTLD | NoR | 166 | 192 | 167 |
| ATLAS | RR | 204 | 14 | 27 |
| ATLAS | RG | 2 | 85 | 8 |
| ATLAS | NoR | 15 | 30 | 143 |
| VZC_TLD | RR | 287 | 30 | 14 |
| VZC_TLD | RG | 35 | 135 | 6 |
| VZC_TLD | NoR | 24 | 24 | 43 |

## Dataset slices

City and size slices are available for DTLD; camera slices for ATLAS; size slices for VZC. A dash denotes undefined three-class mAP. Full class counts/AP and present-class diagnostics are preserved in the JSON/CSV.

| Dataset | Group | Slice | Images | Present classes | mAP | Balanced accuracy |
|:--|:--|:--|--:|--:|--:|--:|
| DTLD | city | Berlin | 1,077 | 3 | 78.07 | 74.10 |
| DTLD | city | Bochum | 109 | 2 | — | 89.63 |
| DTLD | city | Bremen | 222 | 3 | 90.28 | 76.36 |
| DTLD | city | Dortmund | 2,027 | 3 | 72.07 | 70.92 |
| DTLD | city | Duesseldorf | 1,389 | 3 | 71.53 | 72.13 |
| DTLD | city | Essen | 781 | 3 | 63.73 | 62.08 |
| DTLD | city | Frankfurt | 1,677 | 3 | 79.22 | 75.85 |
| DTLD | city | Fulda | 108 | 3 | 78.51 | 58.29 |
| DTLD | city | Hannover | 2,408 | 3 | 73.53 | 70.19 |
| DTLD | city | Kassel | 757 | 3 | 70.45 | 70.44 |
| DTLD | city | Koeln | 1,898 | 3 | 66.03 | 67.00 |
| DTLD | size | small(<16px) | 1,580 | 3 | 67.92 | 70.53 |
| DTLD | size | mid(16-48px) | 7,807 | 3 | 75.86 | 73.43 |
| DTLD | size | large(>=48px) | 3,066 | 3 | 71.82 | 68.43 |
| ATLAS | camera | front_medium | 393 | 3 | 91.76 | 84.57 |
| ATLAS | camera | front_tele | 86 | 3 | 84.07 | 73.82 |
| ATLAS | camera | front_wide | 49 | 3 | 86.15 | 85.00 |
| VZC_TLD | size | small(<16px) | 108 | 3 | 69.49 | 63.33 |
| VZC_TLD | size | mid(16-48px) | 269 | 3 | 73.37 | 64.23 |
| VZC_TLD | size | large(>=48px) | 221 | 3 | 81.94 | 76.05 |

Bochum contains only RR/RG; its mean AP over those two present classes is 98.65%, preserved separately as `mAP_present_classes`.

## Fixed sensitivity and source-split diagnostics

These do not replace primary full-test metrics. Source train/all scores never select the checkpoint.

| Dataset | Diagnostic | Images | mAP | Balanced accuracy | Accuracy | Macro F1 |
|:--|:--|--:|--:|--:|--:|--:|
| DTLD | without relevant-off/unknown-only NoR | 12,240 | 70.28 | 70.79 | 90.96 | 70.95 |
| VZC_TLD | VZC test, without relevant-unknown-only NoR | 551 | 76.54 | 70.38 | 80.40 | 70.84 |
| VZC_TLD | VZC test, unique images | 598 | 79.46 | 70.22 | 77.76 | 71.53 |
| VZC_TLD | source train diagnostic | 2,390 | 71.00 | 63.72 | 75.31 | 65.10 |
| VZC_TLD | source all diagnostic | 2,988 | 72.79 | 65.17 | 75.80 | 66.62 |

## Matched inference latency

RTX 5070; 30 timed iterations after 5 warm-ups; the same 16 JPEGs used in the earlier benchmark. Includes decode, letterbox, metadata, transfers, model, softmax, and CPU result. Measurements ran after all dataset evaluations finished. Batch-4 latency is for the complete batch.

| Backbone | Batch-1 median (ms) | Batch-1 p95 (ms) | Batch-1 FPS | Batch-4 median (ms) | Batch-4 FPS |
|:--|--:|--:|--:|--:|--:|
| ViT-S+ | 42.03 | 43.25 | 23.79 | 173.85 | 23.01 |
| ViT-B | 62.03 | 63.57 | 16.12 | 247.68 | 16.15 |

## Reproducibility and claim scope

Only one city fold and seed are measured for this retained model. ATLAS uses
manual relevance labels, including uncertain annotations; VZC primary metrics
use its published test split. Both external benchmarks influenced development,
so transfer results are exploratory. Low NoR recall remains a limitation.
No full-training-split refit, repeated-seed uncertainty, or independent
confirmatory comparison is claimed.

Reports below are unchanged copies of the corresponding entries in the
[original ViT-B result bundle](../experiments/backbone_capacity/results/full_report.json).
[Provenance](provenance.json) records their hashes. The
[original ViT-S+ release](../experiments/vitsplus_reference/results/README.md)
and [rejected RGB study](../experiments/recall_head/results/README.md) are separate
research records.

[Metrics](metrics.csv) · [Class metrics](per_class_metrics.csv) ·
[Confusion matrices](confusion_matrices.csv) · [Timing samples](efficiency.json) ·
[DTLD report](dtld/report.json) · [ATLAS report](atlas/report.json) ·
[VZC report](vzc_tld/report.json)
