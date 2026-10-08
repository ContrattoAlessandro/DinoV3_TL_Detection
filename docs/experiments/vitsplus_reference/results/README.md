# Recorded Experiment C results

The default predictor is **one fold-0 EMA head, epoch 5**, with 265,374 trainable
parameters and a frozen DINOv3 ViT-S+/16 encoder. Its checkpoint SHA-256 is
`7be4e19ae4ac03a1422983a15aa561f4d2f91a324be8e0e055dad56536c985da`.
Training uses 21,493 DTLD frames; 7,032 frames in Dortmund/Kassel/Fulda select
the epoch by validation mAP (69.37446754%).

These are unchanged post-study diagnostic measurements. C failed the original
validation eligibility rule and was subsequently chosen by the user as the
repository default. The full evaluation plan and prior comparisons remain in
[the experiment archive](../experiments/README.md).

## Primary metrics

All scores below are percentages at raw **T=1**. NoR includes relevant
off/unknown-only images. Each row evaluates the same single checkpoint.

| Dataset | Images | AP RR | AP RG | AP NoR | mAP | Balanced accuracy | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DTLD official test | 12,453 | 92.52 | 94.32 | 23.49 | 70.11 | 68.76 | 88.76 | 70.04 | 9.11 |
| ATLAS manual benchmark | 528 | 92.88 | 84.76 | 82.89 | 86.84 | 76.12 | 74.62 | 71.97 | 15.02 |
| VZC-TLD published test | 598 | 93.01 | 76.03 | 47.47 | 72.17 | 65.54 | 75.42 | 66.18 | 10.42 |

## Signal and NoR operating performance

| Dataset | RR recall | RG recall | Mean RR/RG recall | NoR recall |
|:--|--:|--:|--:|--:|
| DTLD | 86.07 | 94.69 | 90.38 | 25.52 |
| ATLAS | 85.71 | 90.53 | 88.12 | 52.13 |
| VZC-TLD | 85.80 | 77.84 | 81.82 | 32.97 |

C improves signal recall over the matched v5 fold-0 reference, while reducing
NoR recall. B has higher DTLD mAP/macro F1; D has higher VZC mAP; the axial
variant has higher DTLD macro F1 but weaker transfer scores. These tradeoffs
are reported in the [complete comparison](../experiments/diagnostics/README.md).
Architecture choice reflects development priorities rather than dominance
on every metric or success under the original study gates.

## DTLD confusion matrix

Rows are ground truth; columns are argmax predictions.

| Actual / predicted | RR | RG | NoR |
|:--|--:|--:|--:|
| RR | 3,752 | 513 | 94 |
| RG | 248 | 7,167 | 154 |
| NoR | 189 | 202 | 134 |

## Files and provenance

- [All primary metrics](metrics.csv), [class metrics](per_class_metrics.csv), [confusion matrices](confusion_matrices.csv).
- Original C reports: [DTLD](dtld/report.json), [ATLAS](atlas/report.json), [VZC-TLD](vzc_tld/report.json).
- [Report/checkpoint provenance](provenance.json), [efficiency measurements](efficiency.json).
- [Frozen diagnostic plan](../experiments/diagnostics/plan.json), [training history](../experiments/generalization/C/fold0/history.jsonl).
- [Checkpoint identity and selection history](../../metadata/checkpoints.json).

JSON reports preserve calibration metrics, class support, available city/camera
and native lamp-size slices, fixed sensitivity analyses, and prediction hashes.
Raw prediction arrays remain local in `runs/archive/`; images and checkpoints
are excluded from Git. No public weight download is currently provided.

ATLAS labels are manual, include uncertainty, and are not official test ground
truth. ATLAS/VZC and official DTLD test results were available during architecture
choice. Scores are exploratory for that choice; no independent confirmatory
selection evaluation, full-training refit, or four-fold C ensemble is claimed.
