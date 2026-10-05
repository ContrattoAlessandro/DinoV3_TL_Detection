# Recorded v5 results

AP, accuracy, F1, and ECE values are percentages at raw temperature **T=1**.
NLL and Brier score are unitless. ECE, NLL, and Brier are better when lower;
the other displayed scores are better when higher. Full
precision, class-specific precision/recall/F1/support, NLL, Brier score, and
confusion matrices appear in the linked CSV and JSON files.

## Official DTLD test — final fixed evaluation

All **12,453** official test frames are evaluated: 4,359 RR, 7,569 RG, and 525
NoR, across 632 sessions in 11 cities. No images are missing or excluded. The
published train/test partition has zero shared image identities, shared sessions,
or exact prepared-JPEG duplicates. The primary NoR policy includes 213 frames
whose only relevant states are off/unknown.

The four checkpoints and the **four-head mean-logit ensemble** were fixed before
test inference. Epochs, temperatures, thresholds, and ensemble weights were not
selected using test scores. This is the final test of retained city-fold EMA
checkpoints; a model refitted on the full official training split is not claimed.

| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Fold 0 | 90.87 | 94.76 | 24.36 | 70.00 | 75.47 | 83.75 | 68.80 | 6.88 |
| Fold 1 | 89.86 | 93.71 | 15.27 | 66.28 | 70.96 | 77.23 | 63.09 | 18.61 |
| Fold 2 | 90.12 | 94.43 | 24.52 | 69.69 | 74.39 | 83.10 | 67.94 | 7.08 |
| Fold 3 | 89.59 | 94.69 | 20.53 | 68.27 | 74.16 | 83.31 | 67.98 | 6.76 |
| Mean of four | 90.11 | 94.40 | 21.17 | 68.56 | 73.75 | 81.85 | 66.95 | 9.83 |
| Logit ensemble (four) | 91.41 | 95.04 | 24.92 | 70.46 | 75.89 | 84.65 | 69.58 | 11.75 |

Population standard deviation of individual checkpoint mAP: 1.47
percentage points. It measures checkpoint variability, not a confidence interval.
Each individual head has 489,475 trainable parameters. The ensemble holds four
heads (1,957,900 parameters) with one shared frozen encoder at inference. The
shared-encoder logits matched separate full-model forwards bit for bit on the
first batch for every checkpoint.

| Predictor | Weighted F1 (%) | NLL ↓ | Brier ↓ |
|:--|--:|--:|--:|
| Mean of four | 84.49 | 0.5456 | 0.2939 |
| Logit ensemble (four) | 86.52 | 0.4997 | 0.2603 |

### Primary ensemble per-class performance

| Class | Images | AP | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|
| RR | 4,359 | 91.41 | 90.37 | 81.81 | 85.88 |
| RG | 7,569 | 95.04 | 93.31 | 88.15 | 90.66 |
| NoR | 525 | 24.92 | 22.33 | 57.71 | 32.20 |

### Primary ensemble confusion matrix

Rows are ground truth; columns are argmax predictions. Counts refer to the full
official test set at T=1.

| Actual \ Predicted | RR | RG | NoR |
|:--|--:|--:|--:|
| RR | 3,566 | 372 | 421 |
| RG | 264 | 6,672 | 633 |
| NoR | 116 | 106 | 303 |

The fixed off/unknown sensitivity diagnostic removes 213 pseudo-NoR frames,
leaving 12,240 images (312 NoR). It gives ensemble mAP
69.02% and balanced accuracy
76.52%; it does not replace
the full-split result.

The Bochum city slice has no NoR examples. Its `n_classes=2` mAP averages only
RR/RG AP and must be read as a two-class slice diagnostic. Undefined AP values
are blank in the slice CSV and `null` in JSON; the full test set has all three classes.

[All individual and ensemble metrics](dtld_test_metrics.csv) · [Summary](dtld_test_summary.csv)
· [Per-class metrics](dtld_test_per_class_metrics.csv) · [Confusion matrices](dtld_test_confusion_matrices.csv)
· [City and lamp-size slices](dtld_test_slices.csv) · [Off/unknown sensitivity](dtld_test_sensitivity.csv)
· [Frozen protocol](dtld_test_protocol.json) · [Artifact hashes and verification](dtld_test_provenance.json)

The protocol retains its exact execution bytes in Git, including line endings.
Provenance also gives LF-normalized source hashes for checkout verification
across platforms; executed-file hashes remain in the frozen protocol.


## DTLD city-disjoint validation

Each of the 28,525 images is held out in one fold. The selected epochs are
8, 2, 8, and 11 for fold seeds 0, 1, 2, and 3. The mean gives each fold equal
weight; it is not a pooled out-of-fold metric or an ensemble prediction.

| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Fold 0 | 89.92 | 95.73 | 20.50 | 68.72 | 72.27 | 83.15 | 66.89 | 5.42 |
| Fold 1 | 88.60 | 95.30 | 10.00 | 64.63 | 68.13 | 77.23 | 60.09 | 18.22 |
| Fold 2 | 94.06 | 92.80 | 26.51 | 71.12 | 75.33 | 82.70 | 71.20 | 7.06 |
| Fold 3 | 93.50 | 97.96 | 42.11 | 77.85 | 82.44 | 87.04 | 75.38 | 9.10 |
| Mean of four | 91.52 | 95.45 | 24.78 | 70.58 | 74.54 | 82.53 | 68.39 | 9.95 |

Population standard deviation of fold mAP: 4.80 percentage points.
These scores use the validation folds that selected the checkpoints.
The final fixed official-test evaluation is reported separately above.

[Per-fold metrics](dtld_folds.csv) · [Summary](dtld_summary.csv)

## ATLAS transfer

All four checkpoints see the same 528 manually labeled images, including
uncertain annotations. Class counts: 245 RR, 95 RG, 188 NoR. These are the fixed
manual relevance labels defined in [the data protocol](../data.md).

| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Fold 0 | 89.58 | 77.01 | 77.05 | 81.22 | 73.17 | 75.19 | 73.73 | 5.80 |
| Fold 1 | 88.81 | 73.82 | 73.91 | 78.84 | 62.01 | 61.74 | 61.04 | 14.13 |
| Fold 2 | 87.38 | 73.45 | 75.85 | 78.89 | 73.47 | 76.14 | 73.90 | 6.28 |
| Fold 3 | 87.15 | 76.70 | 73.91 | 79.25 | 70.06 | 72.35 | 70.72 | 4.46 |
| Mean of four | 88.23 | 75.25 | 75.18 | 79.55 | 69.68 | 71.35 | 69.85 | 7.67 |

Population standard deviation of mAP across checkpoints: 0.97 percentage points.

[Per-checkpoint metrics](atlas_checkpoints.csv) · [Summary](atlas_summary.csv)

## VZC published test split

All four checkpoints see the same 598 images: 331 RR, 176 RG, 91 NoR. The final
row averages raw logits before softmax and is a separate ensemble predictor.

| Model | AP RR | AP RG | AP NoR | mAP | Bal. acc. | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Fold 0 | 89.49 | 73.14 | 43.87 | 68.83 | 67.10 | 64.21 | 61.13 | 7.27 |
| Fold 1 | 89.07 | 75.46 | 43.32 | 69.28 | 57.52 | 50.67 | 48.41 | 9.68 |
| Fold 2 | 86.13 | 68.69 | 35.36 | 63.39 | 58.03 | 51.51 | 48.93 | 21.03 |
| Fold 3 | 88.82 | 69.37 | 42.14 | 66.78 | 63.19 | 56.69 | 55.77 | 16.12 |
| Mean of four | 88.38 | 71.66 | 41.17 | 67.07 | 61.46 | 55.77 | 53.56 | 13.53 |
| Logit ensemble (four) | 90.10 | 75.24 | 44.60 | 69.98 | 64.02 | 58.36 | 56.25 | 7.90 |

Primary NoR labels include 47 images whose only relevant state is unknown.
Excluding those images leaves 551 samples and reduces mean mAP to 64.65%; the
ensemble has 67.61% mAP under that sensitivity policy. No duplicates occur
within the test split. The source train split is a separate transfer diagnostic
(mean mAP 64.09%), with no training on VZC images.

The four checkpoints predict NoR for 53.18% of test images on average, versus
15.22% in the labels. Mean NoR precision/recall are 26.63%/91.48%; RR and RG
recall are 55.97% and 36.93%. This explains the gap between ranking and operating
metrics. Saved DTLD temperature scaling increases mean test ECE from 13.53% to
21.90%, so raw T=1 is the primary report.

[All metrics and sensitivity policies](vzc_metrics.csv) · [Mean-model summaries](vzc_summary.csv)
· [Class precision/recall/F1](vzc_per_class_metrics.csv) · [Confusion matrices](vzc_confusion_matrices.csv)

## Scope and provenance

ATLAS and VZC results were inspected during architecture selection. They are
exploratory transfer diagnostics, not an independent confirmatory test of that
selection. AP depends on class prevalence and annotation policy; cross-dataset
score differences do not alone measure generalization improvement. See the
[evaluation protocol](../evaluation.md).

[Complete v5 metrics and source hashes](v5_results.json). Rebuild these exports
with `python scripts/summarize_results.py` after restoring the local saved v5
predictions. Checkpoints and image datasets are excluded from Git.
