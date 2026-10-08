# RGB + revised supervision: fixed epoch-two external diagnostic

The human-requested comparison evaluates the combined model's final trained additional-epoch-two EMA against the original frozen ViT-B epoch-seven EMA. The candidate failed the city-validation gates; its selected best.pt is the untouched baseline. This comparison explicitly evaluates last.pt and performs no training, threshold fitting, or checkpoint selection on these benchmarks.

Candidate SHA-256: `61e648bd1ac16fe43fa174aa28ee7d0fbcd94cdf93b0d52ab2f880ae958dfe41`.
Baseline SHA-256: `edb2bb7bfd4a40cf470cd62f1dbd14e25b14d9c8feb107207be03922c528042e`.

Original audited baseline raw logits are reused. Image identities and labels are aligned exactly before paired comparisons. Primary results use raw logits, T=1, and final softmax for AP. ATLAS includes uncertain labels to match the original 528-image benchmark. VZC primary metrics use its published test split; its evaluator also retains a separate train-split transfer diagnostic.

## Primary comparison

Each entry shows baseline → candidate (difference in percentage points).

| Metric | DTLD test | ATLAS | VZC test |
|---|---:|---:|---:|
| RR recall | 87.24 → 88.23 (+0.99) | 83.27 → 87.35 (+4.08) | 86.71 → 88.52 (+1.81) |
| RG recall | 95.63 → 96.31 (+0.69) | 89.47 → 89.47 (+0.00) | 76.70 → 77.84 (+1.14) |
| Mean signal recall | 91.44 → 92.27 (+0.84) | 86.37 → 88.41 (+2.04) | 81.71 → 83.18 (+1.47) |
| NoR recall | 31.81 → 12.38 (-19.43) | 76.06 → 56.38 (-19.68) | 47.25 → 29.67 (-17.58) |
| RR precision | 91.46 → 89.53 (-1.94) | 92.31 → 88.43 (-3.88) | 82.95 → 82.07 (-0.88) |
| RG precision | 91.71 → 90.62 (-1.10) | 65.89 → 52.80 (-13.10) | 71.43 → 66.18 (-5.24) |
| mAP | 72.80 → 72.45 (-0.34) | 90.25 → 89.27 (-0.98) | 79.46 → 77.40 (-2.06) |
| Balanced accuracy | 71.56 → 65.64 (-5.92) | 82.93 → 77.73 (-5.20) | 70.22 → 65.34 (-4.88) |
| Macro F1 | 72.97 → 67.55 (-5.42) | 80.53 → 74.01 (-6.52) | 71.53 → 66.64 (-4.90) |

## DTLD

Images: 12453. External broad gate passes: False. Failed criteria: mAP, acc_bal, macro_F1, recall_NoR, mean_signal_recall_gain_below_1pp.

| True class | Baseline correct → candidate wrong | Baseline wrong → candidate correct |
|---|---:|---:|
| RR | 26 | 69 |
| RG | 72 | 124 |
| NoR | 102 | 0 |

Baseline confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
3803 462 94
189 7238 142
166 192 167
```

Candidate confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
3846 491 22
254 7290 25
196 264 65
```

Paired whole-session bootstrap: 632 sessions, 2000 valid draws, seed 20261007. The following are 95% intervals for candidate-minus-baseline differences, in percentage points.

| Metric | Difference | 95% session-bootstrap interval |
|---|---:|---:|
| mAP | -0.34 | [-1.35, +0.67] |
| Balanced accuracy | -5.92 | [-8.36, -3.80] |
| Macro F1 | -5.42 | [-7.50, -3.13] |
| Mean signal recall | +0.84 | [+0.43, +1.24] |
| RR recall | +0.99 | [+0.43, +1.54] |
| RG recall | +0.69 | [+0.18, +1.22] |
| NoR recall | -19.43 | [-26.69, -13.04] |
| RR precision | -1.94 | [-2.64, -1.22] |
| RG precision | -1.10 | [-1.62, -0.67] |

## ATLAS

Images: 528. External broad gate passes: False. Failed criteria: mAP, acc_bal, macro_F1, recall_NoR, precision_RR, precision_RG.

| True class | Baseline correct → candidate wrong | Baseline wrong → candidate correct |
|---|---:|---:|
| RR | 3 | 13 |
| RG | 1 | 1 |
| NoR | 38 | 1 |

Baseline confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
204 14 27
2 85 8
15 30 143
```

Candidate confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
214 19 12
3 85 7
25 57 106
```

Paired prediction changes are reported without image-independent confidence intervals: trustworthy independent scene/session groupings are unavailable.

## VZC_TLD

Images: 598. External broad gate passes: False. Failed criteria: mAP, acc_bal, macro_F1, recall_NoR, precision_RG.

| True class | Baseline correct → candidate wrong | Baseline wrong → candidate correct |
|---|---:|---:|
| RR | 5 | 11 |
| RG | 6 | 8 |
| NoR | 17 | 1 |

Baseline confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
287 30 14
35 135 6
24 24 43
```

Candidate confusion matrix, true rows and predicted columns RR/RG/NoR:

```text
293 34 4
36 137 3
28 36 27
```

Paired prediction changes are reported without image-independent confidence intervals: trustworthy independent scene/session groupings are unavailable.

## Decision and limitations

Keep the original baseline: the candidate already failed the predeclared city-validation gates, and this external diagnostic does not override that decision. Broad improvement requires at least +1 percentage point mean signal recall on every dataset, no regression in either signal recall, mAP, balanced accuracy, or macro F1, and no more than two points lost in NoR recall or either signal precision.

ATLAS and VZC were already inspected during development, so this is observed transfer on known benchmarks. It does not establish performance on a genuinely untouched dataset. DTLD session-bootstrap intervals measure uncertainty in this paired test-set comparison, not variability across training seeds.

The complete raw reports, predictions, fixed input hashes, status, paired transitions and bootstrap results are retained under runs/recall_head_combined_external_epoch2. The training pilot and historical experiment records remain intact.

## DTLD test slices

Baseline slices use the same candidate-recorded target sizes and city membership after exact identity alignment; size buckets exclude NoR images.

| Slice | Images | Baseline mean signal recall | Candidate mean signal recall | Difference (pp) |
|---|---:|---:|---:|---:|
| city: Berlin | 1077 | 91.77% | 92.45% | +0.68 |
| city: Bochum | 109 | 89.63% | 92.41% | +2.78 |
| city: Bremen | 222 | 70.79% | 70.65% | -0.13 |
| city: Dortmund | 2027 | 96.22% | 96.59% | +0.38 |
| city: Duesseldorf | 1389 | 93.91% | 94.46% | +0.54 |
| city: Essen | 781 | 93.12% | 94.70% | +1.58 |
| city: Frankfurt | 1677 | 94.92% | 95.19% | +0.27 |
| city: Fulda | 108 | 87.44% | 91.85% | +4.41 |
| city: Hannover | 2408 | 89.58% | 91.39% | +1.82 |
| city: Kassel | 757 | 87.48% | 88.70% | +1.22 |
| city: Koeln | 1898 | 88.84% | 89.16% | +0.32 |
| target_lamp_size: small(<16px) | 2053 | 77.95% | 80.56% | +2.61 |
| target_lamp_size: mid(16-48px) | 7426 | 94.15% | 94.71% | +0.57 |
| target_lamp_size: large(>=48px) | 2449 | 96.28% | 96.49% | +0.21 |
