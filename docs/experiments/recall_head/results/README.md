# Frozen ViT-B recall pilot: measured fold-0 results

Completed 8 October 2026 at 11:52 Europe/Rome. All four variants completed two additional epochs from the same original epoch-7 EMA checkpoint, then stopped under patience two. No trained epoch passed every validation gate. Every selected checkpoint is therefore the untouched epoch-zero predictor. The original ViT-B remains selected.

These are raw DTLD city-validation results on 7,032 images: 2,292 RR, 4,437 RG and 303 NoR. They are separate from the earlier DTLD test results. No fold replication, final latency benchmark, or new DTLD test/ATLAS/VZC evaluation ran, because the initial pilot gate failed.

## Baseline and final trained epochs

The table shows the final trained epoch for diagnosis, not an approved or selected replacement. All figures are percentages.

| Variant | RR recall | RG recall | Mean recall | NoR recall | mAP | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 88.83 | 93.37 | 91.10 | 32.34 | 70.13 | 71.52 | 71.16 |
| control | 88.31 | 93.64 | 90.98 | 33.66 | 70.24 | 71.87 | 71.49 |
| supervision | 89.35 | 94.93 | 92.14 | 12.21 | 69.56 | 65.50 | 66.44 |
| rgb | 88.31 | 93.46 | 90.89 | 33.33 | 70.21 | 71.70 | 71.23 |
| combined | 89.53 | 94.86 | 92.20 | 12.87 | 69.74 | 65.75 | 66.84 |

## Interpretation

RGB alone showed no benefit in this short continuation: mean signal recall was 0.22 percentage points below the untouched baseline and 0.09 below the matched control at epoch two. Its learned residual output is nonzero, so the branch was updated. Two epochs and one seed are insufficient to rule out the architecture under a different training schedule.

Revised supervision raised mean signal recall by 1.04 points without RGB and 1.09 with RGB, but NoR recall fell by 20.13 and 19.47 points respectively. The combined model also lost 0.40 points mAP, 5.76 points balanced accuracy, and 4.33 points macro F1. Its RR precision fell by 2.18 points, beyond the allowed two-point loss. Recall gains cannot be considered a broad improvement.

The combined first epoch was less damaging, but still failed: mean signal recall 92.19%, NoR recall 20.13%, mAP 69.99%, balanced accuracy 68.17%, macro F1 69.47%. Further training reduced NoR recall to 12.87%, while mean signal recall increased by only 0.01 points.

Baseline confusion matrix (true rows and predicted columns: RR, RG, NoR):

```text
2036 182  74
 136 4143 158
 116  89  98
```

Combined additional epoch two:

```text
2052 212 28
 182 4209 46
 130 134 39
```

Signal-to-NoR errors decreased from 232 to 74; cross-colour signal errors increased from 318 to 394; correct NoR rejections decreased from 98 to 39. These are aggregate count changes, not paired per-image transitions. The model became more willing to predict a signal, with weaker colour discrimination and relevance rejection.

## Target size and city generalization

Target-size slices use the largest relevant lamp compatible with the image target, after native-frame letterboxing (scale 0.625, image content 640 by 1280 inside the 720 by 1280 canvas). NoR is excluded from these slices. Baseline sizes were reconstructed from the unchanged annotations and aligned cached validation identities; bucket counts match the trained epoch reports.

| Target lamp height | Images | Baseline mean recall | Supervision epoch 2 | Combined epoch 2 |
|---|---:|---:|---:|---:|
| small(<16px) | 1090 | 76.16% | 80.90% | 81.24% |
| mid(16-48px) | 4107 | 93.40% | 94.00% | 94.00% |
| large(>=48px) | 1532 | 96.48% | 96.08% | 96.08% |

The combined model gained 5.08 points on small target lamps, but the supervision-only model gained 4.73. This implicates the loss changes as the primary source of the observed small-lamp gain; the small extra RGB difference is not replicated evidence of architectural superiority.

Combined epoch-two city mean signal recall: Dortmund 93.75% versus baseline 92.91%, Kassel 90.19% versus 87.71%, Fulda 85.66% versus 86.27%. Improvement is uneven: Fulda RR recall falls from 78.79% to 78.35%, RG from 93.75% to 92.97%. Worst-city recall therefore fails its gate.

RGB-only epoch-two clean-training probe mAP was 92.38% versus held-city validation mAP 70.21%, a 22.17-point gap. The probe is training data and has different class/city composition; this is a diagnostic gap, not a direct estimate of unseen-dataset performance. Revised supervision also reduced probe NoR recall between epochs, suggesting an operating-balance shift that affects both training and held-city images.

## Recommended next experiment

1. Retain the original ViT-B checkpoint and the current validation gates.
2. Separate the two supervision changes: original loss, lamp-normalization only, joint-evidence only, and both. The current factorial study separates architecture from a bundled supervision change; it cannot identify which loss causes the NoR collapse.
3. Start loss changes gradually and test stronger balanced supervision of irrelevant/off lamps and false signal evidence. Inspect loss-gradient scale on shared lamp/state/relevance outputs. The lamp loss definitions differ, so their numerical loss magnitudes are not directly comparable.
4. For a predeclared second pilot, evaluate all cells over the same longer window (up to six additional epochs, or a minimum training window before patience begins). Keep checkpoint eligibility separate from the decision to terminate training. This gives a zero-initialized branch time to learn without accepting the observed regressions. Preserve the current pilot as a separate record.
5. Revisit RGB after supervision is controlled; repeat promising candidates on other city folds and seed 1 before external evaluation. Avoid another architecture expansion based only on this single-fold result.

No external-transfer improvement or bootstrap confidence interval is established by this pilot. Selected predictions are identical to baseline, and raw predictions from rejected epochs were not cached, so paired session-bootstrap intervals for those epochs cannot be reconstructed from confusion matrices. They would require a separate read-only checkpoint evaluation.

## Provenance

- `runs/recall_head_pilot/status.json` and `decision.json` record the completed non-promotion decision.
- `runs/recall_head_pilot/fold0/{control,supervision,rgb,combined}/history.jsonl` retain all trained epoch metrics and slices.
- `epoch_metrics.csv` accompanies this report and includes the individual failed gates.
- The control reporting failure was repaired by converting NumPy report metadata to built-in values. Original files and an audit are under `runs/recall_head_pilot/recovery_serialization`; head and EMA tensors were verified identical. Training settings and gates were preserved.
- This report performs no new fitting or external evaluation.
