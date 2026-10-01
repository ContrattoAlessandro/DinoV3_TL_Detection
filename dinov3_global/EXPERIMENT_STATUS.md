# Reviewed labels and paired experiments — 30 September 2026

ATLAS labels are consistent with the visual review and the annotator's two
maneuver confirmations. Image 16 remains RR: the ego vehicle intends to turn
left. Image 19 was corrected from RR to RG for the left-turn movement. Both
notes are saved in the labeling application. Revision 31 contains 28 certain
labels: 15 RR, 10 RG, 3 NoR. Source image hashes match. The evaluation snapshot
is `runs/atlas_review/atlas_labels_revision31.json`; future app edits do not
alter these saved results. Single frames do not establish every intended
maneuver independently, so the confirmed intent remains part of the annotation.

## ATLAS checkpoint results

Fresh GPU inference, EMA `best.pt`, full input resized to its checkpoint's
resolution, no side crop, raw logits at T=1. No ATLAS temperature fitting.

| Checkpoint | AP RR | AP RG | AP NoR | mAP | Balanced accuracy | Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exp2 | 95.9% | 78.2% | 56.7% | **76.9%** | **71.1%** | 18/28 |
| exp3 | 92.7% | 75.0% | 27.1% | 64.9% | 65.6% | 15/28 |
| exp4 | 88.6% | 73.1% | 27.8% | 63.2% | 68.9% | 16/28 |

Exp2 transfers best on these images despite exp4's stronger saved DTLD
validation score. Exp4 incorrectly predicts NoR on 7/15 RR and 3/10 RG images;
exp2 does so on 4/15 RR and 6/10 RG. All checkpoints correctly classify the
three actual NoR examples, but exp3/exp4 rank numerous other images above them
for NoR probability, explaining their poor AP NoR. AP and argmax accuracy
measure different behavior. These are 28 potentially correlated scenes, with
only three NoR examples; this establishes a diagnostic failure pattern, not a
precise estimate of broad transfer. Reports, per-camera metrics, confusion
matrices and raw per-image logits are in `runs/eval_atlas_reviewed/{exp2,exp3,exp4}`.

## Protocol check

[Color Is Not Enough, Section V-A](https://flore.unifi.it/retrieve/5ae436fa-8236-4bb6-8794-05fabc4ae1f2/Color_Is_Not_Enough_Dataset_and_Method_for_Identifying_Relevant_Traffic_Lights_in_Driving_Scenes.pdf)
uses official DTLD training and official validation as final test, with five
seeds. It does not specify city cross-validation. It reports image-level class
AP, their three-class mean and macro recall. Our metrics and 1280×720
preprocessing follow these definitions, but city CV is additional development
validation. Its selected-epoch scores cannot be compared directly with the
paper's final-test mAP.

Local `DTLD_train.json` has 28,525 unique frames and `DTLD_test.json` has 12,453,
with zero frame overlap. Their union equals all 40,978 entries in
`DTLD_all.json`; every train/test image is present. Annotation SHA256 values
and counts are saved in `runs/atlas_review/official_split_audit.json`.
The local holdout is named `test`, whereas the paper calls its holdout
`validation`. No author-supplied frame manifest/checksum was available to
establish exact membership parity; filename convention alone is not proof of
an exact replication. No new predictions on that split were made here.

The repository's `map_to_nor` policy includes 453 training and 213 held-out
frames that have relevant off/unknown signals but no relevant stop/go signal.
The paper's NoR definition says no relevant light, regardless of state; its
handling of relevant off/unknown cases is unspecified. Keep this policy fixed
for the paired experiment and resolve it before claiming strict paper parity.
These ambiguous frames are identified as `pseudo_nor`; their counts are
recorded per city fold and their AP is diagnosed during training.

## Locked city comparison

Only official DTLD training images enter the four folds. Each training and
validation partition contains all three classes and has zero session overlap.
Each city appears in validation once; neither ATLAS labels nor official test
predictions are used for epoch/configuration selection.

| Fold | Validation cities | Train frames | Validation frames | RR / RG / NoR |
| --- | --- | ---: | ---: | --- |
| 0 | Dortmund, Kassel, Fulda | 21,493 | 7,032 | 2,292 / 4,437 / 303 |
| 1 | Koeln, Essen, Bochum | 21,308 | 7,217 | 2,275 / 4,741 / 201 |
| 2 | Hannover, Berlin, Bremen | 21,277 | 7,248 | 2,972 / 3,792 / 484 |
| 3 | Duesseldorf, Frankfurt | 21,497 | 7,028 | 2,031 / 4,613 / 384 |

Both variants use 12 epochs, warmup 3, no early stopping, seed 0 + fold,
learning rate 6e-5, weight decay .05, batch 4 with accumulation 4, EMA .999,
the same sampling and safe augmentation configuration. Head, relevance loss
weighting and branch supervision differ as intended by v5. This is a comparison
of configuration bundles; it cannot attribute an improvement to one component.
Separate ablations follow only if the city results justify them.

The queue alternates v4-safe and v5 within each fold, using one GPU process at
a time. Both variants must accept the same annotation/membership hashes
before training. Config and training-source hashes prevent silently mixing
changed experiments; completed folds save raw validation logits and reports.
An epoch checkpoint preserves optimizer, scaler, EMA, best score, staleness
and random states for resuming. GPU operations can still be nondeterministic.

Run directory: `runs/city_comparison_20260930`. `status.json` identifies the
active job and process IDs; `v4_safe_fold0.log` records the first full run.
Per-variant `cv_city_report.json` is updated when folds finish. On completion,
`comparison.json` records paired fold deltas, mean and worst-fold mAP. Until
then the comparison is pending and no v5 improvement is claimed.

The runner passed a real paired GPU smoke test and 43 regression tests.
Smoke-subset metrics do not estimate full-run performance. To resume an
interrupted comparison after its supervisor and child have exited:

```powershell
python dinov3_global/scripts/compare_city.py --out dinov3_global/runs/city_comparison_20260930 --folds 4 --epochs 12 --resume
```

## Throughput update

At the user's request, v4-safe fold 0 was paused at its complete epoch-6
checkpoint, benchmarked on disposable copies and resumed at epoch 7. The
selected execution settings use two workers, prefetch 4, pinned memory and
one main CPU thread. Batch 4 / accumulation 4 and all scientific settings
remain fixed. Runtime settings persist automatically for the remaining jobs.
Both variants passed bit-identical GPU-update checks; 46 regression tests now
pass. [THROUGHPUT_REPORT.md](THROUGHPUT_REPORT.md) records the approximately
15% measured training-throughput improvement, validation timings, resource
tradeoffs and execution provenance.

## Expanded ATLAS manual evaluation collection

The user supplied 500 additional images in `C:/Users/alexa/Desktop/test`.
All are distinct by SHA-256, with no exact overlap with the original 28.
The combined collection has 528 images at
`datasets/atlas_relevance_expanded`; source images remain unchanged.
Additional camera counts are 380 front_medium, 78 front_tele and 42 front_wide.

The labeler at http://127.0.0.1:8787 now saves to
`runs/atlas_expanded_labels.json`. The 28 original label records, including
notes and maneuver confirmations, were imported identically from revision 31.
The remaining 500 initially started unlabeled. The original `runs/atlas_labels.json` and
the historical 28-image evaluation reports remain unchanged.

After manual labeling and review, future OOD evaluations should explicitly
use the expanded labels file. Keep it evaluation-only, exclude uncertain
records by default, and retain the original/additional partitions from
`runs/atlas_expanded/collection_manifest.json` when comparing with historical
results. More images improve coverage, but nearby frames and camera views
can be correlated; 528 images are not necessarily 528 independent scenes.
No model predictions were loaded into the expanded labeler.

The collection passed source/copy hash checks, image-header verification,
record-preservation checks and browser checks for counts, image display,
navigation, zoom and reload. The page starts at image 29, the first unlabeled
image. Labeler setup made no changes to the active training source or queue.

### Completed expanded annotations and queued test

The user completed all 528 annotations at revision 561. All source hashes
match and the original 28 records are unchanged. Counts are 245 RR, 95 RG,
188 NoR. One additional RG image is marked uncertain:
`front_medium_1722849022-21861000000.jpg`.

The full requested OOD test includes **all 528 images**, including that
uncertain label, with a separate 527-certain-image sensitivity report. It
also reports the original 28 and additional 500 separately. This preserves
comparability with historical results without replacing the expanded score.
The frozen snapshot is
`runs/eval_atlas_expanded_revision561/labels_revision561.json`; later labeler
edits will not silently change this evaluation. `label_audit.json` verifies
both entry counts, hashes and unchanged training sources.

`runs/atlas_expanded/evaluate_suite.py` is queued in a background worker.
It waits until the entire city comparison completes and its training
processes exit, then evaluates exp2, exp3, exp4 and all eight v4-safe/v5 fold
best checkpoints serially on the GPU. It passes the snapshot explicitly
with `--include-uncertain`, uses raw logits at T=1, and saves checkpoint and
label hashes. No GPU is initialized while waiting. Resume the worker with
the same script after a reboot; it reuses only completed matching reports.

Evaluation progress and results live under
`runs/eval_atlas_expanded_revision561` (`status.json`, `worker.log`, per-model
reports and eventual `comparison.json`). The final model trained on all
DTLD training images must also be tested against this same frozen snapshot
when that checkpoint becomes available. ATLAS is not used for training or
epoch selection.
