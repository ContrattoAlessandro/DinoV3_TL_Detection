# Evaluation and reproducibility

## Released predictor

The released model uses frozen DINOv3 ViT-B/16 and the shared evidence head.
`checkpoints/dinoglobal_vitb_fold0.pt` contains the fold-0, seed-0, epoch-7 EMA
head. [Checkpoint metadata](../metadata/checkpoints.json) records its SHA-256.
Its serialized identifier `v7_evidence` is retained for strict loading.

Training used 21,493 frames and validation used 7,032 frames from Dortmund,
Kassel, and Fulda. Highest held-city validation mAP at raw T=1 selected epoch 7.
The configuration and training history are in the
[reproduction record](reproduction/README.md).

## Training partitions and checkpoint selection

Ordinary `train.py` creates a session-disjoint validation partition inside
official DTLD train. `cross_validate.py` partitions whole cities using
[frozen membership](../metadata/dtld_city_folds.json):

| Fold | Held-out cities | Train | Validation | Runner seed |
|:--|:--|--:|--:|--:|
| 0 | Dortmund, Kassel, Fulda | 21,493 | 7,032 | 0 |
| 1 | Koeln, Essen, Bochum | 21,308 | 7,217 | 1 |
| 2 | Hannover, Berlin, Bremen | 21,277 | 7,248 | 2 |
| 3 | Duesseldorf, Frankfurt | 21,497 | 7,028 | 3 |

The runner adds the fold index to the configured base seed. Only fold 0 has a
published checkpoint and completed measurements for this model; the remaining
rows describe supported partitions, not measured results. Neither workflow
uses official test images for fitting or epoch selection.

Each new run selects the highest validation EMA mAP at T=1, retaining the earlier
epoch on ties. Clean training-probe scores are diagnostic only. Scientific
configuration and [execution settings](../configs/runtime.json) are recorded
separately. `last.pt` stores optimizer, scaler, EMA, and RNG state for resume.

## Official DTLD test

Evaluate all **12,453** official frames: 4,359 RR, 7,569 RG, and 525 NoR.
The latter include 213 relevant-off/unknown-only scenes. Missing images,
invalid references, and aliased filenames cause an error rather than silent
membership changes.

The evaluator freezes checkpoint/data/source identities, audits native session
membership and prepared-image overlap, and uses raw logits at **T=1**. Reported
mAP averages three one-vs-rest AP values from final softmax probabilities;
balanced accuracy averages class recall. Argmax supplies class decisions.

One checkpoint produces one EMA predictor. Explicitly supplied compatible
checkpoints can also be evaluated individually and as a mean-raw-logit ensemble.
Shared encoder inference must exactly match separate complete-model forwards
on the first batch. The published scores use one head.

The full official membership is primary. A separately labeled sensitivity
excludes the 213 off/unknown-only NoR frames. City and lamp-size slices are
diagnostics; target-size slices use the largest relevant lamp compatible with
the image label and exclude NoR, which has no compatible target.

## Transfer benchmarks and calibration

ATLAS uses a fixed **528-image manual relevance-label snapshot**, including
uncertain labels in the published primary score. It is not official ATLAS
relevance ground truth. Camera-specific reports remain available.

VZC-TLD uses its **598-image published test** split. Source train (2,390 images)
and combined (2,988 images) are separate transfer diagnostics. All COCO lamp
annotations, including `iscrowd=1`, are retained. Unknown-only and duplicate-image
sensitivities are reported explicitly.

Primary metrics never fit temperatures, thresholds, epochs, or ensemble weights
on test/transfer labels. Optional DTLD `--calibrated` uses the checkpoint's saved
validation temperature. Temperature scaling preserves argmax but may change
cross-image softmax ranking and AP; DTLD calibration need not transfer.

The published transfer data influenced development and model choice, so these
are exploratory benchmarks. They do not establish an independent confirmatory
architecture comparison. The results cover one training seed and city fold;
no full-training-split refit or repeated-seed estimate is claimed. Low NoR recall
remains a limitation.

## Published artifacts and reproduction

[Result tables](results/README.md) include full-precision predictions, class
metrics, confusion matrices, source-split sensitivities, timing, and provenance.
Each result artifact is hashed in `docs/results/provenance.json`.

`summarize_results.py` requires complete evaluation status, checks membership
and exact mean logits, recomputes metrics, and verifies executed source hashes
before exporting tables to a fresh directory. Keep new evaluations separate
from the published measurements. Library/hardware changes may affect numerical
reproduction; the recorded software environment is in
[metadata/environment.json](../metadata/environment.json).

`dinov3_global.comparison` supports exact image/label alignment and paired
whole-session bootstrap intervals. These intervals describe test-sample
uncertainty for fixed predictions; they do not measure training-seed variability.
Reliable external scene groups are unavailable, so ATLAS/VZC images should not
be treated as independent samples for uncertainty claims.
