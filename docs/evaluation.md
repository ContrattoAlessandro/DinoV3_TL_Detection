# Evaluation and reproducibility

## Supported model and recorded checkpoint

The supported method is DinoGlobal-MIL with the original shared evidence head.
The default encoder is frozen DINOv3 ViT-B/16; ViT-S+/16 remains an explicit
reference configuration. The serialized head identifier `v7_evidence` preserves
strict loading of both retained checkpoints. Rejected RGB/evidence-loss
configurations and earlier v5/axial heads are unsupported by the active package.

The retained default is the **fold-0 epoch-7 EMA** trained with ViT-B, identified
in [checkpoint metadata](../metadata/checkpoints.json). It used 21,493 training
images and 7,032 validation images, holding out Dortmund, Kassel and Fulda.
Selection used the highest raw-validation mAP. The original ViT-S+ epoch-5 EMA
and its measurements remain preserved as the reference.

The RGB pilot failed its validation and external improvement criteria. Its
[results and decisions](experiments/recall_head/results/README.md) are archived;
its failure does not change the retained predictor or training objective.

## New training protocol

[Default settings](../configs/default.yaml) retain the original head, data, loss,
and augmentation choices, with ViT-B as the encoder. New runs independently select the highest validation
EMA mAP at T=1; ties retain the earlier epoch. Selection no longer depends on
legacy v5 predictions. The scientific configuration and execution-only
[runtime settings](../configs/runtime.json) are recorded separately.

[City-fold membership](../metadata/dtld_city_folds.json) is frozen:

| Fold | Held-out cities | Train | Validation | Seed |
|:--|:--|--:|--:|--:|
| 0 | Dortmund, Kassel, Fulda | 21,493 | 7,032 | 0 |
| 1 | Koeln, Essen, Bochum | 21,308 | 7,217 | 1 |
| 2 | Hannover, Berlin, Bremen | 21,277 | 7,248 | 2 |
| 3 | Duesseldorf, Frankfurt | 21,497 | 7,028 | 3 |

The membership manifest predates C and contains historical transform fields.
The current audit matches annotation hashes, image identities, class counts, cities,
and sessions; the effective configuration separately records full-frame
letterboxing. Only fold 0 for each retained backbone has recorded completed measurements. Additional
folds and ensembles must be reported as new runs.

Ordinary `train.py` uses a session-disjoint validation partition within official
training data. Both workflows exclude official test from fitting and epoch
selection. Held-out city scores used for epoch selection are validation scores.

## Official DTLD test

Evaluate all 12,453 published DTLD test frames: 4,359 RR, 7,569 RG, and 525 NoR,
including 213 relevant-off/unknown-only NoR frames. The evaluator rejects missing
images, invalid references, and aliased filenames instead of silently excluding
them. It records native session membership and exact prepared-JPEG overlap.

Before inference, `plan.json` freezes checkpoint hashes, epochs, configuration,
annotation/membership hashes, source hashes, runtime, and environment. Reports
use T=1 without fitting thresholds, calibration, or ensemble weights on test.

For one checkpoint, the final predictor is that single EMA head. If multiple
compatible checkpoints are explicitly supplied, evaluate each separately and
average their **raw logits before softmax**. Encoder features are shared only
after bitwise agreement with separate complete-model forwards is checked.
Do not compare a single head with an old four-head ensemble as a matched
architecture ablation. The archived matched reference is v5 fold 0.

The full official membership is primary. A separately labeled sensitivity
analysis excludes the 213 relevant-off/unknown-only frames; it does not replace
the complete-split score. City and resized lamp-size slices are diagnostics. Target-size slices use the largest
relevant lamp compatible with the image label; NoR has no such target and is
excluded from these signal-only slices. Legacy largest-any-lamp slices remain
available.

## Transfer evaluation and claim scope

ATLAS uses the fixed 528-image manual-label snapshot, including uncertain labels
for the recorded primary result. It is not official ATLAS relevance ground truth.
VZC-TLD uses the 598 published test images; source train and combined memberships
are separate diagnostics. Unknown-only and duplicate sensitivities are explicit.

Transfer data did not fit model weights or the diagnostic checkpoint epoch.
However, ATLAS/VZC results influenced development and the choice of the retained backbones, so they
are exploratory transfer benchmarks. Official DTLD test results were also
available before the retained models became repository defaults. A new independent evaluation
is needed for a confirmatory architecture-selection claim. No repeated-seed
uncertainty or four-fold ensemble result is available in the recorded release.

## Metrics

| Metric | Definition |
|:--|:--|
| AP RR / RG / NoR | One-vs-rest average precision using final softmax class scores. |
| mAP | Arithmetic mean of the three class APs; image classification, no box IoU. |
| Balanced accuracy | Mean class recall under three-way argmax. |
| Accuracy | Fraction of correct argmax predictions. |
| Per-class precision / recall / F1 | Operating performance for each class; macro F1 averages equally. |
| Mean RR/RG recall | Arithmetic mean of RR and RG recalls. |
| ECE | Sample-weighted confidence/accuracy gap in 15 equal-width confidence bins. |
| NLL | Mean negative log probability assigned to the true class. |
| Brier | Mean sum of squared three-class probability errors, without dividing by three. |

ECE, NLL, and Brier are better when lower. AP measures ranking and does not
establish the quality of argmax decisions; read NoR recall alongside mAP.
Cross-dataset AP also depends on class prevalence and label policy.
Undefined AP is JSON `null`; slices state their number of present classes.

Optional calibration uses a positive scalar fitted on DTLD validation only.
It preserves argmax but may change cross-image softmax ranking and AP. Raw T=1
remains primary. Library/hardware changes may affect numerical reproduction.

## Artifacts

- Training: selected `best.pt`, resumable `last.pt`, per-epoch `history.jsonl`, clean probe.
- Cross-validation: membership/config/source audits, fold reports, validation logits.
- DTLD test: frozen plan, full membership audit, progress, member/final reports and logits.
- ATLAS: annotation audit, per-camera/class reports, full-precision logit CSV.
- VZC: label hashes, checkpoint/source provenance, reports, sensitivities and logit arrays.
- Publication: [ViT-B result tables](results/README.md), [historical archive](experiments/README.md).

`summarize_results.py` supports one or more DTLD checkpoints, checks complete
status, exact membership and mean logits, recomputes reports, and verifies
protocol/source hashes before exporting to a fresh directory. ATLAS/VZC
evaluators write their own detailed tables. Keep new results separate from the
immutable measurements distributed in `docs/results/` and `docs/experiments/`.

## Paired model comparisons

`dinov3_global.comparison` aligns exact image identities and labels, reports
paired prediction changes and metric differences, and optionally computes
whole-session bootstrap intervals. Supply trustworthy native session IDs and
record the resampling seed. These intervals describe a fixed-checkpoint test
comparison; they do not measure training-seed variability. External scene
groupings are unavailable in the current transfer benchmarks, so report paired
counts without treating their images as independent observations.
