# Evaluation and reproducibility

## Fixed architecture and checkpoints

The supported architecture is DinoGlobal-MIL v5. The default configuration
matches the scientific settings stored in the retained city-fold checkpoints:
12 epochs, a frozen ViT-S+/16 encoder, layer-6/final fusion, three local MIL
branches, scene-conditioned relevance, and bounded scene logits.

[Checkpoint metadata](../metadata/checkpoints.json) identifies the four EMA
checkpoints by SHA-256. Checkpoint files remain local under
`runs/v5/city_cv/fold{0..3}/`; they are excluded from Git. DINOv3 weights are
loaded separately from the pinned revision. A fresh checkout can reproduce
training after obtaining the licensed backbone and datasets.

The [recorded environment](../metadata/environment.json) specifies the versions
used for verification and retained inference. `configs/runtime.json` records
the audited city-fold execution settings: two workers, pinning, prefetch factor
four, no persistent workers, and one CPU thread during training. Scientific
settings and runtime settings are stored separately. Hardware, library versions,
and numerical kernels can affect reproduction across machines.

## DTLD city-disjoint validation

City groups are greedily balanced by frame count. The frozen
[fold manifest](../metadata/dtld_city_folds.json) records annotation and membership
hashes, class counts, and session overlap. Reproduction must match that manifest.

| Fold | Held-out cities | Training images | Validation images | Seed | Selected epoch |
|:--|:--|--:|--:|--:|--:|
| 0 | Dortmund, Kassel, Fulda | 21,493 | 7,032 | 0 | 8 |
| 1 | Koeln, Essen, Bochum | 21,308 | 7,217 | 1 | 2 |
| 2 | Hannover, Berlin, Bremen | 21,277 | 7,248 | 2 | 8 |
| 3 | Duesseldorf, Frankfurt | 21,497 | 7,028 | 3 | 11 |

Each fold trains for 12 epochs. `best.pt` is chosen using its highest EMA
validation mAP at raw temperature $T=1$. The held-out cities serve as validation
for epoch selection, so these out-of-fold results remain **validation results**.
Training/validation sessions have zero overlap in every fold. This evaluation
covers 28,525 images exactly once across folds, although fold scores are averaged
with equal fold weight in the main table.

The ordinary training script instead creates a session-disjoint validation
partition from the official training data. It does not automatically refit a
final model on every training image. The final official-test evaluation below
uses the retained city-fold checkpoints, with their selection already complete.

## Final official DTLD test

The final evaluation uses **DTLD v2.0's published `DTLD_test.json` membership**.
All 12,453 annotated frames have prepared RGB images and valid lamp annotations;
none are missing or excluded. There are 4,359 RR, 7,569 RG, and 525 NoR frames
across 632 sessions and 11 cities. Primary labels retain 213 relevant-off/unknown-only
frames as NoR. The separately labeled sensitivity diagnostic removes those
frames, leaving 12,240 images and 312 NoR examples.

Before test inference, the evaluator freezes the checkpoint hashes, selected
epochs, data membership hashes, source hashes, preprocessing, label policy,
runtime, and environment in a [protocol record](results/dtld_test_protocol.json).
The four city-validation-selected EMA checkpoints are evaluated individually.
The declared primary predictor is their equally weighted **mean-logit ensemble**:

$$
\bar\ell(x)=\frac14\sum_{f=0}^{3}\ell_f(x),\qquad
p(x)=\operatorname{softmax}(\bar\ell(x)),\qquad
\hat y(x)=\operatorname{argmax}_c p_c(x).
$$

All primary scores use $T=1$. No weights, epochs, temperatures, thresholds, or
ensemble coefficients are fitted or selected on the test labels. Every member
and the ensemble use the same audited frame order. The arithmetic mean of
individual checkpoint metrics is also reported, separately from the ensemble.
No best-test checkpoint is promoted as the final model.

The encoder is identical and frozen across checkpoints, so inference reuses one
DINOv3 pass for all four heads. On the first batch, the evaluator requires bitwise
agreement with separate full-model forwards for every head. A single model has
489,475 trainable head parameters; the ensemble holds 1,957,900 head parameters
and one 28,692,864-parameter frozen encoder at inference. This ensemble is not a
single head refitted on the complete official training split.

The audit compares the complete official train and test annotations and hashes
every prepared JPEG. It finds zero shared native image identities, basenames,
sessions, or exact JPEG contents between the splits, and no exact duplicates
within either split. These checks concern the local prepared images and session
identifiers, defined as native `city/route/timestamp` folders. The official test
cities are also represented in the training split.
City-held-out validation instead measures transfer to cities absent from that
fold's training data. Their scores therefore describe different conditions.

Reproduce this fixed evaluation in a fresh output directory:

```sh
python scripts/evaluate.py --ckpt runs/v5/city_cv/fold0/best.pt runs/v5/city_cv/fold1/best.pt runs/v5/city_cv/fold2/best.pt runs/v5/city_cv/fold3/best.pt --runtime-loader configs/runtime.json --out runs/v5/dtld_test_reproduction
```

Evaluation refuses to overwrite an existing run. Scores are recorded with full
precision in the [result tables](results/README.md), including class metrics,
confusion matrices, city/size slices, and the fixed off/unknown sensitivity policy.
[Artifact provenance](results/dtld_test_provenance.json) records completion,
prediction/report hashes, and the encoder-reuse check. Dataset images,
per-image membership manifests, and full-precision logits remain local under
`runs/v5/evaluations/dtld_test/`.

## Transfer benchmarks

Each selected DTLD fold checkpoint is evaluated unchanged on the same transfer
images. Primary tables report the arithmetic mean of the four individual model
metrics. Standard deviations are population standard deviations across these
four checkpoints; they are descriptive variability, not confidence intervals
over independent training repeats.

ATLAS uses the frozen 528-image manual-label snapshot, including uncertainty
flags in the primary benchmark. VZC uses the published 598-image test split as
primary; its train split and the combined set are separate diagnostics. No
transfer data is used to train weights, select the best epoch within a fold, or
fit temperature scaling. However, ATLAS and VZC results were inspected during
the choice of v5 as the retained architecture. These are therefore exploratory
transfer diagnostics; an independent confirmatory architecture evaluation is
still needed for that stronger claim.

The VZC evaluator additionally averages the four models' **raw logits**, then
applies one final softmax. This ensemble is a distinct predictor from a mean of
four metric values or a mean of probabilities. It is labeled separately in the
results. Unknown-only and exact-duplicate sensitivity analyses use explicitly
defined memberships; they do not replace the primary policy.

## Metric definitions

| Metric | Definition and interpretation |
|:--|:--|
| AP RR / RG / NoR | One-vs-rest average precision using the final softmax score for that class. Measures ranking across images. |
| mAP | Arithmetic mean of the three class APs. It is image-classification mAP, with no box-IoU thresholds. |
| Balanced accuracy | Mean class recall of the three-way argmax prediction. Measures the operating decision equally across classes. |
| Accuracy | Fraction of images with a correct argmax prediction. Depends strongly on class prevalence. |
| Precision / recall / F1 | Per-class argmax operating performance; macro F1 averages class F1 equally. |
| ECE | Sample-weighted absolute confidence/accuracy gap across 15 equal-width confidence bins, using maximum softmax confidence. Lower is better. |
| NLL | Mean negative log probability assigned to the true class. Lower is better. |
| Brier score | Mean sum of squared errors over the three class probabilities, without division by the number of classes. Lower is better. |

AP and balanced accuracy answer different questions, so their numerical values
are not directly comparable. A model can rank a class correctly while rarely
choosing it over the other classes. Report both, along with per-class recall and
precision. AP also depends on class prevalence, which changes substantially
between DTLD, ATLAS, and VZC.

All primary metrics use $T=1$. Training fits a scalar temperature on validation
logits and saves its value for optional inference. That calibration never uses
test or transfer labels. A positive scalar temperature preserves argmax, but
can change cross-image softmax rankings and therefore AP. Saved DTLD calibration
increased mean VZC-test ECE from 13.53% to 21.90%; it is not the primary transfer
score.

AP is undefined when a class has no positive or no negative examples. Such
values are serialized as JSON `null`. Generic slice reports average defined
APs and include `n_classes`. OOD reports label incomplete-class mAP as
`mAP_present_classes` rather than comparing it as full three-class mAP. DTLD
city and lamp-size slices with fewer than 20 examples or fewer than two classes
are skipped. Lamp-size buckets use maximum annotated lamp height in the frame;
they are diagnostic proxies, not object-detection evaluations.

## Output artifacts

- Training: `best.pt`, `last.pt`, `history.jsonl`, and validation reports.
- Cross-validation: membership/effective-config audits, per-fold reports and full-precision `val_predictions.npz`.
- DTLD test: frozen `plan.json`, `data_audit.json`, per-image `membership.json`, progress `status.json`, per-member and ensemble `report.json` / `predictions.npz`.
- Folder inference: `predictions.csv`, optionally RR/RG evidence overlays.
- Manual-label evaluation: `report.json` and full-precision logit CSV.
- VZC: frozen label audit, checkpoint/source hashes, reports, logit arrays, metric/per-class/confusion CSVs, and sensitivity reports.

`scripts/summarize_results.py` reconstructs public tables from the preserved v5
predictions. Its output directory defaults to `docs/results/`. These exports
contain full metric precision and portable identifiers; rounded percentages
appear only in Markdown tables.

For new reproduction outputs, `summarize_results.py` accepts `--city-cv`,
`--atlas`, `--vzc`, and `--dtld-test` directory overrides. The official-test
export verifies that recomputed member scores and averaged logits agree with
the saved reports and ensemble. The ATLAS directory must contain
`v5_fold0` through `v5_fold3`, each with `predictions.csv`. VZC evaluation writes
both per-model metrics and mean-model summaries automatically. Use a separate
`--out` directory to retain the recorded public tables alongside new results.

Original local evaluation plans retain their historical source hashes and path
records. They are preserved as provenance under `runs/v5/`, together with the
relocated checkpoint files. Public metadata and commands use the new layout.
The [refactoring verification](../metadata/refactor_verification.json) records
equivalence checks against predictions and gradients captured before cleanup.
