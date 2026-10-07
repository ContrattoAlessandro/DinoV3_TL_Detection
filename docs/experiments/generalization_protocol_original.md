# DTLD recall and generalization study

The retained v5 city models remain the reference. The axial pilot improved DTLD
test mAP but reduced transfer mAP on ATLAS and VZC. Side cropping also removed
label-defining lamps in 367 training and 139 test images. Legacy nearest-lamp
patch assignment supervised conflicting attributes in 6,420 training frames.
These observations motivate the changes below; they do not prove overfitting
or demonstrate that the proposed head improves metrics.

## Implementation

Native full-frame RGB JPEGs are aspect-preservingly letterboxed, starting at
720×1280. Image and box transforms are identical. Pure padding patches are
excluded from losses and MIL candidates; partially occupied boundary patches
remain available so boundary lamps retain supervision. Geometry describes the
unpadded source frame. The official RR/RG/NoR labels and city/session/frame
membership stay fixed. Fold manifests record membership and transformation
hashes separately. Legacy configs and strict checkpoint loading are preserved.

Lamp presence is the union of annotations. State, relevance, direction and
pictogram each receive a separate conflict mask. In the revised loss, each
annotated lamp contributes equally within its image, then images contribute
equally. Shared patches with agreeing attributes contribute to every covering
instance; conflicting attributes contribute to none. The existing auxiliary
coefficients remain, with pictogram CE added at 0.1 over the ten existing
classes and inverse-square-root weights capped at three.

Revised training uses one shuffled pass per epoch, global inverse-square-root
class weights normalized to mean example weight one, and zero logit adjustment.
Two photometric views share the same geometry and erasure. Consistency remains
0.1 and all-lamp erasure remains 2%. Zoom-out has probability 0.5 and scale
0.9–1.0, with placements that retain the entire frame. Each epoch records
component losses and a fixed, clean, class-stratified training probe alongside
validation metrics. The probe is diagnostic and never selects checkpoints.

`v7_evidence` retains frozen DINOv3 layer-6/final fusion, width 192 and dropout
0.2. A single trunk and lamp/state/direction/pictogram predictors feed k=1,2,4
pooling. A width-64 base relevance MLP sees appearance and predicted direction
and pictogram distributions. A separate width-64 correction sees appearance,
scene and geometry; its final layer starts at zero, its output is bounded to
±1 logit, and scene/geometry inputs drop together with probability 0.5.
There are no added absolute sinusoidal positions or direct scene class logits.
Signal evidence scales are positive and NoR explicitly decreases with either
signal score. Inference needs only RGB images and returns RR/RG/NoR logits.

## Frozen experiment protocol

| Pilot | Configuration | Change |
|---|---|---|
| A | `configs/generalization_a.yaml` | v5, full-frame inputs and attribute conflict masks |
| B | `configs/generalization_b.yaml` | A plus revised sampling, loss normalization and zoom-out |
| C | `configs/generalization_c.yaml` | B plus shared v7 head and pictogram supervision |
| D | `configs/generalization_d.yaml` | C at 896×1600 |

All pilots train fold 0 with seed 0 from fresh heads. AdamW uses lr 6e−5,
weight decay 0.05, 12 epochs, three warm-up epochs, effective batch 16 and EMA
0.999. Checkpoints maximize mean RR/RG validation recall subject to mAP at
least the corresponding retained v5 baseline and NoR recall no more than two
percentage points lower. Ties use mAP, then earlier epoch. `best.pt` exists
only after an eligible epoch; `best_unconstrained.pt` is a diagnostic artifact
that cannot be promoted. Failed runs remain unsuccessful.

The completed JPEG-to-probabilities pipeline is timed at batch one and four
before expensive pilots. D is rejected if either median exceeds 2× v5. The
winner is chosen with the same validation rule; its other three city folds use
seeds 1–3. This bounds full training to seven runs. Any failed winning fold
prevents promotion. Sources, configuration, inputs and selected checkpoints
are frozen before official DTLD test and exploratory ATLAS/VZC evaluation.
Primary metrics use raw logits at T=1, without external selection, thresholds
or calibration. Explicit model selection follows the principle discussed in
[DomainBed](https://arxiv.org/abs/2007.01434).

Promotion requires the four-head ensemble to achieve all of these:

- DTLD mAP +≥1 point and mean RR/RG recall +≥2 points, neither signal recall
  declining, and NoR recall falling by at most two points.
- ATLAS and VZC mAP and macro F1 falling by at most one point, with mean RR/RG
  recall preserved.

Individual heads and equivalent four-head ensembles are reported separately.
Validation predictions use each image's held-out head only. Reports include
per-class precision/recall, confusion matrices, ECE/NLL/Brier, city/camera
slices where identifiers are available, native relevant-lamp size, and
off/unknown sensitivity. DTLD uncertainty uses 2,000 paired complete-session
bootstrap samples. This estimates sampling uncertainty conditional on the
weights. Differences across city folds also change seed and training cities,
so they cannot isolate training-seed variability. ATLAS/VZC remain exploratory
because they have already influenced development.

## Execution and artifacts

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -B -u scripts/run_generalization_study.py
# Continue the same immutable study after interruption:
python -B -u scripts/run_generalization_study.py --resume
```

The default output is `runs/generalization_study`. Read `status.json` and the
current stage log for progress. `pilots.json`, `pilot_winner.json`,
`frozen_candidate.json`, `comparison.json`, `decision.json`, `RESULTS.md` and
CSV exports preserve the decisions and measurements. `decision.json` records
whether promotion passed; v5 artifacts are never replaced or deleted. A
failed criterion retains v5 and identifies the unmet metric, without claiming
that the metric alone establishes its causal failure mechanism.

The CPU suite verifies geometry, conflict masks, instance weighting, padding,
selection gates and monotonic NoR. `scripts/smoke_generalization.py` exercises
real CUDA two-view training, clean-probe logging, strict loading and bit-exact
live/EMA resume. `scripts/benchmark_generalization.py` measures the complete
inference pipeline. Early smoke metrics are functional checks, not study
results.

## Post-study diagnostic tests

After all four pilots failed validation eligibility, the user requested testing
each model on DTLD, ATLAS and VZC_TLD. These evaluations are stored separately
in `runs/generalization_study/diagnostic_tests`; the original retention decision
and training artifacts stay intact. The fixed diagnostic EMA checkpoints are
A epoch 11, B epoch 9, C epoch 5 and D epoch 3, selected exclusively by fold-0
validation mAP before test inference.

```powershell
python -B -u scripts/evaluate_generalization_pilots.py
# Continue the same frozen evaluation after interruption:
python -B -u scripts/evaluate_generalization_pilots.py --resume
```

DTLD covers all 12,453 official test frames. VZC_TLD covers only the 598
published test images. ATLAS uses the user-confirmed 528-image manual benchmark;
its relevance labels are not official ATLAS test ground truth. All predictions
use raw logits at T=1. The matched comparison is the retained v5 fold-0 model;
the other v5 folds, four-head reference ensemble and previous axial pilot are
reported separately using verified cached predictions. The three 720×1280
models share encoder computations only after bit-exact agreement with separate
complete-model forwards; D uses its own 896×1600 inputs. No A–D ensemble or
test-based promotion is performed.

Read `diagnostic_tests/status.json` for progress and `diagnostic_tests/README.md`
for the completed tables. `summary.json`, `metrics.csv`, `per_class.csv`,
`confusion_matrices.csv`, dataset/model reports and raw predictions retain the
full measurements. `plan.json` freezes checkpoints, input membership, sources
and the original decision hash.
