# Experiment history

Experiment C became the default on **7 October 2026** at the user's direction.
Only its architecture is implemented in the current source tree. This archive
preserves what was tried, how checkpoints were selected, and what each achieved.
Original source is available at Git commit `54a28b4` (before this cleanup).

## Architectural progression

All variants use a frozen DINOv3 ViT-S+/16 and layer-6/final fusion. Differences
refer to trainable heads, supervision, and preprocessing.

| Experiment | Architecture / training change | Input | Head parameters | Recorded training |
|:--|:--|:--|--:|:--|
| v5 | Three independent attribute-MIL branches (k=1/2/4); CLS gate, absolute relevance positions, bounded direct scene logits; cropped/resized DTLD, replacement sampling and logit adjustment. | 720×1280 | 489,475 | Four 12-epoch city folds |
| Axial (v6) | v5 plus two shared row/column attention blocks affecting relevance only. | 720×1280 | 1,380,739 | One 12-epoch fold-0 pilot |
| A | v5 head with full-frame letterboxing, content-relative geometry, independent attribute conflict masks, and clean training probe. | 720×1280 | 489,475 | One 12-epoch fold-0 pilot |
| B | A plus natural sampling, no logit adjustment, inverse-sqrt image-class weights, per-instance attribute losses, and full-frame zoom-out. | 720×1280 | 489,475 | One 12-epoch fold-0 pilot |
| **C (default)** | B training with one shared appearance/attribute trunk, 10-class pictogram supervision, width-64 base relevance and bounded context correction; positive signal readout and decreasing NoR logit. | 720×1280 | **265,374** | One 12-epoch fold-0 pilot |
| D | C with higher input resolution and a 56×100 patch grid. | 896×1600 | 265,374 | One 12-epoch fold-0 pilot |

For v5, each branch has its own attribute maps and relevance predictor. C shares
one map across k=1/2/4 pooling. C removes the head-added absolute positions,
CLS multiplicative gate, and direct scene-to-class residuals; its scene and
content geometry contribute only through a ±1 relevance-logit correction.

## Selection history

The v5/axial pilots select EMA epochs by validation mAP. The original A–D study
maximized mean RR/RG validation recall subject to mAP at least v5 and NoR recall
no more than two percentage points lower. All four failed eligibility. The
original [decision](generalization/decision.json) retains v5 and remains unchanged.

Subsequent diagnostic evaluations fixed A/B/C/D at their highest validation-mAP
epochs **11/9/5/3** before official DTLD/ATLAS/VZC inference. C's later adoption
as the repository default is a separate development decision, informed by these
comparisons. It does not turn a diagnostic checkpoint into a successful gated
study. New C training selects validation mAP without a legacy baseline dependency.

## Matched single-head results

Each model below is one fold-0 EMA head, at T=1. Values are percentages.
The same images/labels are aligned across models in each benchmark.

| Model | DTLD mAP | DTLD macro F1 | DTLD signal recall | DTLD NoR recall | ATLAS mAP | ATLAS macro F1 | VZC mAP | VZC macro F1 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| v5 fold 0 | 70.00 | 68.80 | 84.25 | 57.90 | 81.22 | 73.73 | 68.83 | 61.13 |
| Axial fold 0 | 71.58 | 72.85 | 88.55 | 52.00 | 68.09 | 65.46 | 58.29 | 46.51 |
| A | 70.24 | 68.82 | 84.37 | 55.05 | 79.00 | 73.09 | 67.49 | 62.06 |
| B | 72.11 | 72.30 | 89.37 | 38.67 | 81.47 | 77.16 | 69.74 | 66.32 |
| **C** | **70.11** | **70.04** | **90.38** | **25.52** | **86.84** | **71.97** | **72.17** | **66.18** |
| D | 69.96 | 65.51 | 91.69 | 9.52 | 86.93 | 64.06 | 75.63 | 62.12 |

Signal recall is the mean of RR/RG recall. C is a compact development choice
with strong transfer ranking and signal recall; its NoR recall is a material
limitation. Dataset-dependent tradeoffs prevent an unqualified best-model claim.
The old four-head v5 ensemble is preserved separately, and is not a matched
comparison with a single C head.

## Archive contents

- [Complete diagnostic comparison](diagnostics/README.md): all A–D, axial, v5 folds, and v5 ensemble results.
- Diagnostic [metrics](diagnostics/metrics.csv), [per-class results](diagnostics/per_class.csv), [confusion matrices](diagnostics/confusion_matrices.csv), [full comparison](diagnostics/comparison.json), and [frozen plan](diagnostics/plan.json).
- [Original configurations](configurations/README.md): exact JSON snapshots of all six configurations, including the old default.
- [Original generalization protocol](generalization_protocol_original.md), [pilot outcomes](generalization/pilots.json), [latency gate](generalization/latency_gate.json), [benchmarks](generalization/benchmark.json).
- Training histories: [A](generalization/A/fold0/history.jsonl), [B](generalization/B/fold0/history.jsonl), [C](generalization/C/fold0/history.jsonl), [D](generalization/D/fold0/history.jsonl), [v5 fold 0](v5/training/fold0/history.jsonl), [axial fold 0](axial/training/fold0/history.jsonl).
- [v5 results](v5/results/README.md), [v5 architecture snapshot](v5/architecture_original.md), [axial results](axial/results/README.md).
- [Preservation manifest](preservation_manifest.json): original path, archive path, and SHA-256 for every preserved artifact.

Frozen snapshots retain historical paths, source hashes, terminology, and
decisions; commands inside them refer to their original Git revision. They are
research records, not entry points for the current package. Original per-image
logits, long console logs and evaluation membership files remain in the ignored
local `runs/archive/`. Checkpoint identities remain in the plans; obsolete
non-C checkpoint files were removed after preserving their records.

To recover an old implementation in a separate checkout, use
`git worktree add --detach ../traffic-light-history 54a28b4`.
Historical configurations are intentionally stored as records rather than
supported runnable configuration choices. Reproduce new work with the current
[default C protocol](../evaluation.md).
