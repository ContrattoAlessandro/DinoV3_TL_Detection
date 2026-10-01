# DTLD and OOD review — 30 September 2026

The main issue is **rare-class relevance reasoning and dependence on scene
features**, not simply too few epochs or too little dropout. A safer training
pipeline and a smaller, constrained v5 head are implemented. **V5 is an
experimental candidate, not a demonstrated accuracy improvement.** It passed
unit tests and a tiny real GPU training/validation/checkpoint smoke run. No
full v5 result is available yet. The paired city comparison is now launched
as described in [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md); no new official-test
evaluation has been run.

## Evidence from this workspace

The current model freezes DINOv3 ViT-S+/16, concatenates layer 6 and final-layer
features, then trains three token-MIL branches (local pooling, k=1/2/4). Each
branch predicts lampness, six states, relevance and four housing directions.
RR/RG evidence is lamp × relevance × state. NoR has an MLP using CLS plus pooled
evidence; CLS can also add unrestricted RR/RG logit offsets.

| Saved experiment | DTLD validation mAP | Saved DTLD test mAP | Test AP NoR | CARLA color-label agreement* |
| --- | ---: | ---: | ---: | ---: |
| exp2 | 66.5% | 64.8% | 12.9% | 193/309 (62.5%) |
| exp3 | 68.2% | 66.0% | 15.8% | 159/309 (51.5%) |
| exp4 | 71.7% | 67.4% | 18.2% | 62/309 (20.1%) |

*CARLA's COCO categories describe color, not ego relevance. These agreements
come from existing external `_metrics.csv` files and are **proxy diagnostics**,
not valid RR/RG/NoR benchmark accuracy. ATLAS now has reviewed manual labels
and measured checkpoint results in [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).
There are **28 images** in
the provided ATLAS tree, including rain and three camera views.

Exp4 test AP RR/RG is already 89.8%/94.1%. Holding those fixed, 81.5% mAP would
require approximately **60.6% AP NoR**, much higher than the current 18.2%.
Moderate regularization alone is unlikely to close that entire gap. Validation
contains only **170 NoR frames**, and adjacent frames can share an intersection;
the effective number of independent NoR scenes is smaller. Exp4 peaked at epoch
10 and stopped at 22; NoR validation AP falls from 27.4% at the selected epoch
to 18.2% on the saved test evaluation. This is consistent with limited transfer
and noisy model selection; it does not isolate one causal mechanism.

### Measured scene dependence

I reran exp4 on all 309 CARLA test images with cached weights. Its original
prediction counts exactly match the saved external run. A diagnostic
counterfactual preserves the token evidence and replaces CLS only at the NoR
MLP input with zero, retaining its two evidence inputs:

| Diagnostic setting | Predicted RR | Predicted RG | Predicted NoR |
| --- | ---: | ---: | ---: |
| Original exp4 | 58 | 6 | 245 |
| Remove direct RR/RG scene prior | 69 | 55 | 185 |
| Neutral CLS in NoR head | 162 | 104 | 43 |
| Both changes | 164 | 137 | 8 |

Mean NoR logit decreases from 2.43 to −0.88 when CLS is neutralized, despite
retaining lamp evidence. The mean direct scene offsets are −0.57 for RR and
−1.33 for RG. This supports constraining the scene paths. **Zeroing components
of a trained model is out of its training distribution:** the counterfactuals
are not validated replacement models and do not establish correct relevance.

Reproduce with `scripts/audit_ood_context.py`. Full per-image diagnostics are
in `runs/context_carla_exp4/per_image.csv` and `summary.json`.

## Implemented changes

### Training safety (also applies to legacy configurations)

- Both consistency views retain the full frame. The previous framing crop
  could remove the only relevant signal while preserving its RR/RG label.
  That is contradictory supervision for an existential label. Cropping remains
  available only as an explicit low-level legacy ablation.
- The engine now honors augmentation strength and enable switches. Previously
  it read `p_erase`, but called `train_view` with default strength and both
  photometry/degradations enabled, regardless of the other YAML settings.
- Augmentation can draw independently per image, preserve clean examples and
  blend transformed images with original pixels. No hue rotations or flips are
  introduced. This reduces destructive chains on very small lamps.
- Optional second-view CE and a consistency warm-up prevent the second view
  from being constrained only by another uncertain prediction.
- Resume preserves historical best score and early-stop patience; incompatible
  data/head/loss/augmentation configurations are rejected. Fresh training into
  an existing run directory is rejected to prevent checkpoint overwrite.
- City-held-out runs now seed Python, NumPy and Torch together, including the
  Python-random photometric draws, for reproducible baseline comparisons.

Historical configuration files and checkpoint layouts are retained, but rerunning
them now uses the corrected view safety behavior. Exact historical training
reproduction would require the previous code revision. Legacy inference remains
compatible: exp2/exp3/exp4 all loaded strictly and passed GPU inference.

### V5 architecture and loss hypotheses

`configs/v5.yaml` retains the frozen backbone, mid-layer fusion, direction task
and local pooling. The head has **489,475 parameters vs 734,275 in v4** (33%
fewer). Its projection is 192-dimensional with dropout 0.2.

- **Appearance separate from position:** lamp/state/direction heads receive
  visual features without the added absolute positional encoding. Position and
  geometry enter relevance. This cannot make DINO features perfectly invariant,
  but removes a direct positional shortcut from appearance classification.
- **Controlled context:** a zero-initialized, bounded scene gate conditions
  relevance features. CLS is dropped per frame with probability 0.3 in training.
  Direct RR/RG scene offsets are limited to ±0.5 logits.
- **NoR anchored to evidence:** its baseline MLP input is `[zero CLS; S_RR; S_RG]`.
  Scene context adds a residual bounded to ±0.5 logits. This addresses the
  measured CARLA failure while preserving a limited scene correction.
- **Each branch learns its own attributes:** auxiliary losses are averaged
  across branches. V4 supervised their averaged raw logits even though global
  evidence multiplies probabilities separately inside each branch. Opposing
  branch errors could therefore cancel in supervision.
- **Actual lamps get explicit relevance loss mass:** 75% on real lamp tokens,
  25% on valid background, with positive weight 2 within the lamp group. Lit,
  irrelevant lamps no longer compete for loss mass with thousands of easy
  background tokens. These weights need validation; they are not derived optima.
- Lamp erasure is reduced from 6% to 2%. A filled hole is a synthetic artifact,
  so it should be a small supplement rather than a large source of NoR examples.

`configs/v4_safe.yaml` uses the v4 head/loss with the same view augmentation,
second-view CE and consistency warm-up as v5. Compare these configurations to
isolate the v5 head and auxiliary-loss changes. Do not resume an old v4 head
into v5: its dimensions differ.

The existing sampler/logit-adjustment pair is preserved as a controlled choice.
It is a heuristic combination: the actual training prior changes after
oversampling and erasure. Balanced-error theory for logit adjustment does not
automatically establish optimal mAP or calibration under those modifications.

## What the literature supports

- [Color Is Not Enough](https://flore.unifi.it/retrieve/5ae436fa-8236-4bb6-8794-05fabc4ae1f2/Color_Is_Not_Enough_Dataset_and_Method_for_Identifying_Relevant_Traffic_Lights_in_Driving_Scenes.pdf)
  supports jointly reasoning about traffic-light attributes and global action.
  It reports that NoR is particularly difficult, and that its trained models
  also transfer poorly between DTLD and VZC-TLD. Strong DTLD scores alone do not
  establish cross-dataset generalization. Its reported comparisons average five
  runs; this workspace's saved experiments are single-seed results.
- [AugMix](https://arxiv.org/abs/1912.02781) motivates diverse augmentation,
  retaining original image content and consistency regularization. This
  implementation is a task-specific two-view adaptation, **not full AugMix**;
  its benchmark improvements should not be assumed to transfer quantitatively.
- [DomainBed / In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)
  motivates carefully controlled baselines and a specified model-selection
  protocol. DTLD city-held-out validation is a useful proxy for transfer, not
  proof of performance on a new sensor or country.
- [SWAD](https://arxiv.org/abs/2102.08604) motivates averaging suitable training
  iterates. The current engine already uses EMA. SWAD is a potential follow-up
  after the view/scene problems are tested; it is not implemented here.

## Reusable ATLAS annotation and evaluation

Start from the repository root:

```powershell
python dinov3_global/scripts/label_atlas.py
```

Open <http://127.0.0.1:8787>. The server binds only to localhost, reads the source
images and existing ATLAS state boxes, and writes
`dinov3_global/runs/atlas_labels.json`. Labels autosave atomically with a previous
revision backup (`.json.bak`). Image hashes prevent accidental reuse with changed
images; concurrent tabs cannot silently overwrite a newer revision.

Use **1/2/3** for RR/RG/NoR, arrows to navigate, U for next unlabeled, zoom for
small lamps, and **uncertain** when ego maneuver/relevance cannot be established.
RR takes priority over green if both relevant signals exist. Off/unknown without
a relevant stop/go signal follows this repository's NoR policy. Existing state
boxes do not imply relevance. Keep model predictions hidden while annotating.

After labeling, evaluate any checkpoint without inspecting every image again:

```powershell
python dinov3_global/scripts/evaluate_ood.py --ckpt dinov3_global/runs/exp4/best.pt --out dinov3_global/runs/eval_atlas_exp4
```

The script excludes unlabeled/uncertain images by default, checks image hashes,
and reports class AP, balanced accuracy, confusion matrix, class counts and
camera slices. It does not report a three-class mAP if a class is missing.
Checkpoint evaluation uses raw logits at T=1; new `infer_folder.py` CSVs preserve
raw logits alongside calibrated probabilities. Existing prediction CSVs can be
evaluated with `--predictions`, but their rounding/calibration may affect AP.
Labels are never loaded into training. Twenty-eight correlated images are a
small diagnostic set; repeatedly selecting models on it makes it development
data rather than an untouched transfer benchmark.

## Recommended experiment sequence

1. Label ATLAS once, keeping uncertain examples separate. Retain exp2 and exp4
   as reference models; report color-proxy CARLA results separately from manual
   relevance results.
2. Compare v4_safe and v5 on the same city-held-out folds and seed. Use all
   classes in each scored fold; inspect AP NoR and class counts as well as mAP.
   The city script uses only official DTLD training images.
3. If v5 improves held-out cities while maintaining RR/RG, ablate scene bounds,
   per-branch supervision and hard-lamp loss separately. Then repeat with 2–3
   seeds. Do not increase backbone size until these simpler causes are tested.
4. Lock the configuration and epoch budget using city validation. For a
   comparison using the paper's training volume, fit on all official training
   images, then evaluate the official held-out split after verifying the
   off/unknown label policy. A session-split final run should be described as
   using less training data. Repeat seeds before claiming a replicated result.

```powershell
python dinov3_global/scripts/compare_city.py --out dinov3_global/runs/city_comparison_20260930 --folds 4 --epochs 12
# After an interrupted supervisor/child has exited:
python dinov3_global/scripts/compare_city.py --out dinov3_global/runs/city_comparison_20260930 --folds 4 --epochs 12 --resume
```

The full paired city experiments are **started**; subsequent ablations and
final training have not started. If improvement stalls, the next
architectural step should be lamp proposals/ROI features or higher-resolution
lamp features: patch-16 tokens still conflate nearby small signal heads.
Increasing global CLS capacity is poorly matched to the observed failure.
ATLAS also includes rain and different fields of view. Pure resize stretches
those inputs; changing inference to letterbox alone is not a reliable fix without
matching training geometry and relevance coordinates.

## Verification performed

- 43 CPU correctness/regression tests pass (legacy units plus robustness and
  annotation tests).
- Final v5 completed a GPU training/validation smoke run (32 train, 128 val,
  one epoch); its tiny-run metrics are not a performance estimate.
- Exp2, exp3, exp4 and the final v5 smoke checkpoint load strictly and produce
  finite GPU inference outputs.
- A headless browser verified real ATLAS rendering, boxes, hidden predictions,
  zoom and navigation with no JavaScript errors. A separate fixture verified
  keyboard labeling, notes, autosave, reload and uncertainty.
- Python compilation and `git diff --check` pass. Source OOD images are unchanged.
