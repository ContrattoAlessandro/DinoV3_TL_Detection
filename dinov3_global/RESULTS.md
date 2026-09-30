# dinov3_global — Final Results Report

**Run:** `runs/exp2` (seed 0) · stopped manually at epoch 9/30 after a 5-epoch plateau · `best.pt` = epoch 8
**Test:** official DTLD test split, evaluated **once** (2026-09-29) via `scripts/evaluate.py` → `runs/eval_test_exp2.json`
**Hardware:** RTX 5070 12 GB · ~19 min/epoch · ≈3 h trained (of a planned 8.5 h)

---

## 1. Headline: one-shot test results (n = 12,453)

| Metric | This model | Trinci *Faster-RCNN w/attr* (Table II) | Trinci *end-to-end* | Beat? |
|---|---|---|---|---|
| mAP | **64.8** | 81.5 | 83.1 | ❌ no (−16.7) |
| acc_bal | **68.9** | 79.3 | 80.8 | ❌ no (−10.4) |

Per-class AP: **RR 87.7 · RG 93.7 · NoR 12.9**

Baseline comparison (session-disjoint **val**, like-for-like):
- exp1 (legacy 1-query decoder head): val mAP **61.5** @ ep15, AP_NoR 13
- exp2 (TokenMILHead, this run): val mAP **66.5** @ ep8, AP_NoR 14 → **+5.0 pts mAP** over the only baseline trajectory available

Val→test generalization gap: −1.7 pts (66.5 → 64.8) — small and healthy; no test-time fitting, no test-driven selection.

## 2. Where the gap lives

mAP = (AP_RR + AP_RG + AP_NoR)/3. RR (87.7) and RG (93.7) are at their
label-noise ceiling; the entire deficit to 81.5 is **AP_NoR = 12.9**.
Reaching 81.5 with RR/RG at current levels would require AP_NoR ≈ 0.63 —
not achievable in this run: NoR is ~5% of frames, must prove *all* lamps
irrelevant, and was flat at 0.13–0.15 from epoch 2 onward.

Confirmed failure mode (val, instrumented — see `scripts/diagnose.py` and
`CONTEXT_v3.md` §4 for the full diagnosis):

1. **The per-token relevance estimate is the bottleneck.** On NoR frames the head
   emits confident red-lamp evidence (max-token evidence > 0.5) in **23.5%** of cases
   — *more often than on true RG frames (13.1%)** — although the ground truth says NoR
   frames contain ~0.8 relevant lamp tokens. The pooled score of NoR frames (0.155)
   therefore exceeds that of RG frames (0.086).
2. **The NoR readout is not the bottleneck** (this refuted the original hypothesis):
   every evidence-derived NoR score is no better than the current CLS-only head
   (`1−max(S_rr,S_rg)` 0.153 vs learned 0.139), and a linear probe on
   `[CLS feats ‖ S ‖ emax]` scores 0.088. Feeding `S` into the NoR MLP is therefore not
   the fix.
3. **Relevance supervision is thin and 30% corrupted**: rel/state losses run on only
   0.28% of tokens (9.9 lamp tokens/frame of 3,600), get no negative examples from the
   background, and `random_framing` (p=0.3) crops the image while the token targets stay
   rasterised on the uncropped frame — top-1 lamp token inside a GT box falls from
   0.780 to 0.005 after one crop.
4. **Label policy fights the metric** for the hardest frames: the 33% of NoR frames that
   are pseudo-NoR (relevant lamp off/unknown) are downweighted to 0.3 in training yet
   counted fully by AP_NoR; their AP is 0.042 vs 0.115 for clean NoR.
5. Logit adjustment (τ=1.0) requires the raw NoR logit to beat the others by ~3.0
   during training — arguably too aggressive for a noisy tail class.
6. One of the two MIL branches is dead by construction: `tau·log N` = 1.82 at the learned
   tau = 0.223, so its `S_rr` separation is 0.053 vs 0.473 for branch 0.
7. Data limits: 1,202 NoR train frames (4.98%), and **zero lamp-free NoR frames** — the
   "empty road" case that matters for other datasets is never seen in training.

The single fix-run (approved scope, not executed per user decision to finalize):
`sampler: sqrt` + `tau_la: 0.5`, optionally feeding `[CLS, S_RR, S_RG]` to the
NoR MLP. Realistic expected outcome: AP_NoR 0.3–0.5 → test mAP ≈ 72–77 — still
short of 81.5, but materially fewer false "relevant lamp present" claims on
empty-road frames in external datasets.

## 3. Slices (test)

**Per city** (selection-free; test cities were never used for model choice):

| city | mAP | acc_bal | n | | city | mAP | acc_bal | n |
|---|---|---|---|---|---|---|---|---|
| Bochum | 0.897 | 0.703 | 109 | | Essen | 0.641 | 0.578 | 781 |
| Hannover | 0.679 | 0.700 | 2408 | | Berlin | 0.644 | 0.724 | 1077 |
| Frankfurt | 0.696 | 0.738 | 1677 | | Dortmund | 0.647 | 0.669 | 2027 |
| Fulda | 0.674 | 0.638 | 108 | | Koeln | 0.619 | 0.643 | 1898 |
| Duesseldorf | 0.676 | 0.654 | 1389 | | Kassel | 0.566 | 0.532 | 757 |
| | | | | | Bremen | 0.532 | 0.429 | 222 |

Spread is mostly sample-size noise at the extremes (Bochum/Fulda/Bremen n≈100–220);
large-n cities cluster 0.62–0.70. No catastrophic city failure.

**Per lamp size:** small(<16px) **0.596** < mid(16–48px) 0.656 ≈ large(≥48px) 0.642.
Small distant lamps are hardest, as expected; this is the slice most relevant to
other datasets with different camera optics.

## 4. Calibration

Temperature scaling on val (no test use): **T = 0.562**, test **ECE = 0.028**
(raw ECE ≈ 0.17 before scaling). Predictions are well-calibrated — confidence
values from `infer_folder.py` can be read approximately as probabilities.

## 5. Configuration & rationale (what this run tested)

- **Backbone:** frozen DINOv3 ViT-S+/16 (`facebook/dinov3-vits16plus-pretrain-lvd1689m`), dim 384 — zero backbone training cost, strong transfer features; the RTX 5070 cannot afford end-to-end ViT training at useful batch sizes.
- **Head:** TokenMILHead — per-token lampness/state/relevance maps + log-sum-exp smooth-max MIL pooling (evidence equation), CLS-based NoR head; replaces the exp1 1-query cross-attn decoder (kept as `decoder.head: attn`).
- **Training-only aux supervision:** DTLD boxes → 45×80 token targets (lamp BCE 0.3 / state CE 0.2 / rel BCE 0.2, 16 px ignore band). **Inference is image-in → image-out: no boxes, no detector, no test-time fitting.**
- **Loss:** logit-adjusted CE (τ=1.0, Menon et al.) instead of inv-freq weights; `label_policy: downweight` (0.3) for pseudo-NoR; two-view consistency KL (0.1).
- **Augmentation:** label-preserving only (WB gains, gamma, contrast, saturation, noise, JPEG, blur, fog, framing). Hue rotation deliberately excluded — it would corrupt red/green semantics.
- **Ops:** uint8 decode (exp1 died of host-RAM OOM on float32 decode), resume + auto-restart watchdog (2 mystery process kills survived without data loss), EMA, per-epoch checkpoints.

Observations from the trajectory: train loss fell monotonically (1.14 → 0.63);
val mAP plateaued at ~0.66 from epoch 5; consistency loss stayed small
(0.006 → 0.032); ECE improved every epoch. The plateau is a tail-class
failure (NoR), not classic overfitting of RR/RG.

## 6. Limitations (carry into any use of these numbers)

1. **AP_NoR is the known weak class** (test 12.9). Treat "no relevant lamp"
   claims as low-confidence; a frame-level "no lamp" decision should threshold
   conservatively or fall back to the lamp heatmaps.
2. AP_NoR is statistically noisy: only 170 val / ~600 test NoR samples.
3. Relevance semantics assume **right-hand-traffic** conventions (RR = relevant
   to the driver's lane/direction as defined in DTLD).
4. DTLD contains **no rain**; weather robustness is simulated by augmentation only.
5. Stopping at epoch 9 forfeited the cosine LR tail; the reported numbers may
   slightly understate a full 30-epoch run (val was flat for 5 epochs, so the
   expected margin is small and mostly RR/RG).
6. Single seed, single run — no ensemble/soup (dropped per compute budget).
7. exp1 baseline numbers are val-only (that run crashed before any test eval),
   so the +5.0 improvement is a val-to-val comparison.

## 7. External-dataset inspection (manual QA)

The model is image-in → image-out; for unlabeled datasets:

```bash
python dinov3_global/scripts/infer_folder.py \
  --ckpt dinov3_global/runs/exp2/best.pt \
  --images <folder_with_images> \
  --overlays                    # add --crop-sides 114 for 2048x1024 DTLD-style images
```

Outputs `predictions.csv` (per-image RR/RG/NoR probabilities + confidence) and
overlay images (lamp/relevance/RR/RG heatmaps) into the run folder. **Recommended
review order:** (1) empty-road frames — watch for false "relevant" claims
(the NoR weakness); (2) small/distant lamps (small-size slice was weakest);
(3) unusual weather/color casts (trained via simulated aug only).

## 8. Deferred / not executed

- Fix-run for NoR (`sampler: sqrt`, `tau_la: 0.5`, NoR-MLP sees pooled evidence) — scoped, approved in principle, skipped by user decision to finalize.
- City-held-out CV (`scripts/crossval_city.py`), label-policy ablation, strict old-head baseline, multi-seed ensemble (`evaluate.py` supports multi-ckpt ensembling if checkpoints ever exist).
