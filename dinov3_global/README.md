# dinov3_global v2 — DinoGlobal-MIL: frozen DINOv3 + token-level MIL head

**Latest robustness iteration:** [ROBUSTNESS_REVIEW.md](ROBUSTNESS_REVIEW.md)
documents the measured CARLA scene-context failure, corrected augmentation,
experimental `configs/v5.yaml`, controlled `configs/v4_safe.yaml` baseline, and
`scripts/label_atlas.py` / `scripts/evaluate_ood.py` for reusable manual labels.
V5 has passed smoke verification; full-training accuracy is not yet measured.

[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md) contains the reviewed ATLAS labels,
fresh exp2/exp3/exp4 results, paper-protocol differences and the running paired
city comparison. `scripts/compare_city.py` queues both variants on the same
checked folds with a matched schedule and resumable progress.

[THROUGHPUT_REPORT.md](THROUGHPUT_REPORT.md) documents the measured loader
speedup and the checks preserving sampling, random states and GPU updates.

> **Start here for the next iteration:** [`CONTEXT_v3.md`](CONTEXT_v3.md) (measured
> exp2 diagnosis + v3 plan) and [`AUDIT_v3.md`](AUDIT_v3.md) (**code audit: two
> token-supervision bugs found & fixed, 18-test correctness suite, the exp3
> change set and run state**). `RESULTS.md` is the frozen exp2 test report;
> `scripts/diagnose.py` regenerates the diagnosis from any checkpoint.

DTLD global **RR/RG/NoR** classification, targeting paper Table II image-level
metrics (`AP_RR/AP_RG/AP_NoR/mAP/acc_bal`). Beat check: Faster-RCNN w/attr
`81.5 mAP / 79.3 acc_bal`; Trinci et al. end-to-end `83.1 / 80.8`
(*Color Is Not Enough*). Test split is loaded **once**, by `scripts/evaluate.py`.

## Why v2 (design rationale)

exp1 (single cross-attention query over 3600 patch tokens) plateaued at
val mAP 0.615 with AP_NoR 0.13 and died of a host-RAM OOM. Diagnosis:

1. **Wrong inductive bias.** "RR" = *∃ a relevant red lamp* — an existential
   quantifier. Softmax attention is a weighted **mean** over tokens and dilutes
   evidence; MIL literature shows max/log-sum-exp pooling is what preserves
   "any instance" presence semantics (Ilse 2018; Kowalski 2019).
2. **Tiny objects.** Lamps are 1–2 patch tokens at 1280×720; pooling averages
   lamp evidence with background. Instance-level supervision (DTLD boxes,
   training-only) teaches the head where lamps are.
3. **Imbalance.** NoR = 5% of frames; 1/freq weights are not consistent for
   balanced error. Logit adjustment (Menon et al., ICLR 2021) is — and
   `acc_bal` *is* balanced error.
4. **Transfer.** Frozen foundation features generalise (Frozen-DETR; DINOv3);
   full fine-tuning does not in low-data regimes (LoRA ≫ FT). Augmentation must
   be label-preserving physics (ISP/white-balance/gamma/noise/JPEG/fog), never
   hue rotation (would flip red↔green semantics).

## Model

```
frozen DINOv3 ViT-S+/16 (29M, dim 384, fp16, sdpa) @1280x720
  -> 3600 patch tokens + CLS
  -> per-token heads (2 averaged branches):
       lamp_t  = sigmoid(·)                    lampness
       state_t = softmax(·, 6)                 green/off/red/red_yellow/unknown/yellow
       rel_t   = sigmoid(· [h_t; geo_t])       relevance (geo = x, y, bearing, 1-y)
     evidence e^RR = lamp*rel*(red+yellow+red_yellow)
              e^RG = lamp*rel*green
     pooling  S_c = tau * (logsumexp(e_c/tau) - log N)      # smooth max in [0,1]
     logits   [a·S_RR+b, a·S_RG+b, MLP(CLS)]                # NoR = global reasoning
```

Losses: logit-adjusted CE (`tau_la·log pi`, label smoothing 0.05) +
`0.3·lamp BCE + 0.2·state CE + 0.2·rel BCE` (DTLD box rasterised to the 45×80
token grid with a 16px ignore band) + `0.1` two-view consistency KL.
**Inference is image-in → image-out**: boxes/supervision never needed at test
time, so the model runs unchanged on unlabeled foreign datasets.

Legacy `decoder.head: attn` (exp1 single-query) is kept for ablation.

## Label policy

Relevant `off/unknown` lamps (~3.6k train) carry no stop/go signal and can only
make a frame NoR ("pseudo-NoR" label ambiguity). `data.label_policy`:
`map_to_nor` (exp1) | `downweight` (default, weight 0.3) | `exclude`.

## Usage (repo root as CWD)

Install with `python -m pip install -r dinov3_global/requirements.txt`.
The native DTLD annotation parser is bundled in `dinov3_global/dtld_native.py`;
the experiment code has no dependency on other research folders.

Raw DTLD conversion tools live in `scripts/convert_tif.py` and
`scripts/convert_tif_sequential.py`, with their supporting parser in
`scripts/dtld_parsing/`. Install `requirements-preprocessing.txt` for these
tools, and run either script with `--help` for input and output arguments.
`scripts/make_1280.py` prepares the cropped image dump from the converted JPEGs.
The shared `datasets/DTLD`, `datasets/DTLD_jpg`, and `datasets/DTLD_1280`
directories remain at the repository root.

- Bench: `python dinov3_global/scripts/bench.py`
- Train: `python dinov3_global/scripts/train.py --config dinov3_global/configs/base.yaml --out dinov3_global/runs/exp2`
  (resume: `--resume dinov3_global/runs/exp2/last.pt`)
- City-held-out CV (generalisation gate, short runs): `python dinov3_global/scripts/crossval_city.py --folds 4 --epochs 12`
- **Final test (run ONCE, after all selection):**
  `python dinov3_global/scripts/evaluate.py --ckpt dinov3_global/runs/exp2/best.pt`
  (multiple `--ckpt` = logits ensemble)
- Token heatmaps: `python dinov3_global/scripts/infer_attn.py --ckpt ... --max-frames 20`
- **Unlabeled foreign datasets:** `python dinov3_global/scripts/infer_folder.py --ckpt ... --images <folder> --overlays`
  → `predictions.csv` (P_RR/P_RG/P_NoR) + lamp/rel/rr/rg heatmaps for manual
  inspection. Use `--crop-sides 114` for 2048×1024 DTLD-style captures.

First DINOv3 download requires accepting the license on HuggingFace
(`facebook/dinov3-vits16plus-pretrain-lvd1689m`), `hf auth login`, or set
`backbone.local_ckpt` to a local snapshot dir.

## Protocol (no test leakage)

- Train/val carved **session-disjoint** from official DTLD train (`val_frac 0.15`;
  neighbouring frames show the same intersection). `best.pt` = best **val** mAP.
- `crossval_city.py` additionally validates on **unseen cities** — the proxy for
  cross-dataset transfer; configs are selected on worst-city mAP.
- Official test: loaded only by `scripts/evaluate.py`, once, for the final
  unbiased report (metrics + per-city/per-size slices + calibration).

## 12GB (RTX 5070) settings

`configs/base.yaml`: batch 4, accum 4 (eff 16), AMP fp16+sdpa, workers 0 on
Windows / 4 on Linux. Measured: 18.7 ms/img forward, ~24 ms/img end-to-end
train step at bs 4 → 10–20 min/epoch at 24k frames.

## Known limitations

- Relevance semantics ("applies to ego lane") encode right-hand-traffic road
  conventions; transfer to left-hand-traffic countries is untested.
- NoR is structurally hardest (must establish *all* lamps are irrelevant) and
  is expected to trail RR/RG AP (same asymmetry reported by Trinci et al.).
- DTLD/LISA/BSTLD contain essentially no rain; weather robustness comes from
  simulation (fog veil, noise, blur), not from data.
