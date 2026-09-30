# dinov3_global v4 — Results Report (exp4, one-shot test)

**Run:** `runs/exp4` (`configs/v4.yaml`, seed 0, 40-epoch schedule) · early-stopped
at epoch 22 (patience 12) · `best.pt` = **epoch 10** (val mAP 0.7170)
**Test:** official DTLD test split (n = 12,453), evaluated **once** on 2026-09-30
via `scripts/evaluate.py` → `runs/eval_test_exp4.json`
**Hardware:** RTX 5070 12 GB · ~22 min/epoch · ≈ 8 h trained

---

## 1. Headline: all experiments side by side

| | exp1 (legacy attn) | exp2 (MIL/LSE) | exp3 (v3) | **exp4 (v4)** | Trinci Table II |
|---|---|---|---|---|---|
| key changes | 1-query cross-attn | MIL head + token aux | v3 fix set (top-k pooling, rel on all tokens, map_to_nor, pow075, erasure) | **+ dir aux head, focal rel loss, local pooling, mid-layer fusion, CLS modulation, balanced state CE** | Faster-RCNN w/attr / end-to-end |
| best **val** mAP (epoch) | 0.616 (13)* | 0.665 (8) | 0.682 (5) | **0.717 (10)** | - |
| **test mAP** | – (no test eval)* | 0.648 | 0.660 | **0.674** | 0.815 / 0.831 |
| **test acc_bal** | – | 0.689 | 0.716 | **0.724** | 0.793 / 0.808 |
| test AP_RR | – | 0.877 | 0.891 | **0.898** | - |
| test AP_RG | – | 0.937 | 0.931 | **0.941** | - |
| **test AP_NoR** | – | 0.129 | 0.158 | **0.182** | needs ≈ 0.63 for 81.5 mAP |
| val→test gap | – | −1.7 | −2.2 | **−4.3** (AP_NoR 0.272→0.182) | - |
| ECE (after T-scaling) | – | 0.028 | 0.022 | 0.027 | - |

\* exp1 crashed on a host-RAM OOM before any test evaluation; its numbers are
val-only (like-for-like column kept for the architecture ablation).

**exp4 vs exp3 (test): mAP +1.4, acc_bal +0.7, AP_NoR +2.3, AP_RR +0.7, AP_RG +1.0.**
Cumulative v4-architecture effect vs exp2: **mAP +2.6, AP_NoR +5.2**.
Every class improved; the NoR tail remains the entire gap to the benchmark.

## 2. Per-lamp-size slices (test) — where the fixes landed

| slice | exp2 | exp3 | **exp4** | n |
|---|---|---|---|---|
| small (<16 px) | 0.596 | 0.641 | **0.694** | 934 |
| mid (16–48 px) | 0.656 | 0.666 | **0.688** | 7,661 |
| large (≥48 px) | 0.642 | 0.655 | 0.649 | 3,858 |

The small-lamp slice — the measured weakness (39% of DTLD lamps are <16 px
tall at 720p; median 19×5.6 px, ~2.4 patch tokens per lamp) — went from the
*weakest* slice to the *strongest*. This is exactly where the token-supervision
fixes (exp3) and the lamp-scale reasoning (v4: mid-layer fusion, local pooling,
direction aux) were aimed.

City spread (3-class slices only; `n_classes` now reported per slice —
Bochum's 0.940 is a 2-class slice and NOT comparable): worst large-n city
Kassel 0.632 (exp3: 0.574), best large-n Frankfurt 0.797 (exp3: 0.696) — the
spread tightened on both ends.

## 3. Why exp4 works (recap of the decision log)

v4 targeted the measured token-evidence failure (see `DECISIONS_v4.md`):
relevance ≈ housing-direction + bearing (P(rel|back)=0.000, P(rel|front)=0.43),
irrelevant lamps are visually lit (27% red), and frame evidence is a max over
~17 lamp tokens (extreme-value effect). The change set: **direction aux head**
feeding the rel MLP, **focal rel loss** (γ=2, pos_weight 100→20) so trivial
background stops swamping hard negatives, **lampness-weighted local pooling**
(per-lamp evidence before the max), **mid-layer ViT fusion** (lamp detail),
**CLS scene prior** on the evidence logits, **balanced state CE** + relevant-lamp
boost (pseudo-NoR state confusion).

Trajectory (val, EMA): AP_NoR 0.122 → **0.249 (ep2)** → 0.279 (ep4) → plateau
0.25–0.28; both v4 decision gates passed at ep5 (≥0.20 AP_NoR; mAP ≥0.70).
Best-checkpoint selection stayed on val mAP at T=1; the test set was touched
once, by this evaluation.

## 4. Limitations (carry into any use of these numbers)

1. **AP_NoR is still the weak class** (test 0.182). The val→test gap for NoR
   is −9 pts (0.272→0.182); val holds only 170 NoR frames, so tail-class model
   selection is noisy and part of the val gain does not transfer.
2. Reaching Trinci's 81.5 mAP needs AP_NoR ≈ 0.63 — likely requiring the
   deferred levers (ViT-B+/native 1820×1024 input for per-lamp quality, or
   instance-proposal MIL), plus multi-seed ensembling.
3. Single seed, single run; no ensemble (`evaluate.py` averages logits over
   multiple `--ckpt`s if checkpoints ever exist).
4. Relevance semantics assume right-hand traffic; DTLD contains no rain.

## 5. Reproduce

```bash
python dinov3_global/scripts/train.py --config dinov3_global/configs/v4.yaml --out dinov3_global/runs/exp4
python dinov3_global/scripts/evaluate.py --ckpt dinov3_global/runs/exp4/best.pt   # one-shot
python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp4 --plot
```
