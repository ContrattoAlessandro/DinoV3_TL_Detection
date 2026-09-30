# DECISIONS_v4 — architecture review and the v4 change set

*Written 2026-09-29, after a full review of the `dinov3_global` architecture
(reads together with `CONTEXT_v3.md` [exp2 diagnosis] and `AUDIT_v3.md` [exp3
change set]). This file is the decision log: what was measured, which levers
were considered, why each was taken or rejected, and what v4 changed.*

---

## 0. Where things stood (exp2 -> exp3)

| | exp2 (MIL/LSE) | exp3 (v3) | Trinci Table II |
|---|---|---|---|
| test mAP | 64.8 | **66.0** | 81.5 (end-to-end 83.1) |
| test acc_bal | 68.9 | **71.6** | 79.3 (80.8) |
| AP_RR / AP_RG | 87.7 / 93.7 | 89.1 / 93.1 | - |
| **AP_NoR** | 12.9 | **15.8** | needs ≈ 63 for 81.5 mAP |

RR/RG are at their label-noise ceiling; **the entire gap is AP_NoR**. exp3's
fixes (token-supervision bug fixes, rel on all tokens, map_to_nor, pow075
sampler, top-k pooling, evidence-aware NoR head, lamp erasure) moved NoR +2.9
test pts and tightened the small-lamp slice (0.596 -> 0.640) - then plateaued.

## 1. New measurements taken for this review (train split, 200,144 lamps)

- **F2 lamp geometry @720p:** median **19 x 5.6 px**, 39.4% of lamps < 16 px
  tall, ~**2.4 patch tokens per lamp**, ~7 lamps/frame (~17 lamp tokens of
  3,600). Patch-16 token features barely resolve the lamp glass/housing.
- **F3 relevance ≈ direction rule** (this is the headline finding):

  | direction | n lamps | P(relevant) |
  |---|---|---|
  | back | 15,471 | **0.000** |
  | left | 18,672 | 0.043 |
  | right | 23,673 | 0.040 |
  | front | 142,328 | 0.426 |

  The relevance head receives `[z_t ; (xc, yc, bearing, 1-yc)]` and was never
  shown the direction attribute - although *most* of its decision is a function
  of housing direction + bearing.
- **F4 irrelevant lamps are visually LIT:** P(state | irrelevant) = 27% red,
  23% green, 37% unknown, 9% off. Suppressing those is entirely `rel_t`'s job
  (evidence = lamp · rel · state).
- **F5 extreme-value effect:** frame evidence is a max over ~17 lamp tokens.
  Even a 90%-accurate per-token estimate produces ≥1 false fire on
  1 − 0.9^17 ≈ **84%** of frames. Aggregation must average a lamp's tokens
  *before* the max, and/or per-token FPR must be driven very low.
- **Corrected diagnostics** (`diagnose.py` fixed, exp3 best.pt, val):
  the old output had recall printed twice (as "precision"), e.g. RG
  r0.838/p0.838 -> true **r0.838 / p0.941**; NoR r0.453 / **p0.161**.
  Sub-population "recall@argmax" was precision: true recalls are clean-NoR
  **0.492** / pseudo-NoR **0.365** (AP 0.148 vs 0.060 - pseudo-NoR remains the
  worst class). The learned NoR logit (AP 0.183) still beats every evidence
  complement (0.163-0.168), confirming CONTEXT §4.3.

## 2. Decision table (explore many -> self-critique -> choose)

| # | Lever | Why it could work | Self-critique / risk | Verdict |
|---|---|---|---|---|
| A | **Direction aux head** + direction posterior as rel input | F3: rel ≈ f(direction, bearing); forces z_t to encode housing orientation; 29% of lamps (back/left/right) become near-decidable | direction may be hard to see at 5.6 px width; aux could be noise | **DO** (cheap; even partial direction info cuts rel FPs) |
| B | **Focal rel loss** (γ=2), pos_weight 100 -> 20 | kills the ~3,500 trivial background negatives/frame that swamp the hard negatives (irrelevant-but-lit lamps, F4) | γ/pw unverifiable without ablation | **DO** (standard; directly targets the measured over-fire) |
| C | **Balanced state CE** (inv-√freq) + boost on relevant-lamp tokens | pseudo-NoR = relevant off/unknown lamp read as red; off is 3.5x rarer than red in the unweighted CE | yellow/red_yellow weights are useless (RR lumps them) | **DO** (mild weights; boost targeted at the failing sub-population) |
| D | **Mid-layer feature fusion** (hidden_states[6] ‖ final) | F2: final layer is semantic, lamp detail lives mid-layer | may add noise | **DO** (learnable projection can ignore it; one shared proj keeps the layout) |
| E | **`pool: local`** lampness-weighted local mean before top-k | F5: pools a lamp's ~2.4 tokens before the frame max; single-token lamps are *not* diluted (weights normalise by lampness mass) | single-token noise survives | **DO** (best risk/reward of the aggregation options) |
| E' | Instance-proposal MIL (grid_sample lamp crops) | most principled fix for F5 | big surgery, no ablation budget, lampness-miss risk | **DEFER** (E is the cheap approximation; escalation if the ep10 gate fails) |
| F | **CLS modulation of RR/RG logits** | which-front-lamp-is-ego's is scene geometry; exp2/3 gave CLS context only to NoR (CONTEXT §4.3 tested only the reverse direction) | shortcut risk for external data | **DO** (cfg-flagged; can be ablated to false) |
| G | Measurement/protocol fixes | numbers must be trustworthy | none | **DO** |
| H | Grid-derived token grid, nearest-centre overlap policy, accum-tail weighting | correctness / foot-guns | none | **DO** |
| I | ViT-B+ backbone or native 1820x1024 input | real gains for F2 | 2-3x compute (12-25 h/run) | **DEFER** (next lever if A-H plateau) |
| J | Multi-seed ensemble / checkpoint soup | +1-2 pts typical | needs several finished runs | **DEFER** (`evaluate.py` averages logits over `--ckpt`s) |
| K | Gentler schedule: lr 6e-5, warmup 4, 40 epochs, patience 12 | exp3 peaked at ep5 and sagged (AP_NoR 0.187 -> 0.146 by ep13) | minor | **DO** (v4 config) |
| L | Bigger head / smarter NoR readout | already tested in exp3 | measured non-fix (+2.9 only) | **SKIP** |
| M | Cache rasterised token targets | planned for speed | parse+raster is ~2-3% of item cost vs JPEG decode; ~0.5 GB RAM | **DROP** (self-critique: not worth it) |

**Chosen: A+B+C+D+E+F+G+H+K** in one v4 config. Everything that changes the
architecture is cfg-gated, so exp2/exp3 checkpoints still load through their
stored cfg (verified: exp2 reproduces val mAP 0.6651 and exp3 0.6820 **to the
digit** through the v4 code paths).

## 3. What changed (code map)

- `dinov3_global/data_global.py`
  - `grid_shape()`/`token_targets(..., patch=)` - the 45x80 grid is derived
    from `target_hw`, not hard-coded (foot-gun fix).
  - **overlap conflict policy:** shared tokens go to the *nearest box centre*
    (was last-writer-wins in file order).
  - **direction targets** (`dir` grid, -1 outside lamp tokens) for the aux head.
  - items cache the parsed `FrameSample` (no per-epoch re-parsing); `state_histogram()`.
- `dinov3_global/backbone.py`
  - `grid_hw` drives N_PATCH; `mid_layer` returns `patches_mid`/`cls_mid`.
- `dinov3_global/decoder.py`
  - `pool: "local"` (5x5 lampness-weighted mean -> top-k), `dir_head`
    (+ direction posterior into the rel MLP), `cls_modulates_evidence`
    (per-class scene prior MLP(CLS) added to the evidence logits),
    mid-layer concatenation through the shared projection (both tokens and
    CLS - the shared proj cannot take them separately; caught by a test).
- `dinov3_global/losses.py`
  - `_bce(..., focal_gamma)` focal BCE for relevance; per-class `state_weights`
    + `state_rel_boost` for the state CE; `w_dir`/`dir_tgt` direction CE.
- `dinov3_global/engine.py`
  - wires the new knobs; `state_balance_weights()` (inv-√freq, normalised over
    occurring classes only); **accumulation-tail fix** (the leftover window at
    epoch end now divides by its actual size); **temperature consistency fix**
    (metrics stay at T=1 for selection/history; the calibrated report is stored
    as `report_calibrated`; `metrics["temperature"]` keeps meaning "divide
    logits by T"); worst-city gate uses 3-class slices only; NaN-safe history.
- `dinov3_global/eval_global.py`
  - `n_classes` per slice (slice mAP is only 3-class comparable with all
    classes present); `to_jsonable()` (NaN -> null, strict JSON).
- `scripts/diagnose.py`
  - **precision/recall swap fixed** (recall was printed twice), sub-population
    "recall@argmax" fixed, and `forward_full` now honours `nor_sees_evidence`,
    CLS modulation and mid fusion (it crashed on every exp3 checkpoint before).
- `scripts/evaluate.py`
  - **temperature is fit on val** (per ckpt, or on the averaged logits for
    ensembles) instead of reusing the last ckpt's stored T; the report carries
    metrics at T=1 and at the fitted T; strict-JSON output.
- `configs/v4.yaml` - all of the above, each with its measured rationale.

Test suite: `python dinov3_global/tests/test_correctness.py` - **29 tests**
(18 existing + 11 new: grid derivation, overlap conflicts, direction targets,
local pooling semantics, dir head/rel input, CLS modulation, mid fusion,
focal down-weighting of easy negatives, state balance/rel boost, dir loss
part, JSON sanitizer).

## 4. The v4 run

**RESULT (one-shot test, n=12,453, 2026-09-30 — budget spent):** early-stopped
at ep22 (best = ep10): **test mAP 67.4 / acc_bal 72.4 / AP_RR 89.8 / AP_RG 94.1
/ AP_NoR 18.2**, T=0.82, ECE 0.027. vs exp3: **mAP +1.4, AP_NoR +2.3**; the
small-lamp slice is now the *strongest* (0.694 vs exp2's 0.596). Val→test gap
−4.3 (NoR 0.272→0.182: tail-class selection on 170 val NoR frames is noisy).
Both decision gates passed at ep5. Full table: `RESULTS_v4.md`.

`runs/exp4`, `configs/v4.yaml`, seed 0, 40 epochs, detached watchdog
(`scripts/watchdog.ps1`, auto-resumes from `last.pt`, honours `STOPPED`).

Key deltas vs exp3: `mid_layer: 6`, `pool: local`, `dir_head: true`,
`cls_modulates_evidence: true`, `token_dir: 0.2`, `rel_pos_weight: 20`,
`rel_focal_gamma: 2.0`, `state_balance: sqrt`, `state_rel_boost: 2.0`,
lr 1e-4 -> 6e-5, warmup 4, epochs 40, patience 12.

**Decision gates (val only, never test):**
- **ep5:** AP_NoR >= 0.20 and rising -> the v4 fixes are working (exp3: 0.183).
- **ep10:** AP_NoR < 0.15 -> stop; escalate to E' (instance-proposal MIL) or I
  (bigger backbone / native resolution).
- **ep20+:** val mAP >= 0.70 -> run the one-shot test evaluation.

Realistic expectation: AP_NoR 0.25-0.40 -> test mAP 0.71-0.77. **Reaching
Trinci's 81.5 still needs AP_NoR ≈ 0.63** - that most likely requires lever I
(per-lamp classification quality) and/or E', and should be stated honestly.

## 5. Monitoring

```bash
python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp4 --plot
python dinov3_global/scripts/diagnose.py --ckpt dinov3_global/runs/exp4/best.pt
# after the gates pass (ONE-SHOT, budget is per final model):
python dinov3_global/scripts/evaluate.py --ckpt dinov3_global/runs/exp4/best.pt
```

## 6. Do not repeat

- Re-implementing head forwards in scripts: `diagnose.py`'s `forward_full`
  drifted from `TokenMILHead.forward` and crashed on every exp3 checkpoint.
  Keep it in sync (or call `dec.forward` and read the aux maps).
- Printing the same subset twice and calling one of them precision.
- Slice mAP without `n_classes` (a 2-class city inflates the comparison).
- Fitting temperatures from checkpoints (fit on val; ensembles need the
  ensemble's own T).
- A shared projection over feature sets of different widths (mid fusion must
  concatenate for tokens *and* CLS).
