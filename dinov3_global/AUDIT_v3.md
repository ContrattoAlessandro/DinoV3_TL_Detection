# AUDIT_v3 — code audit, bug fixes, and the exp3 architecture change set

*Written 2026-09-29 (same day as `CONTEXT_v3.md`), after an independent
step-by-step audit of the `dinov3_global` codebase. Read together with
`CONTEXT_v3.md` (the measured exp2 diagnosis) — this file records what the
audit found in the **code**, what was fixed, and what the exp3 run changed.*

---

## 1. Bugs found (by unit test / real-data audit)

### BUG-A (critical) — per-box ignore band deleted neighbouring lamps' supervision
`data_global.token_targets` applied the 16 px ignore band **per box**: a token
inside lamp box A but within 16 px of box B was marked `valid=False`, silently
removing it from *all three* token losses (lamp BCE, state CE, rel BCE).
DTLD signal heads stack 2–3 lamps (median 5 lamps/frame), so this hit exactly
the clustered-lamp case.

- Measured on **all 28,525 parseable DTLD train frames**
  (`scripts/audit_token_targets.py`): **69,909 of 298,761 lamp-positive tokens
  (23.4%) lost 100% of their supervision, affecting 51.1% of frames.**
- Unit test: `tests/test_correctness.py::ignore_band_neighbour_bug`
  (2/4 lamp tokens lost in a two-lamp cluster).
- **Fix:** accumulate `inside_any`/`band_any` over all boxes; the ignore band
  applies only to `band_any & ~inside_any`. A token inside any lamp box always
  keeps its targets.
- Impact: this corrupted the *relevance/state* signal — the exact component
  the exp2 diagnosis (CONTEXT §4) identified as the NoR bottleneck — and was
  invisible to `diagnose.py` because lampness (a much easier task) still
  reached AUC 0.986.

### BUG-B (minor) — EMA skipped the leftover accumulation window
`engine.fit` updated EMA only inside the `(bi+1) % accum == 0` branch; the
tail partial-window optimizer step at epoch end advanced the weights without
advancing EMA. Fixed by extracting `_optim_step()` (clip → step → EMA) used by
both call sites.

### BUG-C (tooling) — `diagnose.py` crashed on top-k checkpoints
It read `branch.tau`, which raises in `pool: topk` mode (no learnable tau), and
printed `tau*log N` unconditionally. Now guarded; top-k runs print their `ks`.

### Bug-shaped but CORRECT on re-verification (self-critique log)
- **Logit adjustment sign** (`logits + tau_la·log_pi` at train, raw argmax at
  inference) is exactly Menon et al.; verified by test, and reproduces the
  documented behaviour ("raw NoR logit must clear ~3.0").
- **`consistency_kl`** is symmetric and zero for identical inputs (−3e−8 =
  float noise in the log/exp round-trip; my first test's `== 0.0` was wrong).
- **`fit_temperature`** is the standard Guo et al. fit; my first generative
  test was wrong (deterministic labels ⇒ CE optimum at T→0). With labels
  sampled from `softmax(lg/T)` the recovery test passes.
- **Sampler exponent parsing** (`pow075` = 0.75, not 75) — correct; test
  locks it. Independent re-measure of class shares matches the docstring:
  `sqrt → NoR 14.1%`, `pow075 → 22.3%`, `pow100 → 33.3%`.
- **`token_targets` geometry** (114 px crop + 1280/1820 mapping vs
  `make_1280.py`'s baked crop) — consistent; center-in-box rasterisation and
  sub-token snapping verified by tests.
- **exp2 reproducibility under the new code**: `runs/exp2/best.pt` still loads
  (strict) and `scripts/diagnose.py` reproduces the frozen report **exactly**
  (AP_RR 0.8953 / AP_RG 0.9614 / AP_NoR 0.1385 / mAP 0.6651) — the eval
  pipeline (`load_model → predict_split → eval_global`) is verified correct
  end-to-end.

### Two of my own test-design mistakes (kept honest)
First run of the suite: 9/15 passed. Four failures were **test bugs** (missing
`__len__`, `== 0.0` float strictness, unachievable temperature target, `<`
instead of `<=` in warmup) and one was the real BUG-A. The erasure fill test
also caught a real bug in the *new* code (below) that the first version of the
test passed for the wrong reason ("red is gone" also holds for a black hole).

---

## 2. Architecture changes for exp3 (vs exp2)

All changes target the measured root cause (the per-token relevance estimate)
or the structurally unlearned case (lamp-free frames). Config:
`configs/v3.yaml`; run: `runs/exp3` (30 epochs, seed 0, early-stop patience 8).

| # | Change | Rationale (measured) | Risk / mitigation |
|---|---|---|---|
| 1 | BUG-A fix (band ∪ interior semantics) | recovers 23.4% of token supervision | none; test-locked |
| 2 | view 1 never geometrically cropped (already in tree) | framing crop misaligned 30% of steps (0.780→0.005 lamp hit-rate) | diversity kept via view 2 |
| 3 | rel BCE on **all** valid tokens, `token_rel 0.5`, `rel_pos_weight 100` (already in tree) | head was never told background is irrelevant (CONTEXT §4.5) | pos/neg mass ≈ 11/89 |
| 4 | `label_policy: map_to_nor` (full weight for pseudo-NoR) | downweighted frames are the worst-ranked (AP 0.042) yet fully counted | label noise at full weight; hedge = 0.5 |
| 5 | `sampler: pow075` + `tau_la: 0.35` in lockstep | NoR 5.0%→22.3% of batches; don't double-correct with LA | RR/RG keep 73%+ of mass |
| 6 | `pool: topk [1,2,4]` replaces LSE | LSE `-tau·log N` made a dead branch and halved evidence scale (§4.2) | 1/k dilution absorbed by learnable gain |
| 7 | `nor_sees_evidence: true` — NoR MLP input = [CLS ‖ S_RR ‖ S_RG] | "no relevant lamp" *is* the complement of the evidence; jointly trained (post-hoc probe 0.088 is not evidence against joint training) | shortcut risk; can be ablated to `false` |
| 8 | **lamp-erasure augmentation** `p_erase: 0.06` | DTLD has zero lamp-free frames; the empty-road case is the common one on foreign data and the user's manual-inspection case | label-*changing*; verified visually + unit tests; both consistency views share the erased frame/label pair |
| 9 | early-stop patience 8 + `STOPPED` marker honored by the watchdog | decided by val data, not impatience (CONTEXT §P1-3) | marker prevents auto-resume past a deliberate stop |
| 10 | `_optim_step`/EMA consistency, topk-safe diagnose | correctness | — |

Erasure details: every lamp of the frame is painted out via iterative
diffusion infill (normalized convolution repeated 12× at ¼ resolution —
a single pass degenerates to black when the hole exceeds the fill window,
which the test suite caught), the frame is relabelled NoR (honest by the
label definition "2 = no relevant lamp present") and the token targets follow
the pixels (lamp/rel → 0, ignore band released). QA image:
`runs/erase_qa.png`. Effective NoR batch share with 5+6: ≈27%.

## 3. Test suite

`python dinov3_global/tests/test_correctness.py` — 18 tests, all passing:
token geometry/snap/empty, ignore-band neighbour regression, global label
mapping, sampler shares + exponent parsing, logit adjustment, top-k and LSE
pooling, consistency KL, temperature recovery (generative), decoder
forward/grad flow (both pools), token-loss masking, LR schedule, erasure
colour/smoothness/large-hole regressions, erasure label semantics,
exp2-checkpoint layout compatibility.

## 4. Run state & protocol

- **exp3 RESULT (one-shot test, n=12,453, 2026-09-29 — budget spent):**
  **mAP 66.0** / acc_bal **71.6** / AP_RR 89.1 / AP_RG 93.1 / **AP_NoR 15.8**,
  T=0.727, ECE 0.022. Early stopped at ep13 (best = ep5, val mAP 0.682).
  vs exp2 test: **mAP +1.2, acc_bal +2.7, AP_NoR +2.9, AP_RR +1.4, AP_RG −0.6,
  ECE −0.005**. Val→test gap −2.2 (exp2: −1.7) — no leakage.
  Slices: small-lamp 0.640 (**+4.5** vs exp2's 0.596 — the token-supervision
  fixes land where they should), mid 0.666, large 0.655; worst large-n city
  Kassel 0.574 (exp2's worst was Bremen 0.532 — spread tightened).
  Report: `runs/eval_test_exp3.json`.
- Trajectory (val, EMA): mAP 0.509 → 0.652 → 0.672 → 0.680 → **0.682 (ep5)** →
  flat/declining; AP_NoR peaked 0.187 (ep4) and sagged to 0.146 by ep13 — the
  early-stop rule fired correctly at stale=8.
- Honest read: the fixes moved NoR off its 0.13–0.15 exp2 plateau (+2.9 test)
  and tightened the slices, but **NoR remains the bottleneck** (would need
  ≈0.63 to reach Trinci's 81.5). Next levers if another run is wanted:
  direction-attribute aux head (relevance is direction-dependent and the rel
  head only sees 2D geometry), stronger erasure (p 0.1–0.15), or ensembling
  (`evaluate.py` averages logits over multiple `--ckpt`).
- Decision gates used (val only, never test): ep5 `AP_NoR ≥ 0.20` → missed
  (0.183) but rising; ep10 `< 0.15` → not triggered (0.164); early stop
  patience 8 → fired at ep13.
- The known ceiling caveat stands: beating Trinci's 81.5 needs AP_NoR ≈ 0.63.

## 5. Ops note (this session)

A watchdog was already logging in `runs/exp3_watchdog.log` at 16:36 (pid
39076) before this session's launch; it and its python died without writing
`last.pt`/history, so `runs/exp3` started clean (verified: single python
process, `resume=False`, empty run dir). If you see two watchdog entries in a
log, always verify with `Get-CimInstance Win32_Process` **excluding `$PID`**
before killing anything.
