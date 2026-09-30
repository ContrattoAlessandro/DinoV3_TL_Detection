# CONTEXT for the next model (v3) — dinov3_global

> **UPDATE (later, same day):** an independent code audit found and fixed **two
> token-supervision bugs** (a per-box ignore band that deleted 23.4% of lamp
> token supervision across 51.1% of frames - `AUDIT_v3.md` BUG-A) and launched
> the exp3 run with the §6-§7 plan *plus* those fixes, top-k pooling, an
> evidence-informed NoR head and lamp-erasure augmentation. See
> [`AUDIT_v3.md`](AUDIT_v3.md) for the change set, tests and run state.

*Written 2026-09-29 after the exp2 one-shot test evaluation and a full instrumented
diagnosis of the trained checkpoint. This file is a self-contained briefing: a fresh
model/session with no memory of the previous work should be able to continue from it.
Numbers marked **[measured]** come from `scripts/diagnose.py` on the **val** split
(4,378 frames) or from `runs/eval_test_exp2.json`; nothing here uses the test split for
any decision.*

---

## 0. TL;DR state of play

| | |
|---|---|
| System | frozen DINOv3 ViT-S+/16 + `TokenMILHead` → frame-level RR / RG / NoR. Image-in → image-out, no detector, no boxes at inference. |
| Trainable | **0.43 M** head params (backbone 28.7 M frozen). |
| One-shot test (n=12,453) | **mAP 64.8 / acc_bal 68.9** (AP_RR 87.7, AP_RG 93.7, **AP_NoR 12.9**) vs Trinci Table II 81.5 / 79.3. **Benchmark not beaten.** |
| vs baseline | exp1 (legacy attn head) val mAP 61.5 → exp2 val **66.5** (+5.0). Both fail identically on NoR (0.13). |
| Root cause | **[measured]** The per-token *relevance* estimate is the bottleneck: it over-fires on irrelevant/off lamps, so NoR frames look lamp-positive (23.5% of NoR frames get confident red evidence vs 13.1% of true-RG frames). The NoR *readout* is **not** the bottleneck (an evidence-based complement scores no better). |
| Biggest single defect found | **[measured]** 30% of training steps supervise **wrong token targets**: `random_framing` crops the image but the token targets are rasterised on the uncropped frame → top-1 lamp token lands inside a GT box in 78% of clean frames and **0.5%** after one crop. |
| Headline recommendation | Fix token-target/crop alignment, supervise relevance on *all* tokens, realign the pseudo-NoR label policy, and fix the sampler exponent — before touching the architecture. |

---

## 1. Mission, constraints, protocol

**Goal.** Beat Trinci et al. *Color Is Not Enough* on DTLD test (81.5 mAP / 79.3 acc_bal;
their end-to-end model 83.1 / 80.8, Table II) with a *global* relevance classifier, and
generalise to **unlabeled external datasets** that the user inspects manually.

**Hard constraints (do not violate).**
1. **Image-in → image-out.** No detector, no GT boxes, no test-time fitting at inference.
   Boxes are training-only aux supervision.
2. **Test-split discipline.** The official DTLD test split is loaded only by
   `scripts/evaluate.py`, **once per final model**. `best.pt` is selected on the
   session-disjoint val split (carved from official *train*). Never select on test.
3. **Compute:** RTX 5070 12 GB, Windows host. Frozen backbone is what makes this fit.
4. **No hue-rotating augmentation** (would turn red lamps green and corrupt RR/RG).
5. External datasets are **never used for training or threshold fitting** — the user
   looks at outputs qualitatively.

**Metrics** (`eval_global.py`): AP_RR / AP_RG / AP_NoR = one-vs-rest
`average_precision_score` on softmax probs; `mAP` = mean; `acc_bal` =
`balanced_accuracy_score` (macro recall of argmax). Temperature is fit on val and
reported with ECE.

---

## 2. The v2 system, exactly as built

### 2.1 Data
- DTLD v2.0 labels, `label_dir: datasets/DTLD/v2.0`; images are an **offline 1280-wide
  dump** (`datasets/DTLD_1280`, produced by `scripts/make_1280.py`): 2048×1024 →
  crop 114 px per side (2:1 → 16:9) → bicubic resize to 1280×720. `crop_sides: 0` because
  the crop is baked in; `label_crop_sides: 114` drives the label→pixel mapping.
- Frames are decoded to **uint8** CHW tensors and cast to float on GPU (exp1 died of
  host-RAM OOM with float32 CPU decode).
- Global label from per-lamp DTLD attributes: `0=RR` (a **relevant** lamp in
  {red, yellow, red_yellow}), `1=RG` (relevant green, no RR), `2=NoR` (everything else).
- **pseudo-NoR** = frames that are NoR *only* because their relevant lamp is off/unknown
  (no stop/go signal). Policy `downweight` gives them sample weight 0.3.
- Split: val = 15% of **sequences** (never frames — frames 0.66 s apart are the same
  intersection), seed 0, retrying seeds until every class has ≥30 val samples.
  **[measured]** train 24,147 = [RR 8082, RG 14863, **NoR 1202**]; val 4,378 =
  [1488, 2720, **170**]. NoR prevalence 4.98% train / 3.88% val.
- **[measured] Two data facts that dominate everything downstream:**
  - **33% of NoR frames are pseudo-NoR** (401/1202 train, 52/170 val).
  - **Zero NoR frames are lamp-free** — every one of the 1,202 has lamps (median 5/frame).
    DTLD's NoR means "lamps present but not relevant to ego", *never* "empty road".
    The empty-road case that matters for other datasets is **unseen in training**.

### 2.2 Token targets (training-only aux supervision)
`token_targets()` rasterises DTLD lamp boxes onto the 45×80 patch grid: a token is
positive if its centre falls in a box (boxes smaller than a token snap to the token
containing the box centre); tokens within a **16 px ignore band** around a box are
masked out of the loss. Produces `lamp [45,80]∈{0,1}`, `state [45,80]`, `rel [45,80]`.

### 2.3 Backbone
`facebook/dinov3-vits16plus-pretrain-lvd1689m` (ViT-S+/16, dim 384, 29 M), **frozen**,
fp16 + SDPA on CUDA, ImageNet normalisation internal to the wrapper. 1280×720/16 →
**45×80 = 3,600 patch tokens** + CLS. Forward is wrapped in `torch.no_grad()`; patch and
CLS features are cast to fp32 before the head.

### 2.4 Head — `TokenMILHead` (`decoder.py`)
```
h = LayerNorm(Linear(384→256))(patches) + fixed 2D sincos pos        # shared
for each of 2 branches (learnable tau, init 0.05 / 0.30):
    z       = MLP: Linear256→256 → GELU → Dropout0.1 → ×2
    lamp_t  = sigmoid(Linear(z))                     # lamp occupies this patch
    state_t = softmax(Linear(z), 6)                  # green, off, red, red_yellow, unknown, yellow
    rel_t   = sigmoid(Linear([z ; geo_t]))           # geo_t = (xc, yc, xc-0.5, 1-yc)  ← only rel sees geometry
    e_RR    = lamp_t · rel_t · (state[red]+state[yellow]+state[red_yellow])
    e_RG    = lamp_t · rel_t · state[green]
    S_c     = tau · ( logsumexp_t(e_t^c / tau) − log N )              # N = 3600
S = mean over branches
logits = [ a_RR·S_RR + b_RR ,  a_RG·S_RG + b_RG ,  MLP(CLS) ]         # a, b learnable, init 4 / 0
```
The **NoR logit is produced by a 2-layer MLP on the projected CLS token only** — it never
sees the pooled evidence `S`.

### 2.5 Losses (`losses.py`)
- `global_loss`: label-smoothed (0.05) CE with **logit adjustment** (Menon et al.),
  `logits + τ_la·log π`, `τ_la = 1.0`, optional per-sample weight (label policy).
  **[measured]** log π = [−1.094, −0.486, −3.000] → during training the *raw* NoR logit
  must exceed the others by ~3.0 to win an argmax.
- `token_losses`: lampness BCE (w 0.3, `pos_weight` 10) over **all valid** tokens;
  state CE (0.2) and relevance BCE (0.2) over **lamp-positive tokens only**.
- `consistency_kl`: symmetric KL between two augmented views (w 0.1).

### 2.6 Training loop (`engine.fit`)
AdamW on `decoder_parameters()` only (lr 1e-4, wd 0.05, cosine after 3 warm-up epochs,
30 epochs), batch 4 × accum 4 (eff 16), grad-clip 1.0, AMP, **EMA 0.999 on the head**,
val every epoch **on the EMA weights** (live weights are swapped out and back), best.pt on
val mAP, temperature fit on val logits each epoch, `history.jsonl` appended per epoch.
Augmentation (`augment.py`, GPU, whole-batch shared draws, all label-preserving):
brightness, per-channel white balance, gamma, contrast, saturation, sensor noise,
JPEG-ish down/up-sample, defocus blur, fog veil, and `random_framing` (scale 0.85–1.0 +
shift, p = 0.3).

---

## 3. Results ledger

**exp1** (legacy `attn` head, 1 learnable query): val mAP 61.5 @ ep15, AP_RR 0.79,
AP_RG 0.92, AP_NoR 0.13. Crashed on host-RAM OOM (float32 decode) before any test eval.
Softmax attention is a weighted *mean* over 3,600 tokens — wrong inductive bias for an
existential ("ANY relevant lamp") decision. Kept as `decoder.head: attn` ablation.

**exp2 trajectory (val, EMA):**

| ep | train loss | AP_RR | AP_RG | AP_NoR | mAP | acc_bal | worst city |
|---|---|---|---|---|---|---|---|
| 1 | 1.144 | 0.346 | 0.716 | 0.077 | 0.380 | 0.478 | 0.366 |
| 3 | 0.827 | 0.861 | 0.949 | 0.155 | 0.655 | 0.687 | 0.578 |
| 5 | 0.738 | 0.890 | 0.958 | 0.146 | 0.665 | 0.708 | 0.595 |
| **8** | 0.654 | 0.895 | 0.961 | 0.139 | **0.665** | 0.719 | 0.586 ← `best.pt` |
| 9 | 0.631 | 0.894 | 0.962 | 0.130 | 0.662 | 0.710 | 0.564 |

Stopped by the user at ep9/30 (plateau; LR was still ≈9e-5, train loss still falling).

**exp2 one-shot test (n = 12,453):** mAP **64.8**, acc_bal **68.9**, AP_RR 87.7,
AP_RG 93.7, **AP_NoR 12.9**, T = 0.562, **ECE 0.028** (well calibrated — confidences from
`infer_folder.py` are usable). Slices: small lamps (<16 px) 0.596 < mid 0.656 ≈ large
0.642; large-n cities 0.62–0.70 (Bremen 0.532 n=222, Bochum 0.897 n=109 are small-n noise).
Val→test gap −1.7 pts: healthy, no leakage.

---

## 4. Verified diagnosis [all measured on val with `best.pt`]

The instrumented forward in `scripts/diagnose.py` reproduces the training-time val
metrics **exactly** (AP_RR 0.8953, AP_RG 0.9614, AP_NoR 0.1385, mAP 0.6651), so the
numbers below describe the model that was actually evaluated.

### 4.1 The evidence path works for RR/RG and over-fires on NoR
| GT class | mean max-token RR evidence | frames with `emax_rr > 0.5` | **relevant** lamp tokens/frame (GT) | mean lamps/frame |
|---|---|---|---|---|
| RR (n=1488) | 0.842 | 84.7% | 4.34 | 6.5 |
| RG (n=2720) | 0.167 | 13.1% | 4.96 | 6.5 |
| **NoR (n=170)** | **0.269** | **23.5%** | **0.79** | 5.3 |

The ground truth says NoR frames contain ~0.8 relevant lamp tokens, yet the token head
fires confident red evidence on 23.5% of them — *more often than on true RG frames*
(13.1%). Pooled scores agree: `S_rr` = 0.628 (RR) vs 0.155 (NoR) but **0.086 (RG)** —
NoR frames look more "red-relevant" than RG frames do. **The failing component is the
per-token relevance estimate `rel_t`, not the pooling and not the readout.**

### 4.2 One of the two MIL branches is dead, by construction
Learned `tau` = [0.030, 0.223] (init [0.05, 0.30]); learned `a` = [4.47, 4.42].
The LSE score is `S = e_max − tau·log N + tau·log(Σ_t exp((e_t−e_max)/tau))`: the
`−tau·log N` term is only partly cancelled by the diffuse background, so **a large
tau collapses the score towards the frame's mean evidence** instead of its peak.
Measured consequences on val:
- a token peak of 0.842 (RR frames) became `S_rr` = **0.628** at tau = 0.030
  (1.34× attenuation) but only **0.217** at tau = 0.223 (3.9×);
- branch 0 `S_rr` separation RR−NoR = **+0.473** (×a → +2.11 logits) vs branch 1
  **+0.053** (RR 0.217 vs NoR 0.164) — the large-tau branch is nearly constant;
- branch-averaged separation drops to **+0.263**, i.e. half the trunk capacity is
  spent producing a near-constant and the evidence enters the softmax at half scale.
The docstring's "smooth max in [0,1]" therefore only holds when the background is
also near the peak. The learned `a ≈ 4.5` compensates on average, which is why
RR/RG still work. **v3 replaces this with top-k mean pooling (k = 1, 2, 4).**

### 4.3 The NoR readout is NOT the bottleneck (this corrects an earlier hypothesis)
| NoR score used | AP_NoR (val) |
|---|---|
| learned `MLP(CLS)` logit (current) | **0.1385** |
| `1 − max(S_rr, S_rg)`, branch-averaged | 0.1530 |
| `−(S_rr + S_rg)`, branch 0 | 0.1435 |
| `−max(emax_rr, emax_rg)` | 0.1491 |
| `−emax_rr` (best single token) | 0.0415 |
| linear probe on `[CLS-head feats ‖ S ‖ emax]` (fit on 70% of val) | 0.0881 |

Every evidence-derived NoR score is **as bad as or worse than** the current CLS-only
head, and a probe that *adds* the evidence is worse still. Therefore
"the NoR head never sees the evidence" — the hypothesis written into `RESULTS.md` — is
**not** the fix. Feeding `S` into the NoR MLP should be tested cheaply, not assumed.

### 4.4 pseudo-NoR is where the class collapses
| sub-population | n | AP_NoR | recall@argmax |
|---|---|---|---|
| clean NoR (no relevant off/unknown lamp) | 118 | 0.115 | 0.131 |
| **pseudo-NoR** (relevant lamp present, off/unknown) | 52 | **0.042** | **0.048** |
| all NoR | 170 | 0.139 | 0.168 |

The 52 hardest frames are the ones the training loop **downweighted to 0.3** while the
metric counts them at full weight. The policy is aligned with the wrong objective.
Whole-model argmax recalls: RR 0.821, RG 0.824, NoR 0.512 (precisions identical) —
`acc_bal` 0.714 is consistent with this.

### 4.5 Token supervision is sparse *and* 30% corrupted
- **[measured]** lamp tokens/frame: mean **9.93 of 3,600** (0.276%); *relevant* ones 4.59.
  So `rel`/`state` losses run on ~0.28% of tokens — a very thin signal for the exact
  quantity the NoR class depends on.
- **[measured]** background tokens get **no relevance supervision at all**: `token_losses`
  masks state/rel to `valid & lamp_tgt > 0.5`, so the head is never told "this patch is
  *not* a relevant lamp" on the other 99.7% of the grid.
- **[measured]** lampness AUC on valid tokens is 0.986 (the lamp *localisation* works).
- **[measured] the framing bug:** `random_framing` (p = 0.3) crops and resizes the input,
  but `lamp_tgt`/`state_tgt`/`rel_tgt` are rasterised on the **uncropped** frame. Top-1
  lamp token inside a GT box: **0.780** on clean input → **0.005** after one crop
  (s = 0.9, offset 25%; 800 frames). So ~30% of training steps actively teach the token
  heads that the lamp is *not* where it is. Lampness survives it (easy task, 70% clean
  steps); the *relevance* head — the component NoR depends on — does not.

### 4.6 Other facts that shape the plan
- **Undertrained:** stopped at ep9/30, LR ≈9e-5, train loss 0.631 and falling. All tail-class
  conclusions come from a partially converged head — but exp1 (a completely different head)
  reached the same AP_NoR ≈ 0.13 at ep15, so the tail failure is **not head-specific**.
- **Head capacity** 0.43 M vs 28.7 M frozen. Enlarging it is cheap on 12 GB, but nothing
  in 4.1–4.5 points at capacity as the binding constraint.
- `sampler: sqrt` **is implemented** (`engine._make_sampler`) but was never used
  (`sampler: none`). **[measured]** the class share of a `freq**-p` sampler is
  `freq**(1-p)` normalised, which depends on the class **sizes**. On the real train
  histogram (natural RR 33.5 / RG 61.6 / **NoR 5.0%**):
  `sqrt` (p=0.5) → NoR **14.1%**, `pow025` (p=0.25) → NoR **8.5%**,
  **`pow075` (p=0.75) → NoR 22.3%** (the v3 setting), `pow100` → 33.3%.
  *(An earlier draft of this file claimed sqrt → 60% and pow025 → 28%; both were
  normalisation errors, corrected here after a unit test of the sampler.)*
- Logit adjustment already targets the balanced-accuracy optimum, so post-hoc threshold
  tuning is unlikely to buy much; the raw logits are under-confident (T = 0.562 < 1)
  because of the compressed evidence scale, which temperature scaling fixes (ECE 0.028).

---

## 5. Root-cause ranking (what actually blocks the tail class)

1. **The relevance estimate `rel_t` is undertrained and partly mis-supervised.** It is
   trained on 0.28% of tokens, gets no negative examples from the background, and has
   ~30% of its steps pointed at the wrong locations (§4.5). Everything about NoR flows
   from this number.
2. **Class definition vs supervision mismatch.** NoR *is* "all lamps irrelevant/off",
   so the whole burden falls on `rel_t`; the training signal for it is the thinnest in
   the system while the pooled score (which the frame CE also trains) is dominated by
   lampness × state, which NoR frames have plenty of (§4.1).
3. **The label policy fights the metric** for the hardest 33% of NoR frames (§4.4).
4. **Data scarcity / no coverage:** 1,202 NoR frames, 401 of them noisy, and **no
   lamp-free NoR at all** (§2.1) — the external-dataset case is structurally unlearned.
5. **Undertraining** (partial run) and one **dead MIL branch** (§4.2) add noise but are
   not the core failure.

---

## 6. v3 design — prioritised, with rationale and verification

### P0 — do these first; they are cheap, safe and target the measured root causes
**P0-1 · Kill the token-target/crop misalignment.** Two-line change in `engine.fit`:
build view 1 as photometric+degradations **without** `random_framing` (it carries the
token losses), and keep the full `train_view(x)` (with framing) for view 2, which has no
token loss. *Effect:* +~43% correct relevance supervision, no systematic label inversion.
*Alternative if framing diversity is wanted:* have `random_framing` return its crop box
and resample the three target grids with nearest-neighbour.

**P0-2 · Supervise relevance on ALL valid tokens, and raise its weight.** In
`token_losses`, compute the rel BCE over `valid` (not `valid & lamp>0.5`) with target 0 on
background, and set `token_rel` 0.2 → **0.5**. *Rationale:* §4.5 — the head is currently
never told that the ~99.7% background is irrelevant, which is precisely the error mode on
NoR frames. *Risk:* mild class imbalance in the rel task; `pos_weight` on the lamp side
already handles the inverse. *Verify:* `diagnose.py` → lampness AUC must stay ≈0.98 while
`rel` AP on lamp tokens should rise.

**P0-3 · Realign the label policy with the metric.** `policy_weight` 0.3 → **1.0**
(i.e. `label_policy: map_to_nor`), or 0.5 as a hedge. *Rationale:* §4.4 — the
downweighted 33% are the frames the metric counts and the model ranks worst (AP 0.042).
An ablation is the right way to confirm, but the current setting is known-misaligned.

**P0-4 · Fix the imbalance strategy; do not double-correct.** Measured shares (§4.6):
`pow075` puts NoR at **22.3%** of batches — the v3 setting, matching the intended
mild oversampling. (`pow025` → 8.5% is too weak; `pow100` → 33.3% would be a full
rebalance and risks the two classes that already work.) If sampling is used, drop
`tau_la` to **0.35** (logit adjustment already handles imbalance; doing both at full
strength over-corrects and is a likely reason the tail never moved).

### P1 — architecture and protocol, after P0 is measured
**P1-1 · Replace the LSE pooling with top-k mean pooling.** `S_c = mean(top-k e_t^c)` for
k ∈ {1, 2, 4} as the "branches". Full [0,1] range, no `tau·log N` penalty, no dead
branch, and the pooled score becomes human-readable in the heatmaps (useful for the
user's manual inspection). Expect RR/RG unchanged and a genuinely informative
`1 − max(S)` complement for the NoR diagnostics.

**P1-2 · Test the NoR-head variants cheaply before training** (the probe harness in
`diagnose.py` already does this in ~1 min): (a) CLS-only (current, 0.139), (b) CLS ⊕ S,
(c) an explicit complement term `logit_NoR = MLP(CLS) + c·(−(S_RR+S_RG))`, (d) a small
learned attention pooling over tokens for the NoR branch only. Pick by val AP; §4.3
says (b) is *not* obviously better, and exp1's attention head reached the same 0.13, so
expect (c) at best.

**P1-3 · Train the full schedule.** 30 epochs ≈ 8.5 h; do not stop before the LR tail
(ep ≥ 20) unless val mAP is flat for ≥ 6 epochs. Add an early-stop rule on val mAP with
patience 6 so this is decided by the data, not by impatience.

**P1-4 · Make monitor.py show the failure mode live.** Add per-class recall/precision and
**clean vs pseudo AP_NoR** to the per-epoch print. §4.4 was only discoverable *after* the
run; with these two lines the same signal would have been visible at epoch 3.

### P2 — only if the above land
**P2-1 · Manufacture the missing "empty road" class.** Since DTLD contains **no**
lamp-free NoR frame, synthesise them: on a fraction of RR/RG frames, erase the regions the
lampness head marks and relabel to NoR **only if every relevant stop/go lamp was erased**;
otherwise keep the original label. This is the only route to the case the user will
actually inspect on other datasets. It is label-*changing*, so it needs care and its own
sanity check (a lamp-erased frame is a genuine NoR by DTLD's own definition).
**P2-2 · Head capacity** (proj_dim 384–512, 3 pooling branches) — cheap, marginal.
**P2-3 · Inference guard for lamp-free frames:** because lampness is reliable
(AUC 0.986), a deterministic rule "if max lampness < τ_lamp → output NoR" would cover
empty roads without retraining. It cannot be validated on DTLD (no lamp-free frames) and
must be checked on the user's external data by hand.
**P2-4 · Seed ensemble / 3-checkpoint soup** — `evaluate.py` already averages logits over
multiple `--ckpt` paths. Deferred by compute budget.

---

## 7. Proposed single v3 run (what the next training run should be)

```yaml
loss:  {tau_la: 0.35, token_lamp: 0.3, token_state: 0.2, token_rel: 0.5, consist: 0.1}
data:  {sampler: pow075,            # NEW: NoR 5.0% -> 22.3% of batches (measured)
        label_policy: map_to_nor,  # policy_weight 1.0
        policy_weight: 1.0}
optim: {epochs: 30, lr: 1.0e-4, batch: 4, accum: 4, ema_decay: 0.999, seed: 0}
```
plus the P0-1 / P0-2 code changes and (optionally, after the cheap P1-2 probe) P1-1.

**Decision gates while it runs** (decide on val only):
- by **ep 5**: AP_NoR ≥ 0.20 and rising → the P0 fixes are working;
- by **ep 10**: AP_NoR < 0.15 → `rel` supervision is still the problem; stop early and
  escalate to P2-1 (erase-augmentation) rather than burning 8 h;
- at **ep 20+** (LR tail): val mAP ≥ 0.70 → proceed to the one-shot test eval.
Expectation: AP_NoR 0.3–0.5, val mAP 0.72–0.77. **Reaching Trinci's 81.5 still requires
AP_NoR ≈ 0.63** — state that honestly in any report; the realistic value of the v3 work
is NoR robustness on external data, not the headline benchmark.

---

## 8. Operations & environment (learned the hard way)

- **Train detached or not at all.** Background shells owned by the agent session get torn
  down and take the training process with them (happened twice). Use
  `Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','dinov3_global\scripts\watchdog_exp2.ps1' -WindowStyle Hidden`;
  the watchdog resumes from `last.pt` and exits at 30 epochs.
- **Never filter processes by a string that appears in your own command line.** A
  `Get-CimInstance ... | Where CommandLine -like '*watchdog_exp2*'` matched the querying
  shell itself and killed it (exit 255). Always exclude `$PID`.
- PowerShell `*>` writes **UTF-16** logs; `cmd /c ... >>` and `Start-Process -Redirect`
  write UTF-8. `monitor.py` sniffs both.
- `--resume last.pt` restores head + EMA + optimiser + AMP scaler + epoch, and seeds the
  best-mAP tracker from the checkpoint (fixed this session; without it a resumed run could
  overwrite `best.pt` with a worse epoch).
- `num_workers: 0` and `pin_memory: false` on this host (DataLoader workers die
  mysteriously here).
- Timings **[measured]**: 18.7 ms/img forward, ~24 ms/img train step at bs 4, 5.5–5.9 it/s,
  **≈19 min/epoch** on 24,147 frames → 30 epochs ≈ 8.5 h. Val pass ~2 min.
- A silent "process died with no traceback" is this host's failure mode, not a code bug.

## 9. File map & commands

```
dinov3_global/dinov3_global/backbone.py     frozen DINOv3 ViT-S+/16 wrapper
                        decoder.py          TokenMILHead (+ legacy GlobalDecoder "attn")
                        losses.py           logit-adjusted CE, token_losses, consistency_kl
                        engine.py           loaders, train/fit/resume/EMA, predict_split
                        data_global.py      labels, token_targets, uint8 decode, splits
                        eval_global.py      AP/acc_bal, city+size slices, temperature, ECE
                        augment.py          label-preserving GPU augmentation
configs/base.yaml                           all hyper-parameters
scripts/train.py         train / --resume / --seed / --epochs / --max-train (smoke)
scripts/evaluate.py      ONE-SHOT test eval (do not run more than once per final model)
scripts/diagnose.py      instrumented val diagnosis (the source of §4)
scripts/monitor.py       live per-epoch table + verdict, parses the run's logs
scripts/infer_folder.py  predictions.csv + overlays for UNLABELED datasets
scripts/infer_attn.py    lamp/relevance/RR/RG heatmaps
scripts/crossval_city.py city-held-out CV (optional, smoke-validated)
scripts/make_1280.py     offline 2048->1280 dump
scripts/watchdog_exp2.ps1  detached auto-resume watchdog (hardcoded to exp2)
runs/exp2/{best.pt,last.pt,history.jsonl}  the evaluated run (best = ep8)
runs/eval_test_exp2.json                     the one-shot test report
RESULTS.md                                  final results report for exp2
```

```bash
# train (detached, auto-resuming)
powershell -NoProfile -Command "Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','dinov3_global\scripts\watchdog_exp2.ps1' -WindowStyle Hidden"
# live table
python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp2 --plot
# diagnosis (val only, ~6 min)
python dinov3_global/scripts/diagnose.py --ckpt dinov3_global/runs/exp2/best.pt
# ONE-SHOT test eval - already spent for exp2; do not repeat
python dinov3_global/scripts/evaluate.py --ckpt dinov3_global/runs/exp2/best.pt
# external dataset QA (add --crop-sides 114 for raw 2048x1024 DTLD-style images)
python dinov3_global/scripts/infer_folder.py --ckpt dinov3_global/runs/exp2/best.pt --images <folder> --overlays
```

## 10. Open questions (need the user)

1. **Which external datasets**, and what do they contain? Specifically: do they include
   **empty roads / frames without traffic lights** (the concept DTLD never provides), night
   frames, different aspect ratios? This decides whether P2-1/P2-3 are worth the run.
2. Is the target metric for the external data still 3-way RR/RG/NoR, or would a
   "relevant lamp present?" binary be more useful in practice?
3. Should the next run be a single v3 shot (as proposed in §7) or a short 5-epoch probe
   (e.g. `sampler: pow075` + P0 fixes) to verify the AP_NoR trend before committing 8 h?

## 11. Do not repeat

- float32 CPU image decode (host OOM) · PowerShell self-matching process filters ·
  session-bound background shells for long training · sampler exponents that forget the
  `/100` (`"pow075"` must mean 0.75, not 75 — a bug that shipped and was caught by a unit
  test in `engine._make_sampler`) · trusting a class-share number that was normalised
  over per-class weights instead of per-sample weights.
- Hue rotation augmentation (destroys RR/RG semantics) · detector-dependent inference ·
  any test-split model selection · assuming a bigger head or a "smarter NoR readout" fixes
  the tail class when the evidence says the *relevance estimate* is the problem.
