# Frozen ViT-B RGB evidence recall pilot

Historical research record. Experimental runners are archived in [the source snapshot](recall_head/source_snapshot.zip); active training uses [the supported configuration](../../configs/default.yaml). Commands below describe the original study.
This experiment targets missed small lamps and incorrectly relevant signal
evidence while retaining the original frozen DINOv3 encoder. It is a candidate,
not a measured improvement. Historical checkpoints and results remain read-only.

## Architecture and losses

`v8_rgb_evidence` retains the complete semantic evidence head. Two stride-2
3×3 convolutions (16/32 channels, GroupNorm, GELU) encode the same augmented RGB
frame. Aligned 4×4 mean/max pooling and a 32-wide projection produce one feature
per 16-pixel patch. A 64-wide residual MLP consumes these features and semantic
appearance. Its zero-initialized output adds bounded lamp/state corrections
(±3/±1 logits). Relevance inputs and the pooling/readout are unchanged. The
branch adds 22,119 trainable parameters, for a 434,949-parameter ViT-B head.

The optional revised lamp loss assigns 25% mass to instance-averaged positives
and 75% to valid background, normalized within each image and renormalized when
a group is absent. It uses no additional positive multiplier. Joint evidence
BCE, weighted 0.1, supervises the actual lamp × relevance × compatible-state
probabilities on lamp supports. It masks joint state/relevance conflicts and
averages annotated instances, then images. Background, padding, and erased
lamps do not enter this joint loss. Independent attribute masks are preserved.

## Protocol and commands

Run the complete resumable study:

```powershell
python scripts/recall_pilot.py --out runs/recall_head_pilot
python scripts/recall_pilot.py --out runs/recall_head_pilot --resume
```

The initial 2×2 comparison starts from the retained ViT-B fold-0 epoch-7 EMA:
original control, supervision only, RGB only, and both changes. Each uses at most
six additional epochs, one warm-up epoch, patience two, natural sampling,
unchanged augmentation, and effective batch 16. Existing tensors use LR 1e-5;
RGB tensors use LR 6e-5. Both rates share the cosine/warm-up multiplier. EMA,
clipping, checkpointing, and resume include the complete new head. These are
fresh continuation optimizers, rather than resumes of the historical run.

The runner freezes source, annotation, backbone, cached-reference, checkpoint,
and membership identities. Epoch zero preserves the untouched predictor. Raw
city-validation gates preserve RR recall, RG recall, mAP, and worst-city mean
signal recall; NoR recall and each signal precision may decline by at most
two points. Selection maximizes mean signal recall, then worst-city recall,
then mAP, retaining earlier ties. Rejected epochs count toward patience.

A candidate must improve beyond both epoch zero and matched continuation.
Only then does the runner check â‰¤10% batch-1 latency overhead, train matched
baselines/control/candidate on folds 1–3 and fold 0 seed 1, and freeze the final
single fold-0 candidate before external evaluation. It stops at a failed gate.
No external results select a variant or epoch.

The final comparison requires +1 point mean RR/RG recall on each dataset,
nondecreasing individual signal recalls, mAP, balanced accuracy, and macro F1,
and the same two-point NoR/precision tolerance. Passing writes `recommended.pt`;
otherwise the baseline is retained. The default configuration is not rewritten.

`status.json`, per-stage logs, cell plans, complete epoch histories, selected
EMA checkpoints, validation predictions, replication reports, and `decision.json`
record progress and every rejection. External reports include paired prediction
transitions and 2,000-replicate DTLD session-bootstrap difference intervals.
External scene grouping is unavailable, so external uncertainty is not reported
as if images were independent. Existing ATLAS/VZC results remain exploratory.

The standalone `configs/recall_rgb.yaml` supports smoke/inference integration;
the study generates its constrained configurations with immutable references.

```powershell
python scripts/smoke.py --config configs/recall_rgb.yaml --out runs/checks/recall_rgb_smoke
python -m pytest --basetemp runs/checks/recall_tests
```

DTLD and VZC additionally report size slices based on the largest relevant lamp
compatible with the image's target class, measured after resizing. NoR has no
such target (stored as -1); these signal-only slices report their present class
counts. Legacy largest-any-lamp slices remain available. ATLAS lacks equivalent
relevance-box ground truth, so no target-size slice is invented there.

## Verification completed before training

The real 720×1280, batch-4 CUDA smoke verifies bit-exact zero-residual equivalence,
RGB convolution gradients, frozen encoder parameters, both training views,
safe EMA loading, and bit-exact head/EMA restoration after an epoch-boundary
resume. Its peak GPU allocation was 1,417,247,744 bytes. The preliminary architecture
timing used an untrained smoke head; final candidate latency is remeasured only
after selection and is not inferred from this preflight.
