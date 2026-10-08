# DINOv3 capacity pilot — fold 0

Historical research record. Experimental runners are archived in [the source snapshot](recall_head/source_snapshot.zip); active training uses [the supported configuration](../../configs/default.yaml). Commands below describe the original study.
Requested on 7 October 2026. Replace the frozen ViT-S+/16 encoder with
ViT-B/16 and train a fresh Experiment C head. The canonical default and recorded
C results remain the reference. This runner evaluates only fold 0.

| Setting | Reference | Candidate |
|:--|:--|:--|
| Encoder | DINOv3 ViT-S+/16 | DINOv3 ViT-B/16 |
| Hugging Face ID | facebook/dinov3-vits16plus-pretrain-lvd1689m | facebook/dinov3-vitb16-pretrain-lvd1689m |
| Pinned revision | c93d816fc9e567563bc068f01475bec89cc634a6 | 5931719e67bbdb9737e363e781fb0c67687896bc |
| Feature width / fused width | 384 / 768 | 768 / 1536 |
| Encoder depth / fusion | 12 layers; layer 6 and final | Same |
| Head output width / trainable parameters | 192 / 265,374 | 192 / 412,830 |
| Input / batch / accumulation | 720×1280 / 4 / 4 | Same |
| Training split | 21,493 DTLD images | Same |
| Validation split | 7,032 images; Dortmund, Kassel, Fulda | Same |
| Seed / schedule | 0 / 12 epochs, 3 warm-up | Same |
| Checkpoint selection | Highest validation EMA mAP, epoch 5 | Highest validation EMA mAP; earliest tie |

The wider features require a wider input projection. The head's depth, internal
width, loss weights, augmentation, pooling, optimizer, and EMA settings are
unchanged. Both variants use the LVD-1689M pretrained family. Configuration:
[original effective configuration](backbone_capacity/results/training/effective_config.json).

The baseline is the retained `runs/pretrained/experiment_c_fold0.pt` checkpoint,
verified against its recorded SHA-256. Published raw predictions are aligned
against candidate image identities and labels before comparison; mismatches
fail. Their recomputed mAP must agree with the published reports.

| Primary benchmark | Images | Reference mAP | Reference balanced accuracy |
|:--|--:|--:|--:|
| DTLD official test | 12,453 | 70.11% | 68.76% |
| ATLAS fixed manual-label benchmark, including uncertain | 528 | 86.84% | 76.12% |
| VZC-TLD published test | 598 | 72.17% | 65.54% |

All primary scores use raw logits at T=1. Test data never chooses the epoch or
fits calibration. VZC train/all memberships are separate diagnostics. ATLAS and
VZC previously informed development, so these are exploratory comparisons.

The recommendation to try more folds requires improved validation mAP, no test
mAP regression on any of the three datasets, and at least one test mAP increase.
Balanced accuracy, accuracy, macro F1, class AP and RR/RG/NoR recall are reported
alongside mAP. This criterion screens point estimates; one seed and fold do not
establish statistical significance. Folds 1–3 require a subsequent decision and
are never launched by this runner.

## Execution and recovery

Place the licensed candidate snapshot (`config.json` and `model.safetensors`)
at `runs/backbones/dinov3_vitb16`, using the pinned revision above. Run the actual
CUDA smoke test before training:

```sh
python scripts/smoke.py --config configs/backbone_vitb.yaml --out runs/backbone_vitb_fold0/smoke
python scripts/backbone_pilot.py --out runs/backbone_vitb_fold0
```

The runner freezes configuration, dataset manifests, source files, pretrained
weights, baseline checkpoint/results, and the smoke report in `plan.json` before
training. It records its PID, child PID, stage and progress in `status.json`.
Detailed subprocess logs are in `logs/`; epoch checkpoints/history are in
`training/fold0/`. It performs all three evaluations after the 12-epoch run,
then writes `comparison.json`, `comparison.csv`, and `comparison.md`.

To recover an interrupted run with unchanged inputs:

```sh
python scripts/backbone_pilot.py --out runs/backbone_vitb_fold0 --resume
```

Finished stages are skipped. Training resumes from the last completed epoch;
interrupted evaluations use new output directories. Do not start a second
runner while its recorded PID is still active. Generated measurements stay in
`runs/` and do not overwrite the historical result tables.

## Post-training recovery

All 12 epochs completed, selecting epoch 7 by validation mAP (70.13195734%).
The original controller had stopped during epoch 1 after a transient Windows
reader lock prevented replacement of `status.json`; its training subprocess
continued and completed normally. No dataset evaluation had started.

The recovery preserves `plan.json`, the failed status, and the original training
source manifest. `recovery_plan.json` records the two source amendments: bounded
retry of atomic progress-file replacement, and support for an explicit recovery
plan. It verifies the original input identities and records hashes of the
completed checkpoint, history and fold report. Model, loss, augmentation, data
and training configuration remain unchanged.

```sh
python scripts/backbone_pilot.py --out runs/backbone_vitb_fold0 --resume --plan runs/backbone_vitb_fold0/recovery_plan.json
```

The completed training stage is recovered from its verified artifacts, so this
command runs only missing evaluations and the comparison. Regression checks
after the progress-write repair passed all 94 tests.

The [completed results](backbone_capacity/results/README.md) include matched
ViT-S+/ViT-B metrics for all three datasets, class metrics, confusion matrices,
dataset slices, sensitivities, training records, and measured inference latency.
ViT-B passes the recorded criterion for considering the remaining folds; no
additional folds have been launched.
