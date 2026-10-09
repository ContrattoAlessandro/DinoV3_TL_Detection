# Released training record

The released predictor is one fold-0, seed-0, epoch-7 EMA head. It trained for
12 epochs with frozen DINOv3 ViT-B/16 on 21,493 DTLD frames, holding out 7,032
frames from Dortmund, Kassel, and Fulda. Highest validation mAP selected epoch 7
at **70.13195734%**.

| Artifact | Purpose |
|:--|:--|
| [configuration.json](configuration.json) | Released model's scientific configuration; local encoder path normalized to null. |
| [history.jsonl](history.jsonl) | Original per-epoch training and validation measurements. |
| [fold0_report.json](fold0_report.json) | Original selected validation report. |
| [City membership](../../metadata/dtld_city_folds.json) | Frozen annotation and ordered-image identities for the four city partitions. |
| [Environment](../../metadata/environment.json) | Recorded Python/package/CUDA versions. |

The configuration describes the recorded run; `configs/default.yaml` is the
canonical configuration for fresh training. Local file locations are runtime
settings and do not change the scientific model. Raw metric values are preserved.
The histories are descriptive records, not a substitute for resumable optimizer
states. New runs write `best.pt` and `last.pt` to their own output directory.

Use the fold-0 commands in the [main README](../../README.md). Changes in source,
hardware, library versions, or RNG operation order can alter a fresh trajectory;
the bundled checkpoint provides the fixed predictor for the published scores.
Other folds and seeds require new measurements.
