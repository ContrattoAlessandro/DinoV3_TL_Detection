# Throughput audit and safe resume — 30 September 2026

The comparison was paused after the complete epoch-6 checkpoint of
`v4_safe/fold0`. Weights, optimizer, scaler, EMA, best score and random states
were preserved in `runs/city_comparison_20260930/throughput_audit`. The same run
resumes at epoch 7 with faster data delivery. No architecture, loss,
augmentation, precision, resolution, batch size, accumulation or sampler
setting was changed. Both v4-safe and v5 use the same execution settings.

## Measurements

Benchmarks used the actual `engine.fit` two-view pipeline: augmentation,
auxiliary/global losses, backward, clipping, AMP, AdamW, EMA and scalar
logging. Each case restores disposable copies of the saved checkpoint and
random states. Timing follows 16 warmup batches and covers 96 batches;
validation and scientific checkpoint writes are excluded from these training
rates. CPU loader waits and CUDA stage events are recorded. Baseline and the
fastest six-worker case were repeated; the selected two-worker case was also
checked in the GPU equivalence runs. Hardware: RTX 5070 12 GB, eight physical
CPU cores, approximately 31 GiB RAM, PyTorch 2.11.0+cu130.

| Execution | Batch / accumulation | Workers | Prefetch | Images/s |
| --- | --- | ---: | ---: | ---: |
| Original, main CPU threads 8 | 4 / 4 | 0 | — | 21.4; repeats 22.1, 22.2 |
| Main CPU threads 1 | 4 / 4 | 0 | — | 22.1 |
| Threads 1 + pinned memory | 4 / 4 | 0 | — | 22.1 |
| Threads 1 + pinned memory | 4 / 4 | 2 | 2 | 25.0 |
| **Selected: threads 1 + pinned memory** | **4 / 4** | **2** | **4** | **25.2** |
| Threads 1 + pinned memory | 4 / 4 | 4 | 2 | 25.0 |
| Threads 1 + pinned memory | 4 / 4 | 6 | 2 | 25.5 |
| Threads 1 + pinned memory | 4 / 4 | 8 | 2 | 25.4 |
| Exploratory larger batch | 8 / 2 | 6 | 2 | 26.6 |
| Exploratory larger batch | 16 / 1 | 6 | 2 | 26.5 |

The selected configuration delivers approximately **15% more training images
per second**, corresponding to about **13% less training time** under these
measurements. Initial CUDA warmup, Windows worker startup, validation and
checkpoint overhead still contribute to total runtime. A complete resumed
epoch is the final check of the end-to-end improvement.

Training loader wait drops from about 20–22 ms/batch to 0.09 ms/batch. The
selected process tree used roughly 5.1 GiB RSS during the benchmark, versus
9.3 GiB with six workers and 11.3 GiB with eight. Six workers gave less than
1.1% extra steady training throughput over two; their startup and memory
costs outweighed that benefit for this machine.

Validation on the same 1,024 images:

| Execution | Steady images/s | Startup + first 4 batches | Total sample time |
| --- | ---: | ---: | ---: |
| Original | 45.0 | 0.71 s | 23.12 s |
| Selected two workers | 58.5 | 5.67 s | 22.90 s |
| Six workers | 58.0 | 16.04 s | 33.41 s |

The full fold has 7,032 validation images, so faster steady validation matters
more than in this short sample. Two workers improve steady validation by
about 30% and start much faster than six. Fitting temperature took only
approximately 0.03 seconds and is not a worthwhile optimization target.

## Preservation checks

- All 46 regression tests pass, including byte-identical decoded images and
  token targets from parallel workers, and identical main-process Python,
  NumPy and Torch RNG states/sampling order across two train/validation epochs.
- For **both v4-safe and v5**, original and optimized execution produced
  identical sample order and all RNG states, followed by **bit-identical head
  weights after eight real optimizer updates** (32 microbatches). Maximum
  weight difference was zero.
- Validation raw logits on 1,024 images and fitted temperature were also
  bit-identical. The preserved scientific checkpoint file remained unchanged
  throughout benchmarking.

These checks support preserving the training trajectory rather than relying
on a noisy short accuracy test. Full experiment results are still pending.
Persistent workers remain disabled: the original zero-worker loader consumes
a base-seed draw each epoch, and unadapted persistent loaders change that
accounting. Worker startup is a small cost relative to these long epochs.

Larger batches were not adopted. Keeping effective batch 16 does not preserve
every auxiliary loss reduction or the augmentation/dropout draw sequence.
Their additional speed gain here was only around 5% over the selected loader.

## Remaining limit and provenance

After fixing data delivery, the two full-resolution model forward passes take
about 135 ms of a 159 ms batch: roughly 85% of its time. Mixed precision,
SDPA and a frozen backbone are already active. Increasing GPU utilization
alone is not sufficient: batch 16 approached 97% utilization yet delivered
little additional throughput. Caching clean-image backbone features would
change the pixel augmentation training scheme and was not used. The Triton
package required for the usual TorchInductor GPU route is absent in this
Windows environment; no compiler/runtime installation was attempted.

[PyTorch's tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)
motivates overlapping data loading with GPU work and testing pinned-memory
transfers. The decision here is based on local measurements, rather than a
generic worker-count rule.

Scientific configurations and the original model/loss/optimizer source remain
locked. Loader overrides live separately in `runtime_loader.json` and each
fold's `execution_history.jsonl`. A verified execution revision records old
and new source hashes; original source files are retained under
`throughput_audit/source_before_speedup`. Raw timing, resource samples and
equivalence evidence are saved in `throughput_audit/benchmark.json`,
`validation_benchmark.json` and `execution_validation.json`.
