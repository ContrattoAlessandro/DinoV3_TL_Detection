# DinoGlobal-MIL

**Scene-conditioned traffic-light relevance classification with a frozen DINOv3 backbone.**

DinoGlobal-MIL predicts the relevant traffic-light signal for an entire driving
image. A frozen vision transformer supplies dense visual features; three
independent multiple-instance learning (MIL) branches estimate lamp presence,
color state, housing direction, and relevance. Local evidence pooling identifies
small signals, while bounded scene-context paths incorporate global information.
Lamp boxes and attributes supervise the head during training. Inference requires
only an RGB image.

The default **v5** baseline is described below, with canonical configuration
[configs/default.yaml](configs/default.yaml). The optional **v6_axial** variant adds
shared spatial attention to relevance only; its matched fold-0 pilot uses
[configs/spatial_axial.yaml](configs/spatial_axial.yaml).

## Task definition

The output is a three-class softmax, in the fixed order **RR, RG, NoR**.

| Class | Image-level label rule |
|:--|:--|
| **RR** | At least one relevant red, yellow, or red-yellow lamp. |
| **RG** | At least one relevant green lamp, with no relevant RR lamp. |
| **NoR** | No relevant lamp in an RR or RG state. This includes scenes whose only relevant lamps are off or unknown. |

RR takes precedence when relevant lamps have conflicting states. NoR therefore
means *no relevant stop/go signal under this label policy*; it does not assert
that the scene contains no traffic light. Relevance is learned from dataset
annotations. The model is an image classifier, with auxiliary evidence maps.

## Architecture

![DinoGlobal-MIL architecture](docs/assets/architecture.svg)

### 1. Frozen visual encoder and feature fusion

The encoder is DINOv3 **ViT-S+/16**, initialized from
[facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m).
The repository pins the backbone revision in [backbone.py](src/dinov3_global/backbone.py).
All encoder parameters remain frozen, and the encoder stays in evaluation mode
even while the head trains.

An RGB image is prepared at $720\times1280$ and normalized with ImageNet mean
and standard deviation. The $16\times16$ patch size gives a $45\times80$ grid:
**3,600 patch tokens**. The encoder sequence additionally contains a CLS token
and four register tokens, for **3,605 tokens**. Register tokens are discarded
before the head.

For each patch, features from transformer layer 6 and the final output are
concatenated in the order **[final, layer 6]**. The two 384-dimensional vectors
form a 768-dimensional representation. A shared linear projection and layer
normalization produce 192-dimensional patch vectors $h_n$ and a scene vector
$g$ from the corresponding CLS features:

$$
h_n=\operatorname{LN}\!\left(W_p[f_n^{\mathrm{final}};f_n^6]+b_p\right),
\qquad
g=\operatorname{LN}\!\left(W_p[f_{\mathrm{CLS}}^{\mathrm{final}};f_{\mathrm{CLS}}^6]+b_p\right).
$$

### 2. Attribute evidence and scene-conditioned relevance

Each of three independent branches applies a two-layer, width-192 MLP with GELU
and dropout 0.2 to $h_n$, yielding $z_n^{(b)}$. Linear prediction heads estimate:

- lamp probability $\lambda_n$: a sigmoid;
- state distribution $p_n$: a six-way softmax over **green, off, red, red-yellow, unknown, yellow**;
- housing-direction distribution $d_n$: a four-way softmax over **back, front, left, right**;
- relevance probability $r_n$: a sigmoid conditioned on appearance, scene, and geometry.

Lamp, state, and direction heads use $z_n^{(b)}$. The relevance head additionally
uses a bounded multiplicative scene gate, a fixed two-dimensional sinusoidal
position vector $P_n$, normalized geometry $\gamma_n=(x_n,y_n,2x_n-1,1-y_n)$,
and the predicted direction distribution:

$$
\widetilde z_n^{(b)}=
z_n^{(b)}\odot\left(1+0.25\tanh(W_cg+b_c)\right)+P_n,
\qquad
r_n^{(b)}=\sigma\!\left(W_r^{(b)}[\widetilde z_n^{(b)};\gamma_n;d_n^{(b)}]+b_r^{(b)}\right).
$$

The context projection is shared across branches and initialized to zero.
Head-added positions and geometry enter the relevance path; the backbone's own
positional encoding remains part of all visual features. During training, the
entire scene vector is zeroed independently for each image with probability
0.3, without rescaling, across all scene-dependent head paths.

#### Optional relevance-only axial context

`v6_axial` computes two shared axial blocks over the complete 45×80 grid. Each
block uses four-head row attention, four-head column attention, and a width-384
feed-forward network, with pre-LayerNorm, residual connections, and dropout 0.2.
Writing the existing CLS gate as $A$, the shared residual is
$\Delta=\operatorname{Axial}(h\odot A+P)-(h\odot A+P)$.
Each branch adds this residual to $\widetilde z^{(b)}$ before predicting relevance.
Lamp, state, and direction MLP inputs remain unchanged. The head has 1,380,739
trainable parameters, including 891,264 parameters in the shared spatial module.
Missing `decoder.spatial_context` settings preserve v5 checkpoint compatibility.

The frozen DINOv3 features already incorporate spatial attention. This variant
tests whether additional trainable spatial reasoning improves relevance; no
explicit lane geometry or lane-association supervision is added.

Run the matched twelve-epoch pilot and fixed-checkpoint evaluation suite with:

```bash
python scripts/smoke_axial.py
python scripts/run_axial_pilot.py
```

Use `python scripts/run_axial_pilot.py --resume` to resume the pilot. It trains
only fold 0, selects its best EMA checkpoint by validation mAP, then evaluates
DTLD official test, all 528 ATLAS labels, and VZC-TLD. Separate logs and artifacts
are written under `runs/v6_axial_pilot`; the comparison is exported under
`docs/results/spatial_axial_pilot`. The published v5 results remain the baseline.

Each branch constructs evidence for the two signal classes:

$$
e_{n,\mathrm{RR}}^{(b)}=\lambda_n^{(b)}r_n^{(b)}
\left(p_{n,\mathrm{red}}^{(b)}+p_{n,\mathrm{yellow}}^{(b)}+p_{n,\mathrm{red\text{-}yellow}}^{(b)}\right),
\qquad
e_{n,\mathrm{RG}}^{(b)}=\lambda_n^{(b)}r_n^{(b)}p_{n,\mathrm{green}}^{(b)}.
$$

### 3. Local multi-scale MIL pooling

For each class, evidence is reshaped to the patch grid. A lamp-weighted $5\times5$
local mean reduces sensitivity to individual patch activations:

$$
u_{n,c}^{(b)}=
\frac{\operatorname{AvgPool}_{5\times5}(\lambda^{(b)}e_c^{(b)})_n}
{\max\!\left(\operatorname{AvgPool}_{5\times5}(\lambda^{(b)})_n,10^{-4}\right)}.
$$

Pooling has stride 1 and padding 2. The lamp factor appears in both the evidence
definition and the pooling weights, matching the implementation. Branches use
top-$k$ means with **$k\in\{1,2,4\}$**, respectively. Their scores are averaged:

$$
s_c=\frac{1}{3}\sum_{b=1}^{3}
\operatorname{mean}\!\left(\operatorname{TopK}_{k_b}(u_c^{(b)})\right),
\qquad c\in\{\mathrm{RR},\mathrm{RG}\}.
$$

The branches have independent MLPs and attribute heads; they do not merely pool
one shared evidence map at three scales. Exported maps average the three branches.

### 4. Bounded scene paths and image classification

RR and RG logits combine pooled evidence with learned scale and bias, plus a
bounded CLS residual:

$$
\ell_c=a_cs_c+b_c+0.5\tanh\!\left(\operatorname{MLP}_{\mathrm{CLS}}(g)_c\right).
$$

NoR uses an evidence-conditioned base and a bounded scene residual. With
$s=[s_{\mathrm{RR}},s_{\mathrm{RG}}]$ and a shared NoR MLP $f$:

$$
\ell_{\mathrm{NoR}}=f([0;s])+0.5\tanh\!\left(f([g;s])-f([0;s])\right).
$$

The direct scene residual is bounded in magnitude by 0.5 for each class.
Scene information can also affect relevance upstream. The NoR MLP uses separate
dropout draws for its two evaluations during training; evaluation is deterministic.
Final probabilities are $\operatorname{softmax}([\ell_{\mathrm{RR}},\ell_{\mathrm{RG}},\ell_{\mathrm{NoR}}])$.

| Component | Parameters | Optimization |
|:--|--:|:--|
| DINOv3 ViT-S+/16 | 28,692,864 | Frozen |
| Fusion, MIL branches, and scene paths | 489,475 | Trainable |
| Total | 29,182,339 | Head only |

These counts describe one v5 model. The final official-test predictor averages
the raw logits of the four city-fold EMA heads before softmax. Its inference
implementation reuses the common frozen encoder and holds 1,957,900 head
parameters. Individual checkpoint scores are reported alongside the ensemble;
the ensemble is not a single model trained on the entire official training split.

Implementation: [model.py](src/dinov3_global/model.py), [head.py](src/dinov3_global/head.py).

## Training objective

Training uses two independently augmented, full-frame photometric views.
Image-level cross-entropy is averaged across both views. Auxiliary token losses
are computed for each branch on the first view and then averaged across branches.
A symmetric KL penalty encourages consistency between image predictions:

$$
\mathcal L=\tfrac12(\mathcal L_{\mathrm{global}}^{(1)}+\mathcal L_{\mathrm{global}}^{(2)})
+\tfrac13\sum_b\left(0.3\mathcal L_{\mathrm{lamp}}^{(b)}
+0.2\mathcal L_{\mathrm{state}}^{(b)}+0.5\mathcal L_{\mathrm{rel}}^{(b)}
+0.2\mathcal L_{\mathrm{dir}}^{(b)}\right)
+\eta_t\mathcal L_{\mathrm{symKL}}.
$$

Global CE uses label smoothing 0.05 and training-time logit adjustment
$\ell_c+0.35\log\pi_c$, where $\pi_c$ is the natural training-class frequency.
Sampling uses replacement with per-example weight proportional to class
frequency raised to $-0.75$. These are distinct operations.

Lamp BCE covers valid tokens with positive weight 10. State and direction CE
cover annotated lamp tokens. State CE uses inverse-square-root frequency weights
and a $3\times$ multiplier for relevant lamp tokens. Relevance uses focal BCE
($\gamma=2$, positive weight 2 on lamps), combining separately normalized lamp
and background losses with mass 0.75 and 0.25. Conflicting box overlaps and
surrounding ignore bands are masked out. The KL coefficient increases linearly
to 0.1 over the first three epochs.

Photometric and sensor/weather augmentation preserves the full frame. It uses
strength 0.8, clean-view probability 0.2, and minimum clean-image blend 0.35.
Training also applies all-lamp erasure with probability 0.02, updating the image
label and token targets consistently. There are no random geometric crops.

| Setting | Default |
|:--|:--|
| Optimizer | AdamW, learning rate $6\times10^{-5}$, weight decay 0.05 |
| Schedule | 12 epochs; 3 warm-up epochs, then cosine decay |
| Batch | 4 images, accumulation 4; effective batch 16 |
| Precision | Mixed precision on CUDA; float32 on CPU |
| Gradient clipping | Norm 1.0 |
| EMA | Decay 0.999 after every optimizer step |
| Checkpoint selection | Highest validation mAP at temperature $T=1$; early stopping disabled |

See [losses.py](src/dinov3_global/losses.py), [augment.py](src/dinov3_global/augment.py),
and [engine.py](src/dinov3_global/engine.py).

## Repository structure

```text
configs/             Default scientific settings and audited runtime settings
src/dinov3_global/   Encoder, v5 head, data, losses, training, and metrics
scripts/             Data preparation, training, inference, and evaluation
tests/               CPU tests; no encoder download required
metadata/            Frozen folds, checkpoint hashes, labels, environment
docs/                Data protocol, evaluation protocol, figures, results
datasets/            Local datasets; excluded from Git
runs/v5/             Local checkpoints, logs, and predictions; excluded from Git
```

## Installation

Use Python **3.11 or later** and run commands from the repository root:

```sh
python -m pip install -e .
```

Install a PyTorch build appropriate for your CUDA environment using the
[official installation instructions](https://pytorch.org/get-started/locally/).
The tested software versions and GPU are recorded in
[metadata/environment.json](metadata/environment.json).

Access to the DINOv3 weights requires accepting their license on the model page
and authenticating with `hf auth login`. Alternatively, set
`backbone.local_ckpt` to a licensed local Hugging Face snapshot directory.
Encoder weights, dataset images, and trained head checkpoints are not bundled
in Git. The retained local EMA checkpoints and their SHA-256 hashes are listed in
[metadata/checkpoints.json](metadata/checkpoints.json).

## Data preparation

Obtain DTLD under its dataset terms and place the native annotations in
`datasets/DTLD/v2.0/`. Convert Bayer TIFFs and prepare the image dump:

```sh
python -m pip install -e ".[preprocessing]"
python scripts/preprocessing/convert_dtld.py --raw datasets/DTLD --labels datasets/DTLD/v2.0 --out datasets/DTLD_jpg
python scripts/prepare_dtld.py --src datasets/DTLD_jpg --dst datasets/DTLD_1280
```

The deterministic side crop is 114 pixels per side before bicubic resizing.
The configuration distinguishes the crop already baked into the dump
(`label_crop_sides: 114`) from any additional input crop (`crop_sides: 0`).
See [docs/data.md](docs/data.md) for coordinate mapping, annotation policies,
ATLAS provenance, and VZC audits.

## Training and reproduction

Reproduce the four city-disjoint folds using the frozen membership manifest and
the runtime settings used for the retained runs. Choose a fresh output directory:

```sh
python scripts/cross_validate.py --fold-manifest metadata/dtld_city_folds.json --runtime-loader configs/runtime.json --out runs/v5/reproduction --audit-only
python scripts/cross_validate.py --fold-manifest metadata/dtld_city_folds.json --runtime-loader configs/runtime.json --out runs/v5/reproduction
```

Add `--resume` to continue from saved epoch-boundary checkpoints. Fold seeds are
0, 1, 2, and 3. The manifest checks frame membership, annotation checksums, class
coverage, and absence of session overlap. The runtime file controls workers,
prefetching, pinning, and CPU threads without altering the scientific settings.

A separate training run uses a session-disjoint validation partition within the
official DTLD training split:

```sh
python scripts/train.py --out runs/v5/train
python scripts/train.py --out runs/v5/train --resume runs/v5/train/last.pt
```

`best.pt` contains the selected head and EMA parameters; `last.pt` additionally
stores optimizer, mixed-precision scaler, and RNG state for resume. Inference
loads EMA weights. The frozen encoder is loaded separately.

## Inference and evaluation

Predict an image directory, optionally exporting evidence overlays:

```sh
python scripts/infer.py --ckpt runs/v5/city_cv/fold0/best.pt --images path/to/images --out runs/v5/inference --overlays
```

Reproduce the final official DTLD test evaluation of the four retained
checkpoints and their mean-logit ensemble in a fresh output directory:

```sh
python scripts/evaluate.py --ckpt runs/v5/city_cv/fold0/best.pt runs/v5/city_cv/fold1/best.pt runs/v5/city_cv/fold2/best.pt runs/v5/city_cv/fold3/best.pt --runtime-loader configs/runtime.json --out runs/v5/dtld_test_reproduction
```

Primary metrics always use raw logits at $T=1$. Optional `--calibrated` uses a
previously fitted DTLD validation temperature; it never fits calibration on test
or transfer images. Multiple `--ckpt` arguments to evaluation average raw logits
before the final softmax. The evaluator saves individual and ensemble results,
audits official membership and train/test overlap, and freezes checkpoint,
data, source, and environment hashes before inference. The identical frozen
encoder is shared across heads, with exact agreement checked against separate
full-model forwards. Supplying one checkpoint evaluates a single head.

ATLAS evaluation uses the fixed manual annotation snapshot:

```sh
python scripts/evaluate_ood.py --labels metadata/atlas_labels.json --images datasets/atlas_relevance_expanded --ckpt runs/v5/city_cv/fold0/best.pt --include-uncertain --out runs/v5/atlas
```

The retained ATLAS benchmark includes uncertain annotations; omitting
`--include-uncertain` produces the separately defined certain-only diagnostic.
For a new image collection, use `scripts/annotate_ood.py --images path/to/images`.

Download and evaluate the pinned VZC-TLD release:

```sh
python scripts/download_vzc.py
python scripts/evaluate_vzc.py --out runs/v5/vzc_reproduction
```

The VZC evaluator checks file hashes and category semantics, evaluates the four
retained city-fold checkpoints plus their logit ensemble, and exports primary
and sensitivity metrics. No VZC data trains the model or fits temperature scaling.
Details are in [docs/evaluation.md](docs/evaluation.md).

These inference examples use the retained checkpoint paths. To evaluate a new
reproduction run, replace them with `runs/v5/reproduction/fold{0..3}/best.pt`,
supplying all four paths to `evaluate_vzc.py --ckpt` for the same suite.

## Recorded results

All scores below are percentages at $T=1$. The official DTLD test reports the
fixed four-head ensemble and the arithmetic mean of individual checkpoint
metrics separately. The other rows report individual-checkpoint means. DTLD
city validation contains 28,525 unique out-of-fold images; every checkpoint sees
the same official-test or transfer images within its respective benchmark.
mAP averages the three image-level class APs over RR, RG, and NoR.

| Evaluation / predictor | Images | AP RR | AP RG | AP NoR | mAP | Balanced accuracy | Accuracy | Macro F1 | ECE ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **DTLD official test — ensemble** | **12,453** | **91.41** | **95.04** | **24.92** | **70.46** | **75.89** | **84.65** | **69.58** | **11.75** |
| DTLD official test — individual mean | 12,453 | 90.11 | 94.40 | 21.17 | 68.56 | 73.75 | 81.85 | 66.95 | 9.83 |
| DTLD city validation | 28,525 | 91.52 | 95.45 | 24.78 | 70.58 | 74.54 | 82.53 | 68.39 | 9.95 |
| ATLAS transfer | 528 | 88.23 | 75.25 | 75.18 | 79.55 | 69.68 | 71.35 | 69.85 | 7.67 |
| VZC published test split | 598 | 88.38 | 71.66 | 41.17 | 67.07 | 61.46 | 55.77 | 53.56 | 13.53 |

Per-checkpoint scores, per-class precision/recall/F1, confusion matrices, NLL,
Brier score, city/size slices, and sensitivity analyses are provided in
[docs/results/README.md](docs/results/README.md).
Rebuild the tables from local saved predictions with
`python scripts/summarize_results.py`.

The final official-test evaluation covers all 12,453 frames with no missing or
excluded images. Train/test image identities, sessions, and exact prepared-JPEG
contents have zero overlap. Checkpoint selection uses city validation only;
the ensemble, preprocessing, and $T=1$ policy were fixed before test inference.
The [frozen test protocol](docs/results/dtld_test_protocol.json) and
[artifact provenance](docs/results/dtld_test_provenance.json) identify the exact
weights, data, source, environment, and verification used for this result.

The ensemble improves mAP and balanced accuracy over the individual-model mean
by 1.90 and 2.14 percentage points, respectively, while ECE increases. NoR
discrimination remains limited: ensemble NoR precision/recall are 22.33%/57.71%.
The reported predictor uses the retained city-fold models; there is no full-training-split
refit. Official-test cities also occur in the training split, whereas city
validation holds out entire cities. ATLAS and VZC are exploratory transfer
benchmarks inspected during architecture selection. Class prevalence and
annotation policies differ across datasets, so their scores are not directly
interchangeable. See [docs/evaluation.md](docs/evaluation.md) for the complete protocol.

## Development

The DTLD-only recall/generalization study is implemented in
[docs/generalization_study.md](docs/generalization_study.md), with four staged
ablations, validation-gated checkpoint selection and a fixed promotion rule.
Run `python -B -u scripts/run_generalization_study.py` and use `--resume` to
continue the same study. Existing v5 and axial results remain the reference;
the proposed changes require measured acceptance before promotion.

```sh
python -m pip install -e ".[dev]"
python -m pytest
ruff check src scripts tests
ruff format --check src scripts tests
```

Tests cover label priority, box-to-token geometry, loss masks, local pooling,
scene bounds, augmentation consistency, frozen-encoder behavior, fold leakage,
RNG resume, worker invariance, and metric semantics. GitHub Actions runs the CPU
suite and style checks. Refactoring equivalence with the original v5 implementation
is recorded in [metadata/refactor_verification.json](metadata/refactor_verification.json).

## References and license

- Siméoni et al., **DINOv3**, 2025. [Paper](https://arxiv.org/abs/2508.10104).
- DriveU Traffic Light Dataset: [native parser and dataset references](https://github.com/julimueller/dtld_parsing).
- Trinci et al., **Color Is Not Enough: Dataset and Method for Identifying Relevant Traffic Lights in Driving Scenes**, IEEE T-ITS, 2026. [DOI](https://doi.org/10.1109/TITS.2025.3626165), [VZC dataset](https://huggingface.co/datasets/vzc-research-chapter/vzc-traffic-light-dataset).

The repository retains its [AGPL-3.0 license](LICENSE). DINOv3 weights and each
dataset have separate terms. The DTLD conversion helpers retain their upstream
author attribution; see [docs/data.md](docs/data.md).
