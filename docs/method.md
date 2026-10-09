# Method and training protocol


![DinoGlobal-MIL architecture](assets/architecture.svg)

### Full-frame preprocessing and frozen representation

An RGB image is resized isotropically and centered on a **720 Ã— 1280** letterbox
canvas using bicubic interpolation. The complete field of view is retained.
Padding uses RGB (124, 116, 104); a content mask excludes patches entirely within
padding from supervision and evidence pooling. Patch geometry is expressed in
the resized content coordinates, so letterbox margins do not shift its meaning.

The default encoder is **DINOv3 ViT-B/16**, loaded from
[facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
at the revision pinned in [backbone.py](../src/dinov3_global/backbone.py).
All encoder parameters are frozen and the encoder remains in evaluation mode.
Pixels are normalized with ImageNet mean and standard deviation.

The patch grid has **45 Ã— 80 = 3,600 tokens**. The encoder additionally returns
one CLS token and four register tokens; the registers are discarded. Final-layer
and layer-6 features, each 768-dimensional, are concatenated in that order.
A shared projection and LayerNorm map both patch and CLS features to width 192:

$$
h_n = \operatorname{LN}(W_p[f_n^{\mathrm{final}}; f_n^6]+b_p),
\qquad
g = \operatorname{LN}(W_p[f_{\mathrm{CLS}}^{\mathrm{final}}; f_{\mathrm{CLS}}^6]+b_p).
$$

### Shared attribute evidence

One shared two-layer MLP, with width 192, GELU, and dropout 0.2 after each layer,
maps $h_n$ to an appearance representation $z_n$. Linear attribute heads predict:

| Output | Activation | Annotation order |
|:--|:--|:--|
| Lamp presence $\lambda_n$ | Sigmoid | Background / lamp |
| State $p_n$ | Six-way softmax | green, off, red, red-yellow, unknown, yellow |
| Housing direction $d_n$ | Four-way softmax | back, front, left, right |
| Pictogram $q_n$ | Ten-way softmax | left arrow, right arrow, straight arrow, straight-left arrow, bicycle, circle, pedestrian, pedestrian-bicycle, tram, unknown |

Attribute predictors consume appearance features. Direction and pictogram
probabilities also enter the relevance predictor and receive gradients through
that path. The three pooling scales reuse these same predictions.

### Appearance relevance with bounded context correction

Let $a_n=[z_n;d_n;q_n]\in\mathbb{R}^{206}$ and
$\gamma_n=(x_n,y_n,2x_n-1,1-y_n)$ denote content-relative patch geometry.
Two width-64 MLPs estimate base relevance and a context correction:

$$
\delta_n = \tanh f_{\mathrm{ctx}}([a_n;\gamma_n;g]),
\qquad
r_n = \sigma(f_{\mathrm{base}}(a_n)+\delta_n).
$$

Each MLP uses Linearâ€“GELUâ€“Dropout(0.2)â€“Linear. The correction is bounded to
**Â±1 relevance logit**, and its final layer starts at zero. During training,
scene and geometry inputs are jointly zeroed with probability **0.5 per image**,
without rescaling; appearance inputs to the correction remain available.
The head adds no absolute positional embedding or direct CLS-to-class logit path.
The frozen transformer features already contain spatial context.

### Local evidence and multi-scale MIL

Signal evidence combines lamp presence, relevance, and compatible state:

$$
e_{n,\mathrm{RR}}=\lambda_n r_n
(p_{n,\mathrm{red}}+p_{n,\mathrm{yellow}}+p_{n,\mathrm{red\text{-}yellow}}),
\qquad
e_{n,\mathrm{RG}}=\lambda_n r_n p_{n,\mathrm{green}}.
$$

For each class, a lamp-weighted **5 Ã— 5** local average (stride 1, padding 2)
reduces sensitivity to isolated patch activations:

$$
u_{n,c}=
\frac{\operatorname{AvgPool}_{5\times5}(\lambda e_c)_n}
{\max(\operatorname{AvgPool}_{5\times5}(\lambda)_n,10^{-4})},
\qquad
s_c=\frac13\sum_{k\in\{1,2,4\}}
\operatorname{mean}(\operatorname{TopK}_k(u_c)).
$$

Padding-only patches have zero lamp evidence and zero local scores. The lamp
factor appears both in $e_c$ and in the local pooling weights, as implemented.
Top-$k$ aggregation uses **one shared evidence map** at three scales.

### Monotonic image-level readout

Learned positive scales produce RR/RG logits and suppress NoR when either signal
score increases:

$$
\ell_c=\operatorname{softplus}(\alpha_c)s_c+b_c,
\qquad c\in\{\mathrm{RR},\mathrm{RG}\},
$$

$$
\ell_{\mathrm{NoR}}=b_{\mathrm{NoR}}
-\operatorname{softplus}(\beta_{\mathrm{RR}})s_{\mathrm{RR}}
-\operatorname{softplus}(\beta_{\mathrm{RG}})s_{\mathrm{RG}}.
$$

Thus $\partial\ell_{\mathrm{NoR}}/\partial s_c<0$ by construction.
Signal scales initialize to 2.5 and NoR scales to 2.0. Biases initialize to zero.
Probabilities are $\operatorname{softmax}([\ell_{\mathrm{RR}},\ell_{\mathrm{RG}},\ell_{\mathrm{NoR}}])$.
This constraint does not guarantee calibrated probabilities or high NoR recall.

| Component | Parameters | Optimization |
|:--|--:|:--|
| DINOv3 ViT-B/16 | 85,660,416 | Frozen |
| Shared fusion projection and normalization | 295,488 | Trainable |
| Appearance trunk | 74,112 | Trainable |
| Lamp/state/direction/pictogram heads | 4,053 | Trainable |
| Base relevance / context correction | 13,313 / 25,857 | Trainable |
| Image-level readout | 7 | Trainable |
| **Complete head** | **412,830** | **Head only** |
| **Total model** | **86,073,246** | |

Implementation: [model.py](../src/dinov3_global/model.py),
[head.py](../src/dinov3_global/head.py), and [pooling.py](../src/dinov3_global/pooling.py).

## Training objective and optimization

Two independently augmented views retain the complete scene. Image-level
cross-entropy is averaged across both views. The retained protocol applies
attribute supervision to view 1.
A symmetric KL term encourages prediction consistency:

$$
\mathcal{L}=\tfrac12(\mathcal{L}_{\mathrm{CE}}^{(1)}+\mathcal{L}_{\mathrm{CE}}^{(2)})
+0.3\mathcal{L}_{\mathrm{lamp}}+0.2\mathcal{L}_{\mathrm{state}}
+0.5\mathcal{L}_{\mathrm{rel}}+0.2\mathcal{L}_{\mathrm{dir}}
+0.1\mathcal{L}_{\mathrm{pictogram}}+\eta_t\mathcal{L}_{\mathrm{symKL}}.
$$

Global CE uses label smoothing 0.05 and inverse-square-root class-frequency
weights, normalized to expected weight one under natural training frequencies.
Sampling follows the natural image distribution, without logit adjustment.
Lamp BCE uses positive weight 10. Relevance uses focal BCE ($\gamma=2$,
positive weight 2), with separately normalized lamp/background masses 0.75/0.25.
State CE uses inverse-square-root frequency weights and a 3Ã— relevant-lamp
multiplier. Pictogram CE uses inverse-square-root weights capped at 3.
Direction CE is unweighted.

State, direction, pictogram, and lamp-region relevance losses normalize within
lamp instances, then within images. Lamp detection and background relevance
normalize over valid tokens within each image. Shared,
agreeing patches contribute to each applicable instance. Conflicting attributes
are masked independently; padding and boundary ignore bands are excluded.
The KL coefficient ramps to 0.1 over three epochs. See
[losses.py](../src/dinov3_global/losses.py) and [preprocessing.py](../src/dinov3_global/preprocessing.py).

Training letterboxing optionally zooms the complete image to a scale in
[0.9, 1.0] with probability 0.5 and random placement. Photometric and
sensor/weather augmentation uses strength 0.8, clean-view probability 0.2,
and minimum clean-image blend 0.35. All-lamp erasure occurs with probability
0.02 and updates labels and attribute targets. No geometric crop is used.

| Setting | Default |
|:--|:--|
| Optimizer | AdamW, learning rate $6\times10^{-5}$, weight decay 0.05 |
| Schedule | 12 epochs, 3 warm-up epochs, then cosine decay |
| Batch | 4 images, accumulation 4, effective batch 16 |
| Precision / clipping | CUDA mixed precision; gradient norm 1.0 |
| EMA | Decay 0.999 after every optimizer step |
| New-run selection | Highest validation EMA mAP at T=1; earlier epoch wins ties |
| Clean training probe | 1,024 fixed images; diagnostic only |

Training fits only the head on DTLD training data. Checkpoint selection uses
held-out validation mAP; test and transfer predictions do not fit the epoch,
thresholds, or temperature. The released checkpoint uses the recorded fold-0 city-validation partition.
