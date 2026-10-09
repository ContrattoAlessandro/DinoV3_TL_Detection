# Released checkpoint

`dinoglobal_vitb_fold0.pt` is the retained fold-0, seed-0, epoch-7 checkpoint.
It contains head and EMA state dictionaries plus configuration and validation
metadata. **Inference loads EMA.** It contains no DINOv3 encoder weights.

| Property | Value |
|:--|:--|
| Encoder | Frozen DINOv3 ViT-B/16, layer-6/final fusion |
| Trainable head | 412,830 parameters |
| Head identifier | `v7_evidence` |
| Output order | RR, RG, NoR |
| SHA-256 | `edb2bb7bfd4a40cf470cd62f1dbd14e25b14d9c8feb107207be03922c528042e` |

Acquire the encoder separately under its source terms. The loader uses the
licensed local cache at `runs/backbones/dinov3_vitb16/` when present; otherwise
it resolves the pinned Hugging Face identity. Use `torch.load(...,
weights_only=True)` for the checkpoint; strict head loading is enforced.

[Model card](../docs/model_card.md) ·
[Training record](../docs/reproduction/README.md) ·
[Evaluation results](../docs/results/README.md)
