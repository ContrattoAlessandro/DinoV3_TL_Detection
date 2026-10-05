"""Frozen DINOv3 ViT-S+/16, layer-6/final fusion, and the v5 MIL head."""

from __future__ import annotations
import torch
from torch import nn
from .backbone import DinoV3Backbone
from .head import TokenMILHead


class DinoGlobal(nn.Module):
    def __init__(
        self,
        hf_id="facebook/dinov3-vits16plus-pretrain-lvd1689m",
        attn_implementation="sdpa",
        dtype="float16",
        local_ckpt=None,
        grid_hw=(45, 80),
        mid_layer=6,
        **head_kwargs,
    ):
        super().__init__()
        if mid_layer != 6:
            raise ValueError("v5 fuses layer 6 and the final layer")
        self.backbone = DinoV3Backbone(
            hf_id, attn_implementation, dtype, local_ckpt, grid_hw=grid_hw, mid_layer=mid_layer
        )
        self.head = TokenMILHead(in_dim=2 * self.backbone.dim, grid_hw=grid_hw, **head_kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        logits, auxiliary = self.head(
            features["patches"].float(),
            features["cls"].float(),
            patches_mid=features["patches_mid"].float(),
            cls_mid=features["cls_mid"].float(),
        )
        return {"logits": logits, **auxiliary}
