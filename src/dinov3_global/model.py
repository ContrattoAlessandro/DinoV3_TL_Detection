"""Frozen DINOv3 encoder and the shared attribute-evidence MIL head."""

from __future__ import annotations
import torch
from torch import nn
from .backbone import BACKBONE_ID, DinoV3Backbone
from .head import EvidenceMILHead


class DinoGlobal(nn.Module):
    def __init__(
        self,
        hf_id=BACKBONE_ID,
        attn_implementation="sdpa",
        dtype="float16",
        local_ckpt=None,
        grid_hw=(45, 80),
        mid_layer=6,
        head_kind="v7_evidence",
        **head_kwargs,
    ):
        super().__init__()
        if mid_layer != 6:
            raise ValueError("DinoGlobal-MIL fuses layer 6 and the final layer")
        if head_kind != "v7_evidence":
            raise ValueError("Only the DinoGlobal-MIL evidence head is supported")
        self.backbone = DinoV3Backbone(
            hf_id, attn_implementation, dtype, local_ckpt, grid_hw=grid_hw, mid_layer=mid_layer
        )
        self.head = EvidenceMILHead(in_dim=2 * self.backbone.dim, grid_hw=grid_hw, **head_kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def head_parameters(self):
        return self.head.parameters()

    def forward(self, images, content_mask=None, geometry=None):
        with torch.no_grad():
            features = self.backbone(images)
        return self.forward_features(features, content_mask=content_mask, geometry=geometry)

    def forward_features(self, features, content_mask=None, geometry=None):
        """Apply the head to shared frozen features using the normal forward contract."""
        metadata = {
            k: v for k, v in dict(content_mask=content_mask, geometry=geometry).items() if v is not None
        }
        logits, auxiliary = self.head(
            features["patches"].float(),
            features["cls"].float(),
            patches_mid=features["patches_mid"].float(),
            cls_mid=features["cls_mid"].float(),
            **metadata,
        )
        return {"logits": logits, **auxiliary}
