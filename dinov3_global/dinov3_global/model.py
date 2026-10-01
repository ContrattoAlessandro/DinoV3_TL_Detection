"""DinoGlobal: frozen ViT-S/S+ backbone + trainable head. Only the head has grads.

Config `decoder.head` selects the head:
  - "mil"  (default, v2): TokenMILHead - per-token evidence + MIL pooling.
  - "attn" (legacy exp1): single-query cross-attention decoder (ablation).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import DinoV3Backbone
from .decoder import build_decoder


class DinoGlobal(nn.Module):
    def __init__(self, hf_id: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                 attn_implementation: str = "sdpa", dtype: str = "float16",
                 head: str = "mil", local_ckpt: str | None = None,
                 grid_hw: Tuple[int, int] = (45, 80),
                 mid_layer: Optional[int] = None, **head_kw):
        super().__init__()
        self.backbone = DinoV3Backbone(hf_id, attn_implementation, dtype, local_ckpt,
                                       grid_hw=grid_hw, mid_layer=mid_layer)
        self.head_name = head
        # mid-layer fusion concatenates two feature sets -> double input dim
        in_dim = self.backbone.dim * (2 if mid_layer is not None else 1)
        self.decoder = build_decoder(head, in_dim=in_dim,
                                     grid_hw=tuple(grid_hw), **head_kw)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen backbone never leaves eval
        return self

    def decoder_parameters(self):
        return self.decoder.parameters()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            feats = self.backbone(images)
        # head runs in fp32 for stability even when backbone is fp16
        mid = feats.get("patches_mid")
        logits, aux = self.decoder(
            feats["patches"].float(), feats["cls"].float(),
            patches_mid=mid.float() if mid is not None else None,
            cls_mid=(feats["cls_mid"].float() if feats.get("cls_mid") is not None
                     else None))
        return {"logits": logits, "attn": aux.get("attn"),
                "maps": aux.get("maps"), "tokens": aux.get("tokens"),
                "branch_maps": aux.get("branch_maps")}
