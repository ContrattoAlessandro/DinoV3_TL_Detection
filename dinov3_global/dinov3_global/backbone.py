"""Frozen DINOv3 ViT-S backbone wrapper (12GB-safe).

Model: facebook/dinov3-vits16-pretrain-lvd1689m, dim 384, patch 16.
Expects pixel_values [B,3,720,1280] in [0,1]; normalizes internally with
ImageNet mean/std so dataset stays in [0,1] for photometric aug.

Returns dict(patches [B,3600,384], cls [B,384]) and, when ``mid_layer`` is
set, ``patches_mid`` (same layout) from that transformer layer.
Sequence assumption: last N_PATCH tokens are patches, first token is CLS
(N_PATCH derived from ``grid_hw``, default 45x80 = 720/16 x 1280/16).
Verified at runtime in scripts/bench.py.
Requires: transformers>=4.56 + accepted DINOv3 license on first download.

Mid-layer fusion (v4): the final layer of a ViT is optimised for semantics and
abstracts away sub-patch detail - yet 39% of DTLD lamps are <16 px tall at
720p (median 19x5.6 px, ~2.4 patch tokens per lamp), so the lamp glass state
(off vs red) and housing orientation live in *mid-layer* features. Concatenating
layer ``mid_layer`` with the final layer lets the head re-read that detail; the
learned projection can always ignore it.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

PATCH_GRID = (45, 80)  # 720/16, 1280/16


class DinoV3Backbone(nn.Module):
    def __init__(self, hf_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
                 attn_implementation: str = "sdpa",
                 dtype: str = "float16",
                 local_ckpt: str | None = None,
                 grid_hw: Tuple[int, int] = PATCH_GRID,
                 mid_layer: Optional[int] = None):
        super().__init__()
        from transformers import AutoModel
        self.hf_id = local_ckpt or hf_id
        self.grid_hw = tuple(grid_hw)
        self.n_patch = self.grid_hw[0] * self.grid_hw[1]
        self.mid_layer = mid_layer
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        # device placement handled by caller .to(device); no device_map to keep single-GPU simple
        try:
            self.net = AutoModel.from_pretrained(
                self.hf_id, attn_implementation=attn_implementation,
                dtype=torch_dtype, trust_remote_code=False,
            )
        except OSError as e:
            msg = str(e)
            if "gated repo" in msg or "401" in msg:
                raise OSError(
                    "DINOv3 is gated. Fix in 2 min:\n"
                    "  1. Accept license at https://huggingface.co/" + hf_id + "\n"
                    "  2. Run: hf auth login  (paste a read token from "
                    "https://huggingface.co/settings/tokens)\n"
                    "  3. Re-run bench.py\n"
                    "Alternative (no login each run): download once via browser after "
                    "accepting, then set backbone.local_ckpt to the local dir in "
                    "dinov3_global/configs/base.yaml."
                ) from e
            raise
        self.dim = int(self.net.config.hidden_size)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # noqa: D102 — backbone stays in eval (frozen, no dropout)
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """images: [B,3,H,W] in [0,1]. Returns patches/cls on same device as input."""
        x = (images - self._mean.to(images.device)) / self._std.to(images.device)
        out = self.net(pixel_values=x, output_hidden_states=self.mid_layer is not None)
        h = out.last_hidden_state  # [B, S, D]
        if h.shape[1] < self.n_patch + 1:
            raise RuntimeError(f"unexpected seq_len {h.shape[1]}, expected >= {self.n_patch + 1}")
        patches = h[:, -self.n_patch:, :]
        cls = h[:, 0, :]
        res = {"patches": patches, "cls": cls}
        if self.mid_layer is not None:
            hs = out.hidden_states  # tuple: embeddings + one per layer
            res["patches_mid"] = hs[self.mid_layer][:, -self.n_patch:, :]
            res["cls_mid"] = hs[self.mid_layer][:, 0, :]
        return res
