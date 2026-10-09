"""Frozen DINOv3 ViT features at layer 6 and the final layer.
Register tokens are excluded from the downstream head. RGB inputs are
normalized internally with ImageNet mean and standard deviation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

PATCH_GRID = (45, 80)  # 720/16, 1280/16
BACKBONE_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
BACKBONE_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
BACKBONE_SPECS = {
    BACKBONE_ID: {"revision": BACKBONE_REVISION, "dim": 768, "patch": 16},
}


def backbone_spec(hf_id):
    """Return the immutable pretrained identity for a supported encoder."""
    if hf_id not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported DINOv3 backbone: {hf_id}")
    return dict(hf_id=hf_id, **BACKBONE_SPECS[hf_id])


class DinoV3Backbone(nn.Module):
    def __init__(
        self,
        hf_id: str = BACKBONE_ID,
        attn_implementation: str = "sdpa",
        dtype: str = "float16",
        local_ckpt: str | None = None,
        grid_hw: Tuple[int, int] = PATCH_GRID,
        mid_layer: Optional[int] = None,
    ):
        super().__init__()
        from transformers import AutoModel

        spec = backbone_spec(hf_id)
        self.revision = spec["revision"]
        self.hf_id = local_ckpt or hf_id
        self.grid_hw = tuple(grid_hw)
        self.n_patch = self.grid_hw[0] * self.grid_hw[1]
        self.mid_layer = mid_layer
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        # device placement handled by caller .to(device); no device_map to keep single-GPU simple
        try:
            self.net = AutoModel.from_pretrained(
                self.hf_id,
                attn_implementation=attn_implementation,
                dtype=torch_dtype,
                trust_remote_code=False,
                revision=None if local_ckpt else self.revision,
            )
        except OSError as e:
            msg = str(e)
            if "gated repo" in msg or "401" in msg:
                raise OSError(
                    "Access to this DINOv3 checkpoint is gated:\n"
                    "  1. Accept license at https://huggingface.co/" + hf_id + "\n"
                    "  2. Run: hf auth login  (paste a read token from "
                    "https://huggingface.co/settings/tokens)\n"
                    "  3. Re-run the command\n"
                    "Alternative (no login each run): download once via browser after "
                    "accepting, then set backbone.local_ckpt to the local dir in "
                    "configs/default.yaml."
                ) from e
            raise
        self.dim = int(self.net.config.hidden_size)
        if self.dim != spec["dim"]:
            raise ValueError(f"Loaded backbone width {self.dim} differs from {hf_id}: {spec['dim']}")
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
        patches = h[:, -self.n_patch :, :]
        cls = h[:, 0, :]
        res = {"patches": patches, "cls": cls}
        if self.mid_layer is not None:
            hs = out.hidden_states  # tuple: embeddings + one per layer
            res["patches_mid"] = hs[self.mid_layer][:, -self.n_patch :, :]
            res["cls_mid"] = hs[self.mid_layer][:, 0, :]
        return res
