"""v5: attribute evidence, three local MIL branches, and bounded scene context.

Head-level position, geometry, direction, and scene gating affect relevance.
State-dict names preserve the four trained v5 EMA checkpoints exactly.
"""

from __future__ import annotations
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

PATCH_GRID = (45, 80)
RR_STATES = (2, 5, 3)  # red, yellow, red_yellow
GREEN_STATE = 0
N_STATE, N_DIR = 6, 4


def spatial_settings(value, dim):
    """Validate the optional spatial module without changing legacy configs."""
    value = {"mode": "none"} if value is None else value
    if not isinstance(value, dict) or set(value) - {"mode", "depth", "heads", "mlp_ratio"}:
        raise ValueError("spatial_context must contain mode, depth, heads, and/or mlp_ratio")
    mode = value.get("mode", "none")
    if mode not in ("none", "axial"):
        raise ValueError("spatial_context.mode must be none or axial")
    depth, heads, ratio = value.get("depth", 2), value.get("heads", 4), value.get("mlp_ratio", 2)
    if type(depth) is not int or depth < 1 or type(heads) is not int or heads < 1:
        raise ValueError("spatial depth and heads must be positive integers")
    if (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not math.isfinite(ratio)
        or ratio <= 0
    ):
        raise ValueError("spatial mlp_ratio must be finite and positive")
    if mode == "axial" and (dim % heads or int(dim * ratio) < 1):
        raise ValueError("spatial heads must divide proj_dim and the MLP must have positive width")
    return dict(mode=mode, depth=depth, heads=heads, mlp_ratio=ratio)


class AxialBlock(nn.Module):
    """Unmasked row attention, column attention, and a pre-normalized MLP."""

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.row_norm, self.col_norm, self.ffn_norm = [nn.LayerNorm(dim) for _ in range(3)]
        self.row_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, grid):
        batch, h, w, dim = grid.shape
        rows = self.row_norm(grid).reshape(batch * h, w, dim)
        row_out = self.row_attn(rows, rows, rows, need_weights=False)[0]
        grid = grid + self.dropout(row_out.reshape(batch, h, w, dim))
        cols = self.col_norm(grid).permute(0, 2, 1, 3).reshape(batch * w, h, dim)
        col_out = self.col_attn(cols, cols, cols, need_weights=False)[0]
        grid = grid + self.dropout(col_out.reshape(batch, w, h, dim).permute(0, 2, 1, 3))
        return grid + self.ffn(self.ffn_norm(grid))


class AxialSpatialContext(nn.Module):
    """Shared spatial reasoning on a row-major patch grid."""

    def __init__(self, dim, grid_hw, depth=2, heads=4, mlp_ratio=2, dropout=0.2):
        super().__init__()
        settings = spatial_settings(dict(mode="axial", depth=depth, heads=heads, mlp_ratio=mlp_ratio), dim)
        self.grid_hw, self.dim = tuple(grid_hw), dim
        self.blocks = nn.Sequential(
            *[AxialBlock(dim, settings["heads"], settings["mlp_ratio"], dropout) for _ in range(depth)]
        )

    def forward(self, tokens):
        h, w = self.grid_hw
        if tokens.ndim != 3 or tokens.shape[1:] != (h * w, self.dim):
            raise ValueError("Spatial tokens must match the configured grid and projection dimension")
        return self.blocks(tokens.reshape(tokens.shape[0], h, w, self.dim)).reshape_as(tokens)


def build_2d_sincos(h, w, dim):
    """Fixed two-dimensional sinusoidal positions, shaped [h*w, dim]."""
    if dim % 4:
        raise ValueError("proj_dim must be divisible by four")
    d2 = dim // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )

    def encode(position):
        frequency = torch.exp(torch.arange(0, d2, 2, dtype=torch.float32) * -(math.log(10000.0) / d2))
        phase = position.reshape(-1, 1) * frequency.reshape(1, -1)
        return torch.cat([phase.sin(), phase.cos()], dim=1)

    return torch.cat([encode(yy), encode(xx)], dim=1)


def token_geo(h, w):
    """Normalized token centers and bearing: (x, y, 2*x-1, 1-y)."""
    yy, xx = torch.meshgrid(
        (torch.arange(h, dtype=torch.float32) + 0.5) / h,
        (torch.arange(w, dtype=torch.float32) + 0.5) / w,
        indexing="ij",
    )
    x, y = xx.reshape(-1), yy.reshape(-1)
    return torch.stack([x, y, (x - 0.5) * 2, 1 - y], dim=1)


def local_evidence_pool(evidence, lamp, grid_hw, k, content_mask=None):
    batch, count = evidence.shape
    h, w = grid_hw
    eg, lg = evidence.reshape(batch, 1, h, w), lamp.reshape(batch, 1, h, w)
    numerator = F.avg_pool2d(eg * lg, 5, stride=1, padding=2)
    denominator = F.avg_pool2d(lg, 5, stride=1, padding=2).clamp_min(1e-4)
    local = (numerator / denominator).reshape(batch, count)
    if content_mask is not None:
        local = local * content_mask.reshape(batch, count)
    return local.topk(min(k, count), dim=1).values.mean(1)


class MILBranch(nn.Module):
    """Independent attribute predictions and one local top-k pooling scale."""

    def __init__(self, dim, dropout, k, grid_hw=PATCH_GRID):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lamp = nn.Linear(dim, 1)
        self.state = nn.Linear(dim, N_STATE)
        self.dir = nn.Linear(dim, N_DIR)
        self.rel = nn.Linear(dim + 4 + N_DIR, 1)
        self.k, self.grid_hw = int(k), tuple(grid_hw)
        if self.k < 1:
            raise ValueError("top-k scale must be positive")

    def pooled(self, evidence, lamp, content_mask=None):
        """Lamp-weighted 5x5 local mean, followed by the top-k mean."""
        return local_evidence_pool(evidence, lamp, self.grid_hw, self.k, content_mask)


class TokenMILHead(nn.Module):
    """Relevance-conditioned evidence, local MIL, and bounded CLS paths."""

    def __init__(
        self,
        in_dim=768,
        proj_dim=192,
        dropout=0.2,
        topk=(1, 2, 4),
        logit_scale=2.5,
        grid_hw=PATCH_GRID,
        scene_dropout=0.3,
        scene_prior_bound=0.5,
        nor_scene_bound=0.5,
        spatial_context=None,
    ):
        super().__init__()
        if not 0 <= scene_dropout < 1:
            raise ValueError("scene_dropout must be in [0, 1)")
        if scene_prior_bound <= 0 or nor_scene_bound <= 0:
            raise ValueError("Scene bounds must be positive")
        if not topk:
            raise ValueError("At least one pooling scale is required")
        self.grid_hw = tuple(grid_hw)
        if len(self.grid_hw) != 2 or any(type(v) is not int or v < 1 for v in self.grid_hw):
            raise ValueError("grid_hw must contain two positive integers")
        self.scene_dropout = float(scene_dropout)
        self.scene_prior_bound, self.nor_scene_bound = float(scene_prior_bound), float(nor_scene_bound)
        self.proj = nn.Linear(in_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.register_buffer("pos", build_2d_sincos(*self.grid_hw, proj_dim), persistent=False)
        self.register_buffer("geo", token_geo(*self.grid_hw), persistent=False)
        self.rel_context = nn.Linear(proj_dim, proj_dim)
        nn.init.zeros_(self.rel_context.weight)
        nn.init.zeros_(self.rel_context.bias)
        self.branches = nn.ModuleList([MILBranch(proj_dim, dropout, k, self.grid_hw) for k in topk])
        self.nor_head = nn.Sequential(
            nn.Linear(proj_dim + 2, proj_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_dim, 1)
        )
        self.scale = nn.Parameter(torch.full((2,), float(logit_scale)))
        self.bias = nn.Parameter(torch.zeros(2))
        self.cls_mod = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_dim, 2)
        )
        # Construct last: all pre-existing weights retain their seeded initialization.
        settings = spatial_settings(spatial_context, proj_dim)
        self.spatial_context = (
            AxialSpatialContext(
                proj_dim,
                self.grid_hw,
                dropout=dropout,
                **{k: settings[k] for k in ("depth", "heads", "mlp_ratio")},
            )
            if settings["mode"] == "axial"
            else None
        )

    def _branch(self, branch, h, context, spatial_delta=None, content_mask=None, geometry=None):
        batch = h.shape[0]
        z = branch.trunk(h)
        lamp_logit, state_logit = branch.lamp(z).squeeze(-1), branch.state(z)
        rel_z = z * (1 + 0.25 * self.rel_context(context).tanh().unsqueeze(1))
        rel_z = rel_z + self.pos.to(h.device).unsqueeze(0)
        if spatial_delta is not None:
            rel_z = rel_z + spatial_delta
        geo = self.geo.to(h.device).unsqueeze(0).expand(batch, -1, -1) if geometry is None else geometry
        dir_logit = branch.dir(z)
        rel_logit = branch.rel(torch.cat([rel_z, geo, dir_logit.softmax(-1)], dim=-1)).squeeze(-1)
        lamp, state, relevance = lamp_logit.sigmoid(), state_logit.softmax(-1), rel_logit.sigmoid()
        if content_mask is not None:
            lamp = lamp * content_mask.reshape(batch, -1)
        rr = lamp * relevance * state[..., list(RR_STATES)].sum(-1)
        rg = lamp * relevance * state[..., GREEN_STATE]
        scores = torch.stack(
            [branch.pooled(rr, lamp, content_mask), branch.pooled(rg, lamp, content_mask)], dim=1
        )
        maps = dict(
            lamp=lamp,
            rel=relevance,
            state=state,
            lamp_logit=lamp_logit,
            rel_logit=rel_logit,
            state_logit=state_logit,
            dir_logit=dir_logit,
            rr=rr.reshape(batch, *self.grid_hw),
            rg=rg.reshape(batch, *self.grid_hw),
        )
        return scores, maps

    def forward(
        self,
        patches,
        cls,
        patches_mid: Optional[torch.Tensor] = None,
        cls_mid: Optional[torch.Tensor] = None,
        content_mask=None,
        geometry=None,
    ):
        """Return [B,3] logits and evidence; feature order is final, mid."""
        if (patches_mid is None) != (cls_mid is None):
            raise ValueError("patches_mid and cls_mid must be given together")
        if patches_mid is not None:
            patches = torch.cat([patches, patches_mid], dim=-1)
            cls = torch.cat([cls, cls_mid], dim=-1)
        batch = patches.shape[0]
        if patches.ndim != 3 or patches.shape[1] != math.prod(self.grid_hw):
            raise ValueError("Patch count must match the configured grid")
        h = self.proj_norm(self.proj(patches))
        context = self.proj_norm(self.proj(cls))
        if self.training and self.scene_dropout > 0:
            context = context * (torch.rand(batch, 1, device=context.device) >= self.scene_dropout)
        spatial_delta = None
        if self.spatial_context is not None:
            u = h * (1 + 0.25 * self.rel_context(context).tanh().unsqueeze(1)) + self.pos.unsqueeze(0)
            spatial_delta = self.spatial_context(u) - u
        scores, branch_maps = [], []
        for branch in self.branches:
            score, maps = self._branch(branch, h, context, spatial_delta, content_mask, geometry)
            scores.append(score)
            branch_maps.append(maps)
        s = torch.stack(scores).mean(0)
        scene_prior = self.scene_prior_bound * self.cls_mod(context).tanh()
        nor_full = self.nor_head(torch.cat([context, s], dim=-1)).squeeze(-1)
        nor_base = self.nor_head(torch.cat([torch.zeros_like(context), s], dim=-1)).squeeze(-1)
        nor = nor_base + self.nor_scene_bound * (nor_full - nor_base).tanh()
        logits = torch.stack(
            [
                s[:, 0] * self.scale[0] + self.bias[0] + scene_prior[:, 0],
                s[:, 1] * self.scale[1] + self.bias[1] + scene_prior[:, 1],
                nor,
            ],
            dim=1,
        )
        maps = {key: torch.stack([m[key] for m in branch_maps]).mean(0) for key in branch_maps[0]}
        maps["lamp_grid"], maps["rel_grid"] = (
            maps["lamp"].reshape(batch, *self.grid_hw),
            maps["rel"].reshape(batch, *self.grid_hw),
        )
        return logits, dict(maps=maps, tokens=h, branch_maps=branch_maps)
