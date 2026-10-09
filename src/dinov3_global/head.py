"""Compact shared attribute evidence with a monotonic NoR readout."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .pooling import local_evidence_pool, token_geo, RR_STATES, GREEN_STATE


def inverse_softplus(value):
    return math.log(math.expm1(value))


class EvidenceMILHead(nn.Module):
    def __init__(
        self,
        in_dim=1536,
        proj_dim=192,
        dropout=0.2,
        topk=(1, 2, 4),
        logit_scale=2.5,
        grid_hw=(45, 80),
        context_dropout=0.5,
        context_bound=1.0,
    ):
        super().__init__()
        if not 0 <= context_dropout < 1 or context_bound <= 0:
            raise ValueError("Invalid relevance context settings")
        if type(proj_dim) is not int or proj_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid head width/dropout")
        self.grid_hw, self.topk = tuple(grid_hw), tuple(topk)
        if not topk or any(type(k) is not int or k < 1 for k in topk):
            raise ValueError("Pooling scales must be positive integers")
        self.context_dropout, self.context_bound = context_dropout, context_bound
        self.proj, self.proj_norm = nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim)
        self.trunk = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lamp, self.state = nn.Linear(proj_dim, 1), nn.Linear(proj_dim, 6)
        self.dir, self.pictogram = nn.Linear(proj_dim, 4), nn.Linear(proj_dim, 10)
        appearance_dim = proj_dim + 4 + 10

        def relevance_mlp(dim):
            return nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

        self.rel_base = relevance_mlp(appearance_dim)
        self.rel_context = relevance_mlp(appearance_dim + 4 + proj_dim)
        nn.init.zeros_(self.rel_context[-1].weight)
        nn.init.zeros_(self.rel_context[-1].bias)
        self.register_buffer("geo", token_geo(*grid_hw), persistent=False)
        self.scale_raw = nn.Parameter(torch.full((2,), inverse_softplus(logit_scale)))
        self.bias = nn.Parameter(torch.zeros(2))
        self.nor_scale_raw = nn.Parameter(torch.full((2,), inverse_softplus(2.0)))
        self.nor_bias = nn.Parameter(torch.zeros(()))

    def classify(self, scores):
        signal = F.softplus(self.scale_raw) * scores + self.bias
        nor = self.nor_bias - (F.softplus(self.nor_scale_raw) * scores).sum(-1)
        return torch.cat([signal, nor.unsqueeze(-1)], -1)

    def forward(self, patches, cls, patches_mid=None, cls_mid=None, content_mask=None, geometry=None):
        if (patches_mid is None) != (cls_mid is None):
            raise ValueError("Both intermediate feature sets are required")
        if patches_mid is not None:
            patches, cls = torch.cat([patches, patches_mid], -1), torch.cat([cls, cls_mid], -1)
        batch, count, _ = patches.shape
        if count != math.prod(self.grid_hw):
            raise ValueError("Patch count must match the configured grid")
        h = self.proj_norm(self.proj(patches))
        scene = self.proj_norm(self.proj(cls))
        z = self.trunk(h)
        lamp_logit, state_logit = self.lamp(z).squeeze(-1), self.state(z)
        dir_logit, pictogram_logit = self.dir(z), self.pictogram(z)
        direction, pictogram = dir_logit.softmax(-1), pictogram_logit.softmax(-1)
        appearance = torch.cat([z, direction, pictogram], -1)
        geo = self.geo.unsqueeze(0).expand(batch, -1, -1) if geometry is None else geometry
        inputs = torch.cat([geo, scene.unsqueeze(1).expand(-1, count, -1)], -1)
        if self.training and self.context_dropout:
            inputs = inputs * (torch.rand(batch, 1, 1, device=h.device) >= self.context_dropout)
        correction = self.context_bound * self.rel_context(
            torch.cat([appearance, inputs], -1)
        ).tanh().squeeze(-1)
        rel_logit = self.rel_base(appearance).squeeze(-1) + correction
        lamp, state, rel = lamp_logit.sigmoid(), state_logit.softmax(-1), rel_logit.sigmoid()
        if content_mask is not None:
            lamp = lamp * content_mask.reshape(batch, -1)
        rr = lamp * rel * state[..., list(RR_STATES)].sum(-1)
        rg = lamp * rel * state[..., GREEN_STATE]
        pooled = []
        for k in self.topk:
            pooled.append(
                torch.stack(
                    [
                        local_evidence_pool(rr, lamp, self.grid_hw, k, content_mask),
                        local_evidence_pool(rg, lamp, self.grid_hw, k, content_mask),
                    ],
                    -1,
                )
            )
        scores = torch.stack(pooled).mean(0)
        maps = dict(
            lamp=lamp,
            rel=rel,
            state=state,
            lamp_logit=lamp_logit,
            state_logit=state_logit,
            rel_logit=rel_logit,
            dir_logit=dir_logit,
            pictogram_logit=pictogram_logit,
            pictogram=pictogram_logit.softmax(-1),
            rr=rr.reshape(batch, *self.grid_hw),
            rg=rg.reshape(batch, *self.grid_hw),
            lamp_grid=lamp.reshape(batch, *self.grid_hw),
            rel_grid=rel.reshape(batch, *self.grid_hw),
        )
        return self.classify(scores), dict(
            maps=maps, tokens=h, branch_maps=[maps], scores=scores, relevance_correction=correction
        )
