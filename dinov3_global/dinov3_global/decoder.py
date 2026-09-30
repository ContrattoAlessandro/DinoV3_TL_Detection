"""Decoders for global RR/RG/NoR on frozen DINOv3 patch tokens.

Two options (config `decoder.head`):

- ``attn`` (legacy, exp1): 1 learnable GLOBAL query cross-attends the patch
  grid -> Linear 3. Kept only as the ablation baseline. Known failure mode:
  softmax attention is a weighted *mean* over 3600 tokens, which dilutes the
  "ANY relevant red lamp exists" (existential) evidence the task is built on.

- ``mil`` (v2, DinoGlobal-MIL): per-token evidence + smooth-max (log-sum-exp)
  MIL pooling. Instance-level score pooling is the correct inductive bias for
  presence/absence tasks (Ilse et al. 2018; Kowalski 2019: max pooling
  localises, mean-style pooling does not) and stays detector-free at inference.

TokenMILHead anatomy (per branch, 2 branches averaged -> anti-collapse):
  f_t (384) + 2D sincos pos -> shared proj/norm -> per-branch trunk MLP
    lamp_t  = sigmoid(Linear)                 # "a lamp occupies this patch"
    state_t = softmax(Linear, 6)              # green/off/red/red_yellow/unknown/yellow
    rel_t   = sigmoid(Linear([h_t; geo_t]))   # "this lamp is relevant to ego"
                                            # geo_t = normalised (x, y, bearing, 1-y)
  evidence:  e_t^RR = lamp_t*rel_t*(s_red+s_yellow+s_red_yellow)
             e_t^RG = lamp_t*rel_t*s_green
  pooling:   S_c = mean(top-k_t e_t^c)                        # v3 "topk", k=1,2,4
             S_c = tau * (logsumexp_t(e_t^c / tau) - log N)  # v2 "lse" (exp2 default)
  logits:    [a_RR*S_RR + b_RR, a_RG*S_RG + b_RG, MLP(CLS)]    # NoR needs global
                                                          # reasoning -> CLS head

All learned parameters live in the head; the backbone stays frozen.

v4 additions (all cfg-gated; default construction stays byte-compatible with
exp2/exp3 checkpoints, which build the head from their stored cfg):
- ``pool="local"``: lampness-weighted local mean of the evidence before top-k.
  Aggregates a lamp's ~2.4 tokens before the frame max, so one noisy token
  cannot dominate (measured failure: evidence over-fires on NoR frames; with
  ~17 lamp tokens/frame even a 90%-accurate token estimate fires on >80% of
  frames under plain max). No dilution for single-token lamps: the weights
  normalise by the lampness mass, so an isolated lamp keeps its score.
- ``dir_head``: per-token DTLD *direction* aux head (back/front/left/right) and
  the direction posterior fed into the relevance head. Measured on the train
  split: P(relevant | back) = 0.000 (15,471 lamps), | left/right ≈ 0.04,
  | front = 0.43 - relevance is largely a function of housing direction and
  bearing, and the rel head never saw the direction signal before.
- ``cls_modulates_evidence``: per-class scene prior MLP(CLS) added to the
  evidence logits (which-lamp-is-ego's-lane is a scene-level fact; exp2/3 gave
  CLS context only to the NoR logit).
- mid-layer fusion: ``forward`` accepts ``patches_mid`` concatenated with the
  final-layer tokens before the projection (sub-patch lamp detail).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_GRID = (45, 80)  # default: 720/16, 1280/16 (v4: derived from target_hw)
N_PATCH = PATCH_GRID[0] * PATCH_GRID[1]
RR_STATES = (2, 5, 3)      # indices into STATES: red, yellow, red_yellow
GREEN_STATE = 0            # index into STATES: green
N_STATE = 6
N_DIR = 4                  # DIRECTIONS: back, front, left, right


def build_2d_sincos(h: int, w: int, dim: int) -> torch.Tensor:
    """Fixed 2D sincos posemb [h*w, dim], dim even, half for y half for x."""
    assert dim % 4 == 0, "proj_dim must be divisible by 4 for 2D sincos"
    d2 = dim // 2
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [h,w]
    def _enc(pos: torch.Tensor) -> torch.Tensor:
        freq = torch.exp(torch.arange(0, d2, 2, dtype=torch.float32) * -(math.log(10000.0) / d2))
        args = pos.reshape(-1, 1) * freq.reshape(1, -1)  # [h*w, d2/2]
        return torch.cat([args.sin(), args.cos()], dim=1)  # [h*w, d2]
    return torch.cat([_enc(yy.reshape(-1)), _enc(xx.reshape(-1))], dim=1)


def token_geo(h: int, w: int) -> torch.Tensor:
    """Per-token geometry [h*w, 4]: (xc, yc, bearing=(xc-0.5)*2, 1-yc).
    Relevance is positional (ego-lane bearing), state is not -> only the
    relevance head sees this."""
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xc, yc = xx.reshape(-1), yy.reshape(-1)
    return torch.stack([xc, yc, (xc - 0.5) * 2.0, 1.0 - yc], dim=1)


# --------------------------------------------------------------------------- #
# legacy attn head (exp1 baseline, ablation only)
# --------------------------------------------------------------------------- #
class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )
        self.n2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, mem: torch.Tensor):
        a, w = self.cross(q, mem, mem, need_weights=True)  # w: [B,1,3601]
        q = self.n1(q + a)
        q = self.n2(q + self.ff(q))
        return q, w


class GlobalDecoder(nn.Module):
    """Legacy: 1 learnable query -> Linear 3 (kept for ablation)."""

    head_kind = "attn"

    def __init__(self, in_dim: int = 384, proj_dim: int = 256,
                 depth: int = 2, heads: int = 4, ff_dim: int = 1024,
                 dropout: float = 0.1, grid_hw: Tuple[int, int] = PATCH_GRID,
                 **_ignored):
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        self.proj = nn.Linear(in_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.register_buffer("pos", build_2d_sincos(*self.grid_hw, proj_dim), persistent=False)
        self.query = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList(
            [CrossBlock(proj_dim, heads, ff_dim, dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(proj_dim)
        self.head = nn.Linear(proj_dim, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor, cls: torch.Tensor,
                patches_mid: Optional[torch.Tensor] = None,
                cls_mid: Optional[torch.Tensor] = None):
        """patches [B,N,Di], cls [B,Di] -> logits [B,3], aux dict."""
        if patches_mid is not None:
            patches = torch.cat([patches, patches_mid], dim=-1)
            cls = torch.cat([cls, cls_mid], dim=-1)
        B = patches.shape[0]
        gh, gw = self.grid_hw
        pm = self.proj_norm(self.proj(patches)) + self.pos.to(patches.device).unsqueeze(0)
        cm = self.proj_norm(self.proj(cls)).unsqueeze(1)
        mem = torch.cat([cm, pm], dim=1)  # [B,N+1,D]
        mem = self.dropout(mem)
        q = self.query.expand(B, -1, -1)
        w_last = None
        for blk in self.blocks:
            q, w_last = blk(q, mem)
        logits = self.head(self.out_norm(q.squeeze(1)))
        attn = None
        if w_last is not None:
            attn = w_last.squeeze(1)[:, 1:].reshape(B, gh, gw)
        return logits, {"attn": attn, "maps": None, "tokens": None}


# --------------------------------------------------------------------------- #
# v2: token-level MIL head
# --------------------------------------------------------------------------- #
def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x))


class MILBranch(nn.Module):
    """One pooling scale over per-token evidence.

    ``pool="lse"`` (v2/exp2): log-sum-exp smoothing with a learnable tau. The
    score is ``S = e_max - tau*log N + tau*log(sum_t exp((e_t-e_max)/tau))``: the
    ``-tau*log N`` term is only partly cancelled by the diffuse background, so a
    large tau collapses the score towards the frame's mean evidence. Measured on
    exp2 (val): a token peak of 0.842 produced S_rr = 0.628 at the learned
    tau=0.030 but only 0.217 at tau=0.223, and the RR-NoR separation of that
    branch was 0.053 vs 0.473 for the sharp branch - i.e. near-constant.

    ``pool="topk"`` (v3, used by configs/v3.yaml): mean of the k largest token
    evidences. Full [0,1] dynamic range, no tau*log(N) penalty, and the k values
    double as a multi-scale "existential" ensemble. The 1/k dilution of a single
    lamp is fixed (not evidence-dependent) and is absorbed by the learnable gain.
    """

    def __init__(self, dim: int, dropout: float, tau_init: float,
                 pool: str = "lse", k: int = 1,
                 grid_hw: Tuple[int, int] = PATCH_GRID,
                 dir_in: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.lamp = nn.Linear(dim, 1)
        self.state = nn.Linear(dim, N_STATE)
        # v4 aux: housing direction (back/front/left/right). P(relevant|back)=0,
        # P(relevant|left/right)≈0.04, P(relevant|front)=0.43 (measured on the
        # DTLD train split) - the rel head below consumes this posterior.
        self.dir = nn.Linear(dim, N_DIR) if dir_in else None
        # concat geometry (+ the direction posterior when dir_head is on)
        self.rel = nn.Linear(dim + 4 + (N_DIR if dir_in else 0), 1)
        self.pool = pool
        self.k = int(k)
        self.grid_hw = tuple(grid_hw)
        # no parameter in topk mode -> the lse checkpoint layout stays loadable
        self._tau = (None if pool in ("topk", "local")
                     else nn.Parameter(torch.tensor(_inv_softplus(tau_init))))

    @property
    def tau(self) -> torch.Tensor:
        if self._tau is None:
            raise RuntimeError("branch is in topk/local mode (no learnable tau)")
        return F.softplus(self._tau) + 1e-3

    def pooled(self, e: torch.Tensor, lamp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """e [B,N] in [0,1] -> [B] frame-level evidence score.

        ``local`` (v4): lampness-weighted mean over a 5x5 token window before
        top-k, i.e. evidence averaged over the lamp the token belongs to. The
        normalisation is by the window's lampness mass, so an isolated
        single-token lamp is *not* diluted (background lampness ≈ 0 cancels),
        while a 2-4 token lamp cannot fire through one noisy token alone.
        """
        if self.pool == "local":
            if lamp is None:
                raise RuntimeError("pool='local' needs the lampness map")
            B, N = e.shape
            gh, gw = self.grid_hw
            eg = e.reshape(B, 1, gh, gw)
            lg = lamp.reshape(B, 1, gh, gw)
            num = F.avg_pool2d(eg * lg, 5, stride=1, padding=2)
            den = F.avg_pool2d(lg, 5, stride=1, padding=2).clamp_min(1e-4)
            e = (num / den).reshape(B, N)
        if self.pool in ("topk", "local"):
            k = max(1, min(self.k, e.shape[1]))
            return e.topk(k, dim=1).values.mean(1)
        tau = self.tau
        return tau * (torch.logsumexp(e / tau, dim=1) - math.log(e.shape[1]))


class TokenMILHead(nn.Module):
    """Per-token lampness/state/relevance (+direction) + MIL pooling."""

    head_kind = "mil"

    def __init__(self, in_dim: int = 384, proj_dim: int = 256,
                 n_branches: int = 2, dropout: float = 0.1,
                 tau_init: Tuple[float, ...] = (0.05, 0.3),
                 logit_scale: float = 4.0, pool: str = "lse",
                 topk: Tuple[int, ...] = (1, 2, 4),
                 nor_sees_evidence: bool = False,
                 grid_hw: Tuple[int, int] = PATCH_GRID,
                 dir_head: bool = False,
                 cls_modulates_evidence: bool = False,
                 **_ignored):
        super().__init__()
        self.pool = pool
        self.grid_hw = tuple(grid_hw)
        self.nor_sees_evidence = bool(nor_sees_evidence)
        self.dir_head = bool(dir_head)
        self.cls_modulates_evidence = bool(cls_modulates_evidence)
        self.proj = nn.Linear(in_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)
        self.register_buffer("pos", build_2d_sincos(*self.grid_hw, proj_dim), persistent=False)
        self.register_buffer("geo", token_geo(*self.grid_hw), persistent=False)
        if pool == "topk":
            ks = [max(1, int(k)) for k in (topk or (1, 2, 4))]
            self.branches = nn.ModuleList(
                [MILBranch(proj_dim, dropout, 0.1, pool="topk", k=k,
                           grid_hw=self.grid_hw, dir_in=self.dir_head) for k in ks])
        elif pool == "local":
            ks = [max(1, int(k)) for k in (topk or (1, 2, 4))]
            self.branches = nn.ModuleList(
                [MILBranch(proj_dim, dropout, 0.1, pool="local", k=k,
                           grid_hw=self.grid_hw, dir_in=self.dir_head) for k in ks])
        else:
            taus = list(tau_init)[:n_branches]
            while len(taus) < n_branches:
                taus.append(0.1)
            self.branches = nn.ModuleList(
                [MILBranch(proj_dim, dropout, t, grid_hw=self.grid_hw,
                           dir_in=self.dir_head) for t in taus])
        # NoR needs global reasoning -> CLS head. Optionally sees the pooled
        # evidence as well (v3): "no relevant lamp" is by definition the
        # complement of the two evidence channels, and with top-k pooling the
        # scores live in [0,1] and are informative. Off by default so legacy
        # exp2 checkpoints (CLS-only input) keep loading with strict=True.
        nor_in = proj_dim + (2 if self.nor_sees_evidence else 0)
        self.nor_head = nn.Sequential(
            nn.Linear(nor_in, proj_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, 1),
        )
        # learnable per-class gain/bias so bounded evidence can compete with the
        # free-scale NoR logit in one softmax
        self.scale = nn.Parameter(torch.full((2,), float(logit_scale)))
        self.bias = nn.Parameter(torch.zeros(2))
        # v4: scene-level prior for the two evidence classes. Which front-facing
        # lamp belongs to the ego lane is a scene-geometry fact that token-local
        # evidence cannot resolve; exp2/3 exposed CLS context only to NoR.
        if self.cls_modulates_evidence:
            self.cls_mod = nn.Sequential(
                nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(proj_dim, 2),
            )

    def _branch(self, br: MILBranch, h: torch.Tensor):
        """h: [B,N,D] -> pooled scores [B,2], per-token probs + raw logits."""
        B, N, _ = h.shape
        z = br.trunk(h)
        lamp_logit = br.lamp(z).squeeze(-1)                   # [B,N]
        state_logit = br.state(z)                             # [B,N,6]
        geo = self.geo.to(h.device).unsqueeze(0).expand(B, -1, -1)
        if self.dir_head:
            dir_logit = br.dir(z)                             # [B,N,4]
            rel_in = torch.cat([z, geo, torch.softmax(dir_logit, dim=-1)], dim=-1)
        else:
            dir_logit = None
            rel_in = torch.cat([z, geo], dim=-1)
        rel_logit = br.rel(rel_in).squeeze(-1)
        lamp = torch.sigmoid(lamp_logit)
        state = torch.softmax(state_logit, dim=-1)
        rel = torch.sigmoid(rel_logit)
        e_rr = lamp * rel * state[..., list(RR_STATES)].sum(-1)
        e_rg = lamp * rel * state[..., GREEN_STATE]
        s_rr = br.pooled(e_rr, lamp=lamp)
        s_rg = br.pooled(e_rg, lamp=lamp)
        maps = {
            "lamp": lamp, "rel": rel, "state": state,
            "lamp_logit": lamp_logit, "rel_logit": rel_logit,
            "state_logit": state_logit,
            "rr": e_rr.reshape(B, *self.grid_hw), "rg": e_rg.reshape(B, *self.grid_hw),
        }
        if dir_logit is not None:
            maps["dir_logit"] = dir_logit
        return torch.stack([s_rr, s_rg], dim=1), maps

    def forward(self, patches: torch.Tensor, cls: torch.Tensor,
                patches_mid: Optional[torch.Tensor] = None,
                cls_mid: Optional[torch.Tensor] = None):
        """patches [B,N,Di], cls [B,Di] -> logits [B,3], aux dict.

        ``patches_mid``/``cls_mid`` (v4): mid-layer features, same trailing dim
        as their final-layer counterparts. The projection is shared between
        tokens and CLS (as in exp2/3), so fusion concatenates BOTH - the
        projection input dim must match construction.
        """
        if (patches_mid is None) != (cls_mid is None):
            raise ValueError("patches_mid and cls_mid must be given together "
                             "(shared projection)")
        if patches_mid is not None:
            patches = torch.cat([patches, patches_mid], dim=-1)
            cls = torch.cat([cls, cls_mid], dim=-1)
        B = patches.shape[0]
        h = self.proj_norm(self.proj(patches))
        h = h + self.pos.to(h.device).unsqueeze(0)
        cm = self.proj_norm(self.proj(cls))

        scores, maps_list = [], []
        for br in self.branches:
            s, mp = self._branch(br, h)
            scores.append(s)
            maps_list.append(mp)
        s = torch.stack(scores).mean(0)                        # [B,2]
        if self.cls_modulates_evidence:
            mod = self.cls_mod(cm)                             # [B,2] scene prior
        else:
            mod = torch.zeros_like(s)
        if self.nor_sees_evidence:
            nor_in = torch.cat([cm, s], dim=-1)
        else:
            nor_in = cm
        logits = torch.stack([s[:, 0] * self.scale[0] + self.bias[0] + mod[:, 0],
                              s[:, 1] * self.scale[1] + self.bias[1] + mod[:, 1],
                              self.nor_head(nor_in).squeeze(-1)], dim=1)

        # averaged maps for inspection/aux supervision
        maps: Dict[str, torch.Tensor] = {}
        for k in maps_list[0]:
            maps[k] = torch.stack([m[k] for m in maps_list]).mean(0)
        maps["lamp_grid"] = maps["lamp"].reshape(B, *self.grid_hw)
        maps["rel_grid"] = maps["rel"].reshape(B, *self.grid_hw)
        return logits, {"attn": maps["rr"].detach(), "maps": maps, "tokens": h}


def build_decoder(head: str = "mil", **kw) -> nn.Module:
    if head == "attn":
        return GlobalDecoder(**kw)
    return TokenMILHead(**kw)
