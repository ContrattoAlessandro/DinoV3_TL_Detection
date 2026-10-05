"""Correctness tests for dinov3_global (no GPU, no HF download needed).

Run: python -m pytest tests/
Covers the units that decide whether the model is trained against correct
targets: token rasterisation geometry, the ignore-band semantics, label
mapping, sampler math, logit adjustment, pooling, temperature scaling.
"""

from __future__ import annotations


import numpy as np
import torch

from dinov3_global.data import (  # noqa: E402
    global_label_from_states,
    token_targets,
)
from dinov3_global.head import MILBranch, TokenMILHead
from dinov3_global.losses import (  # noqa: E402
    consistency_kl,
    global_loss,
    token_losses,
)
from dinov3_global.metrics import fit_temperature  # noqa: E402
from dinov3_global.engine import _make_sampler, _lr_of  # noqa: E402
from dinov3_global.augment import (  # noqa: E402
    apply_lamp_erasure,
    erase_lamps,
)


# --------------------------------------------------------------------------- #
# 1. token rasterisation geometry
# --------------------------------------------------------------------------- #
def test_token_geometry():
    """A box centred on token (5, 8) must mark exactly that token."""
    sx = 1280 / (2048 - 2 * 114)
    sy = 720 / 1024.0
    # target px centre of token (row 5, col 8): ((8+0.5)*16, (5+0.5)*16)
    tcx, tcy = (8 + 0.5) * 16.0, (5 + 0.5) * 16.0
    # back to source 2048x1024 coords
    scx, scy = tcx / sx + 114.0, tcy / sy
    w, h = 20.0, 30.0
    box = np.array([[scx - w / 2, scy - h / 2, scx + w / 2, scy + h / 2]], np.float32)
    tt = token_targets(box, np.array([1]), np.array([2]), 114)
    ys, xs = np.nonzero(tt["lamp"] > 0.5)
    assert len(ys) >= 1, "no lamp token rasterised"
    assert (5, 8) in set(zip(ys.tolist(), xs.tolist())), (
        f"expected token (5,8) marked, got {list(zip(ys.tolist(), xs.tolist()))}"
    )
    assert tt["state"][5, 8] == 2
    assert tt["rel"][5, 8] == 1.0


def test_token_snap_small_box():
    """A box smaller than one token snaps to the token containing its centre."""
    sx = 1280 / (2048 - 2 * 114)
    sy = 720 / 1024.0
    tcx, tcy = (20 + 0.5) * 16.0, (10 + 0.5) * 16.0
    scx, scy = tcx / sx + 114.0, tcy / sy
    box = np.array([[scx - 1, scy - 1, scx + 1, scy + 1]], np.float32)  # 2px box
    tt = token_targets(box, np.array([0]), np.array([0]), 114)
    assert tt["lamp"][10, 20] == 1.0, "small box did not snap to its centre token"


def test_token_empty_boxes():
    tt = token_targets(np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0), 114)
    assert tt["lamp"].sum() == 0 and tt["valid"].all()
    assert tt["rel"].sum() == 0 and tt["state"].sum() == 0


# --------------------------------------------------------------------------- #
# 2. ignore-band vs neighbouring lamp boxes  (BUG HYPOTHESIS)
# --------------------------------------------------------------------------- #
def _stacked_boxes():
    """Two vertically stacked lamps of one signal head (DTLD's typical layout)."""
    sx = 1280 / (2048 - 2 * 114)
    sy = 720 / 1024.0
    tcx = (30 + 0.5) * 16.0

    # lamp A occupies target y 80..110, lamp B directly below y 112..142
    def src(x1, y1, x2, y2):
        return np.array([x1 / sx + 114.0, y1 / sy, x2 / sx + 114.0, y2 / sy], np.float32)

    a = src(tcx - 12, 80, tcx + 12, 110)
    b = src(tcx - 12, 112, tcx + 12, 142)
    return np.stack([a, b]), np.array([1, 1]), np.array([2, 5])


def test_ignore_band_neighbour_bug():
    """Lamp tokens must keep supervision even when a neighbouring lamp's ignore
    band overlaps them. With the current rasteriser the band of box B can
    invalidate the positive tokens of box A (and vice versa)."""
    boxes, rel, st = _stacked_boxes()
    tt = token_targets(boxes, rel, st, 114)
    lamp = tt["lamp"] > 0.5
    valid = tt["valid"]
    n_supervised = int((lamp & valid).sum())
    n_lamp = int(lamp.sum())
    assert n_lamp > 0, "test setup broken: no lamp tokens"
    assert n_supervised == n_lamp, (
        f"{n_lamp - n_supervised}/{n_lamp} lamp-positive tokens lost their "
        f"supervision to a neighbour's ignore band (valid=False)"
    )


# --------------------------------------------------------------------------- #
# 3. global label mapping
# --------------------------------------------------------------------------- #
def test_global_label():
    rr = np.array([1, 0])
    st_rr = np.array([2, 0])  # relevant red + irrelevant green
    y, dbg = global_label_from_states(rr, st_rr)
    assert y == 0 and dbg["n_rel_rr"] == 1
    y, _ = global_label_from_states(np.array([1, 1]), np.array([0, 2]))  # green + red
    assert y == 0, "RR must take priority over RG"
    y, _ = global_label_from_states(np.array([1]), np.array([0]))
    assert y == 1
    y, dbg = global_label_from_states(np.array([1]), np.array([1]))  # relevant off
    assert y == 2 and dbg["n_rel_off_unknown"] == 1
    y, _ = global_label_from_states(np.array([0, 0]), np.array([2, 2]))  # only irrelevant
    assert y == 2
    y, _ = global_label_from_states(np.zeros(0), np.zeros(0))
    assert y == 2


# --------------------------------------------------------------------------- #
# 4. sampler math
# --------------------------------------------------------------------------- #
class _FakeDS:
    def __init__(self, ys):
        self.items = [{"y": int(y)} for y in ys]

    def __len__(self):
        return len(self.items)


def test_sampler_shares():
    ys = np.concatenate([np.zeros(8082), np.ones(14863), np.full(1202, 2)]).astype(int)
    ds = _FakeDS(ys)
    for mode, exp_nor in (
        ("none", 0.0498),
        ("sqrt", 0.1407),
        ("pow025", 0.0840),
        ("pow075", 0.2230),
        ("pow100", 1 / 3),
    ):
        smp = _make_sampler(ds, mode)
        if mode == "none":
            assert smp is None
            continue
        w = smp.weights.numpy()
        share = np.array([w[ys == c].sum() for c in range(3)])
        share /= share.sum()
        assert abs(share[2] - exp_nor) < 0.005, f"{mode}: NoR share {share[2]:.4f} != expected {exp_nor:.4f}"


def test_sampler_pow_parsing():
    """'pow075' must mean exponent 0.75, not 75 (a bug that once shipped)."""
    ys = np.concatenate([np.zeros(100), np.ones(200), np.full(20, 2)]).astype(int)
    smp = _make_sampler(_FakeDS(ys), "pow075")
    w = smp.weights.numpy()
    # weight ratio NoR/RR per sample = (n_RR/n_NoR)^0.75
    ratio = w[ys == 2][0] / w[ys == 0][0]
    assert abs(ratio - (100 / 20) ** 0.75) < 1e-6, f"exponent parsed wrongly: ratio={ratio}"


# --------------------------------------------------------------------------- #
# 5. logit adjustment
# --------------------------------------------------------------------------- #
def test_logit_adjustment():
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    y = torch.tensor([0])
    log_pi = torch.log(torch.tensor([0.5, 0.4, 0.1]))
    got = global_loss(logits, y, log_pi, tau_la=1.0, label_smoothing=0.0)
    want = torch.nn.functional.cross_entropy(logits + log_pi.view(1, -1), y)
    assert torch.allclose(got, want), "logit adjustment not applied as logits + tau*log_pi"
    # tau_la = 0 disables
    got0 = global_loss(logits, y, log_pi, tau_la=0.0, label_smoothing=0.0)
    want0 = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(got0, want0)
    # sample weights
    gotw = global_loss(logits, y, log_pi, 0.0, 0.0, torch.tensor([0.3]))
    assert torch.allclose(gotw, want0 * 0.3)


# --------------------------------------------------------------------------- #
# 6. pooling
# --------------------------------------------------------------------------- #


def test_consistency_kl():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0, 2.0, 3.0]])
    assert abs(consistency_kl(a, b).item()) < 1e-6  # 0 up to log/exp float noise
    c = torch.tensor([[3.0, 2.0, 1.0]])
    assert abs(consistency_kl(a, c).item() - consistency_kl(c, a).item()) < 1e-6


# --------------------------------------------------------------------------- #
# 7. temperature scaling
# --------------------------------------------------------------------------- #
def test_temperature_fit():
    """Generative calibration test: labels sampled from softmax(lg / T_true),
    so the CE-optimal temperature on the raw logits is exactly T_true."""
    rng = np.random.default_rng(0)
    n = 20000
    lg = rng.normal(size=(n, 3)) * 3.0
    T_true = 0.5
    p = np.exp(lg / T_true - (lg / T_true).max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
    T = fit_temperature(lg, y)
    assert abs(T - T_true) < 0.1, f"recovered T={T:.3f}, expected ~{T_true}"


# --------------------------------------------------------------------------- #
# 8. decoder shapes / grads
# --------------------------------------------------------------------------- #


def test_token_losses_masks():
    torch.manual_seed(0)
    B, N = 2, 3600
    maps = {
        "lamp_logit": torch.randn(B, N, requires_grad=True),
        "rel_logit": torch.randn(B, N, requires_grad=True),
        "state_logit": torch.randn(B, N, 6, requires_grad=True),
    }
    lamp = torch.zeros(B, N)
    lamp[:, 10] = 1.0
    valid = torch.ones(B, N, dtype=torch.bool)
    st = torch.zeros(B, N, dtype=torch.long)
    st[:, 10] = 2
    rel = torch.zeros(B, N)
    rel[:, 10] = 1.0
    out = token_losses(maps, lamp, valid, st, rel)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    assert torch.isfinite(maps["rel_logit"].grad).all()
    # all-valid mask: rel loss must run over N*B tokens -> grad everywhere
    assert maps["rel_logit"].grad.abs().min() > 0, "rel supervision skipped tokens"


# --------------------------------------------------------------------------- #
# 9. LR schedule
# --------------------------------------------------------------------------- #
def test_lr_schedule():
    lr = 1e-4
    assert abs(_lr_of(0, 3, 30, lr) - lr / 3) < 1e-12
    assert abs(_lr_of(3, 3, 30, lr) - lr) < 1e-12  # end of warmup = peak
    assert _lr_of(29, 3, 30, lr) < 1e-5  # cosine tail -> ~0
    assert _lr_of(1, 3, 30, lr) < _lr_of(2, 3, 30, lr) <= _lr_of(3, 3, 30, lr)  # warmup


# --------------------------------------------------------------------------- #
# 10. lamp erasure (manufactured NoR)
# --------------------------------------------------------------------------- #
def test_erase_lamps_removes_color():
    torch.manual_seed(0)
    B, H, W = 1, 720, 1280
    x = torch.full((B, 3, H, W), 0.2)
    # a saturated red lamp covering tokens [10:12, 20:22] = px [160:192, 320:352]
    # (as token_targets would rasterise it: every token whose centre is in the box)
    x[:, 0, 160:192, 320:352] = 0.9
    grid = torch.zeros(B, 45, 80)
    grid[:, 10:12, 20:22] = 1.0
    out = erase_lamps(x, grid)
    region = out[:, :, 160:192, 320:352]
    assert region[:, 0].mean() < 0.35, f"red lamp survived erasure (mean {region[:, 0].mean():.3f})"
    assert abs(region.mean() - 0.2) < 0.05, (
        f"fill is not the surrounding texture (mean {region.mean():.3f}, want ~0.2) - "
        "black-hole fill would also pass a 'red is gone' check"
    )
    assert region.std() < 0.05, f"fill not smooth (std {region.std():.3f})"
    # pixels far away must be untouched
    assert torch.allclose(out[:, :, :100, :100], x[:, :, :100, :100])
    assert (out >= 0).all() and (out <= 1).all()
    # sub-token lamp: box snapped to one token, pixels within it + margin
    x1 = x.clone()
    x1[:, 0, 160:176, 320:336] = 0.9
    g1 = torch.zeros(B, 45, 80)
    g1[:, 10, 20] = 1.0
    out1 = erase_lamps(x1, g1)
    assert out1[:, 0, 160:176, 320:336].mean() < 0.35, "single-token lamp survived"
    # LARGE hole (bigger than any single fill window - regression for the
    # single-pass fill that degenerated to black in the hole's centre)
    x2 = x.clone()
    x2[:, 0, 160:288, 320:448] = 0.9  # 8x8 tokens = 128x128 px
    g2 = torch.zeros(B, 45, 80)
    g2[:, 10:18, 20:28] = 1.0
    out2 = erase_lamps(x2, g2)
    r2 = out2[:, :, 170:278, 330:438]
    assert r2[:, 0].mean() < 0.35, f"large lamp survived (mean {r2[:, 0].mean():.3f})"
    assert abs(r2.mean() - 0.2) < 0.05, f"large-hole fill degenerated (mean {r2.mean():.3f})"
    assert r2.std() < 0.05, f"large-hole fill not smooth (std {r2.std():.3f})"


def test_apply_lamp_erasure_labels():
    torch.manual_seed(0)
    B = 6
    x = torch.rand(B, 3, 720, 1280)
    grid = torch.zeros(B, 45, 80)
    grid[:, 5, 5] = 1.0
    y = torch.tensor([0, 1, 0, 1, 2, 0])
    w = torch.ones(B)
    lamp = grid.clone()
    valid = torch.ones(B, 45, 80, dtype=torch.bool)
    rel = torch.zeros(B, 45, 80)
    rel[:, 5, 5] = 1.0
    x2, y2, w2, lamp2, valid2, rel2 = apply_lamp_erasure(x, grid, y, w, lamp, valid, rel, p_erase=1.0)
    # p=1.0 erases every frame -> all relabelled NoR with zeroed targets
    assert (y2 == 2).all(), "erased frames must be labelled NoR"
    assert lamp2.sum() == 0 and rel2.sum() == 0, "erased frames must drop token targets"
    assert valid2.all()
    # token (5,5) = px [80:96, 80:96]; the hole must be filled with smooth
    # local average (~0.5 for U(0,1) noise), not the original noise and not black
    hole = x2[:, :, 82:94, 82:94]
    assert hole.std() < 0.05, f"fill not smooth (std {hole.std():.3f})"
    assert 0.3 < hole.mean() < 0.7, f"fill degenerated (mean {hole.mean():.3f})"
    assert not torch.allclose(x2, x), "input must be modified"
    # originals untouched (clone semantics)
    assert (y == torch.tensor([0, 1, 0, 1, 2, 0])).all() and lamp.sum() == B
    # p=0 is a no-op
    x3, y3, *_ = apply_lamp_erasure(x, grid, y, w, lamp, valid, rel, p_erase=0.0)
    assert torch.allclose(x3, x) and (y3 == y).all()


# --------------------------------------------------------------------------- #
# 7. grid derivation, overlap conflicts, direction targets
# --------------------------------------------------------------------------- #
def test_grid_from_target_hw():
    """The token grid follows target_hw/patch instead of a hard-coded 45x80."""
    from dinov3_global.data import grid_shape

    assert grid_shape((720, 1280)) == (45, 80)
    assert grid_shape((320, 576)) == (20, 36)
    # a box lands on the grid of a NON-default target size
    tt = token_targets(
        np.array([[160.0, 160.0, 224.0, 224.0]]),
        np.array([1]),
        np.array([2]),
        label_crop_sides=0,
        target_hw=(320, 576),
    )
    assert tt["lamp"].shape == (20, 36), tt["lamp"].shape
    # 2048x1024 source -> 576x320 target: sx = 576/2048, sy = 320/1024
    assert tt["lamp"].sum() > 0


def test_overlap_nearest_center():
    """Shared tokens go to the NEAREST box centre (not the last box in file order)."""
    # two overlapping boxes; shared token centre sits closer to box A's centre
    a = [100.0, 100.0, 150.0, 130.0]  # centre (125, 115)
    b = [130.0, 100.0, 180.0, 130.0]  # centre (155, 115)
    tt = token_targets(
        np.array([a, b]),
        np.array([1, 0]),
        np.array([2, 0]),
        label_crop_sides=0,
        target_hw=(1024, 2048),
        direction=np.array([1, 0]),
    )
    # token centres are 16*i+8 -> shared token has centre x=136 (col 8)
    col = (136 - 8) // 16
    row = (104 - 8) // 16
    assert tt["lamp"][row, col] == 1.0
    assert tt["state"][row, col] == 2, "shared token must follow the nearest box (A)"
    assert tt["rel"][row, col] == 1.0
    assert tt["dir"][row, col] == 1
    # union semantics for lampness: every token of both boxes is positive
    assert tt["lamp"][row, 6:11].sum() == 5
    # and the reverse argument order must not change the outcome
    tt2 = token_targets(
        np.array([b, a]),
        np.array([0, 1]),
        np.array([0, 2]),
        label_crop_sides=0,
        target_hw=(1024, 2048),
        direction=np.array([0, 1]),
    )
    assert tt2["state"][row, col] == 2, "assignment must not depend on label order"


def test_direction_targets():
    """dir grid carries the housing direction on lamp tokens, -1 elsewhere."""
    tt = token_targets(
        np.array([[100.0, 100.0, 130.0, 130.0]]),
        np.array([1]),
        np.array([2]),
        label_crop_sides=0,
        target_hw=(1024, 2048),
        direction=np.array([3]),
    )
    inside = tt["lamp"] > 0.5
    assert (tt["dir"][inside] == 3).all()
    assert (tt["dir"][~inside] == -1).all()
    # without the direction arg no dir grid is produced (old callers unaffected)
    tt2 = token_targets(
        np.array([[100.0, 100.0, 130.0, 130.0]]),
        np.array([1]),
        np.array([2]),
        label_crop_sides=0,
        target_hw=(1024, 2048),
    )
    assert "dir" not in tt2


# --------------------------------------------------------------------------- #
# 8. local (lampness-weighted) pooling
# --------------------------------------------------------------------------- #
def test_local_pooling():
    """pool='local' aggregates a lamp's tokens and keeps singletons intact."""
    br = MILBranch(8, 0.0, k=1, grid_hw=(4, 4))
    e = torch.zeros(1, 16)
    lamp = torch.zeros(1, 16)
    # A: isolated single-token lamp -> NOT diluted by the window mass
    e[0, 5], lamp[0, 5] = 0.9, 1.0
    s = br.pooled(e, lamp=lamp)
    assert abs(float(s) - 0.9) < 1e-5, f"singleton diluted: {float(s)}"
    # B: two-token lamp where only one token fires -> averaged, unlike max-pool
    e2 = torch.zeros(1, 16)
    lamp2 = torch.zeros(1, 16)
    e2[0, 5], e2[0, 6], lamp2[0, 5], lamp2[0, 6] = 0.9, 0.1, 1.0, 1.0
    s2 = br.pooled(e2, lamp=lamp2)
    assert abs(float(s2) - 0.5) < 1e-5, f"expected 0.5, got {float(s2)}"
    assert float(s2) < 0.9, "local pooling must suppress single-token noise"
    # topk on the raw map would have returned the noisy peak
    assert abs(float(e2.max()) - 0.9) < 1e-6


# --------------------------------------------------------------------------- #
# 9. direction head + rel input, CLS modulation, mid fusion
# --------------------------------------------------------------------------- #
def test_dir_head_and_rel_input():
    torch.manual_seed(0)
    h = TokenMILHead(384, 64, topk=(1, 2), grid_hw=(4, 5))
    br = h.branches[0]
    assert br.dir is not None and br.dir.out_features == 4
    assert br.rel.in_features == 64 + 4 + 4, "rel must see [z; geo; dir_post]"
    lg, aux = h(torch.randn(2, 20, 384), torch.randn(2, 384))
    assert lg.shape == (2, 3)
    assert aux["maps"]["dir_logit"].shape == (2, 20, 4)
    lg.sum().backward()
    assert br.dir.weight.grad is not None and br.rel.weight.grad is not None


def test_mid_fusion_shapes():
    torch.manual_seed(0)
    h = TokenMILHead(768, 64, topk=(1,), grid_hw=(4, 5))
    p = torch.randn(2, 20, 384)
    lg, aux = h(p, torch.randn(2, 384), patches_mid=torch.randn(2, 20, 384), cls_mid=torch.randn(2, 384))
    assert lg.shape == (2, 3)
    assert aux["tokens"].shape == (2, 20, 64)
    # patches_mid without cls_mid must fail loudly (shared projection)
    try:
        h(p, torch.randn(2, 384), patches_mid=torch.randn(2, 20, 384))
        raise AssertionError("mismatched fusion must raise")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# 10. focal rel loss, state balancing, dir loss part
# --------------------------------------------------------------------------- #
def test_focal_rel_downweights_easy_negatives():
    """Focal modulation must stop the trivial background from swamping the
    loss so irrelevant lamps remain explicit negatives."""
    from dinov3_global.losses import _bce

    logits = torch.zeros(102)
    logits[:2] = 4.0  # confident positives
    logits[2:] = -8.0  # easy negatives
    y = torch.zeros(102)
    y[:2] = 1.0
    l_plain = _bce(logits, y)
    l_focal = _bce(logits, y, focal_gamma=2.0)
    assert float(l_focal) < 0.01 * float(l_plain), (float(l_plain), float(l_focal))
    # gradient on an easy negative collapses under focal
    lg_p = logits.clone().requires_grad_(True)
    _bce(lg_p, y).backward()
    g_plain = float(lg_p.grad[50])
    lg_f = logits.clone().requires_grad_(True)
    _bce(lg_f, y, focal_gamma=2.0).backward()
    g_focal = float(lg_f.grad[50])
    assert abs(g_focal) < 0.01 * abs(g_plain), (g_plain, g_focal)


def test_state_balance_and_rel_boost():
    torch.manual_seed(0)
    B, H, W = 1, 4, 5
    N = H * W
    maps = {
        "lamp_logit": torch.full((B, N), 4.0),
        "rel_logit": torch.zeros(B, N),
        "state_logit": torch.zeros(B, N, 6),
    }
    lamp = torch.ones(B, H, W)
    lamp[0] = 0.0
    lamp[0, 0, 0] = 1.0  # exactly 2 lamp tokens
    lamp[0, 0, 1] = 1.0
    valid = torch.ones(B, H, W, dtype=torch.bool)
    state = torch.full((B, H, W), 2)  # all red
    rel = torch.zeros(B, H, W)
    rel[0, 0, 0] = 1.0  # one relevant lamp token
    base = token_losses(maps, lamp, valid, state, rel, w_lamp=0.0, w_state=1.0, w_rel=0.0)
    # class weights scale the state loss for the target class
    w = torch.ones(6)
    w[2] = 5.0
    weighted = token_losses(
        maps, lamp, valid, state, rel, w_lamp=0.0, w_state=1.0, w_rel=0.0, state_weights=w
    )
    assert abs(float(weighted["state"]) - 5.0 * float(base["state"])) < 1e-5
    # rel boost up-weights exactly the rel-positive token(s)
    boosted = token_losses(
        maps, lamp, valid, state, rel, w_lamp=0.0, w_state=1.0, w_rel=0.0, state_rel_boost=1.0
    )
    assert abs(float(boosted["state"]) - 1.5 * float(base["state"])) < 1e-5


def test_dir_loss_part():
    torch.manual_seed(0)
    B, H, W = 1, 4, 5
    N = H * W
    maps = {
        "lamp_logit": torch.full((B, N), 4.0),
        "rel_logit": torch.zeros(B, N),
        "state_logit": torch.zeros(B, N, 6),
        "dir_logit": torch.zeros(B, N, 4),
    }
    lamp = torch.ones(B, H, W)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    state = torch.zeros(B, H, W, dtype=torch.long)
    rel = torch.zeros(B, H, W)
    dir_tgt = torch.full((B, H, W), -1)
    dir_tgt[0, 0, 0] = 1  # one supervised direction
    out = token_losses(
        maps, lamp, valid, state, rel, w_lamp=0.0, w_state=0.0, w_rel=0.0, w_dir=1.0, dir_tgt=dir_tgt
    )
    assert float(out["dir"]) > 0
    assert abs(float(out["dir"]) - float(out["total"])) < 1e-6


def test_to_jsonable():
    from dinov3_global.metrics import to_jsonable
    import json

    obj = {
        "a": np.float64("nan"),
        "b": np.float32(1.5),
        "c": np.arange(3),
        "d": float("inf"),
        "e": {"f": np.int64(7)},
    }
    out = to_jsonable(obj)
    assert out["a"] is None and out["d"] is None
    assert out["b"] == 1.5 and out["c"] == [0, 1, 2] and out["e"]["f"] == 7
    json.dumps(out, allow_nan=False)  # must be strict-JSON serialisable
