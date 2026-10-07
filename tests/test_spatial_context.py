"""Spatial mixing, relevance isolation, and legacy checkpoint contracts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dinov3_global.config import architecture_name, load_config, validate_config
from dinov3_global.head import AxialBlock, AxialSpatialContext, TokenMILHead
from dinov3_global.model import DinoGlobal
from scripts.evaluate_vzc import evaluation_jobs

ROOT = Path(__file__).resolve().parents[1]
AXIAL = dict(mode="axial", depth=2, heads=4, mlp_ratio=2)


def heads(grid=(3, 4)):
    torch.manual_seed(0)
    baseline = TokenMILHead(in_dim=8, proj_dim=8, grid_hw=grid, dropout=0, scene_dropout=0).eval()
    torch.manual_seed(0)
    spatial = TokenMILHead(in_dim=8, proj_dim=8, grid_hw=grid, dropout=0, scene_dropout=0,
                           spatial_context=AXIAL).eval()
    return baseline, spatial


def test_disabled_context_preserves_seeded_weights_and_strict_state_dict():
    baseline, spatial = heads()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, spatial.state_dict()[key])
    explicit = TokenMILHead(in_dim=8, proj_dim=8, grid_hw=(3, 4), dropout=0, scene_dropout=0,
                            spatial_context={"mode": "none"}).eval()
    explicit.load_state_dict(baseline.state_dict(), strict=True)
    patches, cls = torch.randn(2, 12, 8), torch.randn(2, 8)
    a, am = baseline(patches, cls)
    b, bm = explicit(patches, cls)
    assert torch.equal(a, b)
    for key in am["maps"]:
        assert torch.equal(am["maps"][key], bm["maps"][key])


def test_context_runs_once_and_only_changes_relevance_inputs():
    baseline, spatial = heads()
    calls = []
    handle = spatial.spatial_context.register_forward_hook(lambda *args: calls.append(1))
    patches, cls = torch.randn(2, 12, 8), torch.randn(2, 8)
    _, old = baseline(patches, cls)
    logits, new = spatial(patches, cls)
    handle.remove()
    assert calls == [1]
    assert torch.equal(old["tokens"], new["tokens"])
    for a, b in zip(old["branch_maps"], new["branch_maps"]):
        for key in ("lamp_logit", "state_logit", "dir_logit"):
            assert torch.equal(a[key], b[key])
        assert not torch.equal(a["rel_logit"], b["rel_logit"])
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in spatial.spatial_context.parameters())


def test_relevance_queries_a_different_column_twenty_rows_below():
    baseline, spatial = heads((24, 3))
    patches, cls = torch.randn(1, 72, 8), torch.randn(1, 8)
    changed = patches.clone()
    changed[:, 20 * 3 + 2] += torch.arange(8)
    old = baseline(patches, cls)[1]["maps"]["rel_logit"][0, 0]
    old_changed = baseline(changed, cls)[1]["maps"]["rel_logit"][0, 0]
    new = spatial(patches, cls)[1]["maps"]["rel_logit"][0, 0]
    new_changed = spatial(changed, cls)[1]["maps"]["rel_logit"][0, 0]
    assert torch.equal(old, old_changed)
    assert (new - new_changed).abs() > 1e-6


def test_axial_rows_and_columns_preserve_batch_and_grid_order():
    captured = []

    class Capture(nn.Module):
        def forward(self, query, key, value, need_weights):
            assert need_weights is False
            captured.append(query.detach().clone())
            return torch.zeros_like(query), None

    block = AxialBlock(4, 1, 2, 0)
    block.row_norm = block.col_norm = block.ffn_norm = nn.Identity()
    block.row_attn = block.col_attn = Capture()
    for p in block.ffn.parameters():
        nn.init.zeros_(p)
    grid = torch.arange(2 * 3 * 5 * 4).reshape(2, 3, 5, 4).float()
    assert torch.equal(block(grid), grid)
    assert torch.equal(captured[0], grid.reshape(6, 5, 4))
    assert torch.equal(captured[1], grid.permute(0, 2, 1, 3).reshape(10, 3, 4))


@pytest.mark.parametrize("value", [dict(mode="bad"), dict(mode="axial", heads=5),
                                    dict(mode="axial", depth=0), dict(mode="axial", mlp_ratio=float('nan'))])
def test_invalid_spatial_config_is_rejected(value):
    cfg = deepcopy(load_config(ROOT / "configs/default.yaml"))
    cfg["decoder"]["spatial_context"] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_grid_mismatch_is_rejected():
    context = AxialSpatialContext(8, (2, 3))
    with pytest.raises(ValueError):
        context(torch.randn(1, 5, 8))
    baseline, _ = heads()
    with pytest.raises(ValueError):
        baseline(torch.randn(1, 11, 8), torch.randn(1, 8))


def test_axial_model_keeps_encoder_frozen(monkeypatch):
    from transformers import AutoModel

    class MockViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=384)
            self.parameter = nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values, output_hidden_states):
            final = torch.ones(pixel_values.shape[0], 11, 384, device=pixel_values.device)
            return SimpleNamespace(last_hidden_state=final, hidden_states=tuple(final for _ in range(13)))

    net = MockViT()
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: net)
    model = DinoGlobal(dtype="float32", grid_hw=(2, 3), proj_dim=8, dropout=0, scene_dropout=0,
                       spatial_context=AXIAL).train()
    model(torch.ones(2, 3, 32, 48))["logits"].sum().backward()
    assert not net.training and net.parameter.grad is None
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert model.head.spatial_context.blocks[0].row_attn.in_proj_weight.grad is not None


def test_evaluation_uses_architecture_names_and_rejects_duplicate_names(tmp_path):
    cfg = load_config(ROOT / "configs/spatial_axial.yaml")
    assert architecture_name(cfg) == "v6_axial"
    assert architecture_name(load_config(ROOT / "configs/default.yaml")) == "v5"
    checkpoint = tmp_path / "best.pt"
    torch.save(dict(cfg=cfg), checkpoint)
    assert evaluation_jobs([checkpoint])[0]["name"] == "v6_axial_fold0"
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation_jobs([checkpoint, checkpoint], ["same", "same"])
