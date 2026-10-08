"""Contracts for the public Experiment C package, frozen backbone, and metric reports."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dinov3_global.config import load_config, validate_config
from dinov3_global.backbone import BACKBONE_SPECS, backbone_spec
from dinov3_global.metrics import report, to_jsonable
from dinov3_global.model import DinoGlobal

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("decoder", "head", "attention"),
        ("decoder", "head", "mil"),
        ("decoder", "spatial_context", {"mode": "axial"}),
        ("decoder", "pool", "topk"),
        ("decoder", "relevance_context", False),
        ("backbone", "mid_layer", None),
        ("backbone", "frozen", False),
        ("backbone", "use_cls", False),
        ("backbone", "hf_id", "facebook/dinov2-base"),
        ("backbone", "dim", 384),
        ("data", "global_classes", ["NoR", "RR", "RG"]),
        ("data", "preprocessing", "legacy"),
        ("data", "label_crop_sides", 114),
    ],
)
def test_default_configuration_rejects_incompatible_models(section, key, value):
    cfg = deepcopy(load_config(ROOT / "configs/default.yaml"))
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


class MockViT(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.parameter = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values, output_hidden_states):
        self.last_input = pixel_values.detach().clone()
        count = pixel_values.shape[0]
        # CLS at 0, four register tokens at 1..4, six image patches at 5..10.
        positions = torch.arange(11, dtype=pixel_values.dtype).reshape(1, 11, 1)
        final = positions.expand(count, -1, self.config.hidden_size) + 12
        hidden = tuple(positions.expand(count, -1, self.config.hidden_size) + layer for layer in range(13))
        return SimpleNamespace(last_hidden_state=final, hidden_states=hidden)


@pytest.mark.parametrize("hf_id", BACKBONE_SPECS)
def test_backbone_discards_registers_and_stays_frozen_during_training(monkeypatch, hf_id):
    from transformers import AutoModel

    spec = backbone_spec(hf_id)
    net = MockViT(spec["dim"])
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return net

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    model = DinoGlobal(hf_id=hf_id, dtype="float32", grid_hw=(2, 3), proj_dim=8, dropout=0)
    model.train()
    assert model.training and model.head.training
    assert not model.backbone.training and not net.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert calls[0]["revision"] == spec["revision"]
    assert calls[0]["trust_remote_code"] is False
    images = torch.ones(2, 3, 32, 48)
    features = model.backbone(images)
    assert features["patches"].shape == (2, 6, spec["dim"])
    assert model.head.proj.in_features == 2 * spec["dim"]
    assert torch.equal(features["patches"][0, :, 0], torch.arange(5, 11) + 12)
    assert torch.equal(features["patches_mid"][0, :, 0], torch.arange(5, 11) + 6)
    assert float(features["cls"][0, 0]) == 12
    assert float(features["cls_mid"][0, 0]) == 6
    expected = (images - model.backbone._mean) / model.backbone._std
    assert torch.equal(net.last_input, expected)
    result = model(images)
    assert result["logits"].shape == (2, 3)
    result["logits"].sum().backward()
    assert net.parameter.grad is None
    assert model.head.proj.weight.grad is not None
    assert {id(p) for p in model.head_parameters()} == {id(p) for p in model.parameters() if p.requires_grad}


def test_backbone_configs_change_only_encoder():
    default = load_config(ROOT / "configs/default.yaml")
    reference = load_config(ROOT / "configs/vitsplus.yaml")
    assert default["backbone"]["dim"] == 768
    assert reference["backbone"]["dim"] == 384
    for section in default.keys() - {"backbone"}:
        assert reference[section] == default[section]
    for key in default["backbone"].keys() - {"hf_id", "dim", "local_ckpt"}:
        assert reference["backbone"][key] == default["backbone"][key]


def test_removed_experimental_supervision_is_rejected():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["loss"]["lamp_positive_fraction"] = 0.25
    with pytest.raises(ValueError, match="unsupported"):
        validate_config(cfg)


def test_checkpoint_metadata_supports_weights_only_loading(tmp_path):
    y = np.tile([0, 1, 2], 10)
    rep = report(y, np.eye(3)[y] * 5, cities=np.asarray(["city"] * len(y)))
    path = tmp_path / "best.pt"
    torch.save(dict(report=to_jsonable(rep), head=dict(weight=torch.ones(1))), path)
    restored = torch.load(path, weights_only=True)
    assert restored["report"]["confusion_matrix"] == rep["confusion_matrix"]
    assert restored["report"]["metrics"]["mAP"] == rep["metrics"]["mAP"]


def test_local_backbone_cannot_silently_use_small_features_for_base(monkeypatch):
    from transformers import AutoModel

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **kw: MockViT())
    with pytest.raises(ValueError, match="Loaded backbone width"):
        DinoGlobal(hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m", local_ckpt="wrong_snapshot")


def test_metrics_distinguish_class_ranking_from_argmax_decisions():
    y = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.75, 0.15, 0.1],
            [0.6, 0.3, 0.1],
            [0.6, 0.35, 0.05],
            [0.55, 0.05, 0.4],
            [0.55, 0.1, 0.35],
        ]
    )
    result = report(y, np.log(probabilities))
    assert result["metrics"]["mAP"] == 1.0
    assert result["metrics"]["acc_bal"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"] == [[2, 0, 0], [2, 0, 0], [2, 0, 0]]
    assert result["per_class"]["NoR"]["recall"] == 0
    assert result["metrics"]["nll"] > 0
    assert 0 <= result["metrics"]["brier"] <= 2


def test_undefined_ap_is_exported_as_strict_json_null():
    result = to_jsonable(report(np.array([0, 1]), np.array([[3.0, 0, 0], [0, 3, 0]])))
    assert result["metrics"]["AP_NoR"] is None
    assert result["metrics"]["n_classes"] == 2


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_calibration_temperatures_are_rejected(temperature):
    with pytest.raises(ValueError):
        report(np.array([0, 1, 2]), np.eye(3), T=temperature)
