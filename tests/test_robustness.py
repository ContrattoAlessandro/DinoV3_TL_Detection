"""Regression tests for spatial labels, shortcut controls and reusable OOD labels."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dinov3_global.augment import train_view
from dinov3_global.head import EvidenceMILHead
from dinov3_global.losses import token_losses
from scripts.annotate_ood import AnnotationStore
from scripts.evaluate_ood import csv_logits, evaluate_labels, read_labels


def test_consistency_never_crops(monkeypatch):
    import dinov3_global.augment as aug

    def bad_crop(*args):
        raise AssertionError("crop destroys existential label")

    monkeypatch.setattr(aug, "random_framing", bad_crop)
    x = torch.rand(3, 3, 32, 48)
    train_view(x)


def test_disabled_augmentation_preserves_input():
    x = torch.rand(3, 3, 32, 48)
    before = x.clone()
    actual = train_view(x, photometric_enabled=False, degradations_enabled=False)
    assert torch.equal(actual, before) and torch.equal(x, before)
    assert torch.equal(train_view(x, strength=0), before)


def test_augmentation_blend_and_per_image(monkeypatch):
    import dinov3_global.augment as aug

    draws = iter([0.0, 1.0])
    monkeypatch.setattr(aug, "photometric", lambda x, strength: torch.full_like(x, next(draws)))
    x = torch.full((2, 3, 32, 48), 0.5)
    actual = train_view(x, degradations_enabled=False, blend_min=0.5)
    assert (actual[0] <= 0.5).all() and (actual[0] >= 0.25).all()
    assert (actual[1] >= 0.5).all() and (actual[1] <= 0.75).all()
    assert torch.equal(x, torch.full_like(x, 0.5))


def test_appearance_heads_do_not_see_absolute_position():
    torch.manual_seed(1)
    head = EvidenceMILHead(in_dim=8, proj_dim=16, grid_hw=(3, 4), dropout=0).eval()
    with torch.no_grad():
        head.rel_context[-1].weight.fill_(0.1)
    patches = torch.ones(1, 12, 8)
    _, aux = head(patches, torch.ones(1, 8))
    for key in ("lamp_logit", "state_logit", "dir_logit", "pictogram_logit"):
        value = aux["maps"][key]
        assert torch.allclose(value[:, 0:1].expand_as(value), value)
    assert not torch.allclose(aux["maps"]["rel_logit"][:, :1].expand(1, 12), aux["maps"]["rel_logit"])


def test_context_is_bounded_and_shared_predictors_receive_gradients():
    head = EvidenceMILHead(in_dim=8, proj_dim=16, grid_hw=(3, 4), dropout=0)
    with torch.no_grad():
        head.rel_context[-1].bias.fill_(1000)
    logits, aux = head(torch.randn(2, 12, 8), torch.randn(2, 8))
    assert (aux["relevance_correction"].abs() <= 1).all()
    lamp = torch.zeros(2, 3, 4)
    lamp[:, 0, 0] = 1
    loss = token_losses(
        aux["maps"],
        lamp,
        torch.ones_like(lamp, dtype=torch.bool),
        lamp.long() * 2,
        lamp,
        w_dir=0.2,
        dir_tgt=lamp.long(),
        w_pictogram=0.1,
        pictogram_tgt=lamp.long(),
    )["total"]
    (loss + logits.square().mean()).backward()
    for module in (head.lamp, head.state, head.dir, head.pictogram, head.rel_base[0]):
        assert module.weight.grad.abs().sum() > 0


def test_hard_lamp_loss_is_not_diluted_by_background_count():
    def loss(n):
        lamp = torch.zeros(1, n)
        lamp[0, :2] = 1
        rel = torch.zeros_like(lamp)
        rel[0, 0] = 1
        maps = {
            "lamp_logit": torch.zeros_like(lamp),
            "rel_logit": torch.zeros_like(lamp),
            "state_logit": torch.zeros(1, n, 6),
        }
        return token_losses(
            maps,
            lamp,
            torch.ones_like(lamp, dtype=torch.bool),
            lamp.long(),
            rel,
            w_lamp=0,
            w_state=0,
            w_rel=1,
            rel_lamp_weight=0.75,
            rel_pos_weight=2,
        )["total"]

    assert torch.allclose(loss(10), loss(1000))


def make_store(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i * 60, 0, 0)).save(root / f"{i}.jpg")
    return AnnotationStore(root, tmp_path / "labels.json")


def test_annotation_roundtrip_revision_and_uncertain_exclusion(tmp_path):
    store = make_store(tmp_path)
    for i, c in enumerate(("RR", "RG", "NoR")):
        store.update({"id": i, "label": c, "notes": "a note", "uncertain": i == 2, "revision": i})
    again = AnnotationStore(store.root, store.path)
    assert again.data["revision"] == 3
    assert again.data["images"]["1.jpg"]["label"] == "RG"
    entries, excluded = read_labels(store.path)
    assert len(entries) == 2 and excluded["uncertain"] == 1
    with pytest.raises(RuntimeError):
        store.update({"id": 0, "label": "RG", "revision": 1})
    with pytest.raises(ValueError):
        store.update({"id": -1, "label": "RR", "revision": 3})


def test_annotation_manifest_rejects_changed_images(tmp_path):
    store = make_store(tmp_path)
    Image.new("RGB", (16, 16), "green").save(store.files[0])
    with pytest.raises(ValueError, match="changed"):
        AnnotationStore(store.root, store.path)


def test_save_failure_rolls_back_and_keeps_previous_file(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    before = store.path.read_bytes()

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write", fail)
    with pytest.raises(OSError):
        store.update({"id": 0, "label": "RR", "revision": 0})
    assert store.data["revision"] == 0 and store.data["images"]["0.jpg"]["label"] is None
    assert store.path.read_bytes() == before


def test_missing_class_does_not_report_three_class_map(tmp_path):
    store = make_store(tmp_path)
    store.update({"id": 0, "label": "RR", "revision": 0})
    store.update({"id": 1, "label": "RG", "revision": 1})
    entries, excluded = read_labels(store.path)
    r = evaluate_labels(entries, np.array([[5, 0, 0], [0, 5, 0]]), excluded)
    assert r["metrics"]["mAP"] is None
    assert r["metrics"]["mAP_present_classes"] == 1
    assert r["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    json.dumps(r, allow_nan=False)


def test_prediction_csv_prefers_raw_logits(tmp_path):
    p = tmp_path / "predictions.csv"
    p.write_text("file,P_RR,P_RG,P_NoR,logit_RR,logit_RG,logit_NoR\na.jpg,0.9,0.05,0.05,1,2,3\n")
    actual = csv_logits(p, [("a.jpg", tmp_path / "a.jpg", 0)])
    assert np.array_equal(actual, [[1, 2, 3]])
    # External ATLAS scripts used an inner dataset root; accept unique suffixes.
    actual = csv_logits(p, [("ATLAS/sample/camera/images/a.jpg", tmp_path / "a.jpg", 0)])
    assert np.array_equal(actual, [[1, 2, 3]])


def test_city_manifest_checks_membership_classes_and_session_leakage():
    from dinov3_global.validation import make_folds, fold_manifest

    items = [
        {"city": c, "y": y, "entry": {"image_path": f"./{c}/route/drive/{y}.tiff"}}
        for c in ("A", "B")
        for y in range(3)
    ]
    folds, _ = make_folds(items, 2)
    manifest = fold_manifest(items, folds, {}, "source-hash")
    assert all(f["session_overlap"] == 0 and f["val_hist"] == [1, 1, 1] for f in manifest["folds"])
    with pytest.raises(ValueError, match="every city"):
        fold_manifest(items, [["A"], ["A", "B"]], {}, "source-hash")
    changed = [dict(it) for it in items]
    changed[3]["entry"] = {"image_path": "./A/route/drive/other.tiff"}
    with pytest.raises(ValueError, match="session overlap"):
        fold_manifest(changed, folds, {}, "source-hash")
    missing_class = [it for it in items if not (it["city"] == "B" and it["y"] == 2)]
    with pytest.raises(ValueError, match="missing global class"):
        fold_manifest(missing_class, folds, {}, "source-hash")


def test_rng_resume_roundtrip_is_weights_only_compatible(tmp_path):
    import random
    from dinov3_global.engine import capture_rng_state, restore_rng_state

    original = capture_rng_state()
    state_file = tmp_path / "state.pt"
    try:
        torch.save(original, state_file)
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(torch.load(state_file, weights_only=True))
        actual = (random.random(), np.random.rand(), torch.rand(3))
        assert actual[:2] == expected[:2]
        assert torch.equal(actual[2], expected[2])
    finally:
        restore_rng_state(original)
