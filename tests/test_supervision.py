"""Full-frame geometry, shared evidence, and auxiliary supervision invariants."""

import random

import numpy as np
from PIL import Image
import pytest
import torch

from dinov3_global.data import DTLDGlobalDataset, collate_global
from dinov3_global.dtld import FrameSample
from dinov3_global.engine import global_balance_weights, _make_sampler, capture_rng_state, restore_rng_state
from dinov3_global.head import EvidenceMILHead
from dinov3_global.losses import masked_mean, token_losses
from dinov3_global.preprocessing import letterbox, rasterize_attributes, preprocess_for_config


@pytest.mark.parametrize("size", [(720, 1280), (896, 1600)])
def test_full_frame_boundary_boxes_and_padding(size):
    image, transform = letterbox(Image.new("RGB", (2048, 1024)), size)
    boxes = transform.boxes(np.array([[0, 0, 8, 8], [2040, 1016, 2048, 1024]], np.float32))
    mask, geo = transform.patch_metadata()
    targets = rasterize_attributes(boxes, [1, 1], [2, 0], [1, 1], [5, 5], size, mask)
    assert image.size == size[::-1] and targets["lamp"].sum() == 2
    assert not targets["valid"][~mask].any() and not targets["lamp"][~mask].any()
    assert geo.shape == (size[0] // 16 * (size[1] // 16), 4)
    assert np.allclose(
        boxes[1, 2:],
        [transform.offset_yx[1] + transform.resized_hw[1], transform.offset_yx[0] + transform.resized_hw[0]],
    )


def test_zoom_preserves_entire_frame_and_rng_resume():
    state = capture_rng_state()
    image = Image.new("RGB", (2048, 1024))
    try:
        random.seed(31)
        snapshot = capture_rng_state()
        first = letterbox(image, zoom=0.9, random_placement=True)[1]
        restore_rng_state(snapshot)
        second = letterbox(image, zoom=0.9, random_placement=True)[1]
        assert first == second
        mapped = first.boxes(np.array([[0, 0, 2048, 1024]], np.float32))[0]
        assert 0 <= mapped[0] < mapped[2] <= 1280 and 0 <= mapped[1] < mapped[3] <= 720
    finally:
        restore_rng_state(state)


def test_independent_conflict_masks_are_order_independent():
    boxes = np.array([[1, 1, 15, 15], [2, 2, 14, 14]])
    attributes = [[1, 1], [2, 0], [1, 1], [5, 6]]
    for order in ([0, 1], [1, 0]):
        result = rasterize_attributes(
            boxes[order], *[np.array(a)[order] for a in attributes], (32, 32), np.ones((2, 2), bool)
        )
        assert result["lamp"][0, 0] == 1
        assert result["rel_valid"][0, 0] and result["dir_valid"][0, 0]
        assert not result["state_valid"][0, 0] and not result["pictogram_valid"][0, 0]


def test_dataset_uses_transformed_targets_and_matches_inference(tmp_path):
    (tmp_path / "train").mkdir()
    path = "./City/route/drive/frame.tiff"
    Image.new("RGB", (2048, 1024), (20, 100, 70)).save(tmp_path / "train/frame.jpg")
    frame = FrameSample(
        path,
        np.array([[0, 0, 8, 8]], np.float32),
        np.array([1]),
        np.array([1]),
        np.array([2]),
        np.array([5]),
        ["a"],
    )
    items = [
        dict(
            entry=dict(image_path=path), frame=frame, y=0, weight=1.0, city="City", n_lamps=1, max_lamp_h=5.0
        )
    ]
    ds = DTLDGlobalDataset(
        "unused",
        str(tmp_path),
        "train",
        items=items,
    )
    sample = ds[0]
    tensor, _, metadata = preprocess_for_config(
        tmp_path / "train/frame.jpg", dict(data=dict(target_hw=[720, 1280], preprocessing="letterbox"))
    )
    assert torch.equal(tensor, sample["image"])
    assert torch.equal(metadata["content_mask"], sample["content_mask"])
    assert sample["lamp_tgt"].sum() == 1 and sample["label"] == 0
    assert sample["pictogram_tgt"][sample["lamp_tgt"] > 0.5].item() == 5
    assert collate_global([sample])["geometry"].shape == (1, 3600, 4)
    assert _make_sampler(ds, "none") is None


def test_instance_reduction_equalizes_lamps_then_images():
    values = torch.tensor([[1.0, 1.0, 1.0, 9.0], [5.0, 5.0, 5.0, 5.0]], requires_grad=True)
    ids = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])
    result = masked_mean(values, torch.ones_like(values, dtype=torch.bool), ids, True)
    assert result.item() == 5.0
    result.backward()
    assert values.grad[0, 3] == 3 * values.grad[0, 0]
    assert masked_mean(values, torch.zeros_like(values, dtype=torch.bool), ids, True).item() == 0


def test_shared_agreeing_patch_contributes_to_each_instance():
    targets = rasterize_attributes(
        np.array([[0, 0, 16, 16], [0, 0, 32, 16]]),
        [1, 1],
        [2, 2],
        [1, 1],
        [5, 5],
        (32, 32),
        np.ones((2, 2), bool),
    )
    # Lamp one averages patch A; lamp two averages A,B: weights A=.75,B=.25.
    assert np.allclose(targets["state_instance_weight"][0], [0.75, 0.25])
    assert targets["state_instance_weight"].sum() == 1


def test_global_weights_mean_example_one():
    pi = torch.tensor([0.33, 0.62, 0.05])
    weights = global_balance_weights(pi.log(), "sqrt")
    torch.testing.assert_close((weights * pi).sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[2] / weights[0], (pi[0] / pi[2]).sqrt())


def test_padding_has_no_mil_evidence():
    head = EvidenceMILHead(in_dim=8, proj_dim=16, grid_hw=(4, 4), dropout=0).eval()
    patches, cls = torch.randn(2, 16, 8), torch.randn(2, 8)
    _, out = head(patches, cls, content_mask=torch.zeros(2, 4, 4, dtype=torch.bool))
    assert not out["maps"]["lamp"].any()
    assert not out["maps"]["rr"].any() and not out["maps"]["rg"].any()


def test_nor_is_explicitly_decreasing_and_context_starts_zero():
    head = EvidenceMILHead(in_dim=8, proj_dim=16, grid_hw=(4, 4), dropout=0).eval()
    scores = torch.tensor([[0.2, 0.3]], requires_grad=True)
    logits = head.classify(scores)
    derivative = torch.autograd.grad(logits[0, 2], scores)[0]
    assert (derivative < 0).all()
    assert (head.classify(scores + 0.1)[0, :2] > logits[0, :2]).all()
    _, out = head(torch.randn(1, 16, 8), torch.randn(1, 8))
    assert out["relevance_correction"].abs().max() == 0
    assert out["maps"]["pictogram_logit"].shape == (1, 16, 10)
    with torch.no_grad():
        head.rel_context[-1].bias.fill_(100)
    _, out = head(torch.randn(1, 16, 8), torch.randn(1, 8))
    assert out["relevance_correction"].abs().max() <= 1


def test_conflicts_and_erasure_remove_attribute_gradients():
    maps = {
        k + "_logit": torch.randn(1, 4, n, requires_grad=True)
        for k, n in (("state", 6), ("dir", 4), ("pictogram", 10))
    }
    maps.update(
        lamp_logit=torch.randn(1, 4, requires_grad=True), rel_logit=torch.randn(1, 4, requires_grad=True)
    )
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    lamp, targets = torch.ones(1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.long)
    masks = {k + "_valid": valid.clone() for k in ("state", "dir", "rel", "pictogram")}
    masks["state_valid"][:] = False
    ids = torch.zeros_like(targets)
    loss = token_losses(
        maps,
        lamp,
        valid,
        targets,
        targets.float(),
        w_dir=0.2,
        dir_tgt=targets,
        valid_masks=masks,
        instance_id=ids,
        reduction="instance",
        pictogram_tgt=targets,
        w_pictogram=0.1,
    )
    loss["total"].backward()
    assert maps["state_logit"].grad.abs().sum() == 0
    assert maps["pictogram_logit"].grad.abs().sum() > 0
    erased = token_losses(
        maps,
        lamp * 0,
        valid,
        targets,
        targets.float(),
        w_dir=0.2,
        dir_tgt=targets,
        valid_masks=masks,
        instance_id=ids,
        reduction="instance",
        pictogram_tgt=targets,
        w_pictogram=0.1,
    )
    assert erased["state"] == erased["dir"] == erased["pictogram"] == 0
