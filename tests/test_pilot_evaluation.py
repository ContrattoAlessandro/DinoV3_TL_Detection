"""Fixed diagnostic test membership and shared-encoder grouping invariants."""

import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.evaluate_generalization_pilots import (
    RGBDataset,
    collate_rgb,
    inference_key,
    pilot_jobs,
    freeze,
    dataset_report,
)


def test_grouping_ignores_training_changes_and_head_kind_but_preserves_geometry():
    cfg = dict(
        backbone=dict(hf_id="fixed", dtype="float16"),
        data=dict(preprocessing="letterbox", target_hw=[720, 1280], img_root="native", crop_sides=0),
        decoder=dict(head="mil"),
        loss=dict(token_pictogram=0),
    )
    other = copy.deepcopy(cfg)
    other["decoder"]["head"] = "v7_evidence"
    other["loss"]["token_pictogram"] = 0.1
    assert inference_key(cfg) == inference_key(other)
    other["data"]["target_hw"] = [896, 1600]
    assert inference_key(cfg) != inference_key(other)


def test_external_batch_preserves_individual_letterbox_metadata(tmp_path):
    entries = []
    for i, size in enumerate(((80, 40), (40, 80))):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", size, (40, 60, 80)).save(path)
        entries.append(dict(image=str(path), label=i, file=path.name))
    ds = RGBDataset(entries, dict(data=dict(preprocessing="letterbox", target_hw=[64, 96])))
    batch = collate_rgb([ds[0], ds[1]])
    assert batch["image"].dtype == torch.uint8 and batch["image"].shape == (2, 3, 64, 96)
    assert batch["geometry"].shape == (2, 24, 4)
    assert not torch.equal(batch["content_mask"][0], batch["content_mask"][1])
    assert batch["label"].tolist() == [0, 1]


def test_frozen_inputs_cannot_change_on_resume(tmp_path):
    path = tmp_path / "plan.json"
    freeze(path, dict(epoch=3))
    freeze(path, dict(epoch=3))
    with pytest.raises(ValueError, match="changed"):
        freeze(path, dict(epoch=4))


def test_diagnostic_checkpoint_is_fixed_from_validation_not_last_epoch(tmp_path):
    for name in "ABCD":
        directory = tmp_path / name / "fold0"
        directory.mkdir(parents=True)
        rows = [dict(epoch=1, mAP=0.6), dict(epoch=2, mAP=0.7), dict(epoch=3, mAP=0.65)]
        (directory / "history.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (directory / "fold_report.json").write_text(json.dumps(dict(best_epoch=2)))
        state = dict(epoch=2, ema={}, cfg=dict(optim=dict(seed=0), decoder=dict(head="mil")))
        torch.save(state, directory / "best_unconstrained.pt")
    jobs = pilot_jobs(tmp_path)
    assert [job["epoch"] for job in jobs] == [2] * 4
    state["epoch"] = 3
    torch.save(state, directory / "best_unconstrained.pt")
    with pytest.raises(ValueError, match="validation choice"):
        pilot_jobs(tmp_path)


def test_vzc_report_only_uses_supplied_test_rows():
    entries = [dict(label=i, max_lamp_h=20, pseudo_nor=i == 2, split="test") for i in range(3)]
    logits = np.eye(3) * 2
    result = dataset_report("VZC", entries, np.arange(3), logits, {})
    assert result["metrics"]["n"] == 3
    assert result["exclude_unknown_relevant_sensitivity"]["metrics"]["n"] == 2
