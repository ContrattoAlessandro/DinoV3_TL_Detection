"""Final evaluation rejects silent omissions and verifies encoder reuse."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from dinov3_global.data import DTLDGlobalDataset
from dinov3_global.evaluation import audit_test_split, predict_heads
from dinov3_global.model import DinoGlobal


def make_dataset(root):
    cfg = {"data": {"label_dir": "labels", "img_root": "images", "target_hw": [32, 48]}}
    (root / "labels").mkdir()
    for split in ("train", "test"):
        (root / "images" / split).mkdir(parents=True)
        entries = []
        for index, state in enumerate(("red", "green", "off")):
            name = f"{split}{index}"
            entries.append(
                dict(
                    image_path=f"./City/{split}/session/{name}.tiff",
                    labels=[
                        dict(x=1000, y=100, w=8, h=24, attributes=dict(relevance="relevant", state=state))
                    ],
                )
            )
            Image.new("RGB", (48, 32), (index * 40 + (5 if split == "train" else 15), 0, 0)).save(
                root / "images" / split / f"{name}.jpg"
            )
        (root / "labels" / f"DTLD_{split}.json").write_text(json.dumps(dict(images=entries)))
    dataset = DTLDGlobalDataset(str(root / "labels"), str(root / "images"), "test", target_hw=(32, 48))
    return cfg, dataset


def test_audit_accounts_for_class_policy_and_records_exact_overlap(tmp_path):
    cfg, dataset = make_dataset(tmp_path)
    source = tmp_path / "images/train/train0.jpg"
    (tmp_path / "images/test/test0.jpg").write_bytes(source.read_bytes())
    audit, membership = audit_test_split(cfg, dataset, tmp_path)
    assert audit["splits"]["test"]["class_counts"] == dict(RR=1, RG=1, NoR=1)
    assert audit["splits"]["test"]["pseudo_nor"] == 1
    assert audit["splits"]["test"]["n_native"] == audit["splits"]["test"]["n_evaluated"] == 3
    assert audit["overlap"]["sessions"] == 0
    assert audit["overlap"]["exact_file_cross_split_pairs"] == 1
    assert len(membership["test"]) == 3  # Audits never silently remove duplicates.
    assert membership["test"][0]["image"] == "images/test/test0.jpg"


def test_missing_test_image_fails_even_when_dataset_silently_skips_it(tmp_path):
    cfg, _ = make_dataset(tmp_path)
    (tmp_path / "images/test/test0.jpg").unlink()
    dataset = DTLDGlobalDataset(
        str(tmp_path / "labels"), str(tmp_path / "images"), "test", target_hw=(32, 48)
    )
    assert len(dataset) == 2
    with pytest.raises(FileNotFoundError, match="Missing official test image"):
        audit_test_split(cfg, dataset, tmp_path)


def test_flattened_image_aliases_are_rejected(tmp_path):
    cfg, dataset = make_dataset(tmp_path)
    path = tmp_path / "labels/DTLD_test.json"
    annotations = json.loads(path.read_text())
    duplicate = deepcopy(annotations["images"][0])
    duplicate["image_path"] = "./Other/drive/session/test0.tiff"
    annotations["images"].append(duplicate)
    path.write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="aliased test annotation"):
        audit_test_split(cfg, dataset, tmp_path)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384)
        self.parameter = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, pixel_values, output_hidden_states):
        self.calls += 1
        features = pixel_values.mean(dim=(1, 2, 3)).reshape(-1, 1, 1).expand(-1, 11, 384)
        return SimpleNamespace(
            last_hidden_state=features + 12, hidden_states=tuple(features + i for i in range(13))
        )


def test_shared_encoder_is_bitwise_equal_to_complete_forwards_and_preserves_order(monkeypatch, tmp_path):
    from transformers import AutoModel

    encoder = TinyEncoder()
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: encoder)
    model = DinoGlobal(dtype="float32", grid_hw=(2, 3), proj_dim=8, dropout=0).eval()
    heads = [model.head, deepcopy(model.head).eval()]
    with torch.no_grad():
        heads[1].bias.add_(0.5)
    batches = []
    for start, count in ((0, 2), (2, 1)):
        batches.append(
            dict(
                image=torch.full((count, 3, 32, 48), 60 + start, dtype=torch.uint8),
                label=torch.arange(start, start + count),
                city=["City"] * count,
                max_lamp_h=torch.full((count,), 12.0),
                path=[str(tmp_path / f"{index}.jpg") for index in range(start, start + count)],
            )
        )
    labels, logits, meta, verification = predict_heads(model, heads, batches, torch.device("cpu"), tmp_path)
    assert encoder.calls == len(batches) + len(heads)
    assert model.head is heads[0]
    assert labels.tolist() == [0, 1, 2]
    assert meta["paths"] == ["0.jpg", "1.jpg", "2.jpg"]
    assert all(member["max_abs_logit_difference"] == 0 for member in verification["members"])
    assert not np.array_equal(logits[0], logits[1])
    for index, head in enumerate(heads):
        model.head = head
        with torch.no_grad():
            expected = np.concatenate(
                [model(batch["image"].float() / 255)["logits"].numpy() for batch in batches]
            )
        np.testing.assert_array_equal(logits[index], expected)
