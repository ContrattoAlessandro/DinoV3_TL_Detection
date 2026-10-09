"""Integrity and numerical reproducibility of the released model artifacts."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dinov3_global.config import load_config
from dinov3_global.head import EvidenceMILHead
from dinov3_global.metrics import report
from scripts.evaluate_vzc import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_only_released_checkpoint_is_declared_and_unchanged():
    metadata = json.loads((ROOT / "metadata/checkpoints.json").read_text())
    assert len(metadata["checkpoints"]) == 1
    checkpoint = metadata["checkpoints"][0]
    path = ROOT / checkpoint["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["checkpoint_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["epoch"] == checkpoint["epoch"] == 7
    assert state["cfg"]["optim"]["seed"] == 0
    head = EvidenceMILHead(grid_hw=(2, 3))
    head.load_state_dict(state["ema"], strict=True)
    assert sum(p.numel() for p in head.parameters()) == 412830
    assert not any("backbone" in key for key in state["ema"])


def test_published_artifact_hashes_cover_the_retained_measurements():
    provenance = json.loads((ROOT / "docs/results/provenance.json").read_text())
    for key in ("files_sha256", "reproduction_sha256"):
        for name, expected in provenance[key].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    verification = json.loads((ROOT / "metadata/publication_verification.json").read_text())
    assert verification["logits_bit_exact"] and verification["all_evidence_maps_bit_exact"]
    assert verification["head_gradients_bit_exact"] and verification["checkpoint_bytes_unchanged"]
    for name, expected in verification["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected


@pytest.mark.parametrize("dataset", ["dtld", "atlas", "vzc_tld"])
def test_published_scores_recompute_from_the_included_predictions(dataset):
    directory = ROOT / "docs/results" / dataset
    result = json.loads((directory / "report.json").read_text())
    if dataset == "atlas":
        with (directory / "predictions.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        labels = np.asarray([("RR", "RG", "NoR").index(row["label"]) for row in rows])
        logits = np.asarray([[float(row[f"logit_{cls}"]) for cls in ("RR", "RG", "NoR")] for row in rows])
    else:
        with np.load(directory / "predictions.npz", allow_pickle=False) as values:
            labels, logits = values["labels"], values["logits"]
            if dataset == "vzc_tld":
                snapshot = json.loads((directory / "labels_snapshot.json").read_text())
                assert values["paths"].tolist() == [entry["file"] for entry in snapshot["images"]]
                mask = np.asarray([entry["split"] == "test" for entry in snapshot["images"]])
                labels, logits = labels[mask], logits[mask]
                result = result["test"]
    actual = report(labels, logits)
    for key in ("mAP", "AP_RR", "AP_RG", "AP_NoR", "acc_bal", "accuracy", "macro_F1", "nll", "brier", "ece"):
        assert actual["metrics"][key] == pytest.approx(result["metrics"][key], abs=1e-12)
    assert actual["confusion_matrix"] == result["confusion_matrix"]


def test_default_matches_the_single_released_architecture():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert cfg["backbone"]["dim"] == 768
    assert cfg["optim"]["early_stop_metric"] == "mAP"
    assert "selection_reference" not in cfg["optim"]
    assert source_hashes()


def test_checkpoint_loader_falls_back_from_missing_relative_cache_without_mutating_metadata(
    monkeypatch, tmp_path
):
    from dinov3_global import engine

    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["backbone"]["local_ckpt"] = "runs/backbones/dinov3_vitb16"
    checkpoint = tmp_path / "head.pt"
    torch.save(dict(cfg=cfg, ema=EvidenceMILHead(grid_hw=(2, 3)).state_dict()), checkpoint)
    captured = []

    class Model:
        head = EvidenceMILHead(grid_hw=(2, 3))

        def eval(self):
            return self

    def build(config, *args):
        captured.append(config)
        return Model()

    monkeypatch.setattr(engine, "_build_model", build)
    _, loaded_cfg, state = engine.load_model(checkpoint, str(tmp_path), torch.device("cpu"))
    assert captured[0]["backbone"]["local_ckpt"] is None
    assert loaded_cfg["backbone"]["local_ckpt"] is None
    assert state["cfg"]["backbone"]["local_ckpt"] == "runs/backbones/dinov3_vitb16"
