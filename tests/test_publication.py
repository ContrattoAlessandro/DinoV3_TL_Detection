"""Release evidence remains intact when runnable experiments are removed."""

import hashlib
import json
from pathlib import Path

from dinov3_global.config import load_config
from dinov3_global.head import EvidenceMILHead
from scripts.evaluate_vzc import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_historical_artifacts_match_their_preservation_hashes():
    manifest = json.loads((ROOT / "docs/experiments/preservation_manifest.json").read_text())
    for record in manifest["files"]:
        actual = hashlib.sha256((ROOT / record["archive"]).read_bytes()).hexdigest()
        assert actual == record["sha256"], record["archive"]


def test_published_C_reports_are_the_original_fixed_checkpoint_measurements():
    checkpoint = json.loads((ROOT / "docs/experiments/vitsplus_reference/checkpoints.json").read_text())[
        "checkpoints"
    ][0]
    for directory, historical in (("dtld", "DTLD"), ("atlas", "ATLAS"), ("vzc_tld", "VZC_TLD")):
        public = ROOT / f"docs/experiments/vitsplus_reference/results/{directory}/report.json"
        original = ROOT / f"docs/experiments/diagnostics/{historical}/C/report.json"
        assert public.read_bytes() == original.read_bytes()
        report = json.loads(public.read_text())
        assert report["epoch"] == checkpoint["epoch"] == 5
        assert report["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]


def test_default_is_vitb_with_original_head_and_independent_validation_selection():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert cfg["decoder"]["head"] == "v7_evidence"
    assert cfg["optim"]["early_stop_metric"] == "mAP"
    assert "selection_reference" not in cfg["optim"]
    assert cfg["backbone"]["dim"] == 768
    assert sum(p.numel() for p in EvidenceMILHead(in_dim=1536).parameters()) == 412830
    assert source_hashes()


def test_retained_vitb_reports_equal_original_measurements():
    source = json.loads((ROOT / "docs/experiments/backbone_capacity/results/full_report.json").read_text())
    meta = json.loads((ROOT / "metadata/checkpoints.json").read_text())
    checkpoint = next(c for c in meta["checkpoints"] if c["name"] == meta["default_checkpoint"])
    assert checkpoint["epoch"] == source["selected_epoch"] == 7
    assert checkpoint["checkpoint_sha256"] == source["checkpoint_sha256"]
    for dataset, folder in (("DTLD", "dtld"), ("ATLAS", "atlas"), ("VZC_TLD", "vzc_tld")):
        public = json.loads((ROOT / f"docs/results/{folder}/report.json").read_text())
        assert public == source["dataset_reports"][dataset]


def test_archived_experimental_source_matches_preserved_manifest():
    import zipfile

    archive = ROOT / "docs/experiments/recall_head/source_snapshot.zip"
    manifest = json.loads(archive.with_name("source_snapshot_manifest.json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with zipfile.ZipFile(archive) as saved:
        for record in manifest["files"]:
            assert hashlib.sha256(saved.read(record["path"])).hexdigest() == record["sha256"]
