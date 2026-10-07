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
    checkpoint = json.loads((ROOT / "metadata/checkpoints.json").read_text())["checkpoints"][0]
    for directory, historical in (("dtld", "DTLD"), ("atlas", "ATLAS"), ("vzc_tld", "VZC_TLD")):
        public = ROOT / f"docs/results/{directory}/report.json"
        original = ROOT / f"docs/experiments/diagnostics/{historical}/C/report.json"
        assert public.read_bytes() == original.read_bytes()
        report = json.loads(public.read_text())
        assert report["epoch"] == checkpoint["epoch"] == 5
        assert report["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]


def test_default_is_C_with_independent_validation_selection_and_complete_source_hashes():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert cfg["decoder"]["head"] == "v7_evidence"
    assert cfg["optim"]["early_stop_metric"] == "mAP"
    assert "selection_reference" not in cfg["optim"]
    assert sum(p.numel() for p in EvidenceMILHead().parameters()) == 265374
    assert source_hashes()
