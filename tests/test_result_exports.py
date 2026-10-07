"""Publication provenance must survive Git normalization and reject corruption."""

import hashlib
import csv
import json

import numpy as np
import pytest

from dinov3_global.evaluation import write_json
from dinov3_global.metrics import report, to_jsonable
from scripts.summarize_results import export_test, lf_source_hashes


def test_source_hashes_allow_git_line_endings_but_reject_changed_code(tmp_path):
    path = tmp_path / "model.py"
    crlf = b"def forward(x):\r\n    return x + 1\r\n"
    digest = hashlib.sha256(crlf).hexdigest()
    path.write_bytes(crlf.replace(b"\r\n", b"\n"))
    normalized = lf_source_hashes({"model.py": digest}, tmp_path)
    assert normalized["model.py"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes().replace(b"+ 1", b"+ 2"))
    with pytest.raises(ValueError, match="source differs"):
        lf_source_hashes({"model.py": digest}, tmp_path)


def test_incomplete_test_run_cannot_be_exported_as_final(tmp_path):
    write_json(tmp_path / "status.json", dict(status="evaluating"))
    with pytest.raises(ValueError, match="has not completed"):
        export_test(tmp_path, tmp_path)


def test_export_rejects_changed_ensemble_logits_even_if_softmax_is_identical(tmp_path):
    y = np.array([0, 1, 2, 0, 1, 2])
    meta = dict(
        labels=y,
        paths=np.array([f"frame{i}.jpg" for i in range(6)]),
        cities=np.array(["City"] * 6),
        max_lamp_h=np.ones(6, dtype=np.float32),
        pseudo_nor=np.zeros(6, dtype=bool),
    )
    checkpoints, members = [], []
    for index in range(4):
        logits = np.eye(3)[y].astype(np.float32) * (index + 1)
        members.append(logits)
        directory = tmp_path / f"member{index}"
        directory.mkdir()
        np.savez_compressed(directory / "predictions.npz", logits=logits, **meta)
        result = report(y, logits, cities=meta["cities"].tolist(), max_lamp_h=meta["max_lamp_h"])
        result["without_pseudo_nor"] = report(y, logits)
        write_json(directory / "report.json", to_jsonable(result))
        checkpoints.append(dict(name=directory.name, seed=index, epoch=1))
    write_json(tmp_path / "status.json", dict(status="complete"))
    write_json(tmp_path / "plan.json", dict(checkpoints=checkpoints))
    ensemble = np.mean(members, axis=0)
    # Adding a constant preserves softmax and all reported metrics, but this
    # array is no longer the exact mean of the declared members' raw logits.
    np.savez_compressed(tmp_path / "predictions.npz", logits=ensemble + 1, **meta)
    with pytest.raises(AssertionError):
        export_test(tmp_path, tmp_path)


@pytest.mark.parametrize("n_heads", [1, 2])
def test_export_supports_single_head_and_explicit_ensembles(tmp_path, n_heads):
    directory, out = tmp_path / "evaluation", tmp_path / "exports"
    directory.mkdir()
    y = np.array([0, 1, 2, 0, 1, 2])
    meta = dict(
        labels=y,
        paths=np.array([f"frame{i}.jpg" for i in range(6)]),
        cities=np.array(["City"] * 6),
        max_lamp_h=np.ones(6),
        pseudo_nor=np.zeros(6, dtype=bool),
    )

    def result(logits):
        value = report(y, logits, cities=meta["cities"].tolist(), max_lamp_h=meta["max_lamp_h"])
        value["without_pseudo_nor"] = report(y, logits)
        return to_jsonable(value)

    members, checkpoints = [], []
    for index in range(n_heads):
        logits = np.eye(3)[y] * (index + 1)
        members.append(logits)
        member = directory / f"member{index}"
        member.mkdir()
        np.savez_compressed(member / "predictions.npz", logits=logits, **meta)
        write_json(member / "report.json", result(logits))
        checkpoints.append(dict(name=member.name, seed=7 + index, epoch=5, head_parameters=265374))
    write_json(directory / "status.json", dict(status="complete"))
    write_json(directory / "plan.json", dict(checkpoints=checkpoints, source_sha256={}))
    for name in ("data_audit.json", "membership.json"):
        write_json(directory / name, {})
    logits = np.mean(members, axis=0)
    np.savez_compressed(directory / "predictions.npz", logits=logits, **meta)
    final = result(logits)
    final["protocol_sha256"] = hashlib.sha256((directory / "plan.json").read_bytes()).hexdigest()
    write_json(directory / "report.json", final)
    exported = export_test(directory, out)
    primary = "single_head" if n_heads == 1 else "ensemble_2"
    assert exported["provenance"]["primary_predictor"] == primary
    assert (out / "protocol.json").read_bytes() == (directory / "plan.json").read_bytes()
    rows = list(csv.DictReader((out / "metrics.csv").open(encoding="utf-8")))
    assert rows[-1]["n_heads"] == str(n_heads)
    assert rows[-1]["head_parameters"] == str(n_heads * 265374)
    assert json.loads((out / "results.json").read_text())["reports"][primary]["metrics"]["mAP"] == 1
