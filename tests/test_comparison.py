"""A capacity comparison must use the same images and labels as its reference."""

import numpy as np
import pytest

from dinov3_global.comparison import aligned_predictions, published_vzc_predictions, paired_comparison


def test_comparison_aligns_reordered_dtld_paths_without_changing_logits():
    reference = dict(paths=["red", "green", "off"], labels=np.array([0, 1, 2]))
    candidate = dict(
        paths=[
            "datasets/DTLD_jpg/test/off.jpg",
            "datasets/DTLD_jpg/test/red.jpg",
            "datasets/DTLD_jpg/test/green.jpg",
        ],
        labels=np.array([2, 0, 1]),
        logits=np.array([[0, 0, 3], [3, 0, 0], [0, 3, 0]]),
    )
    labels, logits = aligned_predictions("DTLD", reference, candidate)
    np.testing.assert_array_equal(labels, [0, 1, 2])
    np.testing.assert_array_equal(logits, 3 * np.eye(3))


@pytest.mark.parametrize("change", ["missing", "labels", "duplicate", "nonfinite"])
def test_comparison_rejects_unmatched_or_invalid_predictions(change):
    reference = dict(paths=["camera/red.jpg", "camera/green.jpg"], labels=np.array([0, 1]))
    candidate = dict(paths=list(reference["paths"]), labels=np.array([0, 1]), logits=np.zeros((2, 3)))
    if change == "missing":
        candidate["paths"][0] = "other/red.jpg"
    elif change == "labels":
        candidate["labels"][0] = 1
    elif change == "duplicate":
        candidate["paths"][0] = candidate["paths"][1]
    else:
        candidate["logits"][0, 0] = np.nan
    with pytest.raises(ValueError):
        aligned_predictions("ATLAS", reference, candidate)


def test_vzc_train_diagnostic_cannot_contaminate_the_primary_test_comparison():
    candidate = dict(paths=np.array(["test.png", "train.png"]), labels=np.array([0, 1]), logits=np.eye(3)[:2])
    snapshot = dict(
        images=[dict(file="test.png", label=0, split="test"), dict(file="train.png", label=1, split="train")]
    )
    primary = published_vzc_predictions(candidate, snapshot)
    assert primary["paths"].tolist() == ["test.png"]
    np.testing.assert_array_equal(primary["logits"], candidate["logits"][:1])
    candidate["paths"] = candidate["paths"][::-1]
    with pytest.raises(ValueError, match="frozen label snapshot"):
        published_vzc_predictions(candidate, snapshot)


def test_paired_session_bootstrap_preserves_zero_difference_and_seed():
    y = np.tile([0, 1, 2], 3)
    logits = np.eye(3)[y] * 5
    result = paired_comparison(y, logits, logits, sessions=np.repeat([0, 1, 2], 3), repetitions=12, seed=7)
    assert result["session_bootstrap"]["seed"] == 7
    assert all(v == [0.0, 0.0] for v in result["session_bootstrap"]["intervals"].values())
    assert result["paired_prediction_transitions"][0] == [[3, 0, 0], [0, 0, 0], [0, 0, 0]]
