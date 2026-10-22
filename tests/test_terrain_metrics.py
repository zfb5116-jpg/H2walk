"""Regression tests for edge failures that global height MAE conceals."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "terrain_metrics", Path(__file__).resolve().parents[1] / "tools/terrain_metrics.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
terrain_quality = _MODULE.terrain_quality


def test_low_global_error_does_not_hide_missing_two_cm_tread():
    gt = np.zeros((17, 33))
    gt[:, -1] = 0.02
    result = terrain_quality(np.zeros_like(gt), gt)
    assert result["mae_m"] < 0.001
    assert result["edge_recall"] == 0
    assert result["edge_f1"] == 0
    assert result["edge_false_negative"] == 17


def test_inverted_step_has_correct_location_but_wrong_sign():
    gt = np.zeros((4, 6))
    gt[:, 3:] = 0.02
    result = terrain_quality(-gt, gt)
    assert result["edge_f1"] == 1
    assert result["edge_sign_accuracy"] == 0


def test_perfect_flat_map_has_no_claimed_edge_detection_score():
    result = terrain_quality(np.zeros((3, 4)), np.zeros((3, 4)))
    assert result["mae_m"] == 0
    assert result["edge_f1"] is None
    assert result["edge_recall"] is None


def test_empty_observed_subset_is_explicit_and_bins_cover_all_cells():
    result = terrain_quality(
        np.ones((2, 3)), np.zeros((2, 3)),
        observed=np.zeros((2, 3)), confidence=np.ones((2, 3)),
    )
    assert result["observed_mae_m"] is None
    assert result["unobserved_mae_m"] == 1
    assert sum(entry["count"] for entry in result["reliability_error_bins"]) == 6
    assert result["reliability_error_bins"][-1]["mae_m"] == 1


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_reject_nonfinite_heights(invalid):
    pred = np.zeros((2, 3))
    pred[0, 0] = invalid
    with pytest.raises(ValueError, match="non-finite"):
        terrain_quality(pred, np.zeros((2, 3)))


def test_reject_bad_mask_and_shape():
    with pytest.raises(ValueError, match="shape"):
        terrain_quality(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="binary"):
        terrain_quality(np.zeros((2, 3)), np.zeros((2, 3)), observed=np.full((2, 3), 0.5))
