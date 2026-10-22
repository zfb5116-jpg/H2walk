"""CPU synthetic checks only; no simulator, model substitutes, training or optimizer."""

import importlib.util
from pathlib import Path

import pytest
import torch


# Import the actual H2 helpers without initializing Isaac Lab's task package.
# Full reconstructor checks live in test_h2_reconstructor.py.
HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/terrain_fusion.py"
)
SPEC = importlib.util.spec_from_file_location("terrain_fusion_under_test", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
fusion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusion)


def _fuse(values, *, ages=None, spans=None):
    heights = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1)
    if ages is None:
        ages = torch.zeros(heights.shape[1], dtype=torch.long)
    return fusion.fuse_height_frames(
        heights, torch.ones_like(heights), ages,
        height_spans=None if spans is None else torch.tensor(spans).reshape_as(heights),
    )


def test_complete_two_centimetre_step_survives_filter():
    height = torch.zeros(1, 1, 7, 9)
    height[..., 4:] = 0.02
    assert torch.equal(fusion.edge_aware_median(height), height)


@pytest.mark.parametrize("single_cell", [True, False])
def test_one_cell_wide_two_centimetre_feature_survives_filter(single_cell):
    height = torch.zeros(1, 1, 7, 9)
    if single_cell:
        height[..., 3, 4] = 0.02
    else:
        height[..., 4] = 0.02
    assert torch.equal(fusion.edge_aware_median(height), height)


def test_repeated_consistent_frames_have_more_confidence_than_conflicting_frames():
    consistent = _fuse([0.1, 0.1, 0.1, 0.1])
    conflicting = _fuse([0.1, 0.1, 0.14, 0.14])
    assert consistent[2].item() == pytest.approx(0.8)
    assert conflicting[2].item() < consistent[2].item() / 5
    assert consistent[3].item() == conflicting[3].item() == 4


def test_two_conflicting_samples_are_not_hidden_by_zero_mad():
    assert _fuse([0.0, 0.05])[2].item() < 0.1
    assert _fuse([0.0, 0.0])[2].item() > 0.6


def test_one_sample_has_finite_conservative_confidence():
    height, hit, confidence, count = _fuse([0.2])
    assert height.item() == pytest.approx(0.2)
    assert hit.item() == count.item() == 1
    assert confidence.item() == pytest.approx(0.5)


def test_all_misses_and_nonfinite_measurements_have_no_support():
    heights = torch.tensor([float("nan"), float("inf"), 0.0]).reshape(1, 3, 1, 1)
    for hits in (torch.zeros_like(heights), torch.tensor([1.0, 1.0, float("nan")]).reshape_as(heights)):
        result = fusion.fuse_height_frames(heights, hits, torch.arange(3))
        assert all(torch.isfinite(tensor).all() for tensor in result)
        assert all(torch.count_nonzero(tensor).item() == 0 for tensor in result)


def test_nonfinite_span_drops_only_its_frame():
    height, hit, confidence, count = _fuse([0.2, 1.0], spans=[0.0, float("nan")])
    assert height.item() == pytest.approx(0.2)
    assert count.item() == hit.item() == 1
    assert confidence.item() == pytest.approx(0.5)


def test_old_or_multilayer_measurements_reduce_confidence():
    fresh = _fuse([0.1, 0.1])[2].item()
    old = _fuse([0.1, 0.1], ages=torch.tensor([50, 60]))[2].item()
    multilayer = _fuse([0.1, 0.1], spans=[0.03, 0.03])[2].item()
    assert old < fresh
    assert multilayer < fresh / 5


def test_history_indices_exclude_current_frame_and_respect_partial_reset():
    # First env reset and just pushed its current frame; second has full history.
    pointers = torch.tensor([1, 2])
    fill = torch.tensor([1, 12])
    indices, alive = fusion.history_frame_indices(pointers, fill, 12, torch.tensor([10]))
    assert not torch.eq(indices[:, 0], (pointers - 1) % 12).any()
    assert alive[:, 0].tolist() == [False, True]
    current = torch.full((2, 1, 1, 1), 0.2)
    historical = torch.full_like(current, 0.1)
    heights = torch.cat([current, historical], dim=1)
    hits = torch.cat([torch.ones_like(current), alive[..., None, None].float()], dim=1)
    _, _, confidence, count = fusion.fuse_height_frames(heights, hits, torch.tensor([0, 10]))
    # Reset env must retain exactly one vote (0.5), not two copies of current.
    assert count[:, 0, 0, 0].tolist() == [1, 2]
    assert confidence[0].item() == pytest.approx(0.5)


def test_confidence_bound_limits_total_correction():
    base = torch.full((1, 1, 1, 3), 0.2)
    confidence = torch.tensor([0.0, 0.99, 1.0]).reshape_as(base)
    candidate = torch.tensor([1.0, -1.0, 1.0]).reshape_as(base)
    result = fusion.project_confidence_bound(base, candidate, confidence)
    assert torch.all((result - base).abs() <= (1 - confidence) * 0.08 + 1e-7)
    assert result[0, 0, 0, 2].item() == base[0, 0, 0, 2].item()
    assert (result - base).flatten()[1].abs().item() == pytest.approx(0.0008, abs=1e-7)


def test_nonfinite_filter_input_has_finite_output():
    height = torch.ones(1, 1, 3, 3)
    height[..., 1, 1] = float("nan")
    result = fusion.edge_aware_median(height)
    assert torch.isfinite(result).all()
    assert result[..., 1, 1].item() == 0


@pytest.mark.parametrize("kwargs", [{"threshold_m": 0}, {"threshold_m": float("nan")}, {"kernel_size": 2}])
def test_filter_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fusion.edge_aware_median(torch.zeros(1, 1, 3, 3), **kwargs)


@pytest.mark.parametrize("kwargs", [{"dispersion_scale_m": 0}, {"layer_scale_m": -1}, {"age_decay_steps": float("inf")}])
def test_fusion_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fusion.fuse_height_frames(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1), torch.tensor([0]), **kwargs)


@pytest.mark.parametrize("ages", [torch.tensor([0]), torch.tensor([10, 10]), torch.tensor([12])])
def test_ring_rejects_current_duplicate_or_out_of_bounds_ages(ages):
    with pytest.raises(ValueError):
        fusion.history_frame_indices(torch.tensor([1]), torch.tensor([1]), 12, ages)
