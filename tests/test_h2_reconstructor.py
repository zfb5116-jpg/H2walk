"""Real H2 reconstructor CPU checks, including its actual U-Net.

The isolated package only avoids Isaac Lab's task-registration __init__. No
network, U-Net, or geometric function is replaced. Inputs/weights are synthetic;
these tests do not validate a checkpoint, simulator, robot, or policy rollout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks"
PACKAGE = "h2_reconstructor_cpu_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(NETWORK_DIR)]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE}.lidar_terrain_reconstructor", NETWORK_DIR / "lidar_terrain_reconstructor.py"
)
assert spec is not None and spec.loader is not None
network = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = network
spec.loader.exec_module(network)
fusion = sys.modules[f"{PACKAGE}.terrain_fusion"]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def _model(*, reliable=True, delay=12):
    return network.LidarTerrainReconstructor(
        n_ch=16, n_az=224, n_frames=4, proprio_dim=48,
        map_feat_dim=8, z_prop_dim=8, style_ch=8, gru_hidden_dim=16,
        use_refine=True, use_height_splat=True, gru_delay_steps=delay,
        use_reliable_fusion=reliable,
    ).eval()


def _inputs(model, batch=2):
    directions = model.ray_dirs.reshape(3, -1).transpose(0, 1)
    # Plane z=0, sensor-ray origin 16 cm above a base at z=0.9 m.
    dz = directions[:, 2]
    distance = -1.06 / dz.clamp(max=-1e-6)
    valid = (dz < 0) & (distance >= 0.1) & (distance <= model.max_range)
    ranges = torch.where(valid, distance, torch.zeros_like(distance)).expand(batch, -1).clone()
    mask = valid.float().expand(batch, -1).clone()
    position = torch.zeros(batch, 3)
    position[:, 2] = 0.9
    quaternion = torch.zeros(batch, 4)
    quaternion[:, 0] = 1
    origin = torch.zeros(batch, 3)
    origin[:, 2] = 0.16
    proprio = torch.zeros(batch, model.proprio_dim)
    return ranges, mask, position, quaternion, position.clone(), quaternion.clone(), proprio, origin


def _forward(model, inputs, hidden=None):
    ranges, mask, sensor_pos, sensor_quat, root_pos, root_quat, proprio, origin = inputs
    with torch.no_grad():
        return model.forward_height_splat(
            ranges, mask, sensor_pos, sensor_quat, root_pos, root_quat, proprio, hidden, origin
        )


@pytest.mark.parametrize("reliable", [False, True])
def test_actual_refine_unet_forward_is_finite_for_full_sensor_grid(reliable):
    model = _model(reliable=reliable)
    assert type(model.refine).__name__ == "_RefineUNet"
    assert type(model.refine).__module__ == f"{PACKAGE}.terrain_reconstructor"
    rough, refined, hidden = _forward(model, _inputs(model))
    assert rough.shape == refined.shape == (2, 17 * 33)
    assert hidden.shape == (1, 2, 16)
    assert all(torch.isfinite(value).all() for value in (rough, refined, hidden))
    assert model._last_hit.any()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_reliable_forward_handles_misses_and_nonfinite_ranges(nonfinite):
    model = _model()
    inputs = list(_inputs(model))
    inputs[0].zero_()
    inputs[1].zero_()
    if nonfinite:
        inputs[0][:, :2] = torch.tensor([float("nan"), float("inf")])
        inputs[1][:, :2] = 1
    rough, refined, hidden = _forward(model, inputs)
    assert all(torch.isfinite(value).all() for value in (rough, refined, hidden))
    assert model._last_hit.count_nonzero() == 0
    assert model._last_confidence.count_nonzero() == 0
    assert model._last_frame_count.count_nonzero() == 0


def test_nontrivial_root_and_sensor_pose_keeps_real_forward_finite():
    model = _model()
    inputs = list(_inputs(model))
    inputs[2][:, 0] = 1.7
    inputs[2][:, 1] = -0.2
    inputs[4] = inputs[2].clone()
    inputs[3][:, 0] = torch.cos(torch.tensor(0.31))
    inputs[3][:, 3] = torch.sin(torch.tensor(0.31))
    inputs[5] = inputs[3].clone()
    assert all(torch.isfinite(value).all() for value in _forward(model, inputs))


def test_real_pose_ring_counts_current_once_and_resets_only_requested_env():
    model = _model(delay=12)
    inputs = _inputs(model)
    hidden = None
    for _ in range(11):
        _, _, hidden = _forward(model, inputs, hidden)
    # At age ten there is one older frame plus current, never two current votes.
    seen = model._last_hit > 0.5
    assert torch.equal(model._last_frame_count[seen], torch.full_like(model._last_frame_count[seen], 2))
    model.reset_temporal_state(torch.tensor([0]))
    assert model._pose_fill.tolist() == [0, 11]
    assert model._last_confidence[0].count_nonzero() == 0
    hidden[:, 0] = 0
    _forward(model, inputs, hidden)
    seen0, seen1 = model._last_hit[0] > 0.5, model._last_hit[1] > 0.5
    assert torch.all(model._last_frame_count[0, seen0] == 1)
    assert torch.all(model._last_frame_count[1, seen1] == 2)


def test_refine_and_filter_cannot_exceed_the_total_confidence_budget():
    model = _model(delay=0)
    inputs = _inputs(model)
    with torch.no_grad():
        model.height_delta.bias.fill_(1.5)
        model.refine.dec[-1].bias.fill_(1.5)
        model.edge_clean.net[-1].bias.fill_(1.5)
    rough, refined, _ = _forward(model, inputs)
    ranges, mask, pos, quat, root, root_quat, _, origin = inputs
    h, hit, _ = model._splat_upper_height_stats(
        ranges, mask, pos, quat, root, root_quat, origin, collect_stats=True
    )
    near = network._nearest_fill(h, hit)
    base = torch.where(hit > 0.5, h, near).flatten(1)
    budget = (1 - model._last_confidence) * model.reliable_residual_limit_m
    assert torch.all((rough - base).abs() <= budget + 1e-6)
    assert torch.all((refined - base).abs() <= budget + 1e-6)


def test_legacy_and_reliable_parameter_keys_and_shapes_are_identical():
    legacy, reliable = _model(reliable=False), _model(reliable=True)
    old_state, new_state = legacy.state_dict(), reliable.state_dict()
    assert old_state.keys() == new_state.keys()
    assert all(old_state[key].shape == new_state[key].shape for key in old_state)
    reliable.load_state_dict(old_state, strict=True)


def test_confidence_filter_preserves_reliable_two_cm_feature_and_denoises_uncertain_spike():
    height = torch.zeros(1, 1, 7, 9)
    height[..., 3, 4] = 0.02
    confidence = torch.ones_like(height)
    torch.testing.assert_close(fusion.edge_aware_median(height, confidence=confidence), height)
    confidence[..., 3, 4] = 0.1
    cleaned = fusion.edge_aware_median(height, confidence=confidence)
    assert cleaned[..., 3, 4] == 0
    bounded = fusion.project_confidence_bound(height, cleaned, confidence)
    assert torch.all((bounded - height).abs() <= (1 - confidence) * 0.08 + 1e-7)


def test_uncertain_neighbourhood_does_not_erase_unconfirmed_narrow_feature():
    height = torch.zeros(1, 1, 5, 5)
    height[..., 2, 2] = 0.02
    confidence = torch.full_like(height, 0.1)
    torch.testing.assert_close(fusion.edge_aware_median(height, confidence=confidence), height)
