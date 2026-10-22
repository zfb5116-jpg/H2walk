"""CPU tensor adapter tests, including real U-Net forwards with plain data containers.

No Isaac Lab imports, checkpoints, fake reconstruction networks or training.
The observation function is extracted with AST solely to omit simulator imports;
its body is unchanged and invokes the actual source reconstructor.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
PACKAGE = "h2_observation_real_networks"
package = ModuleType(PACKAGE)
package.__path__ = [str(TASK / "networks")]
sys.modules[PACKAGE] = package
helpers = importlib.import_module(f"{PACKAGE}.lidar_observation")
network = importlib.import_module(f"{PACKAGE}.lidar_terrain_reconstructor")


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_mesh_misses_are_not_filled_from_world_floor():
    hits = torch.tensor([[[float("inf"), 0, 0], [0, 0, -3.0], [0, 0, 0.5]]])
    starts = torch.tensor([[[0.0, 0, 1.0], [0, 0, -2.0], [0, 0, 1.0]]])
    ranges, mask = helpers.mesh_hit_ranges(hits, starts)
    torch.testing.assert_close(ranges, torch.tensor([[0.0, 1.0, 0.5]]))
    assert mask.tolist() == [[0.0, 1.0, 1.0]]


def test_mesh_ranges_and_scan_convention_are_translation_invariant():
    hits = torch.tensor([[[0.2, 0.0, -0.1], [0.3, 0.0, -0.12]]])
    starts = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(helpers.mesh_hit_ranges(hits, starts)[0], helpers.mesh_hit_ranges(hits + 7, starts + 7)[0])
    body_height = torch.tensor([[-1.0, -0.98, -0.96]])
    output = helpers.body_height_to_observation(body_height, torch.tensor([1.0]), reference_z=torch.tensor([1.1]))
    shifted = helpers.body_height_to_observation(body_height, torch.tensor([8.0]), reference_z=torch.tensor([8.1]))
    torch.testing.assert_close(output, torch.tensor([[0.6, 0.58, 0.56]]))
    torch.testing.assert_close(output, shifted, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(helpers.body_height_to_observation(body_height, torch.tensor([1.0]), convention="body_height"), body_height)


def test_scan_scatter_checks_directions_and_preserves_origin():
    expected = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    keep = torch.tensor([True, False, True])
    origin = torch.tensor([[0.0, 0.0, 0.16], [0.0, 0.0, 0.16]])
    ranges = torch.tensor([[0.5, 0.8]])
    out, mask, origin_out = helpers.validate_and_scatter_scan(ranges, torch.ones_like(ranges), origin, expected[keep], expected, keep)
    torch.testing.assert_close(out, torch.tensor([[0.5, 0.0, 0.8]]))
    torch.testing.assert_close(mask, torch.tensor([[1.0, 0.0, 1.0]]))
    torch.testing.assert_close(origin_out, origin[:1])
    with pytest.raises(ValueError, match="order/FOV"):
        helpers.validate_and_scatter_scan(ranges, torch.ones_like(ranges), origin, expected[keep].flip(0), expected, keep)
    with pytest.raises(ValueError, match="shared ray origin"):
        helpers.validate_and_scatter_scan(ranges, torch.ones_like(ranges), origin + torch.tensor([[0, 0, 0], [0, 0, 0.1]]), expected[keep], expected, keep)


def test_velocity_prior_requires_explicit_estimate():
    like = torch.empty(2, 3)
    assert torch.count_nonzero(helpers.velocity_prior(like)) == 0
    with pytest.raises(ValueError, match="explicit"):
        helpers.velocity_prior(like, "estimate")
    with pytest.raises(ValueError, match="shape/device"):
        helpers.velocity_prior(like, "estimate", torch.zeros(2, 2))
    with pytest.raises(ValueError, match="finite"):
        helpers.velocity_prior(like, "estimate", torch.full((2, 3), float("nan")))
    estimate = torch.randn(2, 3)
    torch.testing.assert_close(helpers.velocity_prior(like, "estimate", estimate), estimate)


@pytest.mark.parametrize("limits", [{"min_range": -1}, {"max_range": float("inf")}, {"min_range": 2, "max_range": 1}])
def test_reject_invalid_range_limits(limits):
    with pytest.raises(ValueError, match="range limits"):
        helpers.mesh_hit_ranges(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), **limits)


def test_partial_cache_invalidation_preserves_other_hidden_rows():
    raw = SimpleNamespace(device="cpu", _h2_lidar_recon_hidden=torch.ones(1, 2, 4))
    state = (torch.ones(2, 3),)
    raw._h2_lidar_recon_cache = (5, (), state, torch.ones(2, 3), torch.ones(2, dtype=torch.bool))
    helpers.invalidate_lidar_observation(raw, [0])
    assert raw._h2_lidar_recon_cache[4].tolist() == [False, True]
    assert torch.count_nonzero(raw._h2_lidar_recon_hidden[:, 0]) == 0
    assert torch.equal(raw._h2_lidar_recon_hidden[:, 1], torch.ones(1, 4))
    assert not helpers.cache_state_matches(raw._h2_lidar_recon_cache, 5, (), state)


def test_cache_exact_equality_treats_unchanged_nan_misses_as_same_frame():
    before = torch.tensor([[float("nan"), 1.0], [float("inf"), 0.2]])
    assert helpers.exact_equal_rows(before, before.clone()).tolist() == [True, True]
    changed = before.clone()
    changed[1, 1] = 0.3
    assert helpers.exact_equal_rows(before, changed).tolist() == [True, False]


def _observation_function():
    source = ast.parse((TASK / "mdp/observations.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "reconstructed_lidar_height_scan")
    # Postponed annotations let the production body run without Isaac type imports.
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    namespace = dict(vars(helpers))
    namespace.update({
        "torch": torch, "LidarTerrainReconstructor": network.LidarTerrainReconstructor,
        "quat_apply": network._quat_apply, "quat_apply_inverse": network._quat_apply_inverse,
        "H2_LEG_JOINT_NAMES": [str(i) for i in range(12)],
        "YuHbLidarPatternCfg": type("UnusedPatternType", (), {}),
    })
    exec(compile(ast.fix_missing_locations(tree), "<production_observation_body>", "exec"), namespace)
    return namespace["reconstructed_lidar_height_scan"]


def _tensor_environment():
    torch.manual_seed(9)
    model = network.LidarTerrainReconstructor(
        n_ch=16, n_az=224, n_frames=2, proprio_dim=48, gru_delay_steps=12,
        use_height_splat=True, use_refine=True, use_reliable_fusion=True,
    ).eval()
    n_env = 2
    dirs = model.ray_dirs.reshape(3, -1).T
    root = torch.tensor([[0.0, 0, 1.0], [2.0, 0, 1.0]])
    quat = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    origin = torch.tensor([0.0, 0, 0.16]).expand(dirs.shape[0], -1).clone()
    world_starts = root[:, None, :] + origin[None]
    hits = world_starts + dirs[None] * 1.0
    data = SimpleNamespace(
        root_pos_w=root, root_quat_w=quat, root_ang_vel_b=torch.zeros(n_env, 3),
        joint_pos=torch.zeros(n_env, 29), default_joint_pos=torch.zeros(n_env, 29), joint_vel=torch.zeros(n_env, 29),
    )
    robot = SimpleNamespace(data=data, find_joints=lambda *args, **kwargs: (list(reversed(range(12))), []))
    sensor = SimpleNamespace(
        data=SimpleNamespace(pos_w=root.clone(), quat_w=quat.clone(), ray_hits_w=hits),
        ray_starts=origin, ray_directions=dirs,
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(min_range=0.1, max_range=2.0)),
    )
    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.sensors = {"lidar": sensor, "height_scanner": SimpleNamespace(data=SimpleNamespace(pos_w=root.clone()))}
    scene.terrain = SimpleNamespace(terrain_types=torch.zeros(n_env, dtype=torch.long))
    raw = SimpleNamespace(
        scene=scene, device="cpu", common_step_counter=5, episode_length_buf=torch.ones(n_env, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(n_env, 3)),
        action_manager=SimpleNamespace(action=torch.arange(29).float().expand(n_env, -1)),
        _h2_lidar_reconstructor=model, _h2_lidar_model_settings=("unused-test-checkpoint", True),
    )
    raw.unwrapped = raw
    return raw, model, sensor


def test_real_model_binding_caches_once_and_preserves_other_row_on_same_step_teleport():
    observation = _observation_function()
    raw, model, sensor = _tensor_environment()
    kwargs = dict(
        sensor_cfg=SimpleNamespace(name="lidar"), checkpoint="unused-test-checkpoint",
        use_reliable_fusion=True, reference_sensor_cfg=SimpleNamespace(name="height_scanner"),
    )
    first = observation(raw, **kwargs)
    assert first.shape == (2, 561) and torch.isfinite(first).all()
    hidden = raw._h2_lidar_recon_hidden.clone()
    pointer = model._pose_ptr.clone()
    fill = model._pose_fill.clone()
    second = observation(raw, **kwargs)
    assert second is first
    torch.testing.assert_close(model._pose_ptr, pointer)
    # Teleport only env0 and move its sensor/hits consistently, within one step.
    raw.scene["robot"].data.root_pos_w[0, 0] += 0.2
    sensor.data.pos_w[0, 0] += 0.2
    sensor.data.ray_hits_w[0, :, 0] += 0.2
    result = observation(raw, **kwargs)
    torch.testing.assert_close(result[1], first[1])
    torch.testing.assert_close(raw._h2_lidar_recon_hidden[:, 1], hidden[:, 1])
    torch.testing.assert_close(model._pose_ptr[1], pointer[1])
    torch.testing.assert_close(model._pose_fill[1], fill[1])


def test_real_model_partial_reset_same_step_does_not_advance_other_environment():
    observation = _observation_function()
    raw, model, _ = _tensor_environment()
    kwargs = dict(sensor_cfg=SimpleNamespace(name="lidar"), checkpoint="unused-test-checkpoint", use_reliable_fusion=True)
    first = observation(raw, **kwargs).clone()
    hidden = raw._h2_lidar_recon_hidden.clone()
    pointer = model._pose_ptr.clone()
    token_pointer = model._tok_ptr.clone()
    helpers.invalidate_lidar_observation(raw, [0])
    raw.episode_length_buf[0] = 0
    result = observation(raw, **kwargs)
    torch.testing.assert_close(result[1], first[1])
    torch.testing.assert_close(raw._h2_lidar_recon_hidden[:, 1], hidden[:, 1])
    torch.testing.assert_close(model._pose_ptr[1], pointer[1])
    torch.testing.assert_close(model._tok_ptr[1], token_pointer[1])
    assert model._pose_fill.tolist() == [1, 1]


def test_real_model_binding_supplies_zero_prior_and_ordered_actions_without_gt_velocity():
    observation = _observation_function()
    raw, model, _ = _tensor_environment()
    captured = {}
    forward = model.forward_height_splat
    def observed_forward(*args, **kwargs):
        captured["proprio"] = args[6].clone()
        return forward(*args, **kwargs)
    model.forward_height_splat = observed_forward
    observation(raw, sensor_cfg=SimpleNamespace(name="lidar"), checkpoint="unused-test-checkpoint", use_reliable_fusion=True)
    torch.testing.assert_close(captured["proprio"][:, 33:45], raw.action_manager.action[:, :12])
    assert torch.count_nonzero(captured["proprio"][:, 45:48]) == 0
    # No root_lin_vel_b field was supplied to the tensor robot container.


def test_same_step_nan_sensor_frame_is_idempotent_but_refreshed_hits_recompute_one_row():
    observation = _observation_function()
    raw, model, sensor = _tensor_environment()
    sensor.data.ray_hits_w[:, 0] = float("nan")
    kwargs = dict(sensor_cfg=SimpleNamespace(name="lidar"), checkpoint="unused-test-checkpoint", use_reliable_fusion=True)
    first = observation(raw, **kwargs)
    pointer = model._pose_ptr.clone()
    hidden = raw._h2_lidar_recon_hidden.clone()
    assert observation(raw, **kwargs) is first
    sensor.data.ray_hits_w[0, 1, 0] += 0.01
    refreshed = observation(raw, **kwargs)
    torch.testing.assert_close(refreshed[1], first[1])
    torch.testing.assert_close(model._pose_ptr[1], pointer[1])
    torch.testing.assert_close(raw._h2_lidar_recon_hidden[:, 1], hidden[:, 1])


def test_same_step_changed_output_contract_cold_restarts_without_second_history_vote():
    observation = _observation_function()
    raw, model, _ = _tensor_environment()
    kwargs = dict(sensor_cfg=SimpleNamespace(name="lidar"), checkpoint="unused-test-checkpoint", use_reliable_fusion=True)
    observation(raw, **kwargs)
    observation(raw, output_convention="body_height", **kwargs)
    assert model._pose_fill.tolist() == [1, 1]
    assert model._tok_fill.tolist() == [1, 1]
