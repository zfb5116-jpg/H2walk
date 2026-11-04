"""CPU geometry/reference checks, without simulator or saved policy execution."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import math

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"
spec = spec_from_file_location("terrain_gait_cpu", TASK / "terrain_gait.py")
gait = module_from_spec(spec)
spec.loader.exec_module(gait)


def poses(n=1, dtype=torch.float64):
    feet = torch.zeros(n, 2, 3, dtype=dtype)
    feet[:, 0, 1], feet[:, 1, 1] = .12, -.12
    feet[..., 2] = -gait.TERRAIN_REWARD_PARAMETERS["sole_z"]
    quats = torch.zeros(n, 2, 4, dtype=dtype)
    quats[..., 0] = 1.
    return feet, quats


def scan(*, step_at=None, height=0., n=1, dtype=torch.float64):
    x = torch.arange(-.25, .701, .05, dtype=dtype)
    y = torch.arange(-.35, .351, .05, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.zeros_like(xx)
    if step_at is not None:
        zz[xx >= step_at] = height
    return torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)[None].repeat(n, 1, 1)


def targets(rays, feet=None, quats=None, heading=None):
    if feet is None:
        feet, quats = poses(rays.shape[0], rays.dtype)
    if heading is None:
        heading = rays.new_tensor([[1., 0.]]).repeat(rays.shape[0], 1)
    return gait.select_swing_targets(rays, feet, quats, heading)


def test_sole_uses_inside_rectangle_and_mirrors_right_foot():
    feet, quats = poses()
    points = gait.sole_samples_world(feet, quats)
    assert points.shape == (1, 2, 18, 3)
    assert points[0, 0, :, 0].min() == pytest.approx(-.05)
    assert points[0, 0, :, 0].max() == pytest.approx(.135)
    torch.testing.assert_close(points[0, 0, :, 1], -points[0, 1, :, 1])
    torch.testing.assert_close(points[..., 2], torch.zeros_like(points[..., 2]))


def test_full_foot_roll_and_nonunit_quaternion_are_respected():
    feet, quats = poses()
    quats[0, 0] = torch.tensor([math.cos(.3), math.sin(.3), 0., 0.], dtype=feet.dtype) * 3.
    points = gait.sole_samples_world(feet, quats)
    local_y = feet.new_tensor(gait.TERRAIN_REWARD_PARAMETERS["sole_left_y"]).repeat(6)
    expected = feet[0, 0, 2] + local_y * math.sin(.6) + gait.TERRAIN_REWARD_PARAMETERS["sole_z"] * math.cos(.6)
    torch.testing.assert_close(points[0, 0, :, 2], expected)


@pytest.mark.parametrize("fault", ["nan_quat", "zero_quat", "inf_position"])
def test_invalid_pose_does_not_poison_other_foot(fault):
    feet, quats = poses()
    if fault == "nan_quat":
        quats[0, 0, 0] = torch.nan
    elif fault == "zero_quat":
        quats[0, 0] = 0.
    else:
        feet[0, 0, 1] = torch.inf
    points = gait.sole_samples_world(feet, quats)
    assert points[0, 0].isnan().all()
    assert points[0, 1].isfinite().all()


def test_nearest_queries_are_horizontal_and_unknown_has_no_zero_floor():
    rays = torch.tensor([[[0., 0., 2.], [.1, 0., 3.], [0., .01, torch.nan]]], dtype=torch.float64)
    query = torch.tensor([[[[.01, 0., 999.], [.099, 0., -10.], [.5, 0., 0.], [torch.nan, 0., 0.]]]], dtype=rays.dtype)
    result = gait.nearest_heights(rays, query)
    torch.testing.assert_close(result[..., :2], torch.tensor([[[2., 3.]]], dtype=rays.dtype))
    assert result[..., 2:].isnan().all()
    assert gait.nearest_heights(rays[:, :0], query).isnan().all()


def test_chunked_queries_match_exhaustive_reference_for_many_points():
    generator = torch.Generator().manual_seed(14)
    rays = torch.rand(3, 57, 3, generator=generator, dtype=torch.float64)
    points = torch.rand(3, 2, 7, 18, 3, generator=generator, dtype=torch.float64)
    flat = points.reshape(3, -1, 3)
    distances = (flat[:, :, None, :2] - rays[:, None, :, :2]).square().sum(dim=-1)
    minimum, index = distances.min(dim=-1)
    expected = torch.where(minimum <= .04**2, rays[..., 2].gather(1, index), torch.nan)
    torch.testing.assert_close(gait.nearest_heights(rays, points).reshape_as(expected), expected, equal_nan=True)


def test_support_fraction_distinguishes_full_partial_missing_and_penetration():
    feet, quats = poses()
    coverage, valid = gait.support_geometry(scan(), feet, quats)
    torch.testing.assert_close(coverage, torch.ones_like(coverage))
    assert valid.all()
    coverage, valid = gait.support_geometry(scan(step_at=.05, height=-.2), feet, quats)
    assert ((coverage > 0) & (coverage < 1)).all()
    assert valid.all()
    empty = scan()[:, :0]
    coverage, valid = gait.support_geometry(empty, feet, quats)
    assert not coverage.any() and not valid.any()
    feet[..., 2] -= .03
    coverage, valid = gait.support_geometry(scan(), feet, quats)
    assert not coverage.any() and valid.all()


def test_flat_target_is_nearest_patch_and_landing_z_is_sole_not_ankle():
    landing, apex, valid = targets(scan())
    assert valid.all()
    torch.testing.assert_close(landing[..., 0], torch.full_like(landing[..., 0], .16))
    torch.testing.assert_close(landing[..., 2], torch.zeros_like(landing[..., 2]))
    torch.testing.assert_close(apex, torch.full_like(apex, .035))


@pytest.mark.parametrize("height", [.12, -.12, .24, -.24])
def test_landing_targets_both_up_and_down_and_do_not_straddle_riser(height):
    landing, apex, valid = targets(scan(step_at=.10, height=height))
    assert valid.all()
    torch.testing.assert_close(landing[..., 2], torch.full_like(landing[..., 2], height))
    torch.testing.assert_close(apex, torch.full_like(apex, max(0., height) + .035))


def test_target_does_not_skip_over_intervening_high_obstacle():
    rays = scan()
    rays[:, (rays[0, :, 0] >= .1) & (rays[0, :, 0] <= .2), 2] = .4
    landing, apex, valid = targets(rays)
    assert not valid.any()
    assert landing.isnan().all() and apex.isnan().all()


def test_target_unknown_and_invalid_direction_are_not_filled():
    rays = scan()
    for heading in (rays.new_tensor([[0., 0.]]), rays.new_tensor([[torch.nan, 0.]])):
        landing, apex, valid = targets(rays, heading=heading)
        assert not valid.any() and landing.isnan().all() and apex.isnan().all()
    landing, apex, valid = targets(rays[:, :0])
    assert not valid.any() and landing.isnan().all() and apex.isnan().all()
    landing, apex, valid = targets(rays[:, rays[0, :, 0] < .10])
    assert not valid.any() and landing.isnan().all() and apex.isnan().all()


def test_world_translation_and_yaw_equivariance():
    rays, (feet, quats) = scan(step_at=.10, height=.12), poses()
    landing, apex, valid = targets(rays, feet, quats)
    angle = .7
    rotation = feet.new_tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    shift = feet.new_tensor([1.7, -.6, 2.3])
    rotated_rays = rays.clone()
    rotated_rays[..., :2] = rays[..., :2] @ rotation.T
    rotated_rays += shift
    rotated_feet = feet.clone()
    rotated_feet[..., :2] = feet[..., :2] @ rotation.T
    rotated_feet += shift
    rotated_quats = quats.clone()
    rotated_quats[..., 0], rotated_quats[..., 3] = math.cos(angle / 2), math.sin(angle / 2)
    result, result_apex, result_valid = targets(rotated_rays, rotated_feet, rotated_quats,
                                               feet.new_tensor([[math.cos(angle), math.sin(angle)]]))
    expected = landing.clone()
    expected[..., :2] = expected[..., :2] @ rotation.T
    expected += shift
    assert torch.equal(valid, result_valid)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result_apex, apex + shift[2])


def test_smooth_reference_reaches_apex_and_lower_landing_without_upper_floor():
    u = torch.tensor([-.1, 0., .25, .5, .75, 1., 1.1], dtype=torch.float64)
    result = gait.swing_height_reference(u.new_tensor(.2), u.new_tensor(0.), u.new_tensor(.235), u)
    assert result[0] == result[1] == pytest.approx(.2)
    assert result[3] == pytest.approx(.235)
    assert result[-1] == result[-2] == 0.
    assert 0 < result[4] < .2
    gradient_u = torch.tensor([0., .5, 1.], dtype=torch.float64, requires_grad=True)
    value = gait.swing_height_reference(gradient_u.new_tensor(.2), gradient_u.new_tensor(0.),
                                        gradient_u.new_tensor(.235), gradient_u)
    value.sum().backward()
    torch.testing.assert_close(gradient_u.grad, torch.zeros_like(gradient_u))


def test_parameter_manifest_keeps_shared_transition_constants_explicit():
    shared_spec = spec_from_file_location("walking_rewards_constants", TASK / "walking_rewards.py")
    shared = module_from_spec(shared_spec)
    shared_spec.loader.exec_module(shared)
    for name, value in shared.WALK_REWARD_PARAMETERS.items():
        if name != "semantic_version":
            assert gait.TERRAIN_REWARD_PARAMETERS[name] == value


@pytest.mark.parametrize("radius", [0., -1., float("nan"), True])
def test_nearest_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="radius"):
        gait.nearest_heights(scan(), torch.zeros(1, 1, 3, dtype=torch.float64), radius)


def test_shapes_are_checked_without_silent_broadcasts():
    feet, quats = poses()
    with pytest.raises(ValueError, match="positions"):
        gait.sole_samples_world(feet[:, :1], quats[:, :1])
    with pytest.raises(ValueError, match="direction"):
        gait.select_swing_targets(scan(), feet, quats, torch.ones(1, 1, dtype=feet.dtype))
