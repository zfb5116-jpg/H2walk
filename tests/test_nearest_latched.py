"""Production nearest reward with synthetic stance/liftoff/landing data."""
from importlib.util import module_from_spec, spec_from_file_location
import pytest
import torch

from test_stair_clearance import TASK, _score, _sole_env


def setup():
    spec = spec_from_file_location("walking_latched_cpu", TASK / "walking_rewards.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    env = _sole_env(foot_z=.0543)
    env.walking_reward_state = module.WalkingRewardState(1, "cpu")
    env.support = torch.ones(1, 2, dtype=torch.bool)
    env.scene["contact_forces"].data.net_forces_w_history[..., 2] = 10.
    return env


def advance(env, *, dx=0., left_dx=0., left_z=None, left_support=True):
    state = env.walking_reward_state
    data = env.asset.data
    valid = torch.ones(1, 2, dtype=torch.bool)
    contacts = env.scene["contact_forces"].data.net_forces_w_history.norm(dim=-1).amax(dim=1) > 1.
    state.begin_transition(data.root_pos_w[:, :2], data.root_quat_w, env.command,
                           data.body_pos_w, contacts, env.support, valid, .02)
    data.root_pos_w[:, 0] += dx
    data.body_pos_w[:, 0, 0] += left_dx
    if left_z is not None:
        data.body_pos_w[:, 0, 2] = left_z
    env.support[:, 0] = left_support
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 0, 2] = 10. if left_support else 0.
    contacts = env.scene["contact_forces"].data.net_forces_w_history.norm(dim=-1).amax(dim=1) > 1.
    state.finish_transition(data.root_pos_w[:, :2], data.body_pos_w, contacts, env.support, valid)
    return _score(env, "nearest_latched", height_reference="sole")


def test_production_reward_holds_first_step_through_six_centimeter_scan_switch():
    env = setup()
    advance(env)  # stance query: retain first step at world z=.08
    first = advance(env, dx=.002, left_dx=.03, left_z=.1693, left_support=False)
    held = advance(env, dx=.002, left_dx=.06, left_support=False)
    assert first.item() == pytest.approx(.5, abs=1e-5)
    torch.testing.assert_close(held, first)
    assert _score(env, "nearest", height_reference="sole").item() < held.item()


def test_production_held_target_does_not_reward_invalid_current_sole_pose():
    env = setup()
    advance(env)
    advance(env, dx=.002, left_dx=.03, left_z=.1693, left_support=False)
    env.asset.data.body_quat_w[:, 0] = float("nan")
    result = advance(env, dx=.002, left_dx=.01, left_support=False)
    torch.testing.assert_close(result, torch.zeros(1))


def test_production_no_initial_air_target_and_no_static_hold_reward():
    env = setup()
    env.support[:, 0] = False
    env.scene["contact_forces"].data.net_forces_w_history[:, :, 0] = 0.
    assert not advance(env, left_dx=.03, left_z=.1693, left_support=False).any()
    env = setup()
    advance(env)
    advance(env, left_dx=.03, left_z=.1693, left_support=False)
    assert not advance(env, left_support=False).any()
