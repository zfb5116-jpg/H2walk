from dataclasses import replace

import mujoco
import numpy as np
import pytest

from mujoco_stairs import H2StairsEnv
from mujoco_stairs.stair_rewards import (SequentialTreadReward, StairGeometry, sole_contact_ratios,
                                         stair_boundary_count, stair_riser_count)


GEOMETRY = StairGeometry()


def points(*rows):
    return [np.asarray(rows, dtype=float)]


def ratios(index=0, left=1.0, right=0.0):
    result = np.zeros((2, 6))
    result[:, index] = left, right
    return result


def test_boundary_is_local_union_count_and_inactive_on_flat():
    feet = points([0.8, 0.8, 0.0], [1.4, -0.8, 0.2], [1.4, 0.1, 0.2],
                  [0.3, 0.8, 0.0], [4.5, 0.8, 0.0])
    assert stair_boundary_count(feet, 0.2, GEOMETRY) == 2
    assert stair_boundary_count(feet, 0.0, GEOMETRY) == 0
    assert stair_boundary_count(feet, 0.2, replace(GEOMETRY, lane_half_width=1.0)) == 0


def test_riser_volume_excludes_top_four_cm_and_other_treads():
    feet = points([0.9, 0.0, 0.10], [0.9, 0.0, 0.18], [1.25, 0.0, 0.30],
                  [1.1, 0.0, 0.10], [0.9, 2.1, 0.1], [0.9, 0.0, -0.01])
    assert stair_riser_count(feet, 0.2, GEOMETRY) == 2
    assert stair_riser_count(feet, 0.0, GEOMETRY) == 0
    assert stair_riser_count(feet, 0.04, GEOMETRY) == 0


def test_contact_ratio_requires_support_height_and_xy_overlap():
    sole = np.array([[1.0, 0.1, 0.2], [1.1, 0.1, 0.2],
                     [0.89, 0.1, 0.2], [1.2, 0.9, 0.2]])
    result = sole_contact_ratios([sole, sole], [True, False], 0.2, GEOMETRY)
    assert result[0, 0] == 0.5
    assert not result[1].any()
    assert not result[:, 1:].any()
    assert not sole_contact_ratios([sole + [0, 0, 0.15], sole], [True, False], 0.2, GEOMETRY).any()
    assert not sole_contact_ratios([sole, sole], [True, True], 0.0, GEOMETRY).any()


def test_stable_contact_advances_in_order_and_does_not_repeat():
    sequence = SequentialTreadReward(GEOMETRY)
    # tau=0.35/(2*0.5*0.01)=35 frames; ceil(0.3*tau)=11.
    assert sequence.step(ratios(1), 0.5, 0.01, active=True) == 0.0
    for i in range(11):
        assert sequence.step(ratios(), 0.5, 0.01, active=True) == pytest.approx(1.0)
        assert sequence.current_index == (1 if i == 10 else 0)
    assert sequence.expected_foot == 1
    for _ in range(100):
        assert sequence.step(ratios(), 0.5, 0.01, active=True) == 0.0
    assert sequence.current_index == 1


def test_alternating_foot_gets_more_than_wrong_foot_and_double_stance():
    good, wrong, both = [SequentialTreadReward(GEOMETRY) for _ in range(3)]
    for sequence in (good, wrong, both):
        for _ in range(11):
            sequence.step(ratios(), 0.5, 0.01, active=True)
    assert good.step(ratios(1, 0.0, 1.0), 0.5, 0.01, active=True) == pytest.approx(1.0)
    assert wrong.step(ratios(1, 1.0, 0.0), 0.5, 0.01, active=True) == pytest.approx(0.3)
    assert both.step(ratios(1, 1.0, 1.0), 0.5, 0.01, active=True) == pytest.approx(0.335)


def test_short_contacts_do_not_complete_tread_and_expired_window_never_reopens():
    sequence = SequentialTreadReward(GEOMETRY)
    for _ in range(40):
        sequence.step(ratios(), 0.5, 0.01, active=True)
        sequence.step(np.zeros((2, 6)), 0.5, 0.01, active=True)
    assert sequence.current_index == 0
    assert sequence.window_opened and sequence.window_remaining == 0
    for _ in range(11):
        assert sequence.step(ratios(), 0.5, 0.01, active=True) == 0.0
    assert sequence.current_index == 1  # Recovery remains possible after expiry.


def test_all_six_treads_complete_once_and_reset_removes_history():
    sequence = SequentialTreadReward(GEOMETRY)
    total = 0.0
    for index in range(6):
        for _ in range(11):
            total += sequence.step(ratios(index, float(index % 2 == 0), float(index % 2 == 1)),
                                   0.5, 0.01, active=True)
    assert sequence.current_index == 6
    assert total == pytest.approx(66.0)
    assert sequence.step(ratios(5), 0.5, 0.01, active=True) == 0.0
    sequence.reset()
    assert sequence.current_index == 0 and sequence.expected_foot is None
    assert sequence.step(ratios(), 0.5, 0.01, active=False) == 0.0
    assert sequence.step(ratios(), 0.0, 0.01, active=True) == 0.0


@pytest.mark.parametrize("speed,required", [(0.1, 15), (0.5, 11), (5.0, 2)])
def test_window_duration_adapts_to_speed_and_control_dt(speed, required):
    sequence = SequentialTreadReward(GEOMETRY)
    sequence.step(ratios(), speed, 0.01, active=True)
    assert sequence.required_hold == required


def test_new_terms_leave_flat_rollouts_identical_with_weights_enabled_or_disabled():
    enabled = H2StairsEnv(terrain_mode="flat")
    disabled = H2StairsEnv(terrain_mode="flat", stair_boundary_weight=0.0,
                           stair_riser_weight=0.0, stair_tread_weight=0.0)
    try:
        enabled.reset(seed=3)
        disabled.reset(seed=3)
        random = np.random.default_rng(8)
        for _ in range(40):
            action = random.uniform(-0.3, 0.3, 12)
            a, b = enabled.step(action), disabled.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            for name in ("stair_boundary", "stair_riser", "stair_tread"):
                assert a[4]["reward_terms"][name] == 0.0
    finally:
        enabled.close()
        disabled.close()


def test_real_first_tread_contact_activates_sequence_and_resets_cleanly():
    env = H2StairsEnv(terrain_mode="20cm")
    try:
        env.reset(seed=0)
        env.data.qpos[0] = 1.04
        env.data.qpos[2] += 0.20
        mujoco.mj_forward(env.model, env.data)
        env._foot_states(1.0)
        active_reward = []
        for _ in range(60):
            _, _, terminated, _, info = env.step(np.zeros(12))
            active_reward.append(info["reward_terms"]["stair_tread"])
            if terminated:
                break
        assert max(active_reward) > 0.0
        assert info["stair_completed_treads"] >= 1
        assert env._world_foot_points(env._foot_volume_local)[0].shape == (45, 3)
        assert env._world_foot_points(env._foot_sole_local)[0].shape == (35, 3)
        env.reset()
        assert env._tread_reward.current_index == 0
        assert not env._tread_reward.window_opened
    finally:
        env.close()


def test_real_collision_samples_produce_negative_riser_and_boundary_terms():
    env = H2StairsEnv(terrain_mode="20cm")
    try:
        env.reset()
        env.data.qpos[0] = 0.77
        mujoco.mj_forward(env.model, env.data)
        feet, _ = env._foot_states(1.0)
        terms, diagnostics = env._stair_reward_terms(feet, 0.01)
        assert diagnostics["stair_riser_points"] > 0
        assert terms["stair_riser"] == -0.5 * diagnostics["stair_riser_points"]
        env.data.qpos[1] = 0.80
        mujoco.mj_forward(env.model, env.data)
        feet, _ = env._foot_states(1.0)
        terms, diagnostics = env._stair_reward_terms(feet, 0.01)
        assert diagnostics["stair_boundary_points"] > 0
        assert terms["stair_boundary"] == -0.05 * diagnostics["stair_boundary_points"]
    finally:
        env.close()


@pytest.mark.parametrize("kwargs", [{"stair_boundary_weight": 0.1}, {"stair_riser_weight": 0.5},
                                     {"stair_tread_weight": -1.0}, {"stair_riser_weight": float("nan")},
                                     {"stair_lane_half_width": 2.0}])
def test_invalid_stair_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        H2StairsEnv(**kwargs)
