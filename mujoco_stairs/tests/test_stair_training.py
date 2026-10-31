from dataclasses import replace

import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from mujoco_stairs import H2StairsEnv
from mujoco_stairs.clearance_reward import FootState, swing_score
from mujoco_stairs.curriculum import StairCurriculum


def score(foot, **kwargs):
    return swing_score(foot, other_foot_support=kwargs.pop("other_foot_support", True),
                       step_height=kwargs.pop("step_height", 0.20), command_speed=0.6, **kwargs)


def test_clearance_rewards_crossing_the_next_riser_and_not_kicking_higher():
    foot = FootState(0.60, 0.86, 0.25, 0.6, False, 0.0)
    useful = score(foot)
    assert useful.target_z == pytest.approx(0.25)
    assert useful.clearance > score(replace(foot, sole_z=0.10)).clearance
    assert useful.clearance > score(replace(foot, sole_z=0.50)).clearance
    assert useful.low_clearance == 0.0
    assert score(replace(foot, sole_z=0.10)).low_clearance > 0.0


def test_target_rises_with_each_supporting_tread():
    for level in range(6):
        edge = 0.9 + level * 0.35
        foot = FootState(edge - 0.30, edge - 0.04, level * 0.20 + 0.25,
                         0.6, False, level * 0.20)
        assert score(foot).target_z == pytest.approx((level + 1) * 0.20 + 0.05)
        assert score(foot).clearance == pytest.approx(1.0)


@pytest.mark.parametrize("change", [{"forward_speed": 0.0}, {"forward_speed": -0.5}, {"support": True}])
def test_static_backward_or_planted_foot_cannot_farm_swing_reward(change):
    foot = replace(FootState(0.60, 0.86, 0.25, 0.6, False, 0.0), **change)
    assert score(foot).clearance == 0.0


def test_jump_and_flat_ground_do_not_trigger_stair_clearance():
    foot = FootState(0.60, 0.86, 0.25, 0.6, False, 0.0)
    assert score(foot, other_foot_support=False).clearance == 0.0
    assert score(foot, step_height=0.0).target_z == pytest.approx(0.06)


def test_failed_episodes_never_raise_difficulty_and_successes_complete_all_stages():
    curriculum = StairCurriculum(window=20)
    for _ in range(200):
        curriculum.record(False)
        assert not curriculum.advance()
    assert curriculum.height == 0.0
    for target in curriculum.HEIGHTS[1:]:
        for _ in range(20):
            curriculum.record(True)
        assert curriculum.advance()
        assert curriculum.height == target
    assert not curriculum.advance()


def test_curriculum_restores_rolling_success_window():
    original = StairCurriculum("low", window=5)
    for outcome in (True, True, False, True):
        original.record(outcome)
    restored = StairCurriculum(window=5)
    restored.restore(original.state())
    assert not restored.advance()
    restored.record(True)
    assert restored.advance()
    assert restored.height == 0.06


@pytest.mark.parametrize("mode", ["flat", "low", "20cm", "curriculum"])
def test_real_environment_and_collision_heights_match_observations(mode):
    env = H2StairsEnv(terrain_mode=mode)
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (56,)
        # Cast at y=0.8 to avoid hitting robot geometry.
        for x in (0.5, 1.0, 1.4, 2.0, 2.9, 3.2, 4.4):
            geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(env.model, env.data, np.array([x, 0.8, 5.0]),
                                     np.array([0.0, 0.0, -1.0]), None, 1, -1, geom)
            assert 5.0 - distance == pytest.approx(env.terrain_height(x), abs=1e-6)
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert reward == pytest.approx(sum(info["reward_terms"].values()))
            if terminated or truncated:
                break
    finally:
        env.close()


def test_resets_alone_do_not_advance_curriculum():
    env = H2StairsEnv(terrain_mode="curriculum")
    try:
        for _ in range(60):
            env.reset()
        assert env._terrain_step_height() == 0.0
    finally:
        env.close()


def test_action_mapping_preserves_flat_gait_and_can_reach_high_step_clearance():
    env = H2StairsEnv(terrain_mode="flat")
    try:
        env.reset()
        action = np.zeros(12)
        action[0], action[3], action[5] = -1.0, 1.0, -2.0 / 3.0
        assert np.allclose(env._action_target(action), env._lower_home + 0.45 * action)
        env.terrain_mode = "20cm"
        env.data.qpos[env._qpos_adrs] = env._action_target(action)
        mujoco.mj_forward(env.model, env.data)
        feet, _ = env._foot_states(1.0)
        assert feet[0].sole_z >= 0.25
        assert env.data.qpos[env._qpos_adrs[3]] == pytest.approx(1.6)
    finally:
        env.close()


def test_landing_requires_a_foot_on_the_tread_not_a_toe_on_the_edge():
    env = H2StairsEnv(terrain_mode="20cm")
    try:
        env.reset()
        planted = FootState(1.64, 1.90, 0.60, 0.0, True, 0.40)
        assert env._landed_level(planted) == 3
        assert env._landed_level(replace(planted, rear_x=1.59)) == 0
        assert env._landed_level(replace(planted, support=False)) == 0
        assert env._landed_level(replace(planted, sole_z=0.72)) == 0
    finally:
        env.close()


def test_top_landing_can_finish_but_cannot_skip_sequential_reward():
    env = H2StairsEnv(terrain_mode="20cm")
    try:
        env.reset(seed=1)
        env.data.qpos[0] = 3.2
        env.data.qpos[2] += 1.2
        mujoco.mj_forward(env.model, env.data)
        env._foot_states(1.0)
        tread_rewards = []
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(np.zeros(12))
            tread_rewards.append(info["reward_terms"]["stair_tread"])
            if terminated or truncated:
                break
        assert info["success"]
        assert info["max_landed_level"] == 6
        # Teleporting directly to the top cannot collect the first five treads.
        assert sum(tread_rewards) == 0.0
        assert info["stair_completed_treads"] == 0
        assert "landing" not in info["reward_terms"]
    finally:
        env.close()


@pytest.mark.parametrize("position", [(3.2, 0.0, 4.0), (3.2, 2.5, 2.23)])
def test_crossing_goal_in_air_or_bypassing_stairs_is_not_success(position):
    env = H2StairsEnv(terrain_mode="20cm")
    try:
        env.reset()
        env.data.qpos[:3] = position
        mujoco.mj_forward(env.model, env.data)
        _, _, _, _, info = env.step(np.zeros(12))
        assert not info["success"]
    finally:
        env.close()
