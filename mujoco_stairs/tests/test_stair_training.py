import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from mujoco_stairs import H2StairsEnv
from mujoco_stairs.curriculum import StairCurriculum
from mujoco_stairs.rewards import REWARD_WEIGHTS, weighted_rewards


def test_failed_episodes_never_raise_difficulty_and_successes_complete_stages():
    curriculum = StairCurriculum(window=20)
    for _ in range(40):
        curriculum.record(False)
        assert not curriculum.advance()
    for target in curriculum.HEIGHTS[1:]:
        for _ in range(20):
            curriculum.record(True)
        assert curriculum.advance() and curriculum.height == target
    assert not curriculum.advance()


def test_curriculum_restores_history():
    original = StairCurriculum("low", window=5)
    for outcome in (True, True, False, True):
        original.record(outcome)
    restored = StairCurriculum(window=5)
    restored.restore(original.state())
    assert not restored.advance()
    restored.record(True)
    assert restored.advance() and restored.height == .06


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("mode", ["flat", "low", "20cm", "curriculum"])
def test_physical_collision_heights_match_truth_and_rollouts_are_finite(mode, direction):
    env = H2StairsEnv(terrain_mode=mode, stair_direction=direction)
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (86,) and env.action_space.shape == (15,)
        for x in (-.5, .5, 1., 1.4, 2., 2.9, 3.2, 4.4):
            geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(env.model, env.data, np.array([x, .8, 5.]),
                                     np.array([0., 0., -1.]), None, 1, -1, geom)
            assert 5. - distance == pytest.approx(env.terrain_height(x), abs=1e-6)
        for _ in range(40):
            obs, reward, done, truncated, info = env.step(np.zeros(15))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert reward == pytest.approx(sum(info["reward_terms"].values()))
            assert set(info["reward_terms"]) == set(REWARD_WEIGHTS)
            if done or truncated:
                break
    finally:
        env.close()


def test_waist_is_dynamic_and_moves_entire_upper_mass():
    env = H2StairsEnv(terrain_mode="flat", stair_direction="up")
    try:
        env.reset(seed=0)
        waist = env._qpos_adrs[-3:]
        # Tilt the waist with a fixed base to isolate its COM effect.
        baseline = env.data.subtree_com[env._upper_body].copy()
        env.data.qpos[waist[-1]] = .35
        mujoco.mj_forward(env.model, env.data)
        assert abs(env.data.subtree_com[env._upper_body, 0] - baseline[0]) > .05
        env.reset(seed=0)
        action = np.zeros(15)
        action[-1] = .5
        for _ in range(10):
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert abs(env.data.qpos[waist[-1]]) > .01
        assert abs(env.data.qvel[env._qvel_adrs[-1]]) > .01
        assert info["waist_controlled"]
        assert not set(waist) & set(env._arm_qpos_adrs)
    finally:
        env.close()


def test_new_policy_observes_base_velocity_and_waist_state():
    env = H2StairsEnv(stair_direction="up")
    try:
        base, _ = env.reset(seed=1)
        env.data.qvel[:6] = [.2, .1, .0, .1, .2, .3]
        env.data.qpos[env._qpos_adrs[-1]] += .2
        mujoco.mj_forward(env.model, env.data)
        changed = env._obs()
        assert not np.array_equal(changed[:6], base[:6])
        assert changed[12 + 14] == pytest.approx(.2)
        with pytest.raises(ValueError, match="15 finite"):
            env.step(np.zeros(12))
    finally:
        env.close()


def test_goal_cannot_skip_unconsumed_treads_and_fall_has_no_extra_bonus():
    env = H2StairsEnv(terrain_mode="20cm", stair_direction="up")
    try:
        env.reset(seed=0)
        env.data.qpos[0] = 3.2
        env.data.qpos[2] += 1.2
        mujoco.mj_forward(env.model, env.data)
        for _ in range(40):
            _, _, done, truncated, info = env.step(np.zeros(15))
            assert not info["success"]
            assert info["stair_completed_treads"] == 0
            assert "success" not in info["reward_terms"] and "fall" not in info["reward_terms"]
            if done or truncated:
                break
        env.reset(seed=0)
        env.data.qpos[2] = .1
        mujoco.mj_forward(env.model, env.data)
        _, _, done, _, info = env.step(np.zeros(15))
        assert done and info["fallen"] and not info["success"]
    finally:
        env.close()


def test_reward_integrates_time_and_stair_terms_are_inactive_on_flat():
    raw = {name: 1. for name in REWARD_WEIGHTS}
    a, b = weighted_rewards(raw, .01), weighted_rewards(raw, .02)
    assert sum(b.values()) == pytest.approx(2 * sum(a.values()))
    env = H2StairsEnv(terrain_mode="flat")
    try:
        env.reset(seed=3)
        _, _, _, _, info = env.step(np.zeros(15))
        for name in ("stair_boundary", "stair_riser", "stair_tread"):
            assert info["reward_terms"][name] == 0.
        assert not {"swing_clearance", "low_clearance", "forward", "landing", "upright"} & set(info["reward_terms"])
        env.reset(seed=3, options={"command": [0., 0., 0.]})
        _, _, _, _, info = env.step(np.zeros(15))
        assert info["reward_raw"]["stand_pose_anchor"] > 0
        assert info["reward_raw"]["walking_action_rate"] == 0
    finally:
        env.close()


def test_mixed_reset_contains_both_directions_and_clears_temporal_state():
    env = H2StairsEnv(terrain_mode="low", stair_direction="mixed")
    try:
        directions = set()
        for seed in range(12):
            obs, info = env.reset(seed=seed)
            directions.add(info["stair_direction"])
            assert not env._previous_action.any() and not env._air_time.any()
            assert env._tread_reward.current_index == 0
        assert directions == {"up", "down"}
    finally:
        env.close()
