import json

import mujoco
import numpy as np
import pytest

from mujoco_stairs import H2StairsEnv
from mujoco_stairs.checkpoint import validate_checkpoint_contract
from mujoco_stairs.landing_stability import landing_costs, stable_support_mask
from mujoco_stairs.rewards import REWARD_WEIGHTS
from mujoco_stairs.stair_rewards import SequentialTreadReward, StairGeometry


def stable(**changes):
    state = dict(support=[True, False], foot_velocity=np.zeros((2, 3)),
                 foot_up=np.array([[0., 0., 1.]] * 2), base_velocity=np.zeros(3),
                 base_angular_velocity=np.zeros(3), pelvis_up=1., torso_up=1.)
    state.update(changes)
    return stable_support_mask(**state)


@pytest.mark.parametrize('change', [dict(pelvis_up=.8), dict(torso_up=.8),
    dict(base_velocity=[0., 0., -.6]), dict(base_angular_velocity=[2., 0., 0.]),
    dict(foot_velocity=np.array([[.3, 0., 0.], [0., 0., 0.]])),
    dict(foot_up=np.array([[.5, 0., .86], [0., 0., 1.]]))])
def test_tilt_drop_rotation_sliding_and_foot_pitch_cannot_complete_tread(change):
    assert stable().tolist() == [True, False]
    assert not stable(**change).any()


def test_stable_hold_requires_continuity_and_recovers_after_window_expiry():
    sequence = SequentialTreadReward(StairGeometry(), minimum_hold=.18)
    ratios = np.zeros((2, 6))
    ratios[0, 0] = 1.
    for _ in range(30):
        for _ in range(17):
            sequence.step(ratios, .3, .01, active=True)
        sequence.step(np.zeros_like(ratios), .3, .01, active=True)
    assert sequence.current_index == 0
    for _ in range(18):
        sequence.step(ratios, .3, .01, active=True)
    assert sequence.current_index == 1


def test_impact_and_fall_are_dt_independent_events():
    totals = []
    for dt in (.01, .02):
        raw = landing_costs([1.35, 0.], 0., True, dt, False)
        totals.append({k: v * REWARD_WEIGHTS[k] * dt for k, v in raw.items()})
    assert totals[0] == pytest.approx(totals[1])
    assert totals[0]['landing_impact'] == pytest.approx(-1.)
    assert totals[0]['fall_event'] == -8.
    assert landing_costs([.2, 0.], 0., False, .01, True)['landing_impact'] == 0.


def test_physical_drop_records_touchdown_before_contact_deceleration():
    env = H2StairsEnv(terrain_mode='flat', stair_direction='up')
    try:
        env.reset(seed=0)
        env.data.qpos[2] += .04
        env.data.qvel[2] = -1.
        mujoco.mj_forward(env.model, env.data)
        peak = 0.
        for _ in range(40):
            _, _, done, truncated, info = env.step(np.zeros(15))
            peak = max(peak, max(info['landing_downward_speed']))
            if done or truncated:
                break
        assert peak > .35
    finally:
        env.close()


def test_environment_masks_unstable_contact_before_sequential_completion():
    env = H2StairsEnv(terrain_mode='low', stair_direction='up')
    try:
        env.reset(seed=0, options={'command': [.3, 0., 0.]})
        soles = [np.array([[1.02, .1, .05], [1.08, .1, .05]]),
                 np.array([[.4, -.1, 0.]])]
        env._world_foot_points = lambda _: soles
        for _ in range(40):
            _, info = env._stair_reward_terms([True, False], .01, [False, False])
            assert info['stair_completed_treads'] == 0
        for _ in range(17):
            env._stair_reward_terms([True, False], .01, [True, False])
        assert env._tread_reward.current_index == 0
        env._stair_reward_terms([True, False], .01, [True, False])
        assert env._tread_reward.current_index == 1
    finally:
        env.close()


def test_v3_migration_keeps_robot_contract_strict(tmp_path):
    env = H2StairsEnv()
    try:
        contract = env.get_training_contract()
        contract['reward_version'] = 'h2_stairs_dense_v3'
        contract['reward_weights'] = {'old': 1.}
        path = tmp_path / 'policy.zip'
        sidecar = path.with_suffix('.curriculum.json')
        sidecar.write_text(json.dumps({'training_contract': contract}))
        with pytest.raises(ValueError):
            validate_checkpoint_contract(path, env)
        validate_checkpoint_contract(path, env, allow_reward_change=True)
        contract['model_sha256'] = 'different'
        sidecar.write_text(json.dumps({'training_contract': contract}))
        with pytest.raises(ValueError):
            validate_checkpoint_contract(path, env, allow_reward_change=True)
    finally:
        env.close()
