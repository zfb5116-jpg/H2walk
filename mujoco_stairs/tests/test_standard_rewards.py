import json

import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env

from mujoco_stairs.rewards import compute_standard_rewards
from mujoco_stairs.checkpoint import validate_checkpoint_contract


@pytest.fixture
def state():
    return dict(command=np.array([.3, 0., 0.]), linear_velocity=np.array([.3, 0., 0.]),
                angular_velocity=np.zeros(3), heading_error=0., joint_deviation=np.zeros(15),
                joint_position=np.zeros(15), joint_bounds=np.tile([-1., 1.], (15, 1)),
                joint_limited=np.ones(15, dtype=bool), joint_velocity=np.zeros(15),
                joint_acceleration=np.zeros(15), velocity_limits=np.full(15, 10.),
                torque=np.zeros(15), torque_limits=np.full(15, 100.),
                action=np.zeros(15), previous_action=np.zeros(15), support=np.ones(2),
                landed_air_time=np.zeros(2), forward_speed=.3, stall_time=0.,
                foot_position=np.array([[0., .105, 0.], [0., -.105, 0.]]),
                foot_velocity=np.zeros((2, 3)), foot_up=np.array([[0., 0., 1.]] * 2),
                sole_plane_error=np.zeros(2), hip_anchor_indices=[1, 2, 7, 8],
                torso_gravity=np.array([0., 0., -1.]), pelvis_gravity=np.array([0., 0., -1.]),
                fallen=False, undesired_contacts=0)


def test_tracking_rejects_sideways_and_spinning_motion(state):
    baseline = compute_standard_rewards(state)
    state['linear_velocity'][1] = .7
    state['angular_velocity'][2] = 1.
    changed = compute_standard_rewards(state)
    assert changed['linear_velocity_tracking'] < .2 * baseline['linear_velocity_tracking']
    assert changed['angular_velocity_tracking'] < .1 * baseline['angular_velocity_tracking']


def test_waist_posture_is_free_while_walking_but_standing_anchor_is_gated(state):
    state['joint_deviation'][-1] = .4
    moving = compute_standard_rewards(state)
    assert moving['stand_pose_l2'] == moving['stand_pose_anchor'] == 0.
    assert moving['hip_deviation'] == 0.
    state['command'][:] = 0.
    standing = compute_standard_rewards(state)
    assert standing['stand_pose_l2'] == pytest.approx(.16)
    assert standing['stand_pose_anchor'] < 1.


def test_air_time_only_pays_at_progressing_landing_and_is_bounded(state):
    state['landed_air_time'][:] = [5., 0.]
    assert compute_standard_rewards(state)['feet_air_time'] == pytest.approx(.35)
    state['forward_speed'] = 0.
    assert compute_standard_rewards(state)['feet_air_time'] == 0.


def test_controlled_waist_obeys_same_safety_costs_as_legs(state):
    state['joint_position'][-1] = 1.
    state['joint_velocity'][-1] = 10.
    state['torque'][-1] = 100.
    raw = compute_standard_rewards(state)
    assert raw['joint_position_limit'] == pytest.approx(.1)
    assert raw['joint_velocity_limit'] == pytest.approx(1.)
    assert raw['torque_limit'] == pytest.approx(10.)


def test_checkpoint_contract_requires_new_version_and_same_model(tmp_path):
    env = make_vec_env('CtrlLab-H2-Stairs-MuJoCo-v1', n_envs=1)
    try:
        path = tmp_path / 'policy.zip'
        sidecar = path.with_suffix('.curriculum.json')
        sidecar.write_text(json.dumps({'training_contract': {'actions': 12}}))
        with pytest.raises(ValueError, match='Incompatible checkpoint'):
            validate_checkpoint_contract(path, env)
        contract = env.env_method('get_training_contract')[0]
        sidecar.write_text(json.dumps({'training_contract': contract}))
        validate_checkpoint_contract(path, env)
        contract['model_sha256'] = 'different-model'
        sidecar.write_text(json.dumps({'training_contract': contract}))
        with pytest.raises(ValueError):
            validate_checkpoint_contract(path, env)
    finally:
        env.close()
