from dataclasses import replace

import numpy as np
import pytest

from mujoco_stairs.dense_stair_reward import DenseStairReward
from mujoco_stairs.stair_rewards import StairGeometry


def sample(reward, x=.7, z=.1, support=(False, True), root=.5, index=0, height=.2, active=True):
    soles = [np.array([[x, .1, z], [x + .05, .1, z]]),
             np.array([[.4, -.1, 0.]])]
    return reward.step(soles, support, root, 0., height, index, .01, active)


def test_lift_and_reach_pay_before_contact_but_cannot_be_farmed():
    reward = DenseStairReward(StairGeometry())
    first = sample(reward)
    assert first['stair_lift'] > 0 and first['stair_reach'] > 0
    assert not any(sample(reward).values())
    sample(reward, z=0.)
    assert not any(sample(reward).values())
    higher = sample(reward, z=.24)
    assert higher['stair_lift'] > 0
    assert first['stair_lift'] + higher['stair_lift'] == pytest.approx(100.)


def test_support_lane_direction_and_command_gates():
    for kwargs in ({'support': (True, True)}, {'support': (False, False)}):
        terms = sample(DenseStairReward(StairGeometry()), **kwargs)
        assert terms['stair_lift'] == terms['stair_reach'] == 0
    for kwargs in ({'height': 0.}, {'active': False}):
        assert not any(sample(DenseStairReward(StairGeometry()), **kwargs).values())
    assert not any(sample(DenseStairReward(replace(StairGeometry(), direction='down'))).values())


def test_each_completion_is_paid_once_including_last():
    reward = DenseStairReward(StairGeometry())
    total = sum(sample(reward, index=i)['stair_completion'] * .01 for i in range(1, 7))
    assert total == 6
    assert not any(sample(reward, index=6).values())


def test_backtracking_does_not_reset_approach_budget():
    reward = DenseStairReward(StairGeometry())
    assert sample(reward, root=.7)['stair_approach'] > 0
    sample(reward, root=.3)
    assert sample(reward, root=.7)['stair_approach'] == 0
