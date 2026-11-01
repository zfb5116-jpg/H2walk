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
        for _ in range(15 if index == 5 else 11):
            total += sequence.step(ratios(index, float(index % 2 == 0), float(index % 2 == 1)),
                                   0.5, 0.01, active=True)
    assert sequence.current_index == 6
    assert total == pytest.approx(70.0)
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




def test_descending_tread_contacts_and_riser_cuboids_use_descending_heights():
    geometry = replace(GEOMETRY, direction="down")
    sole = np.array([[1., 0., 1.0], [1.1, 0., 1.0]])
    ratios = sole_contact_ratios([sole, sole], [True, False], .2, geometry)
    assert ratios[0, 0] == 1. and not ratios[0, 1:].any()
    assert stair_riser_count(points([.9, 0., 1.05], [.9, 0., 1.18]), .2, geometry) == 1
    assert geometry.tread_height(5, .2) == 0.


def test_final_long_landing_uses_actual_region_depth_for_window():
    sequence = SequentialTreadReward(GEOMETRY)
    sequence.current_index = 5
    sequence.step(ratios(5), .5, .01, active=True)
    assert sequence.required_hold == 15
