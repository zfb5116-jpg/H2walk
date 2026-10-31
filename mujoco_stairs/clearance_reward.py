"""Terrain-relative swing rewards, independent of the simulator bindings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FootState:
    rear_x: float
    toe_x: float
    sole_z: float
    forward_speed: float
    support: bool
    support_height: float


@dataclass(frozen=True)
class SwingScore:
    clearance: float
    low_clearance: float
    target_z: float


def swing_score(
    foot: FootState,
    *,
    other_foot_support: bool,
    step_height: float,
    command_speed: float,
    first_edge: float = 0.9,
    tread_depth: float = 0.35,
    step_count: int = 6,
    margin: float = 0.05,
) -> SwingScore:
    """Reward forward swing above the next tread, with support on the other leg.

    A stationary raised leg, a planted foot, and a two-foot jump earn no swing
    reward. The target uses the last supporting tread and the next riser ahead
    of this foot, rather than terrain under the pelvis. Excessively high kicks
    lose their height score; there is no incentive to keep raising the leg.
    """
    next_edge = math.inf
    target_z = foot.support_height + 0.06
    if step_height > 0.0:
        support_level = int(round(foot.support_height / step_height))
        # A long step can already be above the next tread. Never target a
        # surface lower than the terrain below the front of the swinging foot.
        toe_level = max(0, int(math.floor((foot.toe_x - first_edge) / tread_depth)) + 1)
        next_level = min(max(support_level + 1, toe_level), step_count)
        edge = first_edge + (next_level - 1) * tread_depth
        if support_level < step_count and foot.toe_x >= edge - 0.40:
            next_edge = edge
            target_z = next_level * step_height + margin

    if foot.support or not other_foot_support or command_speed <= 0.05:
        return SwingScore(0.0, 0.0, target_z)
    forward_gate = min(max(foot.forward_speed / max(command_speed, 0.15), 0.0), 1.0)
    required_lift = max(target_z - foot.support_height, 0.06)
    lift = min(max((foot.sole_z - foot.support_height) / required_lift, 0.0), 1.0)
    overshoot = max(foot.sole_z - target_z, 0.0)
    clearance = forward_gate * lift * math.exp(-((overshoot / 0.10) ** 2))
    near_edge = max(1.0 - abs(foot.toe_x - next_edge) / 0.20, 0.0)
    shortfall = min(max((target_z - foot.sole_z) / required_lift, 0.0), 1.0)
    return SwingScore(clearance, forward_gate * near_edge * shortfall, target_z)
