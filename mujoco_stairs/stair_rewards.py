"""MuJoCo adaptation of WM-LOCO Section 2.2, Eqs. (2)--(4).

Reference: https://arxiv.org/html/2609.02542v1#S2.SS2
The three raw terms are counts, counts, and a bounded sequential contact score.
This module adds no world model, visual policy, or paper performance claims.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StairGeometry:
    first_edge: float = 0.9
    tread_depth: float = 0.35
    step_count: int = 6
    landing_end: float = 4.3
    physical_half_width: float = 2.0
    lane_half_width: float = 0.60
    riser_half_thickness: float = 0.02
    riser_top_exclusion: float = 0.04
    sole_height_tolerance: float = 0.02

    def tread_bounds(self, index: int) -> tuple[float, float]:
        start = self.first_edge + index * self.tread_depth
        end = self.landing_end if index == self.step_count - 1 else start + self.tread_depth
        return start, end


def stair_boundary_count(foot_volume_points: list[np.ndarray], step_height: float,
                         geometry: StairGeometry) -> int:
    """Count the union of two forbidden lateral strips near the staircase.

    The physical scene is four metres wide, so the H2 adaptation constrains a
    central 1.2 m walking lane instead of waiting for an already terminal
    side excursion. Point-count weighting deliberately depends on sensor size.
    """
    if step_height <= 0.0:
        return 0
    points = np.concatenate(foot_volume_points)
    beside = (points[:, 0] >= geometry.first_edge - 0.20) & (points[:, 0] <= geometry.landing_end)
    outside_lane = np.abs(points[:, 1]) > geometry.lane_half_width
    return int(np.count_nonzero(beside & outside_lane))


def stair_riser_count(foot_volume_points: list[np.ndarray], step_height: float,
                      geometry: StairGeometry) -> int:
    """Count volume points in thin cuboids over risers, excluding top 4 cm.

    An exclusion equal to a shallow riser's full height leaves no cuboid. This
    is intentional; clearance shaping still teaches the 4 cm curriculum stage.
    Overlapping cuboids are a union: each point can be counted only once.
    """
    if step_height <= geometry.riser_top_exclusion:
        return 0
    points = np.concatenate(foot_volume_points)
    inside = np.zeros(len(points), dtype=bool)
    for index in range(geometry.step_count):
        edge = geometry.first_edge + index * geometry.tread_depth
        inside |= ((np.abs(points[:, 0] - edge) <= geometry.riser_half_thickness)
                   & (np.abs(points[:, 1]) <= geometry.physical_half_width)
                   & (points[:, 2] >= index * step_height)
                   & (points[:, 2] < (index + 1) * step_height - geometry.riser_top_exclusion))
    return int(np.count_nonzero(inside))


def sole_contact_ratios(sole_points: list[np.ndarray], supported: list[bool],
                         step_height: float, geometry: StairGeometry) -> np.ndarray:
    """Fraction of each sole near each tread with real upward support contact.

    XY overlap alone cannot credit a foot flying above the tread or pressing
    against a vertical face. Sampling uses transformed collision-box bounds
    in the bundled model, not the paper's detailed foot-mesh surface sensor.
    """
    ratios = np.zeros((2, geometry.step_count))
    if step_height <= 0.0:
        return ratios
    for foot_id, points in enumerate(sole_points):
        if not supported[foot_id] or len(points) == 0:
            continue
        for index in range(geometry.step_count):
            start, end = geometry.tread_bounds(index)
            inside = ((points[:, 0] >= start + 0.005) & (points[:, 0] <= end - 0.005)
                      & (np.abs(points[:, 1]) <= geometry.lane_half_width)
                      & (np.abs(points[:, 2] - (index + 1) * step_height)
                         <= geometry.sole_height_tolerance))
            ratios[foot_id, index] = np.count_nonzero(inside) / len(points)
    return ratios


class SequentialTreadReward:
    """Consume treads in order with alternating feet and finite reward windows.

    A failed window is never rewarded again on that tread. Contact can still
    complete it later, so failure cannot permanently deadlock the sequence.
    After completion, an already consumed tread can never produce more credit.
    These expiry details are explicit H2 choices where the paper is underspecified.
    """

    def __init__(self, geometry: StairGeometry):
        self.geometry = geometry
        self.reset()

    def reset(self) -> None:
        self.current_index = 0
        self.expected_foot: int | None = None
        self.contact_foot: int | None = None
        self.window_opened = False
        self.window_remaining = 0
        self.required_hold = 0
        self.hold_steps = 0

    def step(self, ratios: np.ndarray, command_speed: float, dt: float, *, active: bool) -> float:
        if not active or command_speed <= 0.05 or self.current_index >= self.geometry.step_count:
            return 0.0
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("control dt must be finite and positive")
        rho = ratios[:, self.current_index]
        qualified = rho >= 0.5
        if not self.window_opened:
            if not qualified.any():
                return 0.0
            # Use a nominal tread depth even on the longer finishing platform;
            # otherwise its reward duration would depend on the goal extension.
            tau = float(np.clip(self.geometry.tread_depth / (2.0 * command_speed * dt), 5.0, 50.0))
            self.window_remaining = int(math.ceil(tau))
            self.required_hold = int(math.ceil(0.3 * tau))
            self.window_opened = True
        if self.contact_foot is None or not qualified[self.contact_foot]:
            self.hold_steps = 0
            if qualified.any():
                self.contact_foot = (self.expected_foot if self.expected_foot is not None
                                     and qualified[self.expected_foot] else int(np.argmax(rho)))
        foot = self.contact_foot
        reward = 0.0
        if foot is not None:
            if self.window_remaining > 0:
                alternation = 1.0 if self.expected_foot in (None, foot) else 0.3
                reward = alternation * float(rho[foot]) - 0.7 * max(float(rho[1 - foot]) - 0.05, 0.0)
            self.hold_steps = self.hold_steps + 1 if qualified[foot] else 0
        self.window_remaining = max(self.window_remaining - 1, 0)
        if foot is not None and self.hold_steps >= self.required_hold:
            self.current_index += 1
            self.expected_foot = 1 - foot
            self.contact_foot = None
            self.window_opened = False
            self.window_remaining = 0
            self.hold_steps = 0
        return reward
