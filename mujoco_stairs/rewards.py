"""WM-LOCO Appendix B, Table 3 (Stairs column), adapted to MuJoCo H2.

Table weights are transcribed; standard-term formulas/thresholds not specified
by the paper are explicit H2 choices. See REWARDS.md. No fixed swing-height,
raw forward-velocity, success, fall or upper-body-freeze reward is retained.
"""

from types import MappingProxyType

import numpy as np

REWARD_VERSION = "wm_loco_stairs_h2_v2"
REWARD_WEIGHTS = MappingProxyType({
    "linear_velocity_tracking": 3.0,
    "angular_velocity_tracking": 3.0,
    "heading_error": -1.0,
    "no_progress": -2.0,
    "alive": 0.5,
    "stand_pose_anchor": 1.5,
    "stand_pose_l2": -1.0,
    "stand_feet_contact": 0.4,
    "walking_action_rate": -0.005,
    "stair_boundary": -0.05,
    "stair_riser": -0.5,
    "stair_tread": 0.5,
    "feet_air_time": 0.5,
    "feet_slide": -0.4,
    "feet_flat_orientation": -0.4,
    "feet_on_plane": -0.1,
    "feet_close_xy": 0.4,
    "hip_deviation": -0.5,
    "angular_velocity_xy": -0.05,
    "flat_orientation": -3.0,
    "pelvis_orientation": -3.0,
    "joint_torque": -1.5e-7,
    "joint_acceleration": -1.25e-7,
    "joint_velocity": -1e-4,
    "mechanical_energy": -5e-5,
    "joint_position_limit": -1.0,
    "joint_velocity_limit": -1.0,
    "torque_limit": -0.01,
    "undesired_contact": -1.0,
})


def compute_standard_rewards(s: dict) -> dict[str, float]:
    """Return raw, unweighted terms using one post-physics state sample.

    Exponential bandwidths are variance denominators (not standard deviation).
    Pelvis and upper torso orientation are soft costs; waist position is not
    anchored during walking. All 15 controlled joints receive safety costs.
    """
    moving = float(np.linalg.norm(s["command"]) > 0.05)
    standing = 1.0 - moving
    support = np.asarray(s["support"], dtype=float)
    deviation = np.asarray(s["joint_deviation"])
    pose_error = float(np.square(deviation).sum())
    bounds = np.asarray(s["joint_bounds"])
    middle = bounds.mean(axis=1)
    half = (bounds[:, 1] - bounds[:, 0]) * 0.45
    position_excess = np.maximum(np.abs(s["joint_position"] - middle) - half, 0.0)
    position_excess *= s["joint_limited"]
    velocity_excess = np.maximum(np.abs(s["joint_velocity"]) - 0.9 * s["velocity_limits"], 0.0)
    torque_excess = np.maximum(np.abs(s["torque"]) - 0.9 * s["torque_limits"], 0.0)
    # First-contact air-time credit, bounded and gated by actual root progress.
    air_credit = np.clip(s["landed_air_time"] - 0.25, 0.0, 0.35)
    foot_delta = s["foot_position"][0, :2] - s["foot_position"][1, :2]
    separation_error = foot_delta[0] ** 2 + (abs(foot_delta[1]) - 0.21) ** 2
    raw = {
        "linear_velocity_tracking": np.exp(-np.square(s["linear_velocity"][:2] - s["command"][:2]).sum() / 0.25),
        "angular_velocity_tracking": np.exp(-(s["angular_velocity"][2] - s["command"][2]) ** 2 / 0.25),
        "heading_error": s["heading_error"] ** 2,
        "no_progress": moving * float(s["stall_time"] >= 1.0),
        "alive": float(not s["fallen"]),
        "stand_pose_anchor": standing * np.exp(-pose_error / 0.25),
        "stand_pose_l2": standing * pose_error,
        "stand_feet_contact": standing * support.mean(),
        "walking_action_rate": moving * np.square(s["action"] - s["previous_action"]).sum(),
        "feet_air_time": moving * float(s["forward_speed"] > 0.05) * air_credit.sum(),
        "feet_slide": (np.square(s["foot_velocity"][:, :2]).sum(axis=1) * support).sum(),
        "feet_flat_orientation": (np.square(s["foot_up"][:, :2]).sum(axis=1) * support).sum(),
        "feet_on_plane": (s["sole_plane_error"] * support).sum(),
        "feet_close_xy": moving * np.exp(-separation_error / 0.09),
        "hip_deviation": np.square(deviation[s["hip_anchor_indices"]]).sum(),
        "angular_velocity_xy": np.square(s["angular_velocity"][:2]).sum(),
        "flat_orientation": np.square(s["torso_gravity"][:2]).sum(),
        "pelvis_orientation": np.square(s["pelvis_gravity"][:2]).sum(),
        "joint_torque": np.square(s["torque"]).sum(),
        "joint_acceleration": np.square(s["joint_acceleration"]).sum(),
        "joint_velocity": np.square(s["joint_velocity"]).sum(),
        "mechanical_energy": np.abs(s["torque"] * s["joint_velocity"]).sum(),
        "joint_position_limit": position_excess.sum(),
        "joint_velocity_limit": velocity_excess.sum(),
        "torque_limit": torque_excess.sum(),
        "undesired_contact": float(s["undesired_contacts"]),
    }
    return {name: float(value) for name, value in raw.items()}


def weighted_rewards(raw: dict[str, float], dt: float) -> dict[str, float]:
    """Integrate reward rates at the control timestep, consistently for all terms."""
    if set(raw) != set(REWARD_WEIGHTS):
        raise ValueError("Reward terms do not match the versioned WM-LOCO stairs schedule")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(list(raw.values())).all():
        raise ValueError("Reward values and control timestep must be finite")
    return {name: REWARD_WEIGHTS[name] * raw[name] * dt for name in REWARD_WEIGHTS}
