"""Landing quality measured in world coordinates, independent of reward weights."""
import numpy as np


def stable_support_mask(support, foot_velocity, foot_up, base_velocity,
                        base_angular_velocity, pelvis_up, torso_up):
    """A tread can count only while supported, upright, and moving gently."""
    balanced = (pelvis_up > .90 and torso_up > .90
                and abs(base_velocity[2]) < .45
                and np.linalg.norm(base_angular_velocity[:2]) < 1.5)
    return (np.asarray(support, dtype=bool) & balanced
            & (np.linalg.norm(foot_velocity[:, :2], axis=1) < .20)
            & (foot_up[:, 2] > .90))


def landing_costs(touchdown_speeds, base_vertical_velocity, fallen, dt, near_stairs):
    """Touchdown/fall are events; vertical oscillation is a rate near stairs."""
    return {
        "landing_impact": float(np.square(np.clip(np.asarray(touchdown_speeds) - .35, 0., 3.)).sum()) / dt,
        "base_vertical_motion": float(max(abs(base_vertical_velocity) - .25, 0.) ** 2) * near_stairs,
        "fall_event": float(fallen) / dt,
    }
