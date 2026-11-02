"""Terrain-relative swing learning costs; no joint target or gait is supplied."""
from math import cos
import torch


def stair_outward_tilt_growth(env, margin_angle=0.0):
    """Retain the accepted flat policy's physical outward-tilt-potential rate."""
    gravity = env.asset.data.projected_gravity_b
    angular = env.asset.data.root_ang_vel_b
    outward = torch.relu(gravity[:, 0]*angular[:, 1]-gravity[:, 1]*angular[:, 0])
    return outward * (-gravity[:, 2] < cos(margin_angle))


def stair_body_angular_rate(env):
    return env.asset.data.root_ang_vel_w[:, :2].square().sum(-1)


def swing_clearance_deficit(soles, rotations, origins, heights, corners, previous_soles,
                            filtered_force, weight, minimum_load, dt, *,
                            approach=1.4, depth=0.35, steps=6,
                            lookahead=0.20, clearance=0.02, distance_scale=0.05):
    """Cost low forward-moving unloaded soles near the next real riser.

    The other foot must physically support at least70% body weight. Every sole
    corner must clear the terrain plane ahead. This emits only a nonnegative
    cost: standing, backward swing and flat ground receive no positive reward.
    It neither selects a foot/phase nor puts simulator geometry in observations.
    """
    local = soles-origins[:, None]
    world_corners = torch.einsum('nfij,fkj->nfki', rotations, corners)+local[:, :, None]
    front = world_corners[..., 0].amax(-1)
    bottom = world_corners[..., 2].amin(-1)
    current_level = (torch.floor((front-approach)/depth)+1).clamp(0, steps)
    ahead_level = (torch.floor((front+lookahead-approach)/depth)+1).clamp(0, steps)
    obstacle = ahead_level > current_level
    required_height = ahead_level*heights[:, None]+clearance
    deficit = ((required_height-bottom).clamp_min(0)/distance_scale).clamp_max(4).square()
    fz = filtered_force[..., 2]
    unloaded = fz < minimum_load[:, None]
    other_supported = fz.flip(-1) >= .7*weight[:, None]
    forward_speed = ((soles[..., 0]-previous_soles[..., 0])/dt).clamp_min(0)
    moving = (forward_speed/.3).clamp_max(1)
    return (deficit*moving*obstacle*unloaded*other_supported).sum(-1)


def stair_swing_clearance_deficit(env):
    env._update_stair_progress()
    p = env.stair_progress
    soles, rotations = env._stair_foot_pose()
    cost = swing_clearance_deficit(soles, rotations, p.origins, p.heights, p.corners,
        p.pre_soles, p.ground_force_end, p.weight, p.minimum_load, env.step_dt,
        approach=p.approach, depth=p.depth, steps=p.steps)
    # Negative physical costs are not suppressed after a force-qualification
    # failure. Only invalid physical states are excluded by their own boundary.
    return cost * ~p.ever_invalid
