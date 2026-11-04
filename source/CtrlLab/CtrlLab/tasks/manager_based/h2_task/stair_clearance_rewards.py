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


def full_footprint_riser_target(front, rear, *, approach=1.4, depth=0.35, steps=6,
                               lookahead=0.20, footprint_margin=0.008):
    """Keep the first real riser until the complete sole has crossed it."""
    edges = approach + torch.arange(steps, device=front.device, dtype=front.dtype)*depth
    crossed = (rear[..., None] >= edges + footprint_margin).sum(-1)
    selected_edge = edges[crossed.clamp_max(steps-1)]
    obstacle = (crossed < steps) & (front >= selected_edge-lookahead)
    return (crossed+1).clamp_max(steps), obstacle


def full_footprint_clearance_contract(*, approach=1.4, depth=0.35, steps=6,
                                    preview=0.20, full_target_distance=0.20,
                                    terrain_normalized_envelope=False, motion_floor=0.0):
    contract = dict(kind="earliest_uncrossed_full_footprint_v1", real_riser_count=steps,
                approach_m=approach, depth_m=depth, rear_margin_m=0.008,
                lookahead_m=0.20, extra_height_m=0.02, distance_scale_m=0.05,
                deficit_square_cap=16.0, target_retained_until_whole_sole_crossed=True,
                observation_or_joint_controller=False)
    if terrain_normalized_envelope:
        if not preview > full_target_distance >= 0 or not 0 <= motion_floor <= 1:
            raise ValueError("Clearance preview must precede the full-height requirement")
        contract.update(lookahead_m=preview, distance_scale_m=None, deficit_square_cap=1.0,
            height_normalization="actual_riser_height_plus_clearance",
            outside_preview_reference="selected_lower_plane", uncrossed_only_cost_mask=True,
            approach_envelope=dict(preview_m=preview, full_target_distance_m=full_target_distance,
                                   interpolation="linear_clipped"))
        if motion_floor:
            contract["motion_multiplier"] = dict(kind="preview_supported_motion_floor_v1",
                floor=motion_floor, inside_preview_only=True,
                outside_preview="original_forward_motion", includes_stationary_and_retreat=True)
    elif preview != 0.20 or full_target_distance != 0.20 or motion_floor != 0:
        raise ValueError("Legacy clearance geometry must retain its declared20cm window")
    return contract


def swing_clearance_deficit(soles, rotations, origins, heights, corners, previous_soles,
                            filtered_force, weight, minimum_load, dt, *,
                            approach=1.4, depth=0.35, steps=6,
                            lookahead=0.20, clearance=0.02, distance_scale=0.05,
                            terrain_normalized_envelope=False, full_target_distance=0.20,
                            motion_floor=0.0):
    """Cost low supported/unloaded soles through the next real riser.

    The other foot must physically support at least70% body weight. Every sole
    corner must clear the terrain plane ahead. This emits only a nonnegative
    cost. An explicitly selected preview motion floor also covers a stationary
    or retreating unloaded foot; the other foot must still bear70% weight.
    It neither selects a foot/phase nor puts simulator geometry in observations.
    """
    local = soles-origins[:, None]
    world_corners = torch.einsum('nfij,fkj->nfki', rotations, corners)+local[:, :, None]
    front = world_corners[..., 0].amax(-1)
    rear = world_corners[..., 0].amin(-1)
    bottom = world_corners[..., 2].amin(-1)
    target_level, obstacle = full_footprint_riser_target(front, rear, approach=approach,
        depth=depth, steps=steps, lookahead=lookahead)
    if terrain_normalized_envelope:
        if not lookahead > full_target_distance >= 0 or not 0 <= motion_floor <= 1:
            raise ValueError("Clearance preview must precede the full-height requirement")
        if not bool(torch.isfinite(heights).all() & (heights > 0).all()):
            raise ValueError("Terrain-normalized clearance requires actual positive finite heights")
        last_edge = approach+torch.as_tensor(steps-1, device=front.device, dtype=front.dtype)*depth
        obstacle = rear < last_edge+0.008
        edge = approach+(target_level-1).to(front.dtype)*depth
        distance = edge-front
        alpha = ((lookahead-distance)/(lookahead-full_target_distance)).clamp(0, 1)
        alpha = torch.where(distance <= full_target_distance, 1.0, alpha)
        rise = heights[:, None]+clearance
        required_height = (target_level-1)*heights[:, None]+alpha*rise
        deficit = ((required_height-bottom)/rise).clamp(0, 1).square()
    else:
        if motion_floor != 0:
            raise ValueError("A motion floor requires the declared terrain-normalized profile")
        required_height = target_level*heights[:, None]+clearance
        deficit = ((required_height-bottom).clamp_min(0)/distance_scale).clamp_max(4).square()
    fz = filtered_force[..., 2]
    unloaded = fz < minimum_load[:, None]
    other_supported = fz.flip(-1) >= .7*weight[:, None]
    forward_speed = ((soles[..., 0]-previous_soles[..., 0])/dt).clamp_min(0)
    moving = (forward_speed/.3).clamp_max(1)
    if motion_floor:
        # This enables only the new motion floor, never a whole-cost preview
        # mask: distant below-lower-plane moving deficits retain their cost.
        floor = ((distance <= lookahead) & obstacle).to(moving.dtype)*motion_floor
        moving = torch.maximum(moving, floor)
    return (deficit*moving*obstacle*unloaded*other_supported).sum(-1)


def stair_swing_clearance_deficit(env, preview=0.20, full_target_distance=0.20,
                                 terrain_normalized_envelope=False, motion_floor=0.0):
    env._update_stair_progress()
    p = env.stair_progress
    soles, rotations = env._stair_foot_pose()
    cost = swing_clearance_deficit(soles, rotations, p.origins, p.heights, p.corners,
        p.pre_soles, p.ground_force_end, p.weight, p.minimum_load, env.step_dt,
        approach=p.approach, depth=p.depth, steps=p.steps, lookahead=preview,
        terrain_normalized_envelope=terrain_normalized_envelope,
        full_target_distance=full_target_distance, motion_floor=motion_floor)
    # Negative physical costs are not suppressed after a force-qualification
    # failure. Only invalid physical states are excluded by their own boundary.
    return cost * ~p.ever_invalid


def clearance_progress_components(soles, rotations, origins, heights, corners, *,
                                  approach=1.4, depth=0.35, steps=6):
    """Two geometric potentials; contact and motion gates do not define state.

    The existing real-riser selector and full collision sole are reused.  This
    supplies no desired trajectory, phase, observation or joint target.
    """
    if not bool((heights > 0).all()):
        raise ValueError("Clearance progress needs positive actual riser heights")
    local = soles-origins[:, None]
    world_corners = torch.einsum('nfij,fkj->nfki', rotations, corners)+local[:, :, None]
    front = world_corners[..., 0].amax(-1)
    rear = world_corners[..., 0].amin(-1)
    bottom = world_corners[..., 2].amin(-1)
    level, near = full_footprint_riser_target(front, rear, approach=approach,
                                            depth=depth, steps=steps)
    lower_plane = (level-1)*heights[:, None]
    components = ((bottom-lower_plane)/(heights[:, None]+0.02)).clamp(0, 1)
    return components*near, level, near


def clearance_progress_delta(previous, current, positive_weight, positive_eligible, gamma=0.99):
    """Gate each positive difference separately; never cancel a foot's debt.

    The added real-arithmetic event lies in[-2,1.98]. Clamping the new positive
    multiplier to[0,1] does not alter the original potential or its multiplier.
    """
    raw = gamma*current-previous
    multiplier = positive_weight[:, None].clamp(0, 1)*positive_eligible
    return raw, raw.clamp_max(0)+raw.clamp_min(0)*multiplier
