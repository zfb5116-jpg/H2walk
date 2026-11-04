"""Swing-yaw feasibility across the current and remaining Cartesian corridor.

This module has no simulator startup and never changes a terrain or joint limit.
The caller owns pressure allocation, support feedback and the real flight clock.
"""

import torch
from .stair_trajectory import swing_position_local,smooth,swing_pitch


def _rotation_z(angle):
    result = angle.new_zeros(angle.shape + (3, 3))
    cosine, sine = angle.cos(), angle.sin()
    result[..., 0, 0] = result[..., 1, 1] = cosine
    result[..., 0, 1], result[..., 1, 0] = -sine, sine
    result[..., 2, 2] = 1
    return result


def _quaternion_matrix(quaternion):
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
         2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
         2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1
    ).reshape(q.shape[:-1] + (3, 3))


def _context(gait, ids, soles, actual_gravity=None):
    root_position = gait.robot.data.root_pos_w[ids]
    root_rotation = _quaternion_matrix(gait.robot.data.root_quat_w[ids])
    inverse = root_rotation.transpose(-1, -2)
    heading = _quaternion_matrix(gait.flat.heading[ids])
    actual_q = gait.robot.data.joint_pos[ids][:, gait.joint_ids].reshape(-1, 2, 6)
    if actual_gravity is None:
        zero = root_position.new_zeros((ids.numel(), 2, 3))
        actual_gravity = gait.kin.support_torque(
            actual_q, zero, zero, -inverse[..., 2])
    else:
        if actual_gravity.shape not in ((gait.q.shape[0], 2, 6),
                                        (gait.q.shape[0], 12)):
            raise ValueError("actual_gravity must contain the full N-by-2-by-6 gravity tensor")
        actual_gravity = actual_gravity.reshape(gait.q.shape[0], 2, 6)[ids]
    stiffness = gait.robot.data.joint_stiffness[ids][:, gait.joint_ids]
    stiffness = stiffness.reshape(-1, 2, 6).clamp_min(1e-6)
    _, actual_rotation = gait.kin.forward(actual_q)
    return {
        "root": root_position, "root_rotation": root_rotation, "inverse": inverse,
        "heading": heading, "actual_gravity": actual_gravity,
        "stiffness": stiffness, "soles": soles[ids],
        "actual_foot_rotation": root_rotation[:, None] @ actual_rotation,
    }


def _separation_margin(moving, fixed, moving_rotation, fixed_rotation):
    """Four-axis SAT gap for two horizontal rectangles; positive means apart."""
    axes = torch.cat((moving_rotation[..., :2, :2].transpose(-1, -2),
                      fixed_rotation[..., :2, :2].transpose(-1, -2)), -2)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    moving_projection = (moving[..., :, None, :2] * axes[..., None, :, :]).sum(-1)
    fixed_projection = (fixed[..., :, None, :2] * axes[..., None, :, :]).sum(-1)
    return torch.maximum(
        fixed_projection.amin(-2)-moving_projection.amax(-2),
        moving_projection.amin(-2)-fixed_projection.amax(-2),
    ).amax(-1)


@torch.no_grad()
def pick_swing_yaw(gait, ids, soles, *, current_world=None, actual_gravity=None,
                   iterations=30, chunk_size=16, include_midpoints=True,
                   candidate_values=None, pose_seed=None):
    """Return one yaw feasible at current, lift/cross peak and landing poses.

    Yaw remains within +/-0.4 rad of the saved home sole angle. Both the current
    measured-q gravity bias and each candidate-q gravity bias must fit the real
    position-drive limits. Landing coverage certifies the entire real footprint.

    current_world is a full N-by-2-by-3 fresh Cartesian command, when available;
    the initial call defaults to the measured soles. No phase clock is changed.
    An optional actual_gravity override is full N-by-2-by-6 (or N-by-12), not
    already sliced by ids; the context takes the same subindices exactly once.
    Diagnostic tensors are indexed in the same order as ids.
    """
    if soles.shape != (gait.q.shape[0], 2, 3):
        raise ValueError("soles must be the full N-by-2-by-3 measured sole tensor")
    if current_world is not None and current_world.shape != soles.shape:
        raise ValueError("current_world and soles must both be full N-by-2-by-3 tensors")
    result = gait.foot_yaw[ids, gait.swing_side[ids]].clone()
    feasible = torch.zeros(ids.numel(), dtype=torch.bool, device=gait.device)
    current_solution = gait.q[ids].clone()
    diagnostics = {
        "joint_margin": result.new_full(result.shape, -float("inf")),
        "drive_margin": result.new_full(result.shape, -float("inf")),
        "patch_margin": result.new_full(result.shape, -float("inf")),
        "current_position_error": result.new_full(result.shape, float("inf")),
        "current_rotation_error": result.new_full(result.shape, float("inf")),
        "constant_candidate_count": torch.zeros_like(ids),
        "pose_seed": gait.q.new_zeros((ids.numel(), 6 if include_midpoints else 3, 2, 6)),
    }
    grid = torch.linspace(-.4, .4, 33, device=gait.device, dtype=gait.q.dtype)
    for start in range(0, ids.numel(), chunk_size):
        sub = ids[start:start+chunk_size]
        count = sub.numel()
        row = torch.arange(count, device=gait.device)
        side = gait.swing_side[sub]
        other = 1-side
        context = _context(gait, sub, soles, actual_gravity)
        landing = gait.flat.origin[sub] + (
            context["heading"] @ gait.swing_target[sub, :, None]).squeeze(-1)
        clearance = torch.where(gait.preparing[sub],
                                gait.options["preparation_clearance"],
                                gait.options["clearance"])
        local_peak = gait.swing_target[sub].clone()
        local_peak[:, 2] = torch.maximum(local_peak[:, 2], gait.swing_start[sub, 2]) + clearance
        peak = gait.flat.origin[sub] + (
            context["heading"] @ local_peak[..., None]).squeeze(-1)
        current = (soles[sub, side] if current_world is None
                   else current_world[sub, side])
        points = [current]
        if include_midpoints:
            lifted = gait.flat.origin[sub]+(context["heading"]@gait.lift_target[sub,:,None]).squeeze(-1)
            spent=(gait.cross_elapsed[sub]>0)|(gait.descend_time[sub]>=0)
            lifted=torch.where(spent[:,None],current,lifted)
            middle=.5*(lifted+peak)
            rising=(gait.swing_target[sub,2]-gait.swing_start[sub,2]>.04)&~gait.preparing[sub]
            middle[:,2]=torch.where(rising,
                torch.lerp(lifted[:,2],peak[:,2],smooth(torch.full_like(gait.time[sub],.5/.65))),
                middle[:,2])
            points.extend((lifted,middle))
        points.append(peak)
        if include_midpoints:
            points.append(.5*(peak+landing))
        points.append(landing)
        points = torch.stack(points, 1)
        point_count = points.shape[1]
        current_pitch=swing_pitch(gait,sub)[:,0]
        amplitude=(gait.options.get("swing_toe_down_rad",0.)-gait.options.get("swing_toe_up_rad",0.))
        rising=(gait.swing_target[sub,2]-gait.swing_start[sub,2]>.04)&~gait.preparing[sub]
        amplitude=amplitude*rising.to(gait.time.dtype)
        if include_midpoints:
            lifted_pitch=torch.where(spent,current_pitch,amplitude)
            progress=torch.maximum(torch.full_like(current_pitch,.5),gait.cross_elapsed[sub]/gait.options["cross_duration"])
            middle_pitch=amplitude*(1-smooth(progress/.75))
            point_pitch=torch.stack((current_pitch,lifted_pitch,middle_pitch,
                torch.zeros_like(current_pitch),torch.zeros_like(current_pitch),torch.zeros_like(current_pitch)),-1)
        else:
            point_pitch=torch.stack((current_pitch,torch.zeros_like(current_pitch),torch.zeros_like(current_pitch)),-1)
        current_yaw = gait.foot_yaw[sub, side]
        preferred = gait.swing_yaw_goal[sub]
        if candidate_values is None:
            values = torch.cat((grid[None].expand(count, -1),
                                preferred[:, None], current_yaw[:, None]), 1)
            preferred_index = values.shape[1]-2
        else:
            values = candidate_values[sub]
            preferred_index = 0
        angle_count = values.shape[1]
        yaw = gait.foot_yaw[sub, None].expand(-1, angle_count, -1).clone()
        yaw[row[:, None], torch.arange(angle_count, device=gait.device)[None],
            side[:, None]] = values
        rotations_world = context["heading"][:, None, None] @ (
            gait.flat.home_foot_rotation[None, None] @ _rotation_z(yaw))
        rotations_world=rotations_world[:,:,None].expand(-1,-1,point_count,-1,-1,-1).clone()
        ry=gait.time.new_zeros(count,point_count,3,3)
        ry[...,0,0]=ry[...,2,2]=point_pitch.cos()
        ry[...,0,2],ry[...,2,0],ry[...,1,1]=point_pitch.sin(),-point_pitch.sin(),1
        rindex=row[:,None,None]
        aindex=torch.arange(angle_count,device=gait.device)[None,:,None]
        pindex=torch.arange(point_count,device=gait.device)[None,None,:]
        sindex=side[:,None,None]
        rotations_world[rindex,aindex,pindex,sindex]=rotations_world[rindex,aindex,pindex,sindex]@ry[:,None]
        positions_world = context["soles"][:, None].expand(-1, point_count, -1, -1).clone()
        positions_world[row[:, None], torch.arange(point_count, device=gait.device)[None],
                        side[:, None]] = points
        positions_root = (context["inverse"][:, None, None] @ (
            positions_world-context["root"][:, None, None])[..., None]).squeeze(-1)
        target_position = positions_root[:, None].expand(-1, angle_count, -1, -1, -1)
        target_rotation = context["inverse"][:,None,None,None]@rotations_world
        shape = (count, angle_count, point_count, 2)
        base_seed = (gait.q[sub, None].expand(-1, point_count, -1, -1)
                     if pose_seed is None else pose_seed[sub])
        seed = base_seed[:, None].expand(-1, angle_count, -1, -1, -1)
        candidate, _ = gait.kin.solve(
            target_position.reshape(-1, 2, 3), target_rotation.reshape(-1, 2, 3, 3),
            seed.reshape(-1, 2, 6), iterations=iterations)
        position, rotation = gait.kin.forward(candidate)
        candidate = candidate.reshape(shape+(6,))
        position = position.reshape(shape+(3,))
        rotation = rotation.reshape(shape+(3, 3))
        rr = row[:, None, None]
        aa = torch.arange(angle_count, device=gait.device)[None, :, None]
        pp = torch.arange(point_count, device=gait.device)[None, None]
        ss = side[:, None, None]
        selected_q = candidate[rr, aa, pp, ss]
        position_error = (position-target_position).norm(dim=-1)[rr, aa, pp, ss]
        rotation_error = (rotation-target_rotation).abs().amax((-1, -2))[rr, aa, pp, ss]
        joint_margin = torch.minimum(selected_q-gait.kin.lower[side, None, None],
                                     gait.kin.upper[side, None, None]-selected_q).amin(-1)
        current_bias = context["actual_gravity"][row, side] / context["stiffness"][row, side]
        current_request = selected_q + current_bias[:, None, None]
        drive_margin = torch.minimum(current_request-gait.kin.lower[side, None, None],
                                     gait.kin.upper[side, None, None]-current_request).amin(-1)
        zero = gait.q.new_zeros((count*angle_count*point_count, 2, 3))
        gravity_direction = -context["inverse"][..., 2]
        gravity_direction = gravity_direction[:, None, None].expand(-1, angle_count, point_count, -1)
        future_gravity = gait.kin.support_torque(
            candidate.reshape(-1, 2, 6), zero, zero,
            gravity_direction.reshape(-1, 3)).reshape(shape+(6,))[rr, aa, pp, ss]
        future_request = selected_q + future_gravity/context["stiffness"][row, side][:, None, None]
        future_margin = torch.minimum(future_request-gait.kin.lower[side, None, None],
                                      gait.kin.upper[side, None, None]-future_request).amin(-1)
        drive_margin = torch.minimum(drive_margin, future_margin)
        selected_rotation_world = context["root_rotation"][:, None, None] @ rotation[rr, aa, pp, ss]
        selected_position_world = context["root"][:, None, None] + (
            context["root_rotation"][:, None, None] @ position[rr, aa, pp, ss, :, None]).squeeze(-1)
        moving_corners = selected_position_world[..., None, :] + (
            selected_rotation_world[..., None, :, :] @
            gait.foot_corners[side, None, None, :, :, None]).squeeze(-1)
        covered = gait.map.footprint_supported(sub[:,None].expand(-1,angle_count).reshape(-1),
            moving_corners[:,:,-1].reshape(-1,4,3),
            (landing[:,2]+gait.cfg["ground_contact_depth"])[:,None].expand(-1,angle_count).reshape(-1),
            margin=.008).reshape(count,angle_count)
        # A proven lower bound for the complete oriented footprint, rather
        # than the remaining room in one arbitrary target-centred rectangle.
        patch_margin = torch.where(covered,torch.full_like(values,.008),-float('inf'))
        fixed_rotation = context["actual_foot_rotation"][row, other]
        fixed = context["soles"][row, other, None] + (
            fixed_rotation[:, None] @ gait.foot_corners[other, :, :, None]).squeeze(-1)
        fixed = fixed[:, None, None].expand(-1, angle_count, point_count, -1, -1)
        fixed_rotation = fixed_rotation[:, None, None].expand(-1, angle_count, point_count, -1, -1)
        gap = _separation_margin(moving_corners, fixed, selected_rotation_world, fixed_rotation)
        same_level = (selected_position_world[..., 2]-context["soles"][row, other, 2][:, None, None]).abs() < .04
        collision_free = (~same_level) | (gap > .008)
        valid = ((position_error < .001) & (rotation_error < .005) &
                 (joint_margin > gait.options.get("joint_kinematic_reserve",.008)) & (drive_margin > .004) & collision_free).all(-1)
        valid &= (patch_margin > .004) & (values.abs() <= .400001)
        worst_joint, worst_drive = joint_margin.amin(-1), drive_margin.amin(-1)
        margin = torch.minimum(torch.minimum(worst_joint, worst_drive), 2*patch_margin)
        score = margin-.05*(values-current_yaw[:, None]).abs()
        # Retain the existing constant goal when it still passes every point.
        preferred_valid = valid[:, preferred_index]
        score = torch.where(valid, score, -float("inf"))
        best_score, choice = score.max(-1)
        choice = torch.where(preferred_valid, torch.full_like(choice, preferred_index), choice)
        good = torch.isfinite(best_score)
        result[start:start+count] = values[row, choice]
        feasible[start:start+count] = good
        current_solution[start:start+count] = candidate[row, choice, 0]
        diagnostics["joint_margin"][start:start+count] = worst_joint[row, choice]
        diagnostics["drive_margin"][start:start+count] = worst_drive[row, choice]
        diagnostics["patch_margin"][start:start+count] = patch_margin[row, choice]
        diagnostics["current_position_error"][start:start+count] = position_error[row, choice, 0]
        diagnostics["current_rotation_error"][start:start+count] = rotation_error[row, choice, 0]
        diagnostics["constant_candidate_count"][start:start+count] = valid.sum(-1)
        diagnostics["pose_seed"][start:start+count] = candidate[row, choice]
    return result, feasible, current_solution, diagnostics


def reset_swing_yaw_cache(gait, ids):
    if hasattr(gait, "_swing_yaw_pose_seed_valid"):
        gait._swing_yaw_pose_seed_valid[ids] = False


@torch.no_grad()
def keep_or_pick_swing_yaw(gait, ids, soles, *, current_world=None,
                           actual_gravity=None, chunk_size=16):
    """Recheck a cached constant goal cheaply; search only the invalid subset."""
    total = gait.q.shape[0]
    if not hasattr(gait, "_swing_yaw_pose_seed"):
        gait._swing_yaw_pose_seed = gait.q[:, None].expand(-1, 6, -1, -1).clone()
        gait._swing_yaw_pose_seed_valid = torch.zeros(total, dtype=torch.bool, device=gait.device)
        gait._swing_yaw_pose_seed_angle = gait.swing_yaw_goal.clone()
    cached = (gait._swing_yaw_pose_seed_valid[ids] &
              ((gait._swing_yaw_pose_seed_angle[ids]-gait.swing_yaw_goal[ids]).abs() < 1e-6))
    yaw = gait.foot_yaw[ids, gait.swing_side[ids]].clone()
    good = torch.zeros(ids.numel(), dtype=torch.bool, device=gait.device)
    command = gait.q[ids].clone()
    diagnostic = {
        name: yaw.new_full(yaw.shape, -float("inf"))
        for name in ("joint_margin", "drive_margin", "patch_margin",
                     "current_position_error", "current_rotation_error")
    }
    diagnostic["constant_candidate_count"] = torch.zeros_like(ids)
    for use_cached in (True, False):
        local = (cached if use_cached else ~good).nonzero().flatten()
        if not local.numel():
            continue
        sub = ids[local]
        result = pick_swing_yaw(
            gait, sub, soles, current_world=current_world, actual_gravity=actual_gravity,
            iterations=6 if use_cached else 30, chunk_size=chunk_size,
            candidate_values=gait.swing_yaw_goal[:, None] if use_cached else None,
            pose_seed=gait._swing_yaw_pose_seed if use_cached else None)
        chosen, feasible, target, details = result
        yaw[local], good[local], command[local] = chosen, feasible, target
        for name in diagnostic:
            diagnostic[name][local] = details[name]
        accepted = sub[feasible]
        gait._swing_yaw_pose_seed[accepted] = details["pose_seed"][feasible]
        gait._swing_yaw_pose_seed_angle[accepted] = chosen[feasible]
        gait._swing_yaw_pose_seed_valid[sub] = feasible
    return yaw, good, command, diagnostic


@torch.no_grad()
def rate_limit_drive_safe(gait, ids, desired_q, *, actual_gravity=None):
    """Bound a swing reference before gravity bias, rather than clipping later.

    Replaces only selected swing-leg joints. The other leg is left to the caller.
    Position-drive and joint-rate boxes are intersected explicitly. A false flag
    must stop/replan that swing; it never authorizes a wider joint limit.
    """
    if not ids.numel():
        return desired_q, torch.zeros(0, dtype=torch.bool, device=gait.device)
    if desired_q.shape != (ids.numel(), 2, 6):
        raise ValueError("desired_q must be the ids-by-2-by-6 selector result")
    context = _context(gait, ids, gait.foot_target, actual_gravity)
    row = torch.arange(ids.numel(), device=gait.device)
    side = gait.swing_side[ids]
    bias = context["actual_gravity"][row, side]/context["stiffness"][row, side]
    low = torch.maximum(gait.kin.lower[side]+.008, gait.kin.lower[side]+.004-bias)
    high = torch.minimum(gait.kin.upper[side]-.008, gait.kin.upper[side]-.004-bias)
    maximum_delta = gait.options["maximum_joint_rate"]*gait.env.step_dt
    low = torch.maximum(low, gait.q[ids, side]-maximum_delta)
    high = torch.minimum(high, gait.q[ids, side]+maximum_delta)
    feasible = (low <= high).all(-1)
    result = desired_q.clone()
    value = torch.maximum(torch.minimum(result[row, side], high), low)
    result[row, side] = torch.where(feasible[:, None], value, gait.q[ids, side])
    return result, feasible


def preview_swing_world(gait, ids, soles):
    """Fresh selected-foot Cartesian command, matching the existing profile."""
    world = soles.clone()
    if not ids.numel():
        return world
    moving = swing_position_local(gait,ids)[:,0]
    heading = _quaternion_matrix(gait.flat.heading[ids])
    world[ids, gait.swing_side[ids]] = gait.flat.origin[ids] + (
        heading @ moving[..., None]).squeeze(-1)
    return world


@torch.no_grad()
def prepare_swing_command(gait, ids, soles, *, actual_gravity=None):
    """Select a constant task yaw and a drive-safe, joint-rate-bounded q.

    Pressure/support references are untouched. The caller applies only the
    selected swing leg of command, holds/replans false entries before compose,
    and leaves the real state/flight clock running.
    """
    world = preview_swing_world(gait, ids, soles)
    yaw, feasible, desired, diagnostic = keep_or_pick_swing_yaw(
        gait, ids, soles, current_world=world, actual_gravity=actual_gravity)
    command, drive_feasible = rate_limit_drive_safe(
        gait, ids, desired, actual_gravity=actual_gravity)
    feasible &= drive_feasible
    diagnostic["rate_drive_feasible"] = drive_feasible
    return yaw, feasible, command, diagnostic


@torch.no_grad()
def freeze_landed_yaw(gait, ids, side):
    """Keep the physically achieved foot yaw while its load is established."""
    if not ids.numel():
        return
    root_rotation = _quaternion_matrix(gait.robot.data.root_quat_w[ids])
    actual_q = gait.robot.data.joint_pos[ids][:, gait.joint_ids].reshape(-1, 2, 6)
    _, foot_rotation = gait.kin.forward(actual_q)
    world = root_rotation[:, None] @ foot_rotation
    heading = _quaternion_matrix(gait.flat.heading[ids])
    row = torch.arange(ids.numel(), device=gait.device)
    angle = (torch.atan2(world[row, side, 1, 0], world[row, side, 0, 0]) -
             torch.atan2(heading[:, 1, 0], heading[:, 0, 0]) -
             torch.atan2(gait.flat.home_foot_rotation[side, 1, 0],
                         gait.flat.home_foot_rotation[side, 0, 0]))
    angle = torch.atan2(angle.sin(), angle.cos())
    gait.foot_yaw[ids, side] = angle
    reset_swing_yaw_cache(gait, ids)
