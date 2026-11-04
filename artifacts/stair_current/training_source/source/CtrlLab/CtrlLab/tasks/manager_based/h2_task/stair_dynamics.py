"""Swing-only native inertia/C compensation around the issued stair reference.

No gravity/contact term is generated here. The acceleration approximation holds
the current root actor pose, the other leg and the selected foot yaw fixed.
Native C is evaluated at the actual full articulation velocity. Root and other
joint acceleration coupling are omitted, rather than confused with COM motion.
"""

import torch
from .stair_trajectory import swing_position_local,swing_pitch


def _matrix(quaternion):
    q = quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def _smooth(value):
    u = value.clamp(0, 1)
    return u**3*(10-15*u+6*u*u)


def _full_leg(value, gait, ids, side, name):
    if value.shape not in ((gait.q.shape[0], 2, 6), (gait.q.shape[0], 12)):
        raise ValueError(name+" must be full N-by-2-by-6 or N-by-12")
    return value.reshape(gait.q.shape[0], 2, 6)[ids, side]


def _advance_flag(gait, value, name):
    value = getattr(gait, name) if value is None else value
    if value.shape != (gait.q.shape[0],) or value.dtype != torch.bool:
        raise ValueError(name+" must be a full N bool tensor")
    return value


@torch.no_grad()
def preview_swing_trajectory(gait, ids, *, offsets, cross_advancing=None,
                             descent_advancing=None):
    """Pure Cartesian preview with the real current gate state held locally.

    offsets contains local time offsets, normally [-.01,0,.01]. No real clock,
    map, contact state or root pose is changed. Inputs/outputs use root actor and
    world frames, never root COM coordinates. The center reproduces the current
    SWING profile, including its toe pitch if configured.
    """
    cross_flag = _advance_flag(gait, cross_advancing, "cross_advancing")[ids]
    descent_flag = _advance_flag(gait, descent_advancing, "descent_advancing")[ids]
    offset = gait.q.new_tensor(offsets)
    if offset.ndim != 1:
        raise ValueError("offsets must be one-dimensional")
    time = (gait.time[ids, None]+offset).clamp_min(0)
    start, end = gait.swing_start[ids], gait.swing_target[ids]
    local = swing_position_local(gait,ids,offsets,
        cross_advancing=cross_advancing,descent_advancing=descent_advancing)
    heading = _matrix(gait.flat.heading[ids])
    world = gait.flat.origin[ids, None]+(heading[:, None]@local[..., None]).squeeze(-1)
    side = gait.swing_side[ids]
    yaw = gait.foot_yaw[ids, side]
    rz = gait.q.new_zeros((ids.numel(), 3, 3))
    rz[:, 0, 0] = rz[:, 1, 1] = yaw.cos()
    rz[:, 0, 1], rz[:, 1, 0], rz[:, 2, 2] = -yaw.sin(), yaw.sin(), 1
    pitch = swing_pitch(gait,ids,offsets,cross_advancing=cross_advancing)
    ry = gait.q.new_zeros((ids.numel(), offset.numel(), 3, 3))
    ry[..., 0, 0] = ry[..., 2, 2] = pitch.cos()
    ry[..., 0, 2], ry[..., 2, 0], ry[..., 1, 1] = pitch.sin(), -pitch.sin(), 1
    orientation = (heading@gait.flat.home_foot_rotation[side]@rz)[:, None]@ry
    return world, orientation


@torch.no_grad()
def native_swing_terms(gait, ids, *, mass_matrix=None, coriolis=None):
    """Select current Mss and Cs using actual articulation joint IDs plus six.

    The optional overrides are full native N-by-(J+6)-by-(J+6) and N-by-(J+6)
    tensors, for explicit saved-data CPU replay. No G API is called. Missing or
    wrong native layouts fail explicitly instead of silently using old M/zero C.
    """
    view = gait.robot.root_physx_view
    matrix = view.get_generalized_mass_matrices() if mass_matrix is None else mass_matrix
    force = (view.get_coriolis_and_centrifugal_compensation_forces()
             if coriolis is None else coriolis)
    total, joints = gait.robot.data.joint_pos.shape
    size = joints+6
    if not isinstance(matrix, torch.Tensor) or matrix.shape != (total, size, size):
        raise ValueError("H2 requires native floating-base M with shape N-by-(J+6)-by-(J+6)")
    if not isinstance(force, torch.Tensor) or force.shape != (total, size):
        raise ValueError("H2 requires native floating-base C with shape N-by-(J+6)")
    side = gait.swing_side[ids]
    joint_ids = torch.as_tensor(gait.joint_ids, device=matrix.device, dtype=torch.long)
    if joint_ids.numel() != 12 or joint_ids.unique().numel() != 12:
        raise ValueError("gait.joint_ids must contain the twelve unique articulation leg joint IDs")
    if bool(((joint_ids < 0) | (joint_ids >= joints)).any()):
        raise ValueError("leg joint ID is outside the actual articulation")
    selected = joint_ids.reshape(2, 6)[side]+6
    block = matrix[ids[:, None, None], selected[:, :, None], selected[:, None, :]]
    c = force[ids[:, None], selected]
    block, c = block.to(gait.q), c.to(gait.q)
    if not bool(torch.isfinite(block).all() & torch.isfinite(c).all()):
        raise RuntimeError("Non-finite native swing mass/C data")
    return block, c


@torch.no_grad()
def compute_swing_dynamics(gait, ids, *, preview_dt=.01, physics_dt=.005,
                           cross_advancing=None, descent_advancing=None,
                           static_torque=None, mass_matrix=None, coriolis=None,
                           ik_iterations=12, position_tolerance=.001):
    """Return ids-by-6 moving-leg derivatives and a separate dynamic increment.

    Call after this frame's gait.q, gait.qdot and gait.foot_target are assigned.
    qdot_ref preserves the issued reference (bounded to its existing <=8 rate).
    qdot_preview is the symmetric local derivative; qddot_ref uses that curve
    anchored at the issued q, only if the center FK matches the issued target
    and all preview poses/rates/position-preload requests are feasible.

    torque_raw = Mss*qddot_preview + native Cs; gravity/contact are not added.
    torque_applied is zero on skipped entries and clipped against BOTH the
    original native (static+dynamic) torque budget and original velocity target
    limits. velocity_bias=torque_applied/(D+K*physics_dt) is a conservative
    injection convention, not a claim of exact TGS/continuous-PD equivalence.
    The full native drive force cap remains necessary for feedback torque.

    This function never writes gait.q/qdot, clocks, static FF or another leg.
    """
    if preview_dt <= 0 or physics_dt <= 0:
        raise ValueError("preview_dt and physics_dt must be positive")
    count = ids.numel()
    empty = gait.q.new_zeros((count, 6))
    if not count:
        return {name: empty.clone() for name in (
            "qdot_ref", "qdot_preview", "qddot_preview", "qddot_ref", "inertia_raw",
            "coriolis_raw", "torque_raw", "torque_applied", "velocity_bias", "velocity_target")
        } | {"valid": torch.zeros(0, dtype=torch.bool, device=gait.q.device), "diagnostics": {}}
    side = gait.swing_side[ids]
    row = torch.arange(count, device=gait.q.device)
    issued = gait.q[ids]
    issued_velocity = gait.qdot[ids, side]
    rate = min(float(gait.options["maximum_joint_rate"]), 8.)
    qdot_ref = issued_velocity.clamp(-rate, rate)
    world, world_rotation = preview_swing_trajectory(
        gait, ids, offsets=(-preview_dt, 0., preview_dt),
        cross_advancing=cross_advancing, descent_advancing=descent_advancing)
    root_position = gait.robot.data.root_pos_w[ids]
    root_rotation = _matrix(gait.robot.data.root_quat_w[ids])
    inverse = root_rotation.transpose(-1, -2)
    target_position = (inverse[:, None]@(world-root_position[:, None])[..., None]).squeeze(-1)
    target_rotation = inverse[:, None]@world_rotation
    current_position, current_rotation = gait.kin.forward(issued)
    position = current_position[:, None].expand(-1, 3, -1, -1).clone()
    orientation = current_rotation[:, None].expand(-1, 3, -1, -1, -1).clone()
    position[row[:, None], torch.arange(3, device=gait.q.device)[None], side[:, None]] = target_position
    orientation[row[:, None], torch.arange(3, device=gait.q.device)[None], side[:, None]] = target_rotation
    seed = issued[:, None].expand(-1, 3, -1, -1)
    solution, _ = gait.kin.solve(position.reshape(-1, 2, 3), orientation.reshape(-1, 2, 3, 3),
                                 seed.reshape(-1, 2, 6), iterations=ik_iterations)
    solved_position, solved_rotation = gait.kin.forward(solution)
    solution = solution.reshape(count, 3, 2, 6)
    moving = solution[row[:, None], torch.arange(3, device=gait.q.device)[None], side[:, None]]
    # Remove solver's tiny common center residual before second differentiation.
    # The local curve is anchored at the actual issued q, never an unissued pose.
    difference_minus = moving[:, 0]-moving[:, 1]
    difference_plus = moving[:, 2]-moving[:, 1]
    qdot_raw = (difference_plus-difference_minus)/(2*preview_dt)
    qddot_raw = (difference_plus+difference_minus)/preview_dt**2
    one_side_rate = torch.maximum(difference_minus.abs(), difference_plus.abs()).amax(-1)/preview_dt
    qdot_preview = qdot_raw.clamp(-rate, rate)
    position_error = (solved_position.reshape(count, 3, 2, 3)-position).norm(dim=-1)
    position_error = position_error[row[:, None], torch.arange(3, device=gait.q.device)[None], side[:, None]]
    rotation_error = (solved_rotation.reshape(count, 3, 2, 3, 3)-orientation).abs().amax((-1, -2))
    rotation_error = rotation_error[row[:, None], torch.arange(3, device=gait.q.device)[None], side[:, None]]
    issued_world = root_position+(root_rotation@current_position[row, side, :, None]).squeeze(-1)
    issued_world_rotation = root_rotation@current_rotation[row, side]
    command_error = (issued_world-gait.foot_target[ids, side]).norm(dim=-1)
    profile_error = (world[:, 1]-gait.foot_target[ids, side]).norm(dim=-1)
    command_rotation_error = (issued_world_rotation-world_rotation[:, 1]).abs().amax((-1, -2))
    center_joint_error = (moving[:, 1]-issued[row, side]).abs().amax(-1)
    static = _full_leg(gait.feedforward_torque if static_torque is None else static_torque,
                       gait, ids, side, "static_torque")
    stiffness = gait.robot.data.joint_stiffness[ids][:, gait.joint_ids].reshape(count, 2, 6)[row, side]
    damping = gait.robot.data.joint_damping[ids][:, gait.joint_ids].reshape(count, 2, 6)[row, side]
    native_limit = gait.robot.data.joint_effort_limits[ids][:, gait.joint_ids].reshape(count, 2, 6)[row, side]
    speed_limit = gait.robot.data.joint_vel_limits[ids][:, gait.joint_ids].reshape(count, 2, 6)[row, side]
    speed_limit = speed_limit.clamp_max(30.)
    if not bool(torch.isfinite(static).all() & torch.isfinite(stiffness).all()
                & torch.isfinite(damping).all() & torch.isfinite(native_limit).all()
                & torch.isfinite(speed_limit).all()):
        raise RuntimeError("Non-finite native swing drive parameters")
    if not bool((stiffness > 0).all() & (damping > 0).all()
                & (native_limit > 0).all() & (speed_limit > 0).all()):
        raise ValueError("Native swing K/D/effort/speed limits must be positive")
    anchored = issued[row, side, None]+moving-moving[:, 1, None]
    position_request = anchored+static[:, None]/stiffness[:, None]
    preview_drive_ok = ((position_request >= gait.kin.lower[side, None]+.0001)
                        & (position_request <= gait.kin.upper[side, None]-.0001)).all((1, 2))
    finite = torch.isfinite(qdot_raw).all(-1) & torch.isfinite(qddot_raw).all(-1)
    boundary = gait.time[ids] < preview_dt-1e-6
    rate_ok = (one_side_rate <= rate+1e-4) & (issued_velocity.abs().amax(-1) <= rate+1e-4)
    matched = ((command_error < position_tolerance) & (profile_error < position_tolerance)
               & (command_rotation_error < .005) & (center_joint_error < .001))
    solved = (position_error.amax(-1) < position_tolerance) & (rotation_error.amax(-1) < .005)
    static_budget_ok = (static.abs() <= native_limit+1e-4).all(-1)
    active = (gait.state[ids] == gait.SWING) & ~gait.approach[ids]
    valid = active & ~boundary & finite & rate_ok & matched & solved & preview_drive_ok & static_budget_ok
    block, c = native_swing_terms(gait, ids, mass_matrix=mass_matrix, coriolis=coriolis)
    inertia = (block@qddot_raw[..., None]).squeeze(-1)
    raw = inertia+c
    if not bool(torch.isfinite(raw).all()):
        raise RuntimeError("Non-finite swing dynamic torque")
    denominator = damping+stiffness*physics_dt
    low = torch.maximum(-native_limit-static, (-speed_limit-qdot_ref)*denominator)
    high = torch.minimum(native_limit-static, (speed_limit-qdot_ref)*denominator)
    valid &= (low <= high).all(-1) & (qdot_ref.abs() <= speed_limit).all(-1)
    applied = torch.maximum(torch.minimum(raw, high), low)
    applied = torch.where(valid[:, None], applied, 0)
    velocity_bias = applied/denominator
    velocity_target = qdot_ref+velocity_bias
    qddot_ref = torch.where(valid[:, None], qddot_raw, 0)
    return dict(qdot_ref=qdot_ref, qdot_preview=qdot_preview, qddot_preview=qddot_raw,
        qddot_ref=qddot_ref, inertia_raw=inertia, coriolis_raw=c, torque_raw=raw,
        torque_applied=applied, velocity_bias=velocity_bias, velocity_target=velocity_target,
        valid=valid, diagnostics=dict(
            active=active, boundary_skipped=boundary, rate_ok=rate_ok, center_matched=matched,
            preview_solved=solved, preview_position_drive_ok=preview_drive_ok,
            static_budget_ok=static_budget_ok, command_FK_error_m=command_error,
            profile_center_error_m=profile_error, center_joint_error_rad=center_joint_error,
            preview_IK_error_m=position_error.amax(-1), preview_rotation_error=rotation_error.amax(-1),
            one_side_preview_rate_rad_s=one_side_rate,
            issued_minus_preview_qdot_rad_s=issued_velocity-qdot_raw,
            native_mass_diagonal=block.diagonal(dim1=-2, dim2=-1),
            torque_budget_clipped=((raw+static).abs() > native_limit).any(-1),
            velocity_budget_clipped=((qdot_ref+raw/denominator).abs() > speed_limit).any(-1),
            applied_dynamic_clipped=(applied-raw).abs().amax(-1) > .001,
            total_static_plus_dynamic=static+applied,
            native_effort_limit=native_limit, native_velocity_limit=speed_limit))
