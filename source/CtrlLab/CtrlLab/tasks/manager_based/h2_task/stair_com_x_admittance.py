"""DRAFT: measured normal-CoP compliance only in pelvis IK X.

No simulator access, no nominal CoP/GRF/DCM mutation. A 30mm reference bound
is geometric protection, not a guarantee of IK, drive or physical feasibility.
"""
import torch


def initialize_tactile_cop_latest(tactile):
    n, device = tactile.age.shape[0], tactile.age.device
    for key in ('normal_cop_world_w', 'normal_force_measure_w', 'normal_moment_link_w',
                'filtered_normal_force_w'):
        tactile.latest[key] = tactile.age.new_zeros(n, 2, 3)
    tactile.latest['normal_cop_valid'] = torch.zeros(n, 2, device=device, dtype=torch.bool)
    tactile.latest['normal_read_fresh'] = torch.zeros(n, 2, device=device, dtype=torch.bool)
    tactile.latest['normal_measure_step'] = torch.full((n, 2), -1, device=device, dtype=torch.long)
    tactile.latest['normal_cop_plane_z_m'] = tactile.age.new_zeros(n, 2)


def stamp_tactile_normal_read(tactile, foot, filtered, bad_env, step):
    """Begin a read, not its successful completion (capacity can still fail)."""
    tactile.latest['normal_read_fresh'][:, foot] = False
    tactile.latest['filtered_normal_force_w'][:, foot] = filtered
    tactile.latest['normal_measure_step'][:, foot] = int(step)


def finish_tactile_normal_read(tactile, foot, filtered, unfiltered_fz,
        aggregate, bad_env, zero_contact, step):
    """Complete only a finite full raw aggregate, including true empty reads."""
    sign = tactile.normal_sign[foot]
    agreement = ((float(sign)*aggregate-filtered).norm(dim=-1)
                 <= 1+.02*filtered.norm(dim=-1))
    foot_match = (torch.isfinite(unfiltered_fz)
                  & ((unfiltered_fz-filtered[:,2]).abs()
                     <= 1+.02*filtered[:,2].abs()))
    valid = (~bad_env & torch.isfinite(filtered).all(-1)
             & torch.isfinite(aggregate).all(-1) & agreement & foot_match
             & ((sign != 0) | zero_contact))
    tactile.latest['normal_read_fresh'][:,foot] = valid
    tactile.latest['normal_measure_step'][:,foot] = int(step)


def finish_tactile_empty_normal_read(tactile, foot, filtered, unfiltered_fz,
        bad_env, step):
    finish_tactile_normal_read(tactile,foot,filtered,unfiltered_fz,
        torch.zeros_like(filtered),bad_env,torch.ones_like(bad_env),step)


def normal_cop_from_points(*, n, owner, point, scalar_force, normal,
        separation, filtered_force, unfiltered_fz, origin, rotation,
        mass, normal_sign, positive_control_established, invalid_env, min_separation):
    """Normal-only point/moment CoP on its measured plane, never flatheight.

    Inputs are the already fetched tactile point arrays; no additional query.
    Established SDK foot positive-control, current aggregate-force, load,
    vertical-normal and measured-plane checks are independent of map-support
    polygon/stability admission. One loaded point is an observable pressure
    point; it does not establish any two-dimensional support area.
    """
    if point.shape != normal.shape or point.ndim != 2 or point.shape[1] != 3:
        raise ValueError('Contact point/normal layout must be P by3')
    if scalar_force.shape != owner.shape or separation.shape != owner.shape:
        raise ValueError('Contact scalar and owner layouts differ')
    dtype, device = point.dtype, point.device
    force = torch.zeros(n, 3, device=device, dtype=dtype)
    moment = torch.zeros_like(force)
    weight_z = torch.zeros(n, device=device, dtype=dtype)
    height_weighted = torch.zeros_like(weight_z)
    count = torch.zeros(n, device=device, dtype=torch.long)
    low = point.new_full((n,), float('inf'))
    high = point.new_full((n,), -float('inf'))
    bad_count = torch.zeros(n, device=device, dtype=torch.long)
    if owner.numel():
        finite = (torch.isfinite(point).all(-1) & torch.isfinite(normal).all(-1)
                  & torch.isfinite(scalar_force) & torch.isfinite(separation))
        signed_force = float(normal_sign)*scalar_force[:, None]*normal
        bearing = finite & (scalar_force.abs() > 1e-6)
        good = (bearing & ((normal.norm(dim=-1)-1).abs() < .05)
                & (float(normal_sign)*normal[:, 2] > .98)
                & (signed_force[:, 2] > 0) & (separation >= min_separation)
                & (separation <= .002))
        bad_count.index_add_(0, owner, ((bearing & ~good) | ~finite).long())
        clean = torch.where(finite[:, None], signed_force, 0)
        force.index_add_(0, owner, clean)
        delta = torch.where(finite[:, None], point-origin[owner], 0)
        moment.index_add_(0, owner, torch.cross(delta, clean, dim=-1))
        valid_owner = owner[good]
        count.index_add_(0, valid_owner, torch.ones_like(valid_owner))
        weight_z.index_add_(0, valid_owner, signed_force[good, 2])
        height_weighted.index_add_(0, valid_owner, signed_force[good, 2]*point[good, 2])
        low.scatter_reduce_(0, valid_owner, point[good, 2], reduce='amin', include_self=True)
        high.scatter_reduce_(0, valid_owner, point[good, 2], reduce='amax', include_self=True)
    plane_z = height_weighted/weight_z.clamp_min(1)
    threshold = torch.maximum(torch.full_like(weight_z, 20), mass*.981)
    tolerance = 1+.02*filtered_force.norm(dim=-1)
    agreement = (force-filtered_force).norm(dim=-1) <= tolerance
    positive_match = (torch.isfinite(unfiltered_fz)
                      & ((unfiltered_fz-filtered_force[:, 2]).abs()
                         <= 1+.02*filtered_force[:, 2].abs()))
    valid = ((normal_sign != 0) & positive_control_established.bool() & ~invalid_env.bool()
             & torch.isfinite(filtered_force).all(-1) & torch.isfinite(origin).all(-1)
             & torch.isfinite(rotation).all((-1, -2))
             & agreement & positive_match & (bad_count == 0) & (count >= 1)
             & (high-low < .004) & (force[:, 2] >= threshold)
             & (weight_z >= threshold)
             & (force[:, :2].norm(dim=-1) <= .02*force[:, 2].clamp_min(0)))
    # On measured Zplane: Mx=(Cy-Oy)Fz-(Cz-Oz)Fy; My=(Cz-Oz)Fx-(Cx-Ox)Fz.
    dz = plane_z-origin[:, 2]
    cop = origin.clone()
    cop[:, 0] += (dz*force[:, 0]-moment[:, 1])/force[:, 2].clamp_min(1)
    cop[:, 1] += (moment[:, 0]+dz*force[:, 1])/force[:, 2].clamp_min(1)
    cop[:, 2] = plane_z
    valid &= torch.isfinite(cop).all(-1)
    return dict(valid=valid, cop_world=torch.where(valid[:, None], cop, 0),
                force_world=force, moment_about_link_world=moment,
                measured_plane_z=plane_z, aggregate_force_agreement=agreement,
                point_count=count, plane_spread=high-low)


def record_tactile_normal_cop(tactile, foot, p, f, normal, distance, owner,
        filtered, unfiltered_fz, origin, rotation, positive_control, bad_env, step):
    # Initial sign/foot calibration remains owned by the ORIGINAL current
    # positive-control gate, including its3point/flat requirements. Once that
    # calibration exists, observing pressure at a toe/edge is a distinct task.
    established = (tactile.first_foot_positive_control[foot] is not None
                   or bool(positive_control.any()))
    established = torch.full_like(positive_control,established,dtype=torch.bool)
    result = normal_cop_from_points(n=tactile.age.shape[0], owner=owner,
        point=p, scalar_force=f, normal=normal, separation=distance,
        filtered_force=filtered, unfiltered_fz=unfiltered_fz, origin=origin,
        rotation=rotation, mass=tactile.mass, normal_sign=tactile.normal_sign[foot],
        positive_control_established=established,
        invalid_env=bad_env | ~tactile.latest['normal_read_fresh'][:,foot],
        min_separation=tactile.min_separation)
    tactile.latest['normal_cop_world_w'][:, foot] = result['cop_world']
    tactile.latest['normal_cop_valid'][:, foot] = result['valid']
    tactile.latest['normal_force_measure_w'][:, foot] = result['force_world']
    tactile.latest['normal_moment_link_w'][:, foot] = result['moment_about_link_world']
    tactile.latest['normal_cop_plane_z_m'][:, foot] = result['measured_plane_z']
    tactile.latest['normal_measure_step'][:, foot] = int(step)


def ss_normal_cop_error_x(latest, *, ids, step, stance, origin, heading_matrix,
        reference_cop, planned_load, single_support):
    """Fresh loaded-foot normal CoP error in headingX, only unambiguous SS.

    The reference is the CURRENT FINAL physically projected reference_cop,
    never requested/unclipped CoP. All return arrays have len(ids) shape.
    """
    row = torch.arange(ids.numel(), device=ids.device)
    side = stance[ids]
    cop = latest['normal_cop_world_w'][ids, side]
    local = (heading_matrix.transpose(-1, -2)@(cop-origin)[..., None]).squeeze(-1)
    fresh = (latest['normal_measure_step'][ids] == int(step)).all(-1)
    read = latest['normal_read_fresh'][ids].all(-1)
    other_normal_norm = latest['filtered_normal_force_w'][ids,1-side].norm(dim=-1)
    valid = (single_support & fresh & read & latest['normal_cop_valid'][ids, side]
             & (planned_load[row, side] >= .999) & (other_normal_norm < 5))
    error = local[:, 0]-reference_cop[:, 0]
    valid &= torch.isfinite(error)
    return torch.where(valid, error, 0), valid


class PelvisXAdmittance:
    """Optional two-phase component: preview, select feasible offset or baseline.

    The critical-damped response is integrated exactly for held error/omega.
    Measurement invalid or non-SS phase means zero input (stable leakage),
    not reuse of stale pressure. Episode reset and explicit baseline fallback
    canonicalize the state. A fallback keeps this control counter consumed;
    no second integration is hidden in a recursive/repeated call. Component
    fallback/reset is not C2 and is not a physical-safety guarantee.
    """
    def __init__(self, n, device, *, dt, maximum_offset=.03, dtype=torch.float32):
        if dt <= 0 or not 0 < maximum_offset <= .03:
            raise ValueError('Positive actual control dt and offset bound<=30mm required')
        self.dt, self.limit = float(dt), float(maximum_offset)
        self.x = torch.zeros(n, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.x)
        self.last_step = torch.full((n,), -1, device=device, dtype=torch.long)
        self.baseline_fallback_count = torch.zeros(n, device=device, dtype=torch.long)
        self.reset_count = torch.zeros_like(self.baseline_fallback_count)
        self.new_accepted_count = torch.zeros_like(self.baseline_fallback_count)
        self.old_held_count = torch.zeros_like(self.baseline_fallback_count)
        self.last_fallback_step = torch.full_like(self.last_step, -1)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.x[ids], self.v[ids], self.last_step[ids] = 0, 0, -1
        self.baseline_fallback_count[ids], self.reset_count[ids] = 0, 0
        self.new_accepted_count[ids], self.old_held_count[ids] = 0, 0
        self.last_fallback_step[ids] = -1

    def preview(self, ids, *, step, error, measurement_valid, omega, enabled):
        if error.shape != ids.shape or omega.shape != ids.shape:
            raise ValueError('Scalar state inputs must match selected env IDs')
        if not bool(torch.isfinite(omega).all() & (omega > 0).all()):
            raise RuntimeError('Invalid actual vertical/height natural frequency')
        if bool((self.last_step[ids] > int(step)).any()):
            raise ValueError('Counter went backwards without reset')
        duplicate = self.last_step[ids] == int(step)
        fresh = enabled & measurement_valid & torch.isfinite(error)
        drive = torch.where(fresh, error, 0)
        x0, v0 = self.x[ids], self.v[ids]
        y = x0-drive
        b = v0+omega*y
        decay = torch.exp(-omega*self.dt)
        raw_x = drive+(y+b*self.dt)*decay
        raw_v = (v0-omega*b*self.dt)*decay
        x = raw_x.clamp(-self.limit, self.limit)
        boundary = x != raw_x
        outward = ((x >= self.limit) & (raw_v > 0)) | ((x <= -self.limit) & (raw_v < 0))
        v = torch.where(outward, 0, raw_v)
        x, v = torch.where(duplicate, x0, x), torch.where(duplicate, v0, v)
        return dict(x=x, v=v, acceleration_formula=omega.square()*(drive-x)-2*omega*v,
                    raw_x=raw_x, raw_v=raw_v, bound_active=boundary,
                    outward_velocity_projected=outward, input_valid=fresh,
                    input_error=drive, omega=omega, duplicate=duplicate)

    def commit(self, ids, *, step, proposal, feasible, baseline_fallback=None):
        """Commit a permitted offset, or disable this optional component.

        Both offset choices must pass their extra checks. If neither passes,
        caller publishes the ORIGINAL zero-offset rate-limited baseline. It is
        not promoted to pass the extra diagnostics. Canonical x/v=0 does not
        relax the original drive/rate/clamp/contact/acceptance gates.

        Per-type counters count once per env/control counter. A same-counter
        fallback after a changed original state is still canonicalized, but a
        duplicate fallback cannot increment the event/reset count again.
        """
        if baseline_fallback is None:
            baseline_fallback = torch.zeros_like(feasible)
        if bool((feasible & baseline_fallback).any()):
            raise ValueError('Accepted offset and baseline fallback are exclusive')
        new = ~proposal['duplicate']
        before_x, before_v = self.x[ids].clone(), self.v[ids].clone()
        x = torch.where(feasible, proposal['x'], self.x[ids])
        v = torch.where(feasible, proposal['v'], torch.zeros_like(self.v[ids]))
        self.x[ids] = torch.where(new, x, self.x[ids])
        self.v[ids] = torch.where(new, v, self.v[ids])
        self.x[ids] = torch.where(baseline_fallback, 0, self.x[ids])
        self.v[ids] = torch.where(baseline_fallback, 0, self.v[ids])
        event = baseline_fallback & (self.last_fallback_step[ids] != int(step))
        self.baseline_fallback_count[ids] += event.long()
        self.reset_count[ids] += event.long()
        self.new_accepted_count[ids] += (new & feasible).long()
        self.old_held_count[ids] += (new & ~feasible & ~baseline_fallback).long()
        self.last_fallback_step[ids] = torch.where(event, int(step), self.last_fallback_step[ids])
        self.last_step[ids] = int(step)
        return dict(x=self.x[ids].clone(), v=self.v[ids].clone(),
                    held_for_feasibility=new & ~feasible & ~baseline_fallback,
                    baseline_fallback=baseline_fallback,
                    reset_this_call=event,
                    previous_x=before_x, previous_v=before_v,
                    bound_active=proposal['bound_active'])


def solve_ik_x_with_feasibility(kin, *, position_zero, orientation, inverse,
        root_blend, seed, proposal_x, previous_x, fresh_contact, measured_q,
        moving_mask, moving_command, moving_error, feedforward, stiffness,
        effort_limit, lower, upper, dt, maximum_rate, joint_reserve=.002,
        preload_reserve=.004, maximum_command_error=.002,
        maximum_command_rotation_error=.005):
    """Candidate offset, feasible old offset, or original zero-offset baseline.

    Only root-IKX is changed: position(x)=position(0)-Rinv*ex*(1-blend)*x.
    Nominal foot world targets, pressure targets and the moving-foot branch
    remain unchanged. Checks concern the actual rate-limited command. Residual
    policy is applied later, so this is preload feasibility, not a complete
    future motor/pose guarantee. If both offsets fail, re-evaluate zero and
    return its original command even if that baseline does not pass the EXTRA
    diagnostics. No new HOLD or physical-safety claim is made. No original
    joint/effort/rate/contact/acceptance range is enlarged.
    """
    k = stiffness.clamp_min(1e-9)
    affected = (root_blend < 1-1e-6) & ~fresh_contact
    def evaluate(offset):
        correction = torch.zeros_like(position_zero)
        correction[..., 0] = (1-root_blend)*offset[:, None]
        position = position_zero-(inverse@correction[..., None]).squeeze(-1)
        target, error = kin.solve(position, orientation, seed, iterations=6)
        target = torch.where(moving_mask[..., None], moving_command, target)
        error = torch.where(moving_mask, moving_error, error)
        delta = (target-seed).clamp(-maximum_rate*dt, maximum_rate*dt)
        command = torch.where(fresh_contact[..., None], measured_q, seed+delta)
        command_position, command_rotation = kin.forward(command)
        command_error = (command_position-position).norm(dim=-1)
        rotation_error = (command_rotation-orientation).norm(dim=(-2,-1))
        joint_margin = torch.minimum(command-lower, upper-command).amin(-1)
        preload = command+feedforward/k
        preload_margin = torch.minimum(preload-lower, upper-preload).amin(-1)
        checks = ((joint_margin >= joint_reserve) & (preload_margin >= preload_reserve)
                  & (command_error <= maximum_command_error)
                  & (rotation_error <= maximum_command_rotation_error)
                  & (feedforward.abs() <= effort_limit+1e-5).all(-1)
                  & torch.isfinite(command).all(-1) & torch.isfinite(error))
        feasible = (checks | ~affected).all(-1)
        # Diagnostic bits: joint1/preload2/position4/rotation8/FFcap16/finite32.
        # These describe the extra component tests, never baseline acceptance.
        bad = ((joint_margin < joint_reserve), (preload_margin < preload_reserve),
               (command_error > maximum_command_error),
               (rotation_error > maximum_command_rotation_error),
               (feedforward.abs() > effort_limit+1e-5).any(-1),
               ~torch.isfinite(command).all(-1) | ~torch.isfinite(error))
        bits = torch.zeros_like(feasible, dtype=torch.long)
        for bit, failed in enumerate(bad):
            bits |= ((failed & affected).any(-1).long() << bit)
        return dict(target=target, command=command, error=error, position=position, feasible=feasible,
                    failure_bits=bits,
                    command_joint_margin=joint_margin,
                    command_preload_margin=preload_margin,
                    command_error=command_error,command_rotation_error=rotation_error)
    candidate = evaluate(proposal_x)
    old = evaluate(previous_x)
    accept = candidate['feasible']
    held = ~accept & old['feasible']
    fallback = ~accept & ~old['feasible']
    # Recompute the original baseline, rather than keep a tiny nonzero offset
    # or classify it using equality to floating-point zero.
    baseline = evaluate(torch.zeros_like(previous_x)) if bool(fallback.any()) else old
    def selected(name):
        value = candidate[name]
        mask_new = accept.reshape(accept.shape+(1,)*(value.ndim-1))
        mask_old = held.reshape(held.shape+(1,)*(value.ndim-1))
        return torch.where(mask_new, value, torch.where(mask_old, old[name], baseline[name]))
    return dict(target=selected('target'), rate_limited_command=selected('command'),
                error=selected('error'), position=selected('position'),
                accept_new=accept, held_old_offset=held,
                baseline_only=fallback, baseline_fallback=fallback,
                baseline_evaluated=fallback,
                baseline_feasible=fallback & baseline['feasible'],
                previous_offset_feasible=old['feasible'],
                candidate_failure_bits=candidate['failure_bits'],
                previous_offset_failure_bits=old['failure_bits'],
                baseline_failure_bits=torch.where(fallback, baseline['failure_bits'], 0),
                selection_branch=torch.where(accept, 0, torch.where(held, 1, 2)),
                candidate_joint_margin=candidate['command_joint_margin'],
                candidate_preload_margin=candidate['command_preload_margin'],
                candidate_command_error=candidate['command_error'],
                candidate_command_rotation_error=candidate['command_rotation_error'],
                command_rotation_error=selected('command_rotation_error'))
