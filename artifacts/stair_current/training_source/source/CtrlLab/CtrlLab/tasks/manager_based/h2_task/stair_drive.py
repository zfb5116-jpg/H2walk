"""Original delayed support commands with one shared native/force budget.

Nominal gains stay available to planning. Runtime native gains and drive caps
are weighted only after the received role; direct force shares the original
motor limit. Ordinary posture and boundary recovery remain implicit native PD.
The caller must verify ordinary drive-only cap semantics and native readback.
"""

import torch


def prepare_support_effort_command(gait):
    """Prepare a force/role packet; ordinary IK position/velocity stay intact."""
    k = gait.robot.data.joint_stiffness[:, gait.joint_ids].reshape(gait.q.shape)
    limits = gait.robot.data.joint_effort_limits[:, gait.joint_ids].reshape(gait.q.shape)
    weight = torch.zeros_like(gait.time)
    weight[(~gait.approach) & (gait.state == gait.SWING)] = 1
    landing = (~gait.approach) & (gait.state == gait.LAND)
    u = (gait.time/.20).clamp(0, 1)
    weight[landing] = (1-u**3*(10-15*u+6*u*u))[landing]
    gait.support_drive_role.zero_()
    row = torch.arange(gait.n, device=gait.device)
    for axis in gait.options['support_effort_priority_axes']:
        gait.support_drive_role[row, gait.stance, axis] = weight.clamp(0,1)
    requested = gait.feedforward_torque+k*gait.processed_residual.reshape(gait.q.shape)
    gait.support_effort_requested[:] = requested
    gait.support_effort_target[:] = requested.clamp(-limits, limits)
    gait.support_effort_clipped[:] = ((requested != gait.support_effort_target)
                                      & (gait.support_drive_role > 0))


def received_support_budget_ports(latch, position, velocity, legacy_position,
        legacy_velocity, legacy_effort, priority_effort, received_role,
        nominal_stiffness, nominal_damping, effort_limit, soft_lower, soft_upper,
        speed_limit, *, ordinary_drive_only, physics_dt=.005,
        guard_margin=.002, release_hysteresis=.002, recovery_extra=.002, validate=True):
    """Return a same-received-frame implicit/external split without mutation.

    All tensors broadcast in the selected actuator group's own joint order.
    latch is {-1,0,+1}: lower guard / ordinary / upper guard. Fresh guards are
    enabled by received_role>0; an existing guard survives role0 until released.
    Thus normal role0/approach ports remain exact and cannot initiate a new guard.

    soft_lower/upper are ORIGINAL soft bounds. Guard is2mrad inside them;
    release is a further2mrad inside with strictly inward velocity; the implicit
    recovery target is another2mrad inside so its fixed point crosses release.
    Defaults implement the agreed2/4/6mrad geometry, not a gain sweep.

    Legacy external effort must be zero on these selected axes: this interface
    has no budget for an extra legacy force port and will explicitly reject it.
    Both mode and priority effort must already have traversed the original lag.
    No delay, gain-cache, contact, clock or latch storage is changed here.
    """
    if not ordinary_drive_only:
        raise ValueError('Require verified ordinary independent drive force cap')
    if physics_dt <= 0 or min(guard_margin, release_hysteresis, recovery_extra) <= 0:
        raise ValueError('Require positive actual dt and guard geometry')
    values = torch.broadcast_tensors(position, velocity, legacy_position,
        legacy_velocity, legacy_effort, priority_effort, received_role,
        nominal_stiffness, nominal_damping, effort_limit, soft_lower, soft_upper,
        speed_limit, latch)
    (q, v, oldq, oldv, olde, wanted, role, k, d, limit, lower, upper,
     speed, old_latch) = values
    if validate:
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError('Non-finite received state, command or fixed parameter')
        reserve = guard_margin+release_hysteresis+recovery_extra
        invalid = ((role < 0) | (role > 1) | (k <= 0) | (d <= 0)
            | (limit <= 0) | (speed <= 0) | (upper-lower <= 2*reserve)
            | ((old_latch != -1) & (old_latch != 0) & (old_latch != 1)))
        if bool(invalid.any()):
            raise ValueError('Invalid role/latch/nominal gain/limit/soft interval')
        if bool((olde != 0).any()):
            raise ValueError('Nonzero legacy external effort has no allocated budget')
        if bool((oldv.abs() > speed+1e-6).any()):
            raise ValueError('Ordinary delayed velocity target exceeds original speed limit')

    lo, hi = lower+guard_margin, upper-guard_margin
    release_lo, release_hi = lo+release_hysteresis, hi-release_hysteresis
    recovery_lo, recovery_hi = release_lo+recovery_extra, release_hi-recovery_extra
    predicted = q+physics_dt*v
    enabled = (role > 0) | (old_latch != 0)
    # If a physically extreme step spans both boundaries, current violated side
    # takes precedence over the opposite predicted side, without changing q/qd.
    trigger = torch.where(q >= hi, 1, torch.where(q <= lo, -1,
        torch.where(predicted >= hi, 1, torch.where(predicted <= lo, -1, 0))))
    trigger = torch.where(enabled, trigger, 0)
    release = (((old_latch == 1) & (q <= release_hi) & (v < 0))
        | ((old_latch == -1) & (q >= release_lo) & (v > 0)))
    retained = torch.where(release, 0, old_latch)
    next_latch = torch.where(trigger != 0, trigger, retained).to(torch.int8)
    guard = next_latch != 0
    effective_role = torch.where(guard, torch.zeros_like(role), role)
    ordinary_fraction = 1-effective_role
    native_k, native_d = ordinary_fraction*k, ordinary_fraction*d
    native_cap = ordinary_fraction*limit
    external_budget = effective_role*limit
    # Allocate at most the exact remaining represented budget. Mixed cases get
    # one representable step of conservative rounding, never an expanded cap.
    mixed = (effective_role > 0) & (effective_role < 1)
    remaining = (limit-native_cap).clamp_min(0)
    conservative = torch.nextafter(remaining, torch.zeros_like(remaining))
    external_budget = torch.minimum(external_budget,
        torch.where(mixed, conservative, remaining))
    raw_external = effective_role*wanted
    external = raw_external.clamp(-external_budget, external_budget)
    target = torch.where(next_latch == 1, recovery_hi, recovery_lo)
    qout = torch.where(guard, target, oldq)
    vout = torch.where(guard, torch.zeros_like(oldv), oldv)
    motor_low, motor_high = external-native_cap, external+native_cap
    budget = native_cap+external.abs()
    return dict(effective_role=effective_role, native_stiffness=native_k,
        native_damping=native_d, native_drive_cap=native_cap,
        external_effort=external, position=qout, velocity=vout,
        next_latch=next_latch, guard_active=guard,
        guard_trigger=trigger.to(torch.int8), guard_released=release & ~guard,
        guard_lower=lo, guard_upper=hi, release_lower=release_lo,
        release_upper=release_hi, recovery_lower=recovery_lo,
        recovery_upper=recovery_hi, predicted_position=predicted,
        received_role=role, external_budget=external_budget,
        unclipped_external=raw_external, external_clipped=external != raw_external,
        worst_case_motor_effort_lower=motor_low,
        worst_case_motor_effort_upper=motor_high,
        worst_case_absolute_motor_budget=budget,
        original_effort_limit=limit, original_speed_limit=speed,
        budget_satisfied=(budget <= limit),
        guard_target_inside_release=((next_latch != 1) | (qout < release_hi))
            & ((next_latch != -1) | (qout > release_lo)),
        ordinary_drive_semantic_precondition=True)
