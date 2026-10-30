"""Contact-gated LIPM walking reference and USD-leg inverse kinematics.

This is an explicit physical controller, not an unobserved clock reward for
PPO. The policy remains a bounded local residual with its original inputs.
"""

import torch
from isaaclab.utils.math import (matrix_from_quat, quat_apply, quat_apply_inverse, yaw_quat,
                                quat_mul, quat_conjugate)

from .leg_kinematics import H2LegKinematics


def smooth(value):
    value = value.clamp(0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def lipm(position, velocity, cop, omega, duration):
    c, s = torch.cosh(omega * duration), torch.sinh(omega * duration)
    return (cop + (position - cop) * c + velocity / omega * s,
            (position - cop) * omega * s + velocity * c)


def two_cops(position, velocity, target_position, target_velocity, omega, duration):
    """Two constant CoP intervals solve both terminal COM position and speed."""
    c, s = torch.cosh(omega * duration * 0.5), torch.sinh(omega * duration * 0.5)
    b0, b1 = 1.0 - c, -omega * s
    m00, m01 = c * b0 + s / omega * b1, b0
    m10, m11 = omega * s * b0 + c * b1, b1
    free_position, free_velocity = lipm(position, velocity, torch.zeros_like(position), omega, duration)
    r0, r1 = target_position - free_position, target_velocity - free_velocity
    determinant = m00 * m11 - m01 * m10
    return ((r0 * m11 - m01 * r1) / determinant,
            (m00 * r1 - r0 * m10) / determinant)


def landing_center(position, velocity, next_velocity, omega, duration, foot_rotation):
    """Place a sole so both the current DCM and next CoP have usable margin."""
    c, s = torch.cosh(omega * duration), torch.sinh(omega * duration)
    capture = position + velocity / omega
    next_cop = position + (velocity * c - next_velocity) / (omega * s)
    inverse = foot_rotation.transpose(-1, -2)
    capture_local = (inverse @ capture[..., None]).squeeze(-1)
    cop_local = (inverse @ next_cop[..., None]).squeeze(-1)
    cop_bound = torch.tensor([0.06, 0.025], device=position.device, dtype=position.dtype)
    capture_bound = torch.tensor([0.105, 0.030], device=position.device, dtype=position.dtype)
    lower = torch.maximum(cop_local - cop_bound, capture_local - capture_bound)
    upper = torch.minimum(cop_local + cop_bound, capture_local + capture_bound)
    center = (foot_rotation @ ((lower + upper) * .5)[..., None]).squeeze(-1)
    return center, (upper - lower).amin(-1) * .5


class ContactWalkingReference:
    SETTLE, LOWER, PREPARE, WALK, HOLD = range(5)

    def __init__(self, env, robot, joint_ids, cfg):
        self.env, self.robot, self.joint_ids, self.cfg = env, robot, joint_ids, cfg
        self.device, self.n = env.device, env.num_envs
        self.kin = H2LegKinematics(self.device)
        self.home_foot_rotation = self.kin.forward(self.kin.home[None])[1][0]
        self.sensor = env.scene["contact_forces"]
        self.sensor_feet = self.sensor.find_bodies(env.cfg.feet_body_names, preserve_order=True)[0]
        self.feet = robot.find_bodies(env.cfg.feet_body_names, preserve_order=True)[0]
        self.masses = robot.root_physx_view.get_masses().to(self.device)
        self.total_mass = self.masses.sum(-1)
        self.state = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.time = torch.zeros(self.n, device=self.device)
        self.stance = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.steps = torch.zeros_like(self.stance)
        self.failure = torch.zeros_like(self.stance)
        self.origin = torch.zeros(self.n, 3, device=self.device)
        self.heading = torch.zeros(self.n, 4, device=self.device)
        self.heading[:, 0] = 1.0
        self.anchors = torch.zeros(self.n, 2, 3, device=self.device)
        self.foot_rotation = torch.eye(3, device=self.device).expand(self.n, 2, 3, 3).clone()
        self.height = torch.ones(self.n, device=self.device)
        self.width = torch.full((self.n,), 0.105, device=self.device)
        self.segment_duration = torch.full((self.n,), cfg["step_period"], device=self.device)
        self.segment_start = torch.zeros(self.n, 2, device=self.device)
        self.segment_velocity = torch.zeros_like(self.segment_start)
        self.cop0 = torch.zeros_like(self.segment_start)
        self.cop1 = torch.zeros_like(self.segment_start)
        self.lower_start = torch.zeros(self.n, 3, device=self.device)
        self.swing_start = torch.zeros(self.n, 3, device=self.device)
        self.swing_target = torch.zeros_like(self.swing_start)
        self.swing_time = torch.full((self.n,), -1.0, device=self.device)
        self.air_time = torch.zeros(self.n, device=self.device)
        self.seen_air = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.reference_com = torch.zeros(self.n, 3, device=self.device)
        self.reference_velocity = torch.zeros_like(self.reference_com)
        self.load = torch.full((self.n, 2), 0.5, device=self.device)
        self.q = robot.data.default_joint_pos[:, joint_ids].clone().reshape(self.n, 2, 6)
        self.qdot = torch.zeros_like(self.q)
        self.ik_error = torch.zeros(self.n, 2, device=self.device)
        self.support_fraction = torch.zeros(self.n, device=self.device)
        self.capture_error = torch.zeros(self.n, device=self.device)
        self.actual_com = torch.zeros(self.n, 3, device=self.device)
        self.actual_velocity = torch.zeros_like(self.actual_com)
        self.reference_cop = torch.zeros(self.n, 2, device=self.device)
        self.contact_points = torch.zeros(self.n, 2, 3, device=self.device)
        self.feedforward_torque = torch.zeros(self.n, 2, 6, device=self.device)
        self.plan_violation = torch.zeros(self.n, device=self.device)
        self.hold_capture = torch.zeros(self.n, 2, device=self.device)
        self.foot_target = torch.zeros(self.n, 2, 3, device=self.device)
        self.touchdown_count = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.touchdown_anchor = torch.zeros(self.n, 3, device=self.device)
        self.touchdown_locked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.landing_margin = torch.zeros(self.n, device=self.device)
        self.legacy_preload = torch.tensor([env.cfg.stance_preload_torques[n] for n in self.kin.joint_names],
                                           device=self.device).reshape(1, 2, 6)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.state[ids] = self.SETTLE
        for value in (self.time, self.steps, self.failure, self.air_time):
            value[ids] = 0
        self.stance[ids] = 0
        self.swing_time[ids] = -1.0
        self.seen_air[ids] = False
        self.plan_violation[ids] = 0.0
        self.landing_margin[ids] = 0.0
        self.touchdown_count[ids] = 0
        self.touchdown_locked[ids] = False
        self.q[ids] = self.robot.data.default_joint_pos[ids][:, self.joint_ids].reshape(-1, 2, 6)
        self.qdot[ids] = 0.0

    def _measure(self):
        mass = self.masses[..., None]
        com = (self.robot.data.body_com_pos_w * mass).sum(1) / mass.sum(1)
        velocity = (self.robot.data.body_com_lin_vel_w * mass).sum(1) / mass.sum(1)
        soles = (self.robot.data.body_pos_w[:, self.feet]
                 + quat_apply(self.robot.data.body_quat_w[:, self.feet],
                              self.kin.sole_offset.expand(self.n, -1, -1)))
        force = self.sensor.data.net_forces_w[:, self.sensor_feet, 2].clamp_min(0.0)
        return com, velocity, soles, force

    def _local(self, world):
        if world.ndim == 3:
            return quat_apply_inverse(self.heading[:, None].expand(-1, 2, -1), world - self.origin[:, None])
        return quat_apply_inverse(self.heading, world - self.origin)

    def _world(self, local):
        return self.origin + quat_apply(self.heading, local)

    def _initialize(self, ids, com, soles):
        self.heading[ids] = yaw_quat(self.robot.data.root_quat_w[ids])
        self.origin[ids] = soles[ids].mean(1)
        # This task's reference curriculum is a generated flat mesh. Its plane
        # height is the terrain origin, not the center of a tilted touching foot.
        self.origin[ids, 2] = self.env.scene.env_origins[ids, 2]
        self.anchors[ids] = self._local(soles)[ids]
        self.anchors[ids, :, 2] = -self.cfg["ground_contact_depth"]
        self.foot_rotation[ids] = self.home_foot_rotation
        self.width[ids] = (self.anchors[ids, 0, 1] - self.anchors[ids, 1, 1]).abs().mul(0.5).clamp(0.08, 0.14)
        self.lower_start[ids] = self._local(com)[ids]
        self.height[ids] = (self.lower_start[ids, 2] - self.cfg["pelvis_lowering"]).clamp(0.80, 1.06)
        self.state[ids], self.time[ids] = self.LOWER, 0.0

    def _plan(self, ids, position, velocity, target, target_velocity, duration, lower, upper):
        self.segment_start[ids], self.segment_velocity[ids] = position[ids, :2], velocity[ids, :2]
        self.segment_duration[ids] = duration
        omega = torch.sqrt(9.81 / self.height[ids])[:, None]
        p0, p1 = two_cops(position[ids, :2], velocity[ids, :2], target, target_velocity,
                          omega, torch.full_like(omega, duration))
        # Reject an infeasible terminal-state request. Independent clipping
        # changes the terminal state and can turn a forward step backwards.
        violation = torch.stack(((lower - p0).clamp_min(0), (p0 - upper).clamp_min(0),
                                 (lower - p1).clamp_min(0), (p1 - upper).clamp_min(0)))
        self.plan_violation[ids] = violation.amax(dim=(0, 2))
        self.cop0[ids], self.cop1[ids] = p0, p1
        self.time[ids] = 0.0
        return self.plan_violation[ids] <= 0.001

    def _segment(self):
        omega = torch.sqrt(9.81 / self.height)[:, None]
        half = self.segment_duration[:, None] * 0.5
        t = self.time[:, None].clamp(min=0.0)
        first_t = torch.minimum(t, half)
        position, velocity = lipm(self.segment_start, self.segment_velocity, self.cop0, omega, first_t)
        second_t = (t - half).clamp(min=0.0)
        position, velocity = lipm(position, velocity, self.cop1, omega, second_t)
        cop = torch.where(t <= half, self.cop0, self.cop1)
        return position, velocity, omega.square() * (position - cop), cop

    def _start_prepare(self, ids, com, velocity):
        position = self._local(com)
        speed = quat_apply_inverse(self.heading, velocity)
        omega = torch.sqrt(9.81 / self.height[ids])
        k = torch.tanh(omega * self.cfg["step_period"] * 0.5)
        target = self.anchors[ids, :, :2].mean(1)
        target[:, 1] = 0.0
        end_velocity = torch.zeros_like(target)
        end_velocity[:, 1] = omega * self.width[ids] * k
        lower = self.anchors[ids, :, :2].amin(1) - torch.tensor([0.06, 0.025], device=self.device)
        upper = self.anchors[ids, :, :2].amax(1) + torch.tensor([0.06, 0.025], device=self.device)
        feasible = self._plan(ids, position, speed, target, end_velocity,
                              self.cfg["preparation_time"], lower, upper)
        self.state[ids] = self.PREPARE
        bad = ids[~feasible]
        if bad.numel():
            self._hold(bad, com, self._world_feet(self.anchors), 3)

    def _start_step(self, ids, com, velocity, soles):
        measured_soles = self._local(soles)
        self.anchors[ids] = measured_soles[ids]
        self.anchors[ids, :, 2] = -self.cfg["ground_contact_depth"]
        pos, vel = self._local(com), quat_apply_inverse(self.heading, velocity)
        side = self.stance[ids]
        support = self.anchors[ids, side, :2]
        omega = torch.sqrt(9.81 / self.height[ids])
        half = self.cfg["step_period"] * 0.5
        command = self.env.command_manager.get_command("base_velocity")[ids, 0]
        step_length = (command * self.cfg["step_period"]).clamp(0.0, self.cfg["maximum_step_length"])
        sign = torch.where(side == 0, 1.0, -1.0)
        end_velocity = torch.stack((omega * step_length * 0.5 / torch.tanh(omega * half),
                                    -sign * omega * self.width[ids] * torch.tanh(omega * half)), -1)
        bound = torch.tensor([0.06, 0.025], device=self.device)
        # Choose a realizable CoP for terminal speed, then derive the landing
        # from the resulting terminal state. Position is deliberately free on
        # startup and after disturbances, rather than over-constraining both.
        horizon = self.cfg["step_period"]
        c, s = torch.cosh(omega * horizon)[:, None], torch.sinh(omega * horizon)[:, None]
        requested_cop = pos[ids, :2] + (vel[ids, :2] * c - end_velocity) / (omega[:, None] * s)
        cop = torch.maximum(torch.minimum(requested_cop, support + bound), support - bound)
        self.plan_violation[ids] = (requested_cop - cop).abs().amax(-1)
        self.segment_start[ids], self.segment_velocity[ids] = pos[ids, :2], vel[ids, :2]
        self.segment_duration[ids], self.time[ids] = horizon, 0.0
        self.cop0[ids], self.cop1[ids] = cop, cop
        end_p, end_v = lipm(pos[ids, :2], vel[ids, :2], cop, omega[:, None],
                            torch.full_like(omega[:, None], horizon))
        swing = 1 - side
        self.swing_start[ids] = measured_soles[ids, swing]
        self.swing_target[ids] = self.swing_start[ids]
        next_sign = -sign
        next_velocity = end_velocity.clone()
        next_velocity[:, 1] *= -1
        center, margin = landing_center(end_p, end_v, next_velocity, omega[:, None],
                                        torch.full_like(omega[:, None], horizon),
                                        self.home_foot_rotation[swing, :2, :2])
        self.landing_margin[ids] = margin
        self.swing_target[ids, :2] = center
        self.swing_target[ids, 2] = -self.cfg["ground_contact_depth"]
        self.swing_time[ids], self.air_time[ids] = -1.0, 0.0
        self.seen_air[ids] = False
        self.touchdown_count[ids] = 0
        self.touchdown_locked[ids] = False
        self.state[ids] = self.WALK
        landing_stride = self.swing_target[ids, 0] - support[:, 0]
        signed_width = (center[:, 1] - support[:, 1]) * next_sign
        feasible = ((end_v[:, 0] > 0.04) & (end_v[:, 0] < 0.8)
                    & (landing_stride > 0.03) & (landing_stride < 0.38)
                    & (signed_width >= 0.12) & (signed_width <= 0.28) & (margin > 0))
        bad = ids[~feasible]
        if bad.numel():
            self._hold(bad, com, soles, 3)

    def _world_feet(self, local):
        return self.origin[:, None] + quat_apply(self.heading[:, None].expand(-1, 2, -1), local)

    def _hold(self, ids, com, soles, code):
        self.failure[ids] = code
        self.anchors[ids] = self._local(soles)[ids]
        self.lower_start[ids] = self._local(com)[ids]
        self.height[ids] = self.lower_start[ids, 2]
        speed = quat_apply_inverse(self.heading, self.actual_velocity)[ids, :2]
        capture = self.lower_start[ids, :2] + speed * torch.sqrt(self.height[ids, None] / 9.81)
        bound = torch.tensor([0.06, 0.025], device=self.device)
        low = self.anchors[ids, :, :2].amin(1) - bound
        high = self.anchors[ids, :, :2].amax(1) + bound
        self.hold_capture[ids] = torch.maximum(torch.minimum(capture, high), low)
        self.state[ids], self.time[ids] = self.HOLD, 0.0

    def compute(self):
        dt = self.env.step_dt
        self.time += dt
        com, velocity, soles, force = self._measure()
        self.actual_com, self.actual_velocity = com, velocity
        ids = ((self.state == self.SETTLE) & (self.time >= self.cfg["settling_time"])
               & (force.sum(-1) > .2 * self.total_mass * 9.81)).nonzero().flatten()
        if ids.numel():
            self._initialize(ids, com, soles)
        ids = ((self.state == self.LOWER) & (self.time >= self.cfg["lowering_time"])).nonzero().flatten()
        if ids.numel():
            self._start_prepare(ids, com, velocity)
        ids = ((self.state == self.PREPARE) & (self.time >= self.cfg["preparation_time"])).nonzero().flatten()
        if ids.numel():
            self._start_step(ids, com, velocity, soles)

        # Start flight only after measurable support transfer. The capture check
        # is an unloading check, not a whole-single-support confinement rule.
        walking = self.state == self.WALK
        index = torch.arange(self.n, device=self.device)
        side, swing = self.stance, 1 - self.stance
        com_local, vel_local = self._local(com), quat_apply_inverse(self.heading, velocity)
        capture_y = com_local[:, 1] + vel_local[:, 1] * torch.sqrt(self.height / 9.81)
        fraction = force[index, side] / force.sum(-1).clamp_min(1.0)
        support_y = self.anchors[index, side, 1]
        self.support_fraction = fraction
        self.capture_error = capture_y - support_y
        ready = (walking & (self.swing_time < 0) & (self.time >= 0.02)
                 & (fraction >= self.cfg["support_fraction"])
                 & ((capture_y - support_y).abs() <= self.cfg["capture_tolerance"]))
        self.swing_time[ready] = self.time[ready]
        self.air_time = torch.where(walking & (force[index, swing] < 5.0), self.air_time + dt, 0.0)
        self.seen_air |= self.air_time >= 0.08 - 1e-6
        ids = (walking & (self.swing_time < 0) & (self.time > self.cfg["maximum_unload_wait"])).nonzero().flatten()
        if ids.numel():
            self._hold(ids, com, soles, 1)
        walking = self.state == self.WALK
        # A real landing is anchored where it occurred. Moving a loaded sole
        # back to a missed trajectory point drags the support away. Require a
        # valid flight and physical capture inside the actual new foot instead.
        stable_touch = (walking & self.seen_air & (force[index, swing] > 20.0)
                        & (self.time - self.swing_time >= 0.10))
        self.touchdown_count = torch.where(stable_touch, self.touchdown_count + 1, 0)
        new_touch = (self.touchdown_count >= 2) & ~self.touchdown_locked
        self.touchdown_anchor[new_touch] = self._local(soles)[index[new_touch], swing[new_touch]]
        # Toe contact can leave the sole center several millimetres above the
        # plane. Flattening at that center height lifts the foot back into air.
        self.touchdown_anchor[new_touch, 2] = -self.cfg["ground_contact_depth"]
        self.touchdown_locked |= self.touchdown_count >= 2
        capture_world = com.clone()
        capture_world[:, :2] += velocity[:, :2] * torch.sqrt(self.height[:, None] / 9.81)
        capture_world[:, 2] = soles[index, swing, 2]
        relative_capture = quat_apply_inverse(self.robot.data.body_quat_w[:, self.feet][index, swing],
                                               capture_world - soles[index, swing])
        capturable = (relative_capture[:, 0].abs() <= 0.105) & (relative_capture[:, 1].abs() <= 0.030)
        landed = (walking & (self.time >= self.cfg["step_period"] - .03)
                  & (self.touchdown_count >= 2) & capturable)
        ids = landed.nonzero().flatten()
        if ids.numel():
            self.steps[ids] += 1
            self.stance[ids] = 1 - self.stance[ids]
            self._start_step(ids, com, velocity, soles)
        ids = ((self.state == self.WALK) & (self.time > self.cfg["step_period"] + 0.14)).nonzero().flatten()
        if ids.numel():
            self._hold(ids, com, soles, 2)

        local_com = self.lower_start.clone()
        local_velocity = torch.zeros_like(local_com)
        local_acc = torch.zeros_like(local_com)
        cop = self.anchors[..., :2].mean(1)
        lower = self.state == self.LOWER
        local_com[lower, 2] = torch.lerp(self.lower_start[lower, 2], self.height[lower],
                                         smooth(self.time[lower] / self.cfg["lowering_time"]))
        fraction_lower = (self.time[lower] / self.cfg["lowering_time"]).clamp(0, 1)
        center = self.anchors[lower, :, :2].mean(1)
        displacement = center - self.lower_start[lower, :2]
        local_com[lower, :2] += displacement * smooth(fraction_lower)[:, None]
        local_velocity[lower, :2] = (displacement * (30 * fraction_lower.square()
                                                   * (1 - fraction_lower).square())[:, None]
                                          / self.cfg["lowering_time"])
        active = (self.state == self.PREPARE) | (self.state == self.WALK)
        segment_p, segment_v, segment_a, segment_cop = self._segment()
        local_com[active, :2], local_velocity[active, :2] = segment_p[active], segment_v[active]
        local_acc[active, :2], cop[active] = segment_a[active], segment_cop[active]
        local_com[active, 2] = self.height[active]
        hold = self.state == self.HOLD
        omega_hold = torch.sqrt(9.81 / self.height[hold])[:, None]
        hold_delta = ((self.lower_start[hold, :2] - self.hold_capture[hold])
                      * torch.exp(-omega_hold * self.time[hold, None]))
        local_com[hold, :2] = self.hold_capture[hold] + hold_delta
        local_com[hold, 2] = self.lower_start[hold, 2]
        local_velocity[hold, :2] = -omega_hold * hold_delta
        local_acc[hold, :2] = omega_hold.square() * hold_delta
        self.reference_com = self._world(local_com)
        self.reference_velocity = quat_apply(self.heading, local_velocity)

        foot_target = self.anchors.clone()
        walking = self.state == self.WALK
        flight = walking & (self.swing_time >= 0)
        flight_duration = (self.cfg["step_period"] - self.swing_time - 0.025).clamp_min(0.25)
        progress = ((self.time - self.swing_time) / flight_duration).clamp(0.0, 1.0)
        moving_foot = torch.lerp(self.swing_start, self.swing_target, smooth(progress)[:, None])
        moving_foot[:, 2] += self.cfg["clearance"] * torch.sin(torch.pi * progress).square()
        late = (self.time - self.cfg["step_period"]).clamp(0.0, 0.14)
        moving_foot[:, 2] -= 0.015 * late / 0.14
        foot_target[index[flight], (1 - self.stance)[flight]] = moving_foot[flight]
        locked = flight & self.touchdown_locked
        foot_target[index[locked], (1 - self.stance)[locked]] = self.touchdown_anchor[locked]
        # A failed swing returns to the ground instead of freezing an airborne
        # anchor and demanding support from it.
        foot_target[hold, :, 2] *= 1.0 - smooth(self.time[hold, None] / 0.3)

        # Centroidal feedback closes the floating-base COM mode. For the LIPM
        # this produces e_ddot + Kd e_dot + Kp e = 0 before contact constraints.
        omega_squared = (9.81 / self.height)[:, None]
        error = com_local[:, :2] - local_com[:, :2]
        velocity_error = vel_local[:, :2] - local_velocity[:, :2]
        correction = ((1 + self.cfg["com_position_gain"] / omega_squared) * error
                      + self.cfg["com_velocity_gain"] / omega_squared * velocity_error)
        moving = self.state != self.SETTLE
        cop[moving] += correction[moving]
        bounds = torch.tensor([0.06, 0.025], device=self.device)
        low = self.anchors[..., :2].amin(1) - bounds
        high = self.anchors[..., :2].amax(1) + bounds
        low[walking] = self.anchors[index[walking], self.stance[walking], :2] - bounds
        high[walking] = self.anchors[index[walking], self.stance[walking], :2] + bounds
        cop = torch.maximum(torch.minimum(cop, high), low)

        # One coherent support state controls reference, load/CoP and preload.
        span = (self.anchors[:, 0, 1] - self.anchors[:, 1, 1]).clamp_min(0.10)
        self.load[:, 0] = ((cop[:, 1] - self.anchors[:, 1, 1]) / span).clamp(0.0, 1.0)
        self.load[:, 1] = 1.0 - self.load[:, 0]
        if walking.any():
            self.load[walking] = 0.0
            self.load[index[walking], self.stance[walking]] = 1.0
        self.load[hold] = force[hold] / force[hold].sum(-1, keepdim=True).clamp_min(1.0)
        actual_soles = self._local(soles)
        cop[hold] = self.hold_capture[hold]

        # Decompose the aggregate CoP into realizable per-foot points. Applying
        # the same midpoint to each leg invents a 10 cm ankle moment arm in
        # double support even though that midpoint is outside both soles.
        support_centers = self.anchors.clone()
        support_centers[hold] = actual_soles[hold]
        average_center = (support_centers[..., :2] * self.load[..., None]).sum(1)
        shift = cop - average_center
        points = support_centers.clone()
        points[..., :2] += torch.maximum(torch.minimum(shift, bounds), -bounds)[:, None]
        points[..., 2] = 0.0
        self.reference_cop = (points[..., :2] * self.load[..., None]).sum(1)
        local_acc[moving, :2] = omega_squared[moving] * (com_local[moving, :2] - self.reference_cop[moving])
        self.contact_points = self._world_feet(points)

        actual_root = self._local(self.robot.data.root_pos_w)
        offset = com_local - actual_root
        pelvis_target = local_com - offset
        # During flight use the measured floating root to track the foot in
        # world space. A nominal upright root would miss a landing by 9 cm
        # after only five degrees of body pitch. Blend role changes to avoid
        # an ankle-target discontinuity at contact.
        actual_fraction = torch.zeros(self.n, 2, device=self.device)
        flight_blend = smooth((self.time - self.swing_time) / 0.10)
        actual_fraction[index[flight], (1 - self.stance)[flight]] = flight_blend[flight]
        new_support = walking & (self.steps > 0)
        actual_fraction[index[new_support], self.stance[new_support]] = (
            1.0 - smooth(self.time[new_support] / 0.12))
        actual_fraction[hold] = (force[hold] < 20.0).float()
        local_root_quat = quat_mul(quat_conjugate(self.heading), self.robot.data.root_quat_w)
        local_root_quat = torch.where(local_root_quat[:, :1] < 0, -local_root_quat, local_root_quat)
        upright = torch.zeros_like(local_root_quat)
        upright[:, 0] = 1.0
        blended_quat = torch.lerp(upright[:, None], local_root_quat[:, None], actual_fraction[..., None])
        blended_quat /= blended_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotation_inverse = matrix_from_quat(blended_quat).transpose(-1, -2)
        root_for_leg = torch.lerp(pelvis_target[:, None], actual_root[:, None], actual_fraction[..., None])
        position_target = (rotation_inverse @ (foot_target - root_for_leg)[..., None]).squeeze(-1)
        rotation_target = rotation_inverse @ self.foot_rotation
        q, error = self.kin.solve(position_target, rotation_target, self.q,
                                   iterations=self.cfg["ik_iterations"])
        moving = self.state != self.SETTLE
        q[~moving] = self.robot.data.default_joint_pos[~moving][:, self.joint_ids].reshape(-1, 2, 6)
        self.qdot = ((q - self.q) / dt).clamp(-4.0, 4.0)
        error[~moving] = 0.0
        self.q, self.ik_error = q, error
        total_force = torch.cat((local_acc[:, :2], torch.full((self.n, 1), 9.81, device=self.device)), -1)
        # Inverse statics always uses measured joint geometry and a single
        # physical root frame, independently of the IK task frames above.
        actual_inverse = matrix_from_quat(local_root_quat).transpose(-1, -2)
        force_leg = self.total_mass[:, None, None] * self.load[..., None] * total_force[:, None]
        force_leg = (actual_inverse[:, None] @ force_leg[..., None]).squeeze(-1)
        point = (actual_inverse[:, None] @ (points - actual_root[:, None])[..., None]).squeeze(-1)
        gravity_direction = -actual_inverse[..., 2]
        actual_q = self.robot.data.joint_pos[:, self.joint_ids].reshape(self.n, 2, 6)
        preload = self.kin.support_torque(actual_q, force_leg, point, gravity_direction)
        preload[~moving] = self.legacy_preload
        self.feedforward_torque = preload
        self.foot_target = self._world_feet(foot_target)
        return q.reshape(self.n, 12), self.qdot.reshape(self.n, 12), preload.reshape(self.n, 12)

    def summary(self):
        return {"reference_steps_mean": float(self.steps.float().mean()),
                "reference_steps_min": int(self.steps.min()), "reference_steps_max": int(self.steps.max()),
                "unload_timeout_envs": int((self.failure == 1).sum()),
                "landing_timeout_envs": int((self.failure == 2).sum()),
                "infeasible_plan_envs": int((self.failure == 3).sum()),
                "ik_error_max_m": float(self.ik_error.max()),
                "state_counts": {str(i): int((self.state == i).sum()) for i in range(5)}}
