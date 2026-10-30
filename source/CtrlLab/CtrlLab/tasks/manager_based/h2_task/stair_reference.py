"""Contact-confirmed step-to stair climbing around the unchanged leg residual."""

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

from .walking_reference import ContactWalkingReference, smooth, two_cops, lipm
from .stair_perception import LidarTreadMap
from .foot_geometry import sole_footprint_corners
from .stair_support import (project_wrench_points, ss_affine_native_project,
                            prepare_support_pose, cached_support_torque)
from .stair_dynamics import compute_swing_dynamics
from .stair_impedance import StairImpedance
from .stair_entry import direct_entry
from .stair_load import allocate_two_feet, settle_curve_violation
from .stair_trajectory import swing_position_local,swing_pitch
from .stair_contact_handover import blend_contact_quaternion, rebase_landed_joint_reference
from .stair_com_x_admittance import PelvisXAdmittance, ss_normal_cop_error_x, solve_ik_x_with_feasibility
from .stair_swing_ik import (keep_or_pick_swing_yaw, prepare_swing_command,
                             reset_swing_yaw_cache, freeze_landed_yaw)


def profile(t, duration):
    u = (t / duration).clamp(0.0, 1.0)
    return (smooth(u), 30 * u.square() * (1 - u).square() / duration,
            60 * u * (1 - u) * (1 - 2 * u) / duration**2)


def settle_profile(t, duration, start, speed, acceleration, end):
    """Quintic continuation preserving position, speed and acceleration."""
    u = (t / duration).clamp(0, 1)[:, None]
    scale = duration[:,None] if torch.is_tensor(duration) and duration.ndim else duration
    c1, c2 = speed * scale, .5 * acceleration * scale**2
    d, v, a = end - start - c1 - c2, -c1 - 2*c2, -2*c2
    c3, c4, c5 = 10*d - 4*v + .5*a, -15*d + 7*v - a, 6*d - 3*v + .5*a
    p = start + c1*u + c2*u**2 + c3*u**3 + c4*u**4 + c5*u**5
    pd = (c1 + 2*c2*u + 3*c3*u**2 + 4*c4*u**3 + 5*c5*u**4) / scale
    pdd = (2*c2 + 6*c3*u + 12*c4*u**2 + 20*c5*u**3) / scale**2
    return p, pd, pdd


class _ApproachReference(ContactWalkingReference):
    def _forward_speed(self, ids):
        return torch.minimum(super()._forward_speed(ids), self.stair_owner.approach_speed[ids])

    def _start_step(self, ids, com, velocity, soles):
        owner = self.stair_owner
        support_x = soles[ids, :, 0].amax(-1) - self.env.scene.env_origins[ids, 0]
        known = owner.map.last_seen[ids] >= 0
        obstacle = (known & (owner.map.maximum[ids] > .05)
                    & ((owner.map.y[None] - (soles[ids, :, 1].mean(1)
                         - self.env.scene.env_origins[ids, 1])[:, None]).abs() < .3)
                    & (owner.map.x[None] > support_x[:, None]))
        obstacle_x = torch.where(obstacle, owner.map.x[None], float("inf")).amin(-1)
        distance = obstacle_x-support_x
        owner.approach_obstacle_x[ids] = obstacle_x
        owner.approach_speed[ids] = .15+.15*((distance-.25)/.75).clamp(0,1)
        preparation, first_pair, entry_ready = owner._entry_plan(ids, soles)
        stop = entry_ready
        take = ids[stop]
        if take.numel():
            owner.preparation_world[take] = preparation[stop]
            owner.first_pair_world[take] = first_pair[stop]
            owner._take_over(take, com, velocity, soles)
            self._hold(take, com, soles, 0)
        unsafe = (~stop) & (obstacle_x - support_x < .25)
        blocked = ids[unsafe]
        if blocked.numel():
            owner.approach_block_reason[blocked] = 1
            owner._hold(blocked, com, velocity, soles, 11)
            self._hold(blocked, com, soles, 0)
        rest = ids[~stop & ~unsafe]
        if rest.numel():
            super()._start_step(rest, com, velocity, soles)
            # Retain the accepted flat approach, but never let its next sole
            # sweep cross a higher real return before a stair entry is ready.
            planned = self._world(self.swing_target)[rest]
            side = 1-self.stance[rest]
            rotation = matrix_from_quat(self.heading[rest]) @ self.home_foot_rotation[side]
            corners = (rotation[:,None] @ owner.foot_corners[side,...,None]).squeeze(-1)
            front_edge = planned[:,0] + corners[...,0].amax(-1) - self.env.scene.env_origins[rest,0]
            ray_boundary = obstacle_x[~stop & ~unsafe]
            reject = front_edge >= ray_boundary-.015
            blocked = rest[reject]
            if blocked.numel():
                owner.approach_block_reason[blocked] = 2
                owner._hold(blocked,com,velocity,soles,11)
                self._hold(blocked,com,soles,0)


class StairWalkingReference:
    BRAKE, SHIFT, SWING, TRANSFER, TOP_STAND, HOLD, WAIT_MAP = range(7)
    LAND = 7
    SETTLE = 8
    CONTACT_HANDOVER_FIELDS = ("contact_z_offset", "contact_measured_foot_quaternion",
                              "contact_rebased_this_step", "contact_pose_restore_weight")
    IK_COM_X_FIELDS = ("ik_com_x_delta", "ik_com_x_velocity", "ik_com_x_acceleration_formula",
        "ik_com_x_error", "ik_com_x_valid", "ik_com_x_omega", "ik_com_x_balance_omega", "ik_com_x_bound_active",
        "ik_com_x_feasible", "ik_com_x_held", "ik_com_x_baseline_only",
        "ik_com_x_emergency_disabled", "ik_com_x_command_rotation_error", "ik_com_x_step",
        "ik_com_x_baseline_fallback", "ik_com_x_baseline_evaluated", "ik_com_x_baseline_feasible",
        "ik_com_x_previous_offset_feasible", "ik_com_x_selection_branch",
        "ik_com_x_baseline_fallback_count", "ik_com_x_reset_count",
        "ik_com_x_new_accepted_count", "ik_com_x_old_held_count", "ik_com_x_last_fallback_step",
        "ik_com_x_last_fallback_old_delta", "ik_com_x_last_fallback_old_velocity",
        "ik_com_x_last_fallback_proposed_delta", "ik_com_x_last_fallback_proposed_velocity",
        "ik_com_x_last_fallback_candidate_bits", "ik_com_x_last_fallback_old_bits",
        "ik_com_x_last_fallback_baseline_bits")
    IK_COM_X_COUNTER_FIELDS = ("ik_com_x_baseline_fallback_count", "ik_com_x_reset_count",
        "ik_com_x_new_accepted_count", "ik_com_x_old_held_count", "ik_com_x_last_fallback_step")
    FAILURE_DIAGNOSTIC_FIELDS = ("hold_call_count", "latest_hold_reason",
        "first_failure_control_step", "failure_overwrite_prevented_count")
    RECEIVER_FLOAT_FIELDS = ("native_received_role", "native_received_effort", "native_drive_position",
        "native_delayed_legacy_position", "native_delayed_legacy_velocity",
        "native_drive_velocity", "native_drive_external_effort", "native_drive_current_position",
        "native_drive_current_velocity", "native_drive_effort_request", "native_drive_effort_safe",
        "native_drive_pd_expression", "native_effective_role", "native_effective_stiffness",
        "native_effective_damping", "native_effective_drive_cap", "native_total_motor_budget")
    RECEIVER_BOOL_FIELDS = ("native_drive_barrier", "native_drive_infeasible", "native_drive_barrier_satisfied",
                            "native_command_pre_physics")
    RECEIVER_INT_FIELDS = ("native_drive_delay_steps", "native_drive_physics_step", "native_barrier_latch",
                           "native_command_published_control_step", "native_command_receive_control_step")

    def __init__(self, env, robot, joint_ids, cfg):
        self.env, self.robot, self.joint_ids, self.cfg = env, robot, joint_ids, cfg
        self.device, self.n = env.device, env.num_envs
        self.options = env.cfg.stair_reference
        self.flat = _ApproachReference(env, robot, joint_ids, cfg)
        self.flat.stair_owner = self
        self.kin, self.masses, self.total_mass = self.flat.kin, self.flat.masses, self.flat.total_mass
        self.map = LidarTreadMap(env)
        self.map.connect_feet(robot,self.flat.feet,self.total_mass,matrix_from_quat)
        self.ik_com_x_admittance = PelvisXAdmittance(self.n,self.device,dt=env.step_dt,maximum_offset=.03)
        for name in self.IK_COM_X_FIELDS:
            setattr(self,name,torch.zeros(self.n,device=self.device))
        self.ik_com_x_step = torch.full((self.n,),-1,device=self.device,dtype=torch.long)
        for name in self.IK_COM_X_COUNTER_FIELDS+self.FAILURE_DIAGNOSTIC_FIELDS:
            setattr(self,name,torch.zeros(self.n,device=self.device,dtype=torch.long))
        self.ik_com_x_last_fallback_step.fill_(-1)
        self.first_failure_control_step.fill_(-1)
        for name in ("ik_com_x_selection_branch", "ik_com_x_last_fallback_candidate_bits",
                     "ik_com_x_last_fallback_old_bits", "ik_com_x_last_fallback_baseline_bits"):
            setattr(self,name,torch.zeros(self.n,device=self.device,dtype=torch.long))
        self.foot_corners = sole_footprint_corners(self.device)
        self.approach = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.state = torch.full((self.n,), self.SETTLE, dtype=torch.long, device=self.device)
        self.time = torch.zeros(self.n, device=self.device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.levels_completed = torch.zeros_like(self.steps)
        self.failure = torch.zeros_like(self.steps)
        self.stance = torch.ones_like(self.steps)
        self.swing_side = torch.zeros_like(self.steps)
        self.leading = torch.zeros_like(self.steps)
        self.trailing = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.exit_pair = torch.zeros_like(self.trailing)
        self.bottom_aligned = torch.zeros_like(self.trailing)
        self.preparing = torch.zeros_like(self.trailing)
        self.clearance_passed = torch.zeros_like(self.trailing)
        self.cross_time = torch.full_like(self.time, -1)
        self.cross_elapsed = torch.zeros_like(self.time)
        self.descend_time = torch.full_like(self.time, -1)
        self.descent_elapsed = torch.zeros_like(self.time)
        self.q = self.flat.q.clone()
        self.qdot = torch.zeros_like(self.q)
        self.anchors = torch.zeros(self.n, 2, 3, device=self.device)
        self.foot_target = torch.zeros_like(self.anchors)
        self.reference_com = torch.zeros(self.n, 3, device=self.device)
        self.reference_velocity = torch.zeros_like(self.reference_com)
        self.actual_com = torch.zeros_like(self.reference_com)
        self.actual_velocity = torch.zeros_like(self.reference_com)
        self.feedforward_torque = torch.zeros_like(self.q)
        self.dynamic_qddot = torch.zeros_like(self.q)
        self.dynamic_qdot_preview = torch.zeros_like(self.q)
        self.dynamic_torque_raw = torch.zeros_like(self.q)
        self.dynamic_torque_applied = torch.zeros_like(self.q)
        self.dynamic_inertia_torque = torch.zeros_like(self.q)
        self.dynamic_coriolis_torque = torch.zeros_like(self.q)
        self.dynamic_velocity_bias = torch.zeros_like(self.q)
        self.dynamic_velocity_target = torch.zeros_like(self.q)
        self.dynamic_mass_diagonal = torch.zeros_like(self.q)
        self.dynamic_peak_raw = torch.zeros_like(self.q)
        self.dynamic_peak_applied = torch.zeros_like(self.q)
        self.dynamic_applied_steps = torch.zeros(self.n,2,dtype=torch.long,device=self.device)
        self.dynamic_clip_steps = torch.zeros_like(self.dynamic_applied_steps)
        self.drive_velocity_target = torch.zeros_like(self.q)
        for name in ("support_drive_role", "support_pose_pd_torque", "support_effort_requested",
                     "support_effort_target", "support_effort_expression", "support_drive_velocity"):
            setattr(self, name, torch.zeros_like(self.q))
        for name in ("support_position_clipped", "support_velocity_clipped", "support_effort_clipped"):
            setattr(self, name, torch.zeros_like(self.q, dtype=torch.bool))
        for name in self.RECEIVER_FLOAT_FIELDS:
            setattr(self, name, torch.zeros_like(self.q))
        for name in self.RECEIVER_BOOL_FIELDS:
            setattr(self, name, torch.zeros_like(self.q, dtype=torch.bool))
        for name in self.RECEIVER_INT_FIELDS:
            setattr(self, name, torch.zeros_like(self.q, dtype=torch.long))
        self.dynamic_valid = torch.zeros(self.n,2,dtype=torch.bool,device=self.device)
        self.dynamic_diagnostics = torch.zeros(self.n,2,8,device=self.device)
        self.cross_advancing = torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.descent_advancing = torch.zeros_like(self.cross_advancing)
        self.processed_residual = torch.zeros(self.n, 12, device=self.device)
        self.contact_points = torch.zeros_like(self.anchors)
        self.pressure_centers = torch.zeros(self.n, 2, 2, device=self.device)
        self.pressure_radii = torch.zeros_like(self.pressure_centers)
        self.feedforward_effort_ratio = torch.zeros_like(self.q)
        self.reference_cop = torch.zeros(self.n, 2, device=self.device)
        self.requested_cop = torch.zeros_like(self.reference_cop)
        self.requested_acceleration_local = torch.zeros_like(self.reference_cop)
        self.pressure_seed_box_applied = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.legacy_feedback_support_force = torch.zeros(self.n, 2, device=self.device)
        self.control_root_position = torch.zeros(self.n, 3, device=self.device)
        self.control_root_quaternion = torch.zeros(self.n, 4, device=self.device)
        self.control_frame_origin = torch.zeros_like(self.control_root_position)
        self.control_frame_heading = torch.zeros_like(self.control_root_quaternion)
        self.control_root_link_velocity = torch.zeros_like(self.control_root_position)
        self.control_root_angular_velocity = torch.zeros_like(self.control_root_position)
        self.control_joint_position = torch.zeros_like(self.q)
        self.control_joint_velocity = torch.zeros_like(self.q)
        self.control_soles = torch.zeros_like(self.anchors)
        self.control_foot_quaternion = torch.zeros(self.n, 2, 4, device=self.device)
        self.control_sample_step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.control_sample_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.support_fraction = torch.zeros(self.n, device=self.device)
        self.capture_error = torch.zeros(self.n, device=self.device)
        self.ik_error = torch.zeros(self.n, 2, device=self.device)
        self.ik_solver_error = torch.zeros_like(self.ik_error)
        self.rate_limit_count = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.load = torch.full((self.n, 2), .5, device=self.device)
        self.nominal_height = torch.full((self.n,), .94, device=self.device)
        self.height = self.nominal_height.clone()
        # XY plans the total COM; Z plans root height plus a fixed offset.
        self.body_z_offset = torch.zeros(self.n, device=self.device)
        self.body_plan_position = torch.zeros_like(self.reference_com)
        self.body_plan_velocity = torch.zeros_like(self.reference_com)
        self.body_plan_acceleration = torch.zeros_like(self.reference_com)
        self.start_com = torch.zeros_like(self.reference_com)
        self.start_velocity = torch.zeros_like(self.reference_com)
        self.start_acceleration = torch.zeros_like(self.reference_com)
        self.reference_acceleration_local = torch.zeros_like(self.reference_com)
        self.landing_pitch = torch.zeros_like(self.time)
        self.contact_age = torch.full((self.n,2),100.,device=self.device)
        self.contact_pitch = torch.zeros_like(self.contact_age)
        # Pressure anchors remain unchanged; these belong only to the foot pose.
        self.contact_z_offset = torch.zeros_like(self.contact_age)
        self.contact_measured_foot_quaternion = torch.zeros(self.n,2,4,device=self.device)
        self.contact_measured_foot_quaternion[...,0] = 1
        self.contact_rebased_this_step = torch.zeros(self.n,2,dtype=torch.bool,device=self.device)
        self.contact_pose_restore_weight = torch.ones_like(self.contact_age)
        self.end_com = torch.zeros_like(self.reference_com)
        self.segment_h = self.nominal_height.clone()
        self.cop0 = torch.zeros(self.n, 2, device=self.device)
        self.cop1 = torch.zeros_like(self.cop0)
        self.segment_duration = torch.ones(self.n, device=self.device)
        self.segment_c2 = torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.projected_contact_feasible = torch.ones(self.n,2,dtype=torch.bool,device=self.device)
        self.pressure_projection_shift = torch.zeros(self.n,2,device=self.device)
        self.swing_start = torch.zeros_like(self.reference_com)
        self.swing_target = torch.zeros_like(self.reference_com)
        self.lift_target = torch.zeros_like(self.reference_com)
        self.lift_boundary = torch.zeros_like(self.time)
        self.lift_boundary_observed = torch.zeros_like(self.trailing)
        self.lift_source_evidence = torch.zeros(self.n,6,device=self.device)
        self.pair_targets = torch.zeros_like(self.anchors)
        self.preparation_world = torch.zeros_like(self.anchors)
        self.first_pair_world = torch.zeros_like(self.anchors)
        self.foot_yaw = torch.zeros(self.n, 2, device=self.device)
        self.swing_yaw_goal = torch.zeros(self.n, device=self.device)
        self.yaw_checked_time = torch.full_like(self.time, -1)
        self.swing_ik_diagnostics = torch.zeros(self.n,6,device=self.device)
        self.approach_speed = torch.full_like(self.time,.30)
        self.approach_obstacle_x = torch.full_like(self.time,float("inf"))
        self.approach_block_reason = torch.zeros_like(self.steps)
        self.entry_plan_diagnostics = torch.zeros(self.n,6,device=self.device)
        self.entry_direct = torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.entry_leading = torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.entry_root_height = torch.full_like(self.time,self.options["pelvis_height"])
        self.entry_foot_yaw = torch.zeros(self.n,2,device=self.device)
        self.entry_kinematics = torch.zeros(self.n,6,device=self.device)
        self.support_backshift = torch.full_like(self.time,.035)
        self.air_time = torch.zeros(self.n, device=self.device)
        self.seen_air = torch.zeros_like(self.trailing)
        self.touchdown_count = torch.zeros_like(self.steps)
        self.map_wait = torch.zeros_like(self.time)
        self.plan_violation = torch.zeros_like(self.time)
        self.landing_margin = torch.zeros_like(self.time)
        self.touchdown_locked = torch.zeros_like(self.trailing)
        self.swing_time = torch.full_like(self.time, -1)
        self.impedance = StairImpedance(self)
        self.drive_stiffness = torch.zeros_like(self.q)
        self.drive_damping = torch.zeros_like(self.q)
        self.drive_gain_blend = torch.zeros_like(self.time)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.flat.reset(env_ids)
        self.map.reset(env_ids)
        self.ik_com_x_admittance.reset(ids)
        for name in self.IK_COM_X_FIELDS+self.FAILURE_DIAGNOSTIC_FIELDS:
            getattr(self,name)[ids] = 0
        self.ik_com_x_last_fallback_step[ids] = -1
        self.first_failure_control_step[ids] = -1
        self.approach[ids] = True
        self.state[ids] = self.SETTLE
        for item in (self.time, self.steps, self.levels_completed, self.failure, self.air_time,
                     self.touchdown_count, self.map_wait):
            item[ids] = 0
        self.trailing[ids] = self.exit_pair[ids] = self.seen_air[ids] = False
        self.bottom_aligned[ids] = self.preparing[ids] = self.clearance_passed[ids] = False
        self.foot_yaw[ids] = 0
        self.segment_c2[ids] = False
        self.projected_contact_feasible[ids] = True
        self.pressure_projection_shift[ids] = 0
        self.requested_cop[ids] = self.requested_acceleration_local[ids] = 0
        self.pressure_seed_box_applied[ids] = False
        for name in ("legacy_feedback_support_force", "control_root_position", "control_root_quaternion",
                     "control_root_link_velocity", "control_root_angular_velocity", "control_joint_position",
                     "control_joint_velocity", "control_soles", "control_foot_quaternion", "control_sample_step",
                     "control_sample_valid", "control_frame_origin", "control_frame_heading"):
            getattr(self, name)[ids] = 0
        self.body_z_offset[ids] = 0
        self.body_plan_position[ids] = self.body_plan_velocity[ids] = self.body_plan_acceleration[ids] = 0
        for name in ("dynamic_qddot","dynamic_qdot_preview","dynamic_torque_raw","dynamic_torque_applied",
                     "dynamic_inertia_torque","dynamic_coriolis_torque","dynamic_velocity_bias",
                     "dynamic_velocity_target","dynamic_mass_diagonal","dynamic_valid","dynamic_diagnostics",
                     "dynamic_peak_raw","dynamic_peak_applied","dynamic_applied_steps","dynamic_clip_steps",
                     "drive_velocity_target","cross_advancing","descent_advancing",
                     "support_drive_role", "support_pose_pd_torque", "support_effort_requested",
                     "support_effort_target", "support_effort_expression", "support_drive_velocity",
                     "support_position_clipped", "support_velocity_clipped", "support_effort_clipped"):
            getattr(self,name)[ids] = 0
        reset_swing_yaw_cache(self,ids)
        self.contact_age[ids],self.contact_pitch[ids] = 100,0
        self.contact_z_offset[ids] = 0
        self.contact_measured_foot_quaternion[ids] = 0
        self.contact_measured_foot_quaternion[ids,:,0] = 1
        self.contact_rebased_this_step[ids] = False
        self.contact_pose_restore_weight[ids] = 1
        self.approach_block_reason[ids] = 0
        self.entry_direct[ids] = False
        self.entry_leading[ids] = 0
        self.entry_root_height[ids] = self.options["pelvis_height"]
        self.entry_foot_yaw[ids],self.entry_kinematics[ids] = 0,0
        self.support_backshift[ids]=.035
        self.lift_target[ids],self.lift_boundary[ids],self.lift_boundary_observed[ids] = 0,0,False
        self.lift_source_evidence[ids]=0
        self.swing_time[ids] = -1
        for name in self.RECEIVER_FLOAT_FIELDS+self.RECEIVER_BOOL_FIELDS+self.RECEIVER_INT_FIELDS:
            getattr(self, name)[ids] = 0
        self.native_command_published_control_step[ids] = -1
        self.q[ids] = self.flat.q[ids]
        self.impedance.reset(env_ids)

    def _local(self, world):
        return self.flat._local(world)

    def _plan_position(self, ids, com):
        point = self._local(com)[ids].clone()
        point[:, 2] = self._local(self.robot.data.root_pos_w)[ids, 2] + self.body_z_offset[ids]
        return point

    def _plan_velocity(self, ids, velocity):
        speed = quat_apply_inverse(self.flat.heading, velocity)[ids].clone()
        root_speed = quat_apply_inverse(self.flat.heading, self.robot.data.root_link_lin_vel_w)[ids]
        speed[:, 2] = root_speed[:, 2]
        return speed

    def _world(self, local):
        return self.flat._world(local)

    def _feet_world(self, local):
        return self.flat._world_feet(local)

    def _over_target_patch(self, ids, soles):
        selected = self.swing_side[ids]
        quat = self.robot.data.body_quat_w[ids][:, self.flat.feet][torch.arange(ids.numel(), device=self.device), selected]
        corners = self.foot_corners[selected]
        world = soles[ids, selected, None] + quat_apply(quat[:, None].expand(-1, 4, -1), corners)
        target = self.flat.origin[ids] + (matrix_from_quat(self.flat.heading[ids]) @ self.swing_target[ids, :, None]).squeeze(-1)
        return self.map.footprint_supported(ids,world,
            target[:,2]+self.cfg["ground_contact_depth"],margin=.008)

    def _next_tread(self, ids, soles):
        origins = self.env.scene.env_origins[ids]
        local = soles[ids] - origins[:, None]
        support_z = local[..., 2].amin(1)
        old_x = local[..., 0].amax(1)
        rear = local[..., 0].argmin(1)
        return self.map.choose(ids, minimum_x=old_x + .10, maximum_x=old_x + .75,
                               foot_y=local[torch.arange(ids.numel(), device=self.device), rear, 1], low_z=support_z + .05,
            high_z=support_z + self.options["maximum_rise"],interior=True)

    def _entry_plan(self, ids, soles):
        """Verify one observed preparation+first-tread plan before braking."""
        origin = self.env.scene.env_origins[ids]
        feet = soles[ids] - origin[:, None]
        batch = torch.arange(ids.numel(), device=self.device)
        rear = feet[..., 0].argmin(1)
        front = 1 - rear
        x = feet[batch, front, 0]
        z = feet[..., 2].mean(1)
        first, high_ok = self.map.choose(ids, minimum_x=x+.10, maximum_x=x+.75,
            foot_y=feet[batch, front, 1], low_z=z+.05, high_z=z+self.options["maximum_rise"],interior=True)
        high = first - origin
        partner, partner_ok = self.map.choose(ids, minimum_x=high[:, 0]-.03, maximum_x=high[:, 0]+.03,
            foot_y=feet[batch, rear, 1], low_z=high[:, 2]-.018, high_z=high[:, 2]+.018, desired_x=high[:, 0])
        # A preparation step may go slightly ahead of the former front foot,
        # provided its actual observed floor and subsequent high pair all fit.
        prep, low_ok = self.map.choose(ids, minimum_x=torch.maximum(x-.15, feet[batch, rear, 0]+.025),
            maximum_x=torch.minimum(x+.18, high[:, 0]-.22), foot_y=feet[batch, rear, 1],
            low_z=z-.025, high_z=z+.025, desired_x=high[:, 0]-.35)
        prep_local = prep-origin
        prep_other, other_low_ok = self.map.choose(ids,minimum_x=prep_local[:,0]-.025,maximum_x=prep_local[:,0]+.025,
            foot_y=feet[batch,front,1],low_z=z-.025,high_z=z+.025,desired_x=prep_local[:,0])
        prep_pair = torch.zeros_like(feet)
        prep_pair[batch,rear],prep_pair[batch,front] = prep,prep_other
        pair = torch.zeros(ids.numel(), 2, 3, device=self.device)
        pair[batch, front], pair[batch, rear] = first, partner
        stride = (prep - soles[ids, rear]).norm(dim=-1)
        gap = first[:, 0] - torch.maximum(prep[:, 0], soles[ids, front, 0])
        ready = high_ok & partner_ok & low_ok & other_low_ok & (stride <= .40) & (gap >= .12)
        ready &= (prep_other-soles[ids,front]).norm(dim=-1)<=.40
        ready &= gap <= self.options["maximum_first_stride"]
        self.entry_plan_diagnostics[ids] = torch.stack((high_ok,partner_ok,low_ok,other_low_ok,stride,gap),-1)
        self.entry_direct[ids] = False
        self.entry_leading[ids] = rear
        observed = high_ok&partner_ok
        observed &= (pair-soles[ids]).norm(dim=-1).amax(-1)<=self.options["entry_stride_search"]
        observed &= pair[:,:,0].amin(-1)-soles[ids,:,0].amax(-1)>=.10
        candidates = ids[observed]
        if candidates.numel():
            plan = direct_entry(self,candidates,soles,pair[observed])
            good = candidates[plan["valid"]]
            self.entry_direct[good] = True
            self.entry_leading[good] = plan["leading"][plan["valid"]]
            self.entry_root_height[good] = plan["root_height"][plan["valid"]]
            self.entry_foot_yaw[good] = plan["yaw"][plan["valid"]]
            self.support_backshift[good]=plan["support_backshift"][plan["valid"]]
            self.entry_kinematics[candidates] = torch.stack((plan["valid"].to(self.time.dtype),
                plan["root_height"],plan["IK_error"],plan["joint_margin"],plan["COM_error"],plan["support_backshift"]),-1)
        ready |= self.entry_direct[ids]
        return prep_pair, pair, ready

    def _take_over(self, ids, com, velocity, soles):
        self.body_z_offset[ids] = com[ids, 2] - self.robot.data.root_pos_w[ids, 2]
        self.approach[ids] = False
        self.state[ids], self.time[ids] = self.BRAKE, 0.0
        self.anchors[ids] = self._local(soles)[ids]
        self.anchors[ids, :, 2] = -self.cfg["ground_contact_depth"]
        self.q[ids] = self.flat.q[ids]
        foot_rotation = matrix_from_quat(self.robot.data.body_quat_w[ids][:, self.flat.feet])
        heading_rotation = matrix_from_quat(self.flat.heading[ids])
        heading_yaw = torch.atan2(heading_rotation[:,1,0], heading_rotation[:,0,0])
        home_yaw = torch.atan2(self.flat.home_foot_rotation[:,1,0], self.flat.home_foot_rotation[:,0,0])
        foot_yaw = torch.atan2(foot_rotation[:,:,1,0], foot_rotation[:,:,0,0]) - heading_yaw[:,None] - home_yaw[None]
        self.foot_yaw[ids] = torch.atan2(foot_yaw.sin(), foot_yaw.cos())
        # Keep the front foot as support; this per-environment leading foot
        # then remains fixed for L1/R1/L2/R2 (or its mirror) throughout ascent.
        self.leading[ids] = self.entry_leading[ids]
        self.start_com[ids] = self._plan_position(ids, com)
        self.start_velocity[ids] = self._plan_velocity(ids, velocity)
        omega = torch.sqrt(9.81 / self.flat.height[ids])
        self.end_com[ids] = self.start_com[ids]
        self.end_com[ids, :2] += self.start_velocity[ids, :2] / omega[:, None]
        com_root_offset = com[ids, 2] - self.robot.data.root_pos_w[ids, 2]
        pelvis = torch.where(self.entry_direct[ids],self.entry_root_height[ids],self.options["pelvis_height"])
        self.nominal_height[ids] = (pelvis + com_root_offset).clamp(.78, 1.06)
        self.end_com[ids, 2] = self.nominal_height[ids]

    def _hold(self, ids, com, velocity, soles, code):
        fresh = ids[self.approach[ids]]
        self.body_z_offset[fresh] = com[fresh, 2] - self.robot.data.root_pos_w[fresh, 2]
        # failure is the FIRST original reason, not a mutable latest label.
        # Repeated HOLD calls retain their original state/anchor behavior,
        # while diagnostics count a would-be reason overwrite separately.
        recorded = self.failure[ids].clone()
        fresh_failure = recorded == 0
        first = ids[fresh_failure]
        self.failure[first] = code
        self.first_failure_control_step[first] = int(self.env.common_step_counter)
        self.latest_hold_reason[ids] = code
        self.hold_call_count[ids] += 1
        self.failure_overwrite_prevented_count[ids] += ((recorded > 0) & (recorded != code)).long()
        self.approach[ids] = False
        self.state[ids], self.time[ids] = self.HOLD, 0.0
        self.anchors[ids] = self._local(soles)[ids]
        self.start_com[ids] = self._plan_position(ids, com)
        speed = quat_apply_inverse(self.flat.heading, velocity)[ids]
        h = (self._local(com)[ids, 2] - self.anchors[ids, :, 2].mean(1)).clamp(.60, 1.10)
        self.end_com[ids] = self.start_com[ids]
        self.end_com[ids, :2] += speed[:, :2] * torch.sqrt(h[:, None] / 9.81)

    def _pair(self, ids, soles):
        origins = self.env.scene.env_origins[ids]
        feet = soles[ids] - origins[:, None]
        old_x = feet[..., 0].amax(1)
        old_z = feet[..., 2].mean(1)
        leader = self.leading[ids]
        batch = torch.arange(ids.numel(), device=self.device)
        exit_pair = self.levels_completed[ids] >= self.options["requested_levels"]
        target, valid = self.map.choose(
            ids, minimum_x=old_x + .12, maximum_x=old_x + self.options["maximum_first_stride"],
            foot_y=feet[batch, leader, 1], low_z=torch.where(exit_pair, old_z - .025, old_z + .05),
            high_z=torch.where(exit_pair, old_z + .025, old_z + self.options["maximum_rise"]),
            desired_x=old_x + .35,interior=True)
        preparation = ~self.bottom_aligned[ids]&~self.entry_direct[ids]
        if preparation.any():
            align = self.preparation_world[ids,leader]
            align_valid = self.map.contains(ids, align)
            align_valid &= (align - soles[ids, leader]).norm(dim=-1) <= .40
            target = torch.where(preparation[:, None], align, target)
            valid = torch.where(preparation, align_valid, valid)
        desired_local = target - origins
        right, right_valid = self.map.choose(
            ids, minimum_x=desired_local[:, 0] - .03, maximum_x=desired_local[:, 0] + .03,
            foot_y=feet[batch, 1 - leader, 1], low_z=desired_local[:, 2] - .018,
            high_z=desired_local[:, 2] + .018, desired_x=desired_local[:, 0])
        valid &= torch.where(preparation, torch.ones_like(right_valid), right_valid)
        right = torch.where(preparation[:, None], soles[ids, 1 - leader], right)
        first_level = (self.bottom_aligned[ids]|self.entry_direct[ids]) & (self.levels_completed[ids] == 0)
        if first_level.any():
            cached = self.first_pair_world[ids]
            first_target, first_other = cached[batch, leader], cached[batch, 1-leader]
            first_valid = self.map.contains(ids, first_target) & self.map.contains(ids, first_other)
            first_gap = first_target[:,0]-(old_x+origins[:,0])
            reach = torch.where(self.entry_direct[ids],self.options["entry_stride_search"],self.options["maximum_first_stride"])
            first_valid &= (first_gap>=.10)&(first_gap<=reach)
            first_valid &= (first_target-soles[ids,leader]).norm(dim=-1)<=reach
            target = torch.where(first_level[:, None], first_target, target)
            right = torch.where(first_level[:, None], first_other, right)
            valid = torch.where(first_level, first_valid, valid)
        if valid.any():
            good = ids[valid]
            pair = torch.zeros(ids.numel(), 2, 3, device=self.device)
            pair[batch, leader], pair[batch, 1 - leader] = target, right
            relative = pair - self.flat.origin[ids, None]
            inverse = matrix_from_quat(self.flat.heading[ids]).transpose(-1, -2)
            self.pair_targets[good] = (inverse[:, None] @ relative[..., None]).squeeze(-1)[valid]
            self.pair_targets[good, :, 2] -= self.cfg["ground_contact_depth"]
            self.exit_pair[good] = exit_pair[valid]
            self.preparing[good] = preparation[valid]
            self.trailing[good] = False
        return valid


    def _shift(self, ids, support, com, velocity, z_target, state):
        self.start_com[ids] = self._plan_position(ids, com)
        self.start_velocity[ids] = self._plan_velocity(ids, velocity)
        self.start_acceleration[ids] = 0
        self.end_com[ids] = self.start_com[ids]
        self.end_com[ids,:2] = self.anchors[ids,support,:2]
        self.end_com[ids,0]-=torch.where(self.preparing[ids],0,self.support_backshift[ids])
        medial=self.options["support_medial_offset"]
        self.end_com[ids,1] += torch.where(support==0,-medial,medial)
        self.end_com[ids,2] = z_target
        self.segment_h[ids] = (self._local(com)[ids,2]-self.anchors[ids,support,2]).clamp(.65,1.10)
        self.segment_c2[ids] = ~self.preparing[ids]
        duration = torch.where(self.preparing[ids],1.55,self.options["transfer_duration"])
        omega = torch.sqrt(9.81/self.segment_h[ids])[:,None]
        bounds = torch.tensor([.06,.025],device=self.device)
        actual_span = self.anchors[ids,0,1]-self.anchors[ids,1,1]
        span = actual_span.clamp_min(.001)
        def violation_at(point):
            return allocate_two_feet(point,self.anchors[ids,:,:2],bounds)[2]
        for attempt in range(4):
            p0,p1 = two_cops(self.start_com[ids,:2],self.start_velocity[ids,:2],
                self.end_com[ids,:2],torch.zeros((ids.numel(),2),device=self.device),omega,duration[:,None])
            violation = torch.maximum(violation_at(p0),violation_at(p1))
            high_violation = torch.zeros_like(violation)
            if self.segment_c2[ids].any():
                high_violation = settle_curve_violation(
                    self.start_com[ids],self.start_velocity[ids],self.start_acceleration[ids],
                    self.end_com[ids],duration,omega,self.anchors[ids,:,:2],bounds,
                    settle_profile=settle_profile)
                violation = torch.where(self.segment_c2[ids],high_violation,violation)
            violation = torch.maximum(violation,(.10-actual_span).clamp_min(0))
            if attempt<3:
                duration += (violation>.001)*torch.where(self.preparing[ids],.20,.30)
        high = self.segment_c2[ids]
        if high.any():
            # A longer Hermite transition is not necessarily safer when its
            # initial velocity is nonzero. Search feasible durations instead
            # of making an overshooting continuation progressively longer.
            for trial in torch.linspace(1.,4.,13,device=self.device):
                candidate_duration = torch.ones_like(duration)*trial
                candidate_violation = settle_curve_violation(
                    self.start_com[ids],self.start_velocity[ids],self.start_acceleration[ids],
                    self.end_com[ids],candidate_duration,omega,self.anchors[ids,:,:2],bounds,
                    settle_profile=settle_profile,
                    vertical_limit=self.options["maximum_vertical_acceleration"])
                candidate_violation = torch.maximum(candidate_violation,(.10-actual_span).clamp_min(0))
                better = candidate_violation<violation-1e-6
                better |= (candidate_violation<=.001)&((violation>.001)|(candidate_duration<duration))
                better &= high
                duration = torch.where(better,candidate_duration,duration)
                violation = torch.where(better,candidate_violation,violation)
        self.segment_duration[ids],self.plan_violation[ids] = duration,violation
        self.cop0[ids],self.cop1[ids] = p0,p1
        self.stance[ids] = support
        self.state[ids],self.time[ids] = state,0
        bad = ids[violation>.001]
        if bad.numel():
            _,_,soles,_ = self.flat._measure()
            self._hold(bad,com,velocity,soles,14)

    def _swing(self, ids, side, com, soles):
        self.swing_side[ids] = side
        self.stance[ids] = 1 - side
        self.swing_start[ids] = self._local(soles)[ids, side]
        self.swing_target[ids] = self.pair_targets[ids, side]
        clearance=torch.where(self.preparing[ids],self.options["preparation_clearance"],self.options["clearance"])
        self.lift_target[ids] = self.swing_start[ids]
        self.lift_target[ids,2] = torch.maximum(self.swing_start[ids,2],self.swing_target[ids,2])+clearance
        rising=(self.swing_target[ids,2]-self.swing_start[ids,2]>.04)&~self.preparing[ids]
        high=ids[rising]
        if high.numel():
            from .stair_lift import pre_riser_lift
            heading=matrix_from_quat(self.flat.heading[high])
            world_start=self.flat.origin[high]+(heading@self.swing_start[high,:,None]).squeeze(-1)
            world_end=self.flat.origin[high]+(heading@self.swing_target[high,:,None]).squeeze(-1)
            point,evidence=pre_riser_lift(self.map,high,world_start,world_end,
                self.foot_corners[self.swing_side[high]],max_clearance=.03,margin=.02)
            self.lift_target[high]=(heading.transpose(-1,-2)@(point-self.flat.origin[high])[...,None]).squeeze(-1)
            self.lift_boundary[high]=evidence["raw_boundary_world_x"]
            self.lift_boundary_observed[high]=evidence["raw_boundary_valid"]
            self.lift_source_evidence[high]=torch.stack([evidence[k].to(self.time.dtype) for k in
                ("raw_cell_index","raw_minimum","raw_maximum","raw_last_seen","radius_m","advance_m")],-1)
        self.start_com[ids] = self._plan_position(ids, com)
        self.end_com[ids] = self.start_com[ids]
        support = 1 - side
        self.end_com[ids,:2] = self.anchors[ids,support,:2]
        self.end_com[ids,0]-=torch.where(self.preparing[ids],0,self.support_backshift[ids])
        medial=self.options["support_medial_offset"]
        self.end_com[ids,1] += torch.where(support==0,-medial,medial)
        # Keep body height during the short flight. The next double-support
        # transfer raises the body after both feet have reached the new tread.
        self.state[ids], self.time[ids] = self.SWING, 0.0
        self.seen_air[ids] = False
        self.air_time[ids] = 0
        self.touchdown_count[ids] = 0
        self.swing_time[ids] = 0
        self.clearance_passed[ids] = False
        self.cross_time[ids], self.descend_time[ids] = -1.0, -1.0
        self.cross_elapsed[ids] = 0
        self.descent_elapsed[ids] = 0
        distance = (self.swing_target[ids]-self.swing_start[ids]).norm(dim=-1)
        allowed = torch.where(self.preparing[ids],.40,self.options["maximum_first_stride"])
        allowed = torch.where(self.entry_direct[ids]&(self.levels_completed[ids]==0),
                              self.options["entry_stride_search"],allowed)
        reachable = distance<=allowed
        bad = ids[~reachable]
        if bad.numel():
            self._hold(bad,com,self.actual_velocity,soles,15)
        good = ids[reachable]
        if good.numel():
            reset_swing_yaw_cache(self,good)
            self.swing_yaw_goal[good] = torch.where(self.entry_direct[good],
                self.entry_foot_yaw[good,self.swing_side[good]],self.swing_yaw_goal[good])
            yaw, feasible = self._pick_swing_yaw(good,soles)
            self.swing_yaw_goal[good],self.yaw_checked_time[good] = yaw,0
            bad = good[~feasible]
            if bad.numel():
                self._hold(bad,com,self.actual_velocity,soles,15)

    def _pick_swing_yaw(self, ids, soles):
        # The preceding DS feedforward still includes support force.
        yaw,feasible,_,_ = keep_or_pick_swing_yaw(self,ids,soles)
        return yaw,feasible

    def compute(self):
        dt = self.env.step_dt
        for name in ("dynamic_qddot","dynamic_qdot_preview","dynamic_torque_raw","dynamic_torque_applied",
                     "dynamic_inertia_torque","dynamic_coriolis_torque","dynamic_velocity_bias",
                     "dynamic_velocity_target","dynamic_mass_diagonal","dynamic_valid","dynamic_diagnostics",
                     "cross_advancing","descent_advancing"):
            getattr(self,name).zero_()
        self.contact_rebased_this_step.zero_()
        self.ik_com_x_emergency_disabled.zero_()
        self.map.update()
        self.contact_age += dt
        self.time[~self.approach] += dt
        if self.approach.any():
            self.flat.compute()
            ids = self.approach.nonzero().flatten()
            for name in ("q", "qdot", "feedforward_torque", "foot_target", "reference_com", "reference_velocity",
                         "actual_com", "actual_velocity", "contact_points", "reference_cop", "support_fraction",
                         "capture_error", "ik_error", "height", "steps", "stance", "plan_violation", "landing_margin"):
                getattr(self, name)[ids] = getattr(self.flat, name)[ids]
            self.state[ids] = torch.where(self.flat.state[ids] == self.flat.SETTLE,
                                          self.SETTLE, self.flat.state[ids] + 20)
            bad = ids[self.flat.failure[ids] > 0]
            if bad.numel():
                com, velocity, soles, _ = self.flat._measure()
                self._hold(bad, com, velocity, soles, 10)
        self.impedance.update()
        self.drive_stiffness[:] = self.robot.data.joint_stiffness[:,self.joint_ids].reshape(-1,2,6)
        self.drive_damping[:] = self.robot.data.joint_damping[:,self.joint_ids].reshape(-1,2,6)
        self.drive_gain_blend[:] = (self.impedance.age/self.options["stair_drive_handover_s"]).clamp(0,1)
        active_ids = (~self.approach).nonzero().flatten()
        if not active_ids.numel():
            return self.q.reshape(self.n, 12), self.qdot.reshape(self.n, 12), self.feedforward_torque.reshape(self.n, 12)
        com, velocity, soles, force = self.flat._measure()
        self.actual_com, self.actual_velocity = com, velocity
        # Store the same physical sample that feeds this controller, rather
        # than pairing it later with post-step trajectory poses.
        self.control_root_position[:] = self.robot.data.root_pos_w
        self.control_root_quaternion[:] = self.robot.data.root_quat_w
        self.control_frame_origin[:] = self.flat.origin
        self.control_frame_heading[:] = self.flat.heading
        self.control_root_link_velocity[:] = self.robot.data.root_link_lin_vel_w
        self.control_root_angular_velocity[:] = self.robot.data.root_ang_vel_w
        self.control_joint_position[:] = self.robot.data.joint_pos[:, self.joint_ids].reshape(-1, 2, 6)
        self.control_joint_velocity[:] = self.robot.data.joint_vel[:, self.joint_ids].reshape(-1, 2, 6)
        self.control_soles[:] = soles
        self.control_foot_quaternion[:] = self.robot.data.body_quat_w[:, self.flat.feet]
        self.control_sample_step[:] = int(self.env.common_step_counter)
        self.control_sample_valid[:] = True
        local_com = self._local(com)
        local_vel = quat_apply_inverse(self.flat.heading, velocity)
        index = torch.arange(self.n, device=self.device)
        low_surface = self.anchors[..., 2].amin(1) + self.cfg["ground_contact_depth"]
        ready = (~self.approach & (self.state == self.BRAKE) & (self.time >= self.options["brake_duration"]))
        ready |= ~self.approach & (self.state == self.WAIT_MAP)
        ids = ready.nonzero().flatten()
        if ids.numel():
            valid = self._pair(ids, soles)
            good = ids[valid]
            if good.numel():
                self._shift(good, 1 - self.leading[good], com, velocity,
                             low_surface[good] + self.nominal_height[good], self.SHIFT)
            wait = ids[~valid]
            self.state[wait] = self.WAIT_MAP
            self.map_wait[wait] += dt
            bad = wait[self.map_wait[wait] > self.options["map_wait_timeout"]]
            if bad.numel():
                self._hold(bad, com, velocity, soles, 11)

        transition = (~self.approach & ((self.state == self.SHIFT) | (self.state == self.TRANSFER))
                      & (self.time >= self.segment_duration))
        side = self.stance
        h = (local_com[:, 2] - self.anchors[index, side, 2]).clamp(.60, 1.15)
        capture = local_com[:, :2] + local_vel[:, :2] * torch.sqrt(h[:, None] / 9.81)
        delta = capture - self.anchors[index, side, :2]
        fraction = force[index, side] / force.sum(-1).clamp_min(1)
        foot_rotation = matrix_from_quat(self.flat.heading).transpose(-1,-2)[:,None] @ matrix_from_quat(
            self.robot.data.body_quat_w[:,self.flat.feet])
        foot_delta = (foot_rotation[index,side,:2,:2].transpose(-1,-2) @ delta[...,None]).squeeze(-1)
        stable = (foot_delta[:, 0].abs() < .085) & (foot_delta[:, 1].abs() < .035) & (fraction > .65)
        self.capture_error = foot_delta[:, 1]
        self.support_fraction = fraction
        ids = (transition & stable).nonzero().flatten()
        if ids.numel():
            side = torch.where(self.state[ids] == self.SHIFT, self.leading[ids], 1 - self.leading[ids])
            self._swing(ids, side, com, soles)
        bad = (transition & (self.time > self.segment_duration + .8)).nonzero().flatten()
        if bad.numel():
            self._hold(bad, com, velocity, soles, 12)

        swing = ~self.approach & (self.state == self.SWING)
        foot = self.swing_side
        swing_ids = swing.nonzero().flatten()
        if swing_ids.numel():
            selected_foot = foot[swing_ids]
            quat = self.robot.data.body_quat_w[swing_ids][:, self.flat.feet][torch.arange(swing_ids.numel(), device=self.device), selected_foot]
            sole = soles[swing_ids, selected_foot]
            corners = self.foot_corners[selected_foot]
            world_corners = sole[:, None] + quat_apply(quat[:, None].expand(-1, 4, -1), corners)
            target_world = self.flat.origin[swing_ids] + (matrix_from_quat(self.flat.heading[swing_ids]) @ self.swing_target[swing_ids, :, None]).squeeze(-1)
            clear = world_corners[..., 2].amin(1) >= target_world[:, 2] + self.cfg["ground_contact_depth"] + .012
            rising=(self.swing_target[swing_ids,2]-self.swing_start[swing_ids,2]>.04)&~self.preparing[swing_ids]
            lifted=~rising|(self.time[swing_ids]>=self.options["flight_duration"]*.5)
            released = swing_ids[clear & lifted & ~self.clearance_passed[swing_ids]]
            self.clearance_passed[released] = True
            self.cross_time[released] = self.time[released]
            over = self._over_target_patch(swing_ids, soles)
            self.cross_advancing[swing_ids] = self.clearance_passed[swing_ids] & (clear | over)
            self.cross_elapsed[swing_ids] += dt*self.cross_advancing[swing_ids]
            descend = swing_ids[over & self.clearance_passed[swing_ids] & (self.descend_time[swing_ids] < 0)
                                & (self.cross_elapsed[swing_ids] >= self.options["cross_duration"] * .7)]
            self.descend_time[descend] = self.time[descend]
            self.descent_advancing[swing_ids] = (self.descend_time[swing_ids] >= 0) & over
            self.descent_elapsed[swing_ids] += dt*self.descent_advancing[swing_ids]
        air = force[index, foot] < 5
        self.air_time = torch.where(swing & air, self.air_time + dt, torch.zeros_like(self.air_time))
        self.seen_air |= swing & (self.air_time >= .08 - 1e-6)
        threshold = torch.maximum(torch.full_like(self.time, 20), .10 * self.total_mass * 9.81)
        foot_local = self._local(soles)[index, foot]
        near = ((foot_local[:, :2] - self.swing_target[:, :2]).norm(dim=-1) < .035)
        near &= (foot_local[:, 2] - self.swing_target[:, 2]).abs() < .04
        check_ids = (swing | (self.state == self.LAND)).nonzero().flatten()
        if check_ids.numel():
            near[check_ids] &= self._over_target_patch(check_ids, soles)
        # Initial contact on the verified patch enters a separate load phase.
        # A zero-load swing cannot be required to carry10% body weight before
        # receiving any planned GRF. Keep the old foot supporting throughout.
        contact_ids = (swing & self.seen_air & near & (force[index, foot] > 5)).nonzero().flatten()
        if contact_ids.numel():
            freeze_landed_yaw(self,contact_ids,foot[contact_ids])
            contact_time = self.time[contact_ids].clone()
            rising = self.swing_target[contact_ids, 2] - self.swing_start[contact_ids, 2] > .04
            self.landing_pitch[contact_ids] = swing_pitch(self,contact_ids)[:,0]
            self.contact_age[contact_ids,foot[contact_ids]] = 0
            self.contact_pitch[contact_ids,foot[contact_ids]] = self.landing_pitch[contact_ids]
            self.anchors[contact_ids, foot[contact_ids]] = foot_local[contact_ids]
            self.anchors[contact_ids, foot[contact_ids], 2] = self.swing_target[contact_ids, 2]
            landed = foot[contact_ids]
            self.contact_z_offset[contact_ids,landed] = (foot_local[contact_ids,2]
                - self.anchors[contact_ids,landed,2])
            measured_world = self.control_foot_quaternion[contact_ids,landed]
            measured_heading = quat_mul(quat_conjugate(self.flat.heading[contact_ids]),measured_world)
            self.contact_measured_foot_quaternion[contact_ids,landed] = (measured_heading
                / measured_heading.norm(dim=-1,keepdim=True).clamp_min(1e-9))
            self.contact_rebased_this_step[contact_ids,landed] = True
            # Only the newly contacting foot changes seed; the other leg keeps
            # its previous reference. This is a position-command rebase, not a
            # claim that the published joint reference is globallyC2.
            self.q[contact_ids,landed] = self.control_joint_position[contact_ids,landed]
            self.start_com[contact_ids] = self.body_plan_position[contact_ids]
            self.start_velocity[contact_ids] = self.body_plan_velocity[contact_ids]
            self.start_acceleration[contact_ids] = self.body_plan_acceleration[contact_ids]
            self.state[contact_ids], self.time[contact_ids] = self.LAND, 0.0
        foot_rotation_now = matrix_from_quat(self.robot.data.body_quat_w[:,self.flat.feet])[index,foot]
        touch = ((self.state == self.LAND) & self.seen_air & near & (force[index, foot] > threshold)
                 & (foot_rotation_now[:,2,2]>.975))
        self.touchdown_count = torch.where(touch, self.touchdown_count + 1, 0)
        ids = (touch & (self.touchdown_count >= 4) & (self.time >= .20)).nonzero().flatten()
        if ids.numel():
            landed_side = foot[ids]
            was_trailing = self.trailing[ids].clone()
            was_preparing = self.preparing[ids].clone()
            self.anchors[ids, landed_side] = foot_local[ids]
            self.anchors[ids, landed_side, 2] = self.swing_target[ids, 2]
            self.steps[ids] += 1
            prepared = ids[was_preparing]
            if prepared.numel():
                sole_now = self._local(soles)[prepared]
                self.bottom_aligned[prepared] = (sole_now[:,0,0]-sole_now[:,1,0]).abs()<=.04
                self.preparing[prepared] = False
                self.leading[prepared] = 1 - self.leading[prepared]
                self.state[prepared], self.time[prepared], self.map_wait[prepared] = self.WAIT_MAP, 0, 0
                self.start_com[prepared] = self._plan_position(prepared, com)
                self.end_com[prepared] = self.start_com[prepared]
            lead = ids[~was_trailing & ~was_preparing]
            if lead.numel():
                self.trailing[lead] = True
                old_surface = self.anchors[lead, 1 - self.leading[lead], 2] + self.cfg["ground_contact_depth"]
                target_z = old_surface + self.nominal_height[lead]
                self._shift(lead, self.leading[lead], com, velocity, target_z, self.TRANSFER)
            rear = ids[was_trailing & ~was_preparing]
            if rear.numel():
                complete = rear[self.exit_pair[rear]]
                if complete.numel():
                    self.state[complete], self.time[complete] = self.TOP_STAND, 0.0
                    self.start_com[complete] = self._plan_position(complete, com)
                    self.end_com[complete] = self.anchors[complete].mean(1)
                    offset = self.body_z_offset[complete]
                    self.end_com[complete, 2] += self.cfg["ground_contact_depth"] + self.options["top_pelvis_height"] + offset
                again = rear[~self.exit_pair[rear]]
                if again.numel():
                    self.levels_completed[again] += 1
                    self.state[again], self.time[again], self.map_wait[again] = self.WAIT_MAP, 0.0, 0.0
                    self.start_com[again] = self._plan_position(again, com)
                    self.end_com[again] = self.start_com[again]
        bad = (((self.state == self.SWING) & (self.time > self.options["flight_duration"] + .30))
               | ((self.state == self.LAND) & (self.time > .60))).nonzero().flatten()
        if bad.numel():
            self._hold(bad, com, velocity, soles, 13)

        self._compose(active_ids, com, velocity, soles, force)
        return self.q.reshape(self.n, 12), self.qdot.reshape(self.n, 12), self.feedforward_torque.reshape(self.n, 12)

    def _compose(self, ids, com, velocity, soles, force):
        t = self.time[ids]
        state = self.state[ids]
        p = self.start_com[ids].clone()
        v, a = torch.zeros_like(p), torch.zeros_like(p)
        brake = (state == self.BRAKE) | (state == self.WAIT_MAP) | (state == self.HOLD)
        if brake.any():
            sub = ids[brake]
            omega = torch.sqrt(9.81 / self.nominal_height[sub])[:, None]
            e = (self.start_com[sub, :2] - self.end_com[sub, :2]) * torch.exp(-omega * t[brake, None])
            p[brake, :2] = self.end_com[sub, :2] + e
            v[brake, :2], a[brake, :2] = -omega * e, omega.square() * e
            z, zd, za = profile(t[brake], self.options["brake_duration"])
            dz = self.end_com[sub, 2] - self.start_com[sub, 2]
            p[brake, 2] += dz * z
            v[brake, 2], a[brake, 2] = dz * zd, dz * za
        shift = (state == self.SHIFT) | (state == self.TRANSFER)
        if shift.any():
            sub = ids[shift]
            omega = torch.sqrt(9.81 / self.segment_h[sub])[:, None]
            half = self.segment_duration[sub, None] * .5
            segment_time = torch.minimum(t[shift, None], self.segment_duration[sub, None])
            first = torch.minimum(segment_time, half)
            xy, speed = lipm(self.start_com[sub, :2], self.start_velocity[sub, :2], self.cop0[sub], omega, first)
            second = (segment_time - half).clamp_min(0)
            xy, speed = lipm(xy, speed, self.cop1[sub], omega, second)
            cop = torch.where(t[shift, None] <= half, self.cop0[sub], self.cop1[sub])
            p[shift, :2], v[shift, :2], a[shift, :2] = xy, speed, omega.square() * (xy - cop)
            ended = t[shift] >= self.segment_duration[sub]
            selected = shift.nonzero().flatten()[ended]
            p[selected, :2] = self.end_com[sub[ended], :2]
            v[selected, :2], a[selected, :2] = 0, 0
            z, zd, za = profile(t[shift], self.segment_duration[sub])
            dz = self.end_com[sub, 2] - self.start_com[sub, 2]
            p[shift, 2] += dz * z
            v[shift, 2], a[shift, 2] = dz * zd, dz * za
            high = self.segment_c2[sub]
            if high.any():
                selected = shift.nonzero().flatten()[high]
                high_ids = sub[high]
                p[selected],v[selected],a[selected] = settle_profile(t[selected],self.segment_duration[high_ids],
                    self.start_com[high_ids],self.start_velocity[high_ids],self.start_acceleration[high_ids],self.end_com[high_ids])
        swinging = state == self.SWING
        landing = state == self.LAND
        foot_target = self.anchors[ids].clone()
        pose_restore = smooth(self.contact_age[ids]/.30)
        foot_target[...,2] += self.contact_z_offset[ids]*(1-pose_restore)
        if swinging.any():
            sub = ids[swinging]
            u = t[swinging] / self.options["flight_duration"]
            blend, speed, acc = profile(t[swinging], self.options["flight_duration"])
            dz = self.end_com[sub, 2] - self.start_com[sub, 2]
            p[swinging] = self.end_com[sub]
            p[swinging, 2] = self.start_com[sub, 2] + dz * blend
            v[swinging, 2], a[swinging, 2] = dz * speed, dz * acc
            moving = swing_position_local(self,sub)[:,0]
            foot_target[swinging, self.swing_side[sub]] = moving
        if landing.any():
            sub = ids[landing]
            p[landing], v[landing], a[landing] = settle_profile(
                t[landing], .30, self.start_com[sub], self.start_velocity[sub],
                self.start_acceleration[sub], self.end_com[sub])
        top = state == self.TOP_STAND
        if top.any():
            sub = ids[top]
            blend, speed, acc = profile(t[top], 1.5)
            delta = self.end_com[sub] - self.start_com[sub]
            p[top] += blend[:, None] * delta
            v[top], a[top] = speed[:, None] * delta, acc[:, None] * delta

        actual = self._local(com)[ids]
        actual_v = quat_apply_inverse(self.flat.heading, velocity)[ids]
        mean_surface = (self.anchors[ids, :, 2] * self.load[ids]).sum(-1) + self.cfg["ground_contact_depth"]
        height = (actual[:, 2] - mean_surface).clamp(.60, 1.10)
        root = self._local(self.robot.data.root_pos_w)[ids]
        root_v = quat_apply_inverse(self.flat.heading, self.robot.data.root_link_lin_vel_w)[ids]
        vertical = (9.81 + a[:, 2] + 35 * (p[:, 2] - root[:, 2] - self.body_z_offset[ids])
                    + 10 * (v[:, 2] - root_v[:, 2])).clamp(3.0, self.options["maximum_vertical_acceleration"])
        self.body_plan_position[ids], self.body_plan_velocity[ids] = p, v
        self.body_plan_acceleration[ids] = a
        # Reconstruct the actual total-COM reference for logging and XY hip
        # feedback. The Z acceleration above is root-reference acceleration;
        # moving-base/shape-inertia feedforward is not claimed here.
        p[:, 2] += actual[:, 2] - root[:, 2] - self.body_z_offset[ids]
        v[:, 2] += actual_v[:, 2] - root_v[:, 2]
        omega2 = vertical / height
        acceleration = a[:, :2] + self.cfg["com_position_gain"] * (p[:, :2] - actual[:, :2])
        acceleration += self.cfg["com_velocity_gain"] * (v[:, :2] - actual_v[:, :2])
        cop = actual[:, :2] - acceleration / omega2[:, None]
        self.requested_cop[ids] = cop
        self.requested_acceleration_local[ids] = acceleration
        self.pressure_seed_box_applied[ids] = ~swinging
        bounds = torch.tensor([.06, .025], device=self.device)
        low = self.anchors[ids, :, :2].amin(1) - bounds
        high = self.anchors[ids, :, :2].amax(1) + bounds
        side = self.stance[ids]
        # SS pressure is limited by the actual rotated sole and loaded native
        # joint limits below. An additional world-axis rectangle can turn a
        # valid braking request into acceleration away from the support.
        boxed_cop = torch.maximum(torch.minimum(cop, high), low)
        cop = torch.where(swinging[:, None], cop, boxed_cop)
        load,projected_cop,_ = allocate_two_feet(cop,self.anchors[ids,:,:2],bounds,
                                               preferred=self.load[ids,0])
        cop = torch.where((swinging|landing)[:,None],cop,projected_cop)
        load[swinging] = 0
        load[swinging, side[swinging]] = 1
        if landing.any():
            sub = ids[landing]
            landing_share = .12 * smooth(t[landing] / .20)
            load[landing] = 0
            load[landing, self.stance[sub]] = 1 - landing_share
            load[landing, self.swing_side[sub]] = landing_share
        hold = state == self.HOLD
        load[hold] = force[ids[hold]] / force[ids[hold]].sum(-1, keepdim=True).clamp_min(1)
        center = (self.anchors[ids, :, :2] * load[..., None]).sum(1)
        points = self.anchors[ids].clone()
        offset = cop - center
        offset = torch.where(swinging[:, None], offset, offset.clamp(-bounds, bounds))
        points[..., :2] += offset[:, None]
        points[..., 2] += self.cfg["ground_contact_depth"]
        actual_cop = (points[..., :2] * load[..., None]).sum(1)
        self.pressure_centers[ids] = self.anchors[ids,:,:2]
        self.pressure_radii[ids] = bounds
        support_z = (points[..., 2] * load).sum(1)
        height = (actual[:, 2] - support_z).clamp(.60, 1.10)
        omega2 = vertical / height
        acceleration = omega2[:, None] * (actual[:, :2] - actual_cop)
        self.reference_com[ids] = self.flat.origin[ids] + (matrix_from_quat(self.flat.heading[ids]) @ p[..., None]).squeeze(-1)
        self.reference_velocity[ids] = (matrix_from_quat(self.flat.heading[ids]) @ v[..., None]).squeeze(-1)
        self.reference_acceleration_local[ids] = a
        self.reference_cop[ids], self.load[ids], self.height[ids] = actual_cop, load, height
        self.contact_points[ids] = self.flat.origin[ids, None] + (matrix_from_quat(self.flat.heading[ids])[:, None] @ points[..., None]).squeeze(-1)
        pelvis = p - (actual - root)
        qroot = quat_mul(quat_conjugate(self.flat.heading[ids]), self.robot.data.root_quat_w[ids])
        qroot = torch.where(qroot[:, :1] < 0, -qroot, qroot)
        actual_inverse = matrix_from_quat(qroot).transpose(-1,-2)
        actual_q = self.robot.data.joint_pos[ids][:,self.joint_ids].reshape(-1,2,6)
        limits = self.robot.data.joint_effort_limits[ids][:,self.joint_ids].reshape(-1,2,6)
        heading = matrix_from_quat(self.flat.heading[ids])
        actual_foot_rotation = heading[:,None].transpose(-1,-2)@matrix_from_quat(
            self.robot.data.body_quat_w[ids][:,self.flat.feet])
        actual_soles = self._local(soles)[ids]
        proposed_points = points.clone()
        single_feasible = None
        if swinging.any():
            # In SS the force changes with the CoP, but the complete joint
            # torque remains affine in its two horizontal coordinates. Solve
            # that coupled constraint directly with the original native caps.
            selected, _, single_feasible, _ = ss_affine_native_project(
                self.kin, actual_q[swinging], actual_inverse[swinging], points[swinging],
                root[swinging], actual[swinging], self.total_mass[ids[swinging]],
                vertical[swinging], (vertical / height)[swinging], side[swinging],
                actual_soles[swinging], actual_foot_rotation[swinging],
                limits[swinging], self.foot_corners)
            points[swinging] = selected
        # Actual q/root gravity are fixed across these three pressure passes.
        support_pose = prepare_support_pose(self.kin,actual_q,-actual_inverse[...,2])
        for projection in range(3):
            actual_cop = (points[...,:2]*load[...,None]).sum(1)
            support_z = (points[...,2]*load).sum(1)
            height = (actual[:,2]-support_z).clamp(.60,1.10)
            acceleration = (vertical/height)[:,None]*(actual[:,:2]-actual_cop)
            grf = torch.cat((acceleration,vertical[:,None]),-1)
            forces = self.total_mass[ids,None,None]*load[...,None]*grf[:,None]
            forces = (actual_inverse[:,None]@forces[...,None]).squeeze(-1)
            points,_,valid = project_wrench_points(self.kin,actual_q,actual_inverse,forces,points,
                root,actual_soles,actual_foot_rotation,limits,self.foot_corners,
                support_pose=support_pose)
        actual_cop = (points[...,:2]*load[...,None]).sum(1)
        acceleration = (vertical/height)[:,None]*(actual[:,:2]-actual_cop)
        grf = torch.cat((acceleration,vertical[:,None]),-1)
        forces = self.total_mass[ids,None,None]*load[...,None]*grf[:,None]
        forces = (actual_inverse[:,None]@forces[...,None]).squeeze(-1)
        points_root = (actual_inverse[:,None]@(points-root[:,None])[...,None]).squeeze(-1)
        self.feedforward_torque[ids] = cached_support_torque(support_pose,forces,points_root)
        self.feedforward_effort_ratio[ids] = self.feedforward_torque[ids]/limits.clamp_min(1e-6)
        self.projected_contact_feasible[ids] = valid&(self.feedforward_torque[ids].abs()<=limits+.05).all(-1)
        if single_feasible is not None:
            strict = (self.feedforward_torque[ids[swinging]].abs() <= limits[swinging]).all((1, 2))
            self.projected_contact_feasible[ids[swinging]] = (single_feasible & strict)[:, None]
        self.pressure_projection_shift[ids] = (points-proposed_points).norm(dim=-1)
        self.reference_cop[ids] = actual_cop
        self.contact_points[ids] = self.flat.origin[ids,None]+(heading[:,None]@points[...,None]).squeeze(-1)
        swing_ids = ids[swinging]
        if swing_ids.numel():
            selected_yaw,feasible,swing_command,swing_diagnostic = prepare_swing_command(
                self,swing_ids,soles,actual_gravity=self.feedforward_torque)
            accepted = swing_ids[feasible]
            self.swing_yaw_goal[accepted] = selected_yaw[feasible]
            self.foot_yaw[accepted,self.swing_side[accepted]] = selected_yaw[feasible]
            self.yaw_checked_time[swing_ids] = self.time[swing_ids]
            for column,key in enumerate(("joint_margin","drive_margin","patch_margin",
                                         "current_position_error","constant_candidate_count","rate_drive_feasible")):
                self.swing_ik_diagnostics[swing_ids,column] = swing_diagnostic[key].to(self.time.dtype)
            rejected = swing_ids[~feasible]
            if rejected.numel():
                self._hold(rejected,com,velocity,soles,15)
                self._compose(ids,com,velocity,soles,force)
                return
        blend = 1-smooth(self.contact_age[ids]/.30)
        blend[swinging, self.swing_side[ids[swinging]]] = 1
        identity = torch.zeros_like(qroot)
        identity[:, 0] = 1
        rotation = torch.lerp(identity[:, None], qroot[:, None], blend[..., None])
        rotation /= rotation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        inverse = matrix_from_quat(rotation).transpose(-1, -2)
        leg_root = torch.lerp(pelvis[:, None], root[:, None], blend[..., None])
        position = (inverse @ (foot_target - leg_root)[..., None]).squeeze(-1)
        yaw = self.foot_yaw[ids]
        c,s = yaw.cos(),yaw.sin()
        turn = torch.zeros(ids.numel(),2,3,3,device=self.device)
        turn[...,0,0],turn[...,0,1],turn[...,1,0],turn[...,1,1],turn[...,2,2] = c,-s,s,c,1
        foot_rotation = self.flat.home_foot_rotation[None] @ turn
        pitch = self.contact_pitch[ids]*(1-smooth(self.contact_age[ids]/.30))
        if swinging.any():
            sub = ids[swinging]
            pitch[swinging,self.swing_side[sub]] = swing_pitch(self,sub)[:,0]
        c,s = pitch.cos(),pitch.sin()
        toe_up = torch.zeros(ids.numel(),2,3,3,device=self.device)
        toe_up[...,0,0],toe_up[...,0,2],toe_up[...,1,1],toe_up[...,2,0],toe_up[...,2,2] = c,s,1,-s,c
        foot_rotation = foot_rotation @ toe_up
        # Restore complete measured heading-frame rotation, including roll.
        # A moving SWING foot retains its original trajectory orientation.
        if swinging.any():
            pose_restore[swinging,self.swing_side[ids[swinging]]] = 1
        self.contact_pose_restore_weight[ids] = pose_restore
        nominal_quaternion = quat_from_matrix(foot_rotation)
        foot_rotation = matrix_from_quat(blend_contact_quaternion(
            self.contact_measured_foot_quaternion[ids],nominal_quaternion,pose_restore))
        orientation = inverse @ foot_rotation
        # Independent measured-pressure compliance acts ONLY on virtual pelvis
        # IKX. All nominal p/v/a, CoP, loads and GRF above are unchanged.
        step = int(self.env.common_step_counter)
        heading = matrix_from_quat(self.flat.heading[ids])
        measured_error,measurement_valid = ss_normal_cop_error_x(self.map.tactile.latest,
            ids=ids,step=step,stance=self.stance,origin=self.flat.origin[ids],
            heading_matrix=heading,reference_cop=self.reference_cop[ids],
            planned_load=self.load[ids],single_support=swinging)
        balance_omega = torch.sqrt(vertical/height)
        compliance_omega = torch.sqrt(balance_omega*self.options["stair_drive_frequency_rad_s"])
        proposal = self.ik_com_x_admittance.preview(ids,step=step,error=measured_error,
            measurement_valid=measurement_valid,omega=compliance_omega,
            enabled=~self.approach[ids])
        moving_mask = torch.zeros_like(blend,dtype=torch.bool)
        moving_command = self.q[ids].clone()
        moving_error = torch.zeros_like(blend)
        if swing_ids.numel():
            row = torch.arange(swing_ids.numel(),device=self.device)
            selected = self.swing_side[swing_ids]
            moving_mask[swinging,selected] = True
            moving_command[swinging,selected] = swing_command[row,selected]
            moving_error[swinging,selected] = swing_diagnostic["current_position_error"]
        fresh_contact = self.contact_rebased_this_step[ids]
        runtime_soft = self.robot.data.soft_joint_pos_limits[:,self.joint_ids].reshape(-1,2,6,2)[ids]
        offset_ik = solve_ik_x_with_feasibility(self.kin,position_zero=position,
            orientation=orientation,inverse=inverse,root_blend=blend,seed=self.q[ids],
            proposal_x=proposal["x"],previous_x=self.ik_com_x_admittance.x[ids],
            fresh_contact=fresh_contact,measured_q=self.control_joint_position[ids],
            moving_mask=moving_mask,moving_command=moving_command,moving_error=moving_error,
            feedforward=self.feedforward_torque[ids],stiffness=self.drive_stiffness[ids],
            effort_limit=self.robot.data.joint_effort_limits[:,self.joint_ids].reshape(-1,2,6)[ids],
            lower=runtime_soft[...,0],upper=runtime_soft[...,1],dt=self.env.step_dt,
            maximum_rate=self.options["maximum_joint_rate"],
            joint_reserve=self.options["joint_kinematic_reserve"])
        # Optional compliance cannot introduce a controller HOLD. If neither
        # bounded offset passes its extra checks, publish the original
        # zero-offset/rate-limited baseline and canonicalize ONLY the component
        # below. This fallback is not C2 or a physical-safety guarantee; the
        # original actuator, effort, joint, contact and acceptance gates remain.
        target,error,position = offset_ik["target"],offset_ik["error"],offset_ik["position"]
        raw_delta = torch.where(fresh_contact[...,None],torch.zeros_like(target),target-self.q[ids])
        maximum_rate = self.options["maximum_joint_rate"]
        delta = raw_delta.clamp(-maximum_rate * self.env.step_dt, maximum_rate * self.env.step_dt)
        self.rate_limit_count[ids] += (raw_delta - delta).abs().amax(dim=(1, 2)) > 1e-6
        self.qdot[ids] = (delta / self.env.step_dt).clamp(-maximum_rate, maximum_rate)
        self.q[ids] += delta
        self.q[ids],self.qdot[ids] = rebase_landed_joint_reference(
            self.q[ids],self.qdot[ids],self.control_joint_position[ids],
            self.control_joint_velocity[ids],fresh_contact,maximum_rate)
        committed = self.ik_com_x_admittance.commit(ids,step=step,proposal=proposal,
            feasible=offset_ik["accept_new"],baseline_fallback=offset_ik["baseline_fallback"])
        for name,value in (("ik_com_x_delta",committed["x"]),("ik_com_x_velocity",committed["v"]),
                ("ik_com_x_acceleration_formula",proposal["acceleration_formula"]),
                ("ik_com_x_error",proposal["input_error"]),("ik_com_x_valid",proposal["input_valid"]),
                ("ik_com_x_omega",proposal["omega"]),("ik_com_x_balance_omega",balance_omega),("ik_com_x_bound_active",proposal["bound_active"]),
                ("ik_com_x_feasible",offset_ik["accept_new"]),("ik_com_x_held",committed["held_for_feasibility"]),
                ("ik_com_x_baseline_only",offset_ik["baseline_only"]),
                ("ik_com_x_baseline_fallback",offset_ik["baseline_fallback"]),
                ("ik_com_x_baseline_evaluated",offset_ik["baseline_evaluated"]),
                ("ik_com_x_baseline_feasible",offset_ik["baseline_feasible"]),
                ("ik_com_x_previous_offset_feasible",offset_ik["previous_offset_feasible"]),
                ("ik_com_x_selection_branch",offset_ik["selection_branch"]),
                ("ik_com_x_command_rotation_error",torch.where((blend<1-1e-6)&~fresh_contact,offset_ik["command_rotation_error"],0).amax(-1))):
            getattr(self,name)[ids] = value.to(getattr(self,name).dtype)
        event = committed["reset_this_call"]
        event_ids = ids[event]
        # Latch event inputs/reasons until the next event, so a20ms fallback
        # remains explainable even when telemetry is sampled every100ms.
        for name,value in (("ik_com_x_last_fallback_old_delta",committed["previous_x"]),
                ("ik_com_x_last_fallback_old_velocity",committed["previous_v"]),
                ("ik_com_x_last_fallback_proposed_delta",proposal["x"]),
                ("ik_com_x_last_fallback_proposed_velocity",proposal["v"]),
                ("ik_com_x_last_fallback_candidate_bits",offset_ik["candidate_failure_bits"]),
                ("ik_com_x_last_fallback_old_bits",offset_ik["previous_offset_failure_bits"]),
                ("ik_com_x_last_fallback_baseline_bits",offset_ik["baseline_failure_bits"])):
            getattr(self,name)[event_ids] = value[event].to(getattr(self,name).dtype)
        for name,key in (("ik_com_x_baseline_fallback_count","baseline_fallback_count"),
                         ("ik_com_x_reset_count","reset_count"),
                         ("ik_com_x_new_accepted_count","new_accepted_count"),
                         ("ik_com_x_old_held_count","old_held_count"),
                         ("ik_com_x_last_fallback_step","last_fallback_step")):
            getattr(self,name)[ids] = getattr(self.ik_com_x_admittance,key)[ids]
        self.ik_com_x_step[ids] = step
        self.ik_solver_error[ids] = error
        actual_command_position, _ = self.kin.forward(self.q[ids])
        self.ik_error[ids] = (actual_command_position - position).norm(dim=-1)
        self.foot_target[ids] = self.flat.origin[ids, None] + (matrix_from_quat(self.flat.heading[ids])[:, None] @ foot_target[..., None]).squeeze(-1)
        if swing_ids.numel() and self.options.get("moving_inertia_feedforward",False):
            dynamic = compute_swing_dynamics(self,swing_ids,
                preview_dt=self.options["kinematic_preview_dt"],physics_dt=self.env.cfg.sim.dt)
            side = self.swing_side[swing_ids]
            for destination,key in (("dynamic_qddot","qddot_ref"),("dynamic_qdot_preview","qdot_preview"),
                                    ("dynamic_torque_raw","torque_raw"),("dynamic_torque_applied","torque_applied"),
                                    ("dynamic_inertia_torque","inertia_raw"),("dynamic_coriolis_torque","coriolis_raw"),
                                    ("dynamic_velocity_bias","velocity_bias"),("dynamic_velocity_target","velocity_target")):
                getattr(self,destination)[swing_ids,side] = dynamic[key]
            self.dynamic_valid[swing_ids,side] = dynamic["valid"]
            accepted = swing_ids[dynamic["valid"]]
            accepted_side = self.swing_side[accepted]
            self.dynamic_peak_raw[accepted,accepted_side] = torch.maximum(
                self.dynamic_peak_raw[accepted,accepted_side],dynamic["torque_raw"][dynamic["valid"]].abs())
            self.dynamic_peak_applied[accepted,accepted_side] = torch.maximum(
                self.dynamic_peak_applied[accepted,accepted_side],dynamic["torque_applied"][dynamic["valid"]].abs())
            self.dynamic_applied_steps[swing_ids,side] += dynamic["valid"].long()
            self.dynamic_clip_steps[swing_ids,side] += (dynamic["valid"]&dynamic["diagnostics"]["applied_dynamic_clipped"]).long()
            self.dynamic_mass_diagonal[swing_ids,side] = dynamic["diagnostics"]["native_mass_diagonal"]
            for column,key in enumerate(("boundary_skipped","rate_ok","center_matched","preview_solved",
                                         "static_budget_ok","command_FK_error_m","one_side_preview_rate_rad_s",
                                         "applied_dynamic_clipped")):
                self.dynamic_diagnostics[swing_ids,side,column] = dynamic["diagnostics"][key].to(self.time.dtype)

    def summary(self):
        return {"mode": "lidar_step_to_stairs_with_frozen_residual", "network_structure_changed": False,
                "level_counts": {str(i): int((self.levels_completed == i).sum()) for i in range(7)},
                "state_counts": {str(i): int((self.state == i).sum()) for i in range(9)},
                "failure_counts": {str(i): int((self.failure == i).sum()) for i in (0, 10, 11, 12, 13, 14, 15)},
                "ik_error_max_m": float(self.ik_error.max()), "perception": self.map.summary()}
