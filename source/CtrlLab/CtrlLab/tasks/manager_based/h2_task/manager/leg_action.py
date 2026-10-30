"""Residual joint positions bounded by the articulation's own soft limits."""

import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.math import quat_apply_inverse, quat_apply, yaw_quat


class BoundedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._orientation_feedback = getattr(env.cfg, "stance_orientation_feedback", None)
        self._foot_ids = env.scene["contact_forces"].find_bodies(env.cfg.feet_body_names, preserve_order=True)[0]
        self._robot_feet = self._asset.find_bodies(env.cfg.feet_body_names, preserve_order=True)[0]
        self._masses = self._asset.root_physx_view.get_masses().to(self.device)
        torques = getattr(env.cfg, "stance_preload_torques", {})
        self._preload_groups = []
        for actuator in self._asset.actuators.values():
            joint_ids = (list(range(self._asset.num_joints)) if isinstance(actuator.joint_indices, slice)
                         else [int(i) for i in actuator.joint_indices])
            action_indices, local_indices, values = [], [], []
            for action_index, (joint_id, name) in enumerate(zip(self._joint_ids, self._joint_names)):
                if name in torques and int(joint_id) in joint_ids:
                    action_indices.append(action_index)
                    local_indices.append(joint_ids.index(int(joint_id)))
                    values.append(torques[name])
            if action_indices:
                self._preload_groups.append((action_indices, local_indices, actuator,
                                             torch.tensor(values, device=self.device)))
        self.gait_reference = None
        self._support_priority_groups = []
        reference_cfg = getattr(env.cfg, "contact_gait_reference", None)
        if reference_cfg is not None:
            if getattr(env.cfg, "stair_reference", None) is not None:
                from ..stair_reference import StairWalkingReference
                self.gait_reference = StairWalkingReference(env, self._asset, self._joint_ids, reference_cfg)
            else:
                from ..walking_reference import ContactWalkingReference
                self.gait_reference = ContactWalkingReference(env, self._asset, self._joint_ids, reference_cfg)
            self._residual_limits = torch.tensor(reference_cfg["residual_angle_limits"] * 2,
                                                 device=self.device)
            if getattr(env.cfg, "stair_reference", None) is not None:
                selected = set(env.cfg.stair_reference["support_effort_priority_axes"])
                for action_ids, local_ids, actuator, _ in self._preload_groups:
                    from ..stair_receiver import attach_receiver
                    attach_receiver(actuator, self.gait_reference, action_ids, local_ids)
                    pairs = [(a, j) for a, j in zip(action_ids, local_ids) if a % 6 in selected]
                    if not pairs:
                        continue
                    actions, local = map(list, zip(*pairs))
                    native = (list(range(self._asset.num_joints)) if isinstance(actuator.joint_indices, slice)
                              else [int(j) for j in actuator.joint_indices])
                    actuator.enable_support_effort_priority(self.gait_reference, actions, local,
                        self._asset.data.soft_joint_pos_limits[:, native].clone(),
                        self._asset.data.joint_vel_limits[:, native].clone())
                    self._support_priority_groups.append((actuator, actions, local))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if getattr(self, "gait_reference", None) is not None:
            self.gait_reference.reset(env_ids)

    def apply_actions(self):
        super().apply_actions()
        if self.gait_reference is not None:
            velocity = self.gait_reference.qdot.reshape(self.num_envs,12)
            if getattr(self._env.cfg,"stair_reference",None) is not None:
                velocity = velocity+self.gait_reference.dynamic_velocity_bias.reshape(self.num_envs,12)
                limit = self._asset.data.joint_vel_limits[:,self._joint_ids].clamp_max(30.)
                velocity = velocity.clamp(-limit,limit)
                self.gait_reference.drive_velocity_target[:] = velocity.reshape(self.num_envs,2,6)
            self._asset.set_joint_velocity_target(velocity,joint_ids=self._joint_ids)

    def process_actions(self, actions):
        super().process_actions(actions)
        preload = None
        if self.gait_reference is not None:
            reference, _, preload = self.gait_reference.compute()
            residual = self._residual_limits * torch.tanh(self._raw_actions)
            stair_cfg = getattr(self._env.cfg, "stair_reference", None)
            if stair_cfg is not None:
                gait = self.gait_reference
                # Preserve the frozen residual's nominal torque budget when
                # the native stair drive becomes stiffer after the approach.
                residual *= gait.impedance.base_ratio.reshape(self.num_envs,12)
                moving = (~gait.approach & (gait.state == gait.SWING)).nonzero().flatten()
                shaped = residual.view(self.num_envs, 2, 6)
                active = (~gait.approach).nonzero().flatten()
                u = (gait.contact_age[active]/.30).clamp(0,1)
                blend = u**3*(10-15*u+6*u**2)
                shaped[active] *= blend[...,None]
                shaped[moving, gait.swing_side[moving]] *= stair_cfg["swing_residual_scale"]
                gait.processed_residual[:] = residual
            self._processed_actions[:] = reference + residual
        # Preload is distinct from initialization/observation home. It provides
        # the nominal static support torque without requiring a large PD error.
        for action_ids, local_ids, actuator, torques in self._preload_groups:
            requested = torques if preload is None else preload[:, action_ids]
            self._processed_actions[:, action_ids] += requested / actuator.stiffness[:, local_ids].clamp_min(1e-6)
        if self._orientation_feedback is not None:
            self._apply_stance_orientation_feedback()
        if getattr(self._env.cfg, "lateral_com_feedback", None) is not None:
            self._apply_lateral_com_feedback()
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._processed_actions[:] = torch.maximum(
            torch.minimum(self._processed_actions, limits[..., 1]), limits[..., 0])
        if getattr(self._env.cfg, "stair_reference", None) is not None:
            from ..stair_drive import prepare_support_effort_command
            prepare_support_effort_command(self.gait_reference)
            for _, _, actuator, _ in self._preload_groups:
                actuator.stair_publication_epoch.fill_(int(self._env.common_step_counter))
            for actuator, actions, local in self._support_priority_groups:
                actuator.support_priority_command[:, local, 0] = self.gait_reference.support_effort_target.reshape(self.num_envs,12)[:, actions]
                actuator.support_priority_command[:, local, 1] = self.gait_reference.support_drive_role.reshape(self.num_envs,12)[:, actions]

    def _apply_stance_orientation_feedback(self):
        """A bounded IMU PD correction, only through loaded ankle joints."""
        cfg = self._orientation_feedback
        gravity = self._asset.data.projected_gravity_b
        tilt = torch.stack((torch.atan2(-gravity[:, 1], -gravity[:, 2]),
                            torch.atan2(gravity[:, 0], -gravity[:, 2])), dim=-1)
        omega = self._asset.data.root_ang_vel_b[:, :2]
        kp = torch.tensor(cfg["kp"], device=self.device)
        kd = torch.tensor(cfg["kd"], device=self.device)
        cap = torch.tensor(cfg["maximum_torque"], device=self.device)
        total_torque = torch.maximum(torch.minimum(kp * tilt + kd * omega, cap), -cap)
        force = self._env.scene["contact_forces"].data.net_forces_w[:, self._foot_ids, 2].clamp_min(0.0)
        force = torch.where(force > 20.0, force, 0.0)
        force = self._support_role_force(force)
        share = force / force.sum(-1, keepdim=True).clamp_min(1.0)
        corrections = {f"{side}_ankle_{axis}_joint": total_torque[:, axis_index] * share[:, side_index]
                       for side_index, side in enumerate(("left", "right"))
                       for axis_index, axis in enumerate(("roll", "pitch"))}
        for action_ids, local_ids, actuator, _ in self._preload_groups:
            for action_id, local_id in zip(action_ids, local_ids):
                name = self._joint_names[action_id]
                if name in corrections:
                    self._processed_actions[:, action_id] += corrections[name] / actuator.stiffness[:, local_id].clamp_min(1e-6)

    def _support_role_force(self, force):
        """Legacy stance assists belong to flat control and the flat approach.

        The active stair reference already owns body-pose IK, COM feedback,
        G/GRF feedforward and native joint PD. Adding the legacy ankle/hip
        assists there requests a second, incompatible share of the same drive
        budget. Transfer both assists together at the ownership boundary.
        """
        if getattr(self._env.cfg, "stair_reference", None) is not None:
            gait = self.gait_reference
            force = force.clone()
            force[~gait.approach] = 0
            gait.legacy_feedback_support_force[:] = force
        return force

    def _apply_lateral_com_feedback(self):
        """Center lateral mass and momentum over the current support feet."""
        cfg = self._env.cfg.lateral_com_feedback
        force = self._env.scene["contact_forces"].data.net_forces_w[:, self._foot_ids, 2].clamp_min(0)
        force = self._support_role_force(torch.where(force > 20, force, 0))
        contact = force > 0
        weight = contact.to(force.dtype)
        if getattr(self._env.cfg, "stair_reference", None) is not None:
            active = ~self.gait_reference.approach
            # Hip-angle feedback represents one common pelvis displacement,
            # not a net torque to divide between the two support legs.
            body_weight = self._masses.sum(-1)*9.81
            u = ((force-.10*body_weight[:,None])/(.10*body_weight[:,None])).clamp(0,1)
            ramp = u**3*(10-15*u+6*u**2)
            age = (self.gait_reference.contact_age/.30).clamp(0,1)
            age = age**3*(10-15*age+6*age**2)
            weight[active] *= (ramp*age)[active]
        n_contact = contact.sum(-1).clamp_min(1)
        feet = self._asset.data.body_pos_w[:, self._robot_feet]
        foot_quat = self._asset.data.body_quat_w[:, self._robot_feet]
        # Sole centers relative to the two ankle-link frames, in metres.
        offset = (self.gait_reference.kin.sole_offset if self.gait_reference is not None else
                  torch.tensor([[0.04, 0.037, -0.048], [0.04, -0.037, -0.048]], device=self.device))
        sole = feet + quat_apply(foot_quat, offset.expand_as(feet))
        support = (sole * contact[..., None]).sum(1) / n_contact[:, None]
        reference_velocity = torch.zeros_like(support)
        if self.gait_reference is not None:
            ready = self.gait_reference.state != self.gait_reference.SETTLE
            support = torch.where(ready[:, None], self.gait_reference.reference_com, support)
            reference_velocity = torch.where(ready[:, None], self.gait_reference.reference_velocity,
                                               reference_velocity)
        mass = self._masses[..., None]
        com = (self._asset.data.body_com_pos_w * mass).sum(1) / mass.sum(1)
        com_velocity = (self._asset.data.body_com_lin_vel_w * mass).sum(1) / mass.sum(1)
        q_yaw = yaw_quat(self._asset.data.root_quat_w)
        error_xy = quat_apply_inverse(q_yaw, com - support)[:, :2]
        velocity_xy = quat_apply_inverse(q_yaw, com_velocity - reference_velocity)[:, :2]
        error, velocity = error_xy[:, 1], velocity_xy[:, 1]
        correction = (cfg["position_gain"] * error + cfg["velocity_gain"] * velocity).clamp(
            -cfg["maximum_angle"], cfg["maximum_angle"])
        for side_index, side in enumerate(("left", "right")):
            action_id = self._joint_names.index(f"{side}_hip_roll_joint")
            ratio = (self.gait_reference.impedance.base_ratio[:,side_index,1]
                     if getattr(self._env.cfg,"stair_reference",None) is not None else 1)
            self._processed_actions[:, action_id] += correction * weight[:, side_index] * ratio
            # A global forward/backward rocking mode is not observable as a
            # joint-position error. Close the sagittal COM loop as well; the
            # hip-pitch Jacobian has the opposite sign to hip roll.
            sagittal = -(cfg.get("sagittal_position_gain", 0.0) * error_xy[:, 0]
                          + cfg.get("sagittal_velocity_gain", 0.0) * velocity_xy[:, 0])
            sagittal = sagittal.clamp(-cfg["maximum_angle"], cfg["maximum_angle"])
            pitch_id = self._joint_names.index(f"{side}_hip_pitch_joint")
            ratio = (self.gait_reference.impedance.base_ratio[:,side_index,0]
                     if getattr(self._env.cfg,"stair_reference",None) is not None else 1)
            self._processed_actions[:, pitch_id] += sagittal * weight[:, side_index] * ratio
