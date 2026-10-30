"""Solver-integrated PD drives with the same sampled command transport delay."""

import torch

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer


class DelayedImplicitActuator(ImplicitActuator):
    """Delay all setpoints, then let PhysX integrate the bounded PD drive."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.support_priority_buffer = None
        self.stair_epoch_buffer = None

    def enable_support_effort_priority(self, reference, action_ids, local_ids, soft_limits, speed_limits):
        """Transport effort/role once; share one physical motor budget on receipt."""
        self._support_reference = reference
        self._support_action_ids, self._support_local_ids = list(action_ids), list(local_ids)
        self._support_soft_limits = soft_limits
        self._support_speed_limits = speed_limits.clamp_max(30.)
        if not bool(torch.isfinite(soft_limits).all() & torch.isfinite(speed_limits).all()
                    & (soft_limits[..., 1]-soft_limits[..., 0] > .012).all()
                    & (speed_limits > 0).all() & (self.stiffness > 0).all()
                    & (self.damping > 0).all() & (self.effort_limit > 0).all()):
            raise ValueError("Invalid fixed stair effort-drive parameters")
        self.support_priority_command = torch.zeros(*self.stiffness.shape, 2, device=self._device)
        self.support_priority_buffer = DelayBuffer(self.cfg.max_delay, self._num_envs, device=self._device)
        self.support_priority_buffer.set_time_lag(self.positions_delay_buffer.time_lags.clone())
        self._support_effective_role = torch.zeros_like(self.stiffness)
        self._support_barrier_latch = torch.zeros_like(self.stiffness, dtype=torch.int8)
        self._support_joint_ids = (list(range(reference.robot.num_joints)) if isinstance(self.joint_indices, slice)
                                   else [int(i) for i in self.joint_indices])
        self.support_native_verification = dict(writes=0, maximum_K_error=0.,
                                                maximum_D_error=0., maximum_drive_cap_error=0.)
        view = reference.robot.root_physx_view
        self._support_native_readback = tuple(value[:,self._support_joint_ids].to(self._device).clone()
            for value in (view.get_dof_stiffnesses(),view.get_dof_dampings(),view.get_dof_max_forces()))

    def reset(self, env_ids=None):
        count = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        delay = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (count,),
                              dtype=torch.int, device=self._device)
        buffers = [self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer]
        if self.support_priority_buffer is not None:
            buffers.append(self.support_priority_buffer)
            ids = slice(None) if env_ids is None else env_ids
            self.support_priority_command[ids] = 0
            self._support_effective_role[ids] = 0
            self._support_barrier_latch[ids] = 0
            self.apply_support_native_budget(env_ids)
        if self.stair_epoch_buffer is not None:
            buffers.append(self.stair_epoch_buffer)
            self.stair_publication_epoch[slice(None) if env_ids is None else env_ids] = -1
        for buffer in buffers:
            buffer.set_time_lag(delay, env_ids)
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        if self.stair_epoch_buffer is not None:
            from ..stair_receiver import record_receiver
            epoch = self.stair_epoch_buffer.compute(self.stair_publication_epoch)
            record_receiver(self, control_action, joint_pos, joint_vel, epoch)
        if self.support_priority_buffer is not None:
            from ..stair_drive import received_support_budget_ports
            packet = self.support_priority_buffer.compute(self.support_priority_command)
            if not bool(torch.isfinite(packet).all() & torch.isfinite(joint_pos).all()
                        & torch.isfinite(joint_vel).all() & (control_action.joint_efforts == 0).all()):
                raise RuntimeError("Non-finite stair drive data or unbudgeted legacy external effort")
            encoded = received_support_budget_ports(self._support_barrier_latch,
                joint_pos, joint_vel, control_action.joint_positions, control_action.joint_velocities,
                control_action.joint_efforts, packet[..., 0], packet[..., 1], self.stiffness,
                self.damping, self.effort_limit, self._support_soft_limits[..., 0], self._support_soft_limits[..., 1],
                self._support_speed_limits, ordinary_drive_only=True,
                physics_dt=self._support_reference.env.cfg.sim.dt,
                guard_margin=self._support_reference.options["joint_kinematic_reserve"], validate=False)
            encoded["legacy_position"] = control_action.joint_positions
            encoded["legacy_velocity"] = control_action.joint_velocities
            changed = (self._support_effective_role != encoded["effective_role"]).any(-1).nonzero().flatten()
            self._support_effective_role[:] = encoded["effective_role"]
            self._support_barrier_latch[:] = encoded["next_latch"]
            if changed.numel():
                self.apply_support_native_budget(changed)
            control_action.joint_positions = encoded["position"]
            control_action.joint_velocities = encoded["velocity"]
            # Final software force-port clamp is its allocated fraction, not
            # ImplicitActuator.applied_effort (which never clips returned actions).
            force_budget = encoded["effective_role"]*self.effort_limit
            control_action.joint_efforts = encoded["external_effort"].clamp(-force_budget, force_budget)
            encoded["external_effort"] = control_action.joint_efforts
            if not bool((self._support_native_readback[2]+control_action.joint_efforts.abs()
                         <= self.effort_limit+1e-5).all()):
                raise RuntimeError("Native PD plus external support motor budget exceeds original limit")
            encoded["expression"] = (encoded["native_stiffness"]*(encoded["position"]-joint_pos)
                + encoded["native_damping"]*(encoded["velocity"]-joint_vel))
            self._record_support_receiver(packet, encoded, joint_pos, joint_vel)
            # Keep the public estimator consistent with the effective native
            # gains and its separate cap; these remain request estimates.
            result = super().compute(control_action, joint_pos, joint_vel)
            self.computed_effort = encoded["expression"]+control_action.joint_efforts
            cap = encoded["native_drive_cap"]
            self.applied_effort = encoded["expression"].clamp(-cap,cap)+control_action.joint_efforts
            return result
        return super().compute(control_action, joint_pos, joint_vel)

    def apply_support_native_budget(self, env_ids=None):
        """Aggregate all enabled groups; preserve every other native property.

        Do not use Articulation.write_* here: its data/cache values are the
        nominal gains needed by planning and the implicit recovery controller.
        No simulation step occurs between these writes and their readback.
        """
        robot = self._support_reference.robot
        ids = (torch.arange(self._num_envs, dtype=torch.long) if env_ids is None or isinstance(env_ids,slice)
               else torch.as_tensor(env_ids, dtype=torch.long).cpu().unique())
        if not ids.numel():
            return
        view = robot.root_physx_view
        matrices = [view.get_dof_stiffnesses().detach().cpu().clone(),
                    view.get_dof_dampings().detach().cpu().clone(),
                    view.get_dof_max_forces().detach().cpu().clone()]
        for actuator in robot.actuators.values():
            if getattr(actuator,"support_priority_buffer",None) is None:
                continue
            columns = actuator._support_joint_ids
            fraction = 1-actuator._support_effective_role.detach().cpu()
            for matrix,nominal in zip(matrices,(actuator.stiffness,actuator.damping,actuator.effort_limit)):
                matrix[:,columns] = nominal.detach().cpu()*fraction
        for setter,value in zip((view.set_dof_stiffnesses,view.set_dof_dampings,view.set_dof_max_forces),matrices):
            setter(value,indices=ids)
        readbacks = [view.get_dof_stiffnesses().detach().cpu(),
                     view.get_dof_dampings().detach().cpu(),view.get_dof_max_forces().detach().cpu()]
        errors = [float((actual[ids]-wanted[ids]).abs().max()) for actual,wanted in zip(readbacks,matrices)]
        if max(errors) > .001:
            raise RuntimeError("Native support gain/force-budget readback mismatch")
        self.support_native_verification["writes"] += 1
        for key,error in zip(("maximum_K_error","maximum_D_error","maximum_drive_cap_error"),errors):
            self.support_native_verification[key] = max(self.support_native_verification[key],error)
        for actuator in robot.actuators.values():
            if getattr(actuator,"support_priority_buffer",None) is not None:
                actuator._support_native_readback = tuple(value[:,actuator._support_joint_ids].to(actuator._device).clone()
                                                           for value in readbacks)

    def _record_support_receiver(self, packet, encoded, joint_pos, joint_vel):
        gait, actions, local = self._support_reference, self._support_action_ids, self._support_local_ids
        values = dict(native_received_role=packet[..., 1], native_received_effort=packet[..., 0],
            native_delayed_legacy_position=encoded["legacy_position"],
            native_delayed_legacy_velocity=encoded["legacy_velocity"],
            native_drive_position=encoded["position"], native_drive_velocity=encoded["velocity"],
            native_drive_external_effort=encoded["external_effort"],
            native_drive_current_position=joint_pos, native_drive_current_velocity=joint_vel,
            native_drive_effort_request=encoded["unclipped_external"], native_drive_effort_safe=encoded["external_effort"],
            native_drive_pd_expression=encoded["expression"], native_drive_barrier=encoded["guard_active"],
            native_drive_barrier_satisfied=encoded["budget_satisfied"]&encoded["guard_target_inside_release"],
            native_drive_infeasible=~encoded["budget_satisfied"],
            native_effective_role=encoded["effective_role"],
            native_effective_stiffness=self._support_native_readback[0],
            native_effective_damping=self._support_native_readback[1],
            native_effective_drive_cap=self._support_native_readback[2],
            native_total_motor_budget=encoded["worst_case_absolute_motor_budget"],
            native_barrier_latch=encoded["next_latch"].long())
        for key, value in values.items():
            getattr(gait, key).reshape(gait.n, 12)[:, actions] = value[:, local]
        gait.native_drive_delay_steps.reshape(gait.n, 12)[:, actions] = self.positions_delay_buffer.time_lags[:, None].long()
        # Counter names the physics step receiving this command. Its input
        # joint state precedes that5ms step; wrench data measures its endpoint.
        gait.native_drive_physics_step.reshape(gait.n, 12)[:, actions] = int(gait.env._sim_step_counter)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 6
