"""Read-only lower-body command provenance through the existing transport.

The extra buffer transports an integer publication identifier only. It samples
no delay, writes no drive and never replaces a position/velocity/effort packet.
An age must be computed from the publication and receiving counters, rather
than assuming that a DelayBuffer invocation always denotes one physics step.
"""

import torch
from isaaclab.utils.buffers import DelayBuffer


def attach_receiver(actuator, reference, action_ids, local_ids):
    actuator._stair_receiver_reference = reference
    actuator._stair_receiver_actions = list(action_ids)
    actuator._stair_receiver_local = list(local_ids)
    actuator.stair_publication_epoch = torch.full(
        (actuator._num_envs, 1), -1, dtype=torch.long, device=actuator._device)
    actuator.stair_epoch_buffer = DelayBuffer(
        actuator.cfg.max_delay, actuator._num_envs, device=actuator._device)
    actuator.stair_epoch_buffer.set_time_lag(
        actuator.positions_delay_buffer.time_lags.clone())


def record_receiver(actuator, command, position, velocity, epoch):
    """Copy original delayed ordinary ports; priority axes overwrite them later."""
    gait = actuator._stair_receiver_reference
    actions, local = actuator._stair_receiver_actions, actuator._stair_receiver_local
    zeros = torch.zeros_like(position)
    expression = (actuator.stiffness * (command.joint_positions-position)
                  + actuator.damping * (command.joint_velocities-velocity))
    budget = actuator.effort_limit+command.joint_efforts.abs()
    fields = dict(
        native_received_role=zeros, native_received_effort=zeros,
        native_delayed_legacy_position=command.joint_positions,
        native_delayed_legacy_velocity=command.joint_velocities,
        native_drive_position=command.joint_positions,
        native_drive_velocity=command.joint_velocities,
        native_drive_external_effort=command.joint_efforts,
        native_drive_current_position=position, native_drive_current_velocity=velocity,
        native_drive_effort_request=command.joint_efforts,
        native_drive_effort_safe=command.joint_efforts,
        native_drive_pd_expression=expression,
        native_effective_role=zeros,
        native_effective_stiffness=actuator.stiffness,
        native_effective_damping=actuator.damping,
        native_effective_drive_cap=actuator.effort_limit,
        native_total_motor_budget=budget,
        native_barrier_latch=torch.zeros_like(position, dtype=torch.long),
        native_drive_barrier=torch.zeros_like(position, dtype=torch.bool),
        native_drive_barrier_satisfied=budget <= actuator.effort_limit,
        native_drive_infeasible=budget > actuator.effort_limit)
    for key, value in fields.items():
        getattr(gait, key).reshape(gait.n, 12)[:, actions] = value[:, local]
    gait.native_command_published_control_step.reshape(gait.n, 12)[:, actions] = epoch
    received_step = int(gait.env.common_step_counter)
    physics_step = int(gait.env._sim_step_counter)
    gait.native_command_receive_control_step.reshape(gait.n, 12)[:, actions] = received_step
    gait.native_command_pre_physics.reshape(gait.n, 12)[:, actions] = (
        physics_step > 0 and received_step == (physics_step-1)//int(gait.env.cfg.decimation))
    gait.native_drive_delay_steps.reshape(gait.n, 12)[:, actions] = (
        actuator.positions_delay_buffer.time_lags[:, None].long())
    gait.native_drive_physics_step.reshape(gait.n, 12)[:, actions] = physics_step
