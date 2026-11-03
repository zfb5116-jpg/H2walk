"""Bounded env0 copying of already produced policy/native tensors, never a controller."""
import copy
import json
from pathlib import Path

from stair_execution_binding import (execution_config_errors, execution_snapshot_errors,
                                     finite_json, selected_controls_after)


def _value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


class StairExecutionProbe:
    def __init__(self, raw, runner, credit, directory, binding):
        if (execution_config_errors(binding) or raw.num_envs != 1024
                or raw.step_dt != .02 or raw.physics_dt != .005 or raw.cfg.decimation != 4):
            raise RuntimeError("Execution observer requires the original1024/.02/4x.005 source")
        self.raw, self.runner, self.credit = raw, runner, credit
        self.contract = copy.deepcopy(binding)
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.stream = (self.directory / "env0_control_records.jsonl").open("x")
        self.closed = False
        self.controls_seen = self.recorded_controls = self.recorded_physics_substeps = 0
        self.first_recorded_control = self.last_recorded_control = None
        self.pending = None
        self.asset = raw.asset
        term = raw.action_manager.get_term("JointPositionAction")
        self.joint_ids = [int(x) for x in term._joint_ids]
        self.joint_names = [self.asset.joint_names[i] for i in self.joint_ids]
        if len(self.joint_ids) != 12 or len(set(self.joint_ids)) != 12:
            raise RuntimeError("Execution observer did not bind the actual12 action joints")
        self.groups = []
        covered = []
        for name, actuator in self.asset.actuators.items():
            group_ids = list(range(self.asset.num_joints))[actuator.joint_indices] if isinstance(actuator.joint_indices, slice) else [int(i) for i in actuator.joint_indices]
            selected = [(joint, group_ids.index(joint)) for joint in self.joint_ids if joint in group_ids]
            if selected:
                if (not hasattr(actuator, "positions_delay_buffer")
                        or getattr(actuator, "support_priority_buffer", None) is not None
                        or getattr(actuator, "stair_epoch_buffer", None) is not None):
                    raise RuntimeError("Execution observer requires the original ordinary delayed native drive")
                self.groups.append((name, actuator, selected));covered.extend(j for j, _ in selected)
        if sorted(covered) != sorted(self.joint_ids):
            raise RuntimeError("Execution observer action joints are not covered once by native groups")
        self.original_write = self.asset.write_data_to_sim
        self.had_instance_write = "write_data_to_sim" in self.asset.__dict__
        self.original_instance_write = self.asset.__dict__.get("write_data_to_sim")
        if (getattr(getattr(self.original_write, "__func__", None), "__name__", None) != "write_data_to_sim"
                or self.original_write.__func__.__module__ != "isaaclab.assets.articulation.articulation"
                or self.original_write.__func__.__qualname__ != "Articulation.write_data_to_sim"
                or self.had_instance_write):
            raise RuntimeError("Execution observer refuses an already wrapped native write source")

        def observed_original_write():
            result = self.original_write()
            if self.pending is not None:
                self._copy_after_native_write()
            return result

        self.observed_write = observed_original_write
        self.asset.write_data_to_sim = observed_original_write
        self._write_manifest()

    def begin(self):
        control = int(self.raw.common_step_counter) + 1
        if self.closed or self.pending is not None or control != self.controls_seen + 1:
            raise RuntimeError("Execution observer control lifecycle differs")
        self.controls_seen = control
        update = int(self.runner.alg.completed_updates)
        wanted = selected_controls_after(control)
        if wanted == self.recorded_controls:
            return
        if not (480 <= update <= 500 and wanted == self.recorded_controls + 1
                and self.recorded_controls < 1000 and self.credit.expected_counter == control):
            raise RuntimeError("Execution observer missed its exact existing window1 boundary")
        transition = self.runner.alg.transition
        for tensor in (transition.action_mean, transition.action_sigma, transition.actions):
            if tuple(tensor.shape) != (1024, 12):
                raise RuntimeError("Execution observer has no actual existing PPO transition")
        self.pending = dict(control=control, completed_updates=update,
            actor_optimizer_steps=int(self.runner.alg.actor_optimizer_steps),
            cached_transition_control_id=control,
            episode_reset_generation=int(self.credit.pre_generation[0]),
            policy_action_mean=_value(transition.action_mean[0]),
            policy_action_sigma=_value(transition.action_sigma[0]),
            policy_sampled_action=_value(transition.actions[0]),
            physics_counter_before=int(self.raw._sim_step_counter), physics_substeps=[])

    def _copy_after_native_write(self):
        record = self.pending
        step = int(self.raw._sim_step_counter)
        phase = len(record["physics_substeps"]) + 1
        if phase > 4 or step != record["physics_counter_before"] + phase:
            raise RuntimeError("Execution observer native write is not one original physics substep")
        data = self.asset.data
        ids = self.joint_ids
        groups = []
        for name, actuator, selected in self.groups:
            local = [i for _, i in selected]
            groups.append(dict(name=name, global_joint_ids=[j for j, _ in selected],
                local_joint_ids=local,
                position_delay_steps=int(actuator.positions_delay_buffer.time_lags[0]),
                velocity_delay_steps=int(actuator.velocities_delay_buffer.time_lags[0]),
                effort_delay_steps=int(actuator.efforts_delay_buffer.time_lags[0]),
                existing_effort_limit=_value(actuator.effort_limit[0, local]),
                existing_stiffness=_value(actuator.stiffness[0, local]),
                existing_damping=_value(actuator.damping[0, local])))
        computed = _value(data.computed_torque[0, ids])
        clipped = _value(data.applied_torque[0, ids])
        record["physics_substeps"].append(dict(physics_counter=step, substep=phase,
            phase="after_original_native_setters_before_physics",
            actual_delayed_position_target=_value(self.asset._joint_pos_target_sim[0, ids]),
            actual_delayed_velocity_target=_value(self.asset._joint_vel_target_sim[0, ids]),
            actual_feedforward_effort_target=_value(self.asset._joint_effort_target_sim[0, ids]),
            same_prephysics_joint_position=_value(data.joint_pos[0, ids]),
            same_prephysics_joint_velocity=_value(data.joint_vel[0, ids]),
            implicit_PD_computed_estimate=computed,
            implicit_PD_clipped_estimate=clipped,
            implicit_PD_estimate_clipped=[a != b for a, b in zip(computed, clipped)],
            native_groups=groups, true_PhysX_PD_drive_torque=None,
            true_motor_saturation=None))

    def finish_control(self, existing_credit_record):
        if self.pending is None:
            return
        record = self.pending
        if (existing_credit_record["control"] != record["control"]
                or existing_credit_record["completed_updates"] != record["completed_updates"]
                or existing_credit_record["episode_reset_generation"] != record["episode_reset_generation"]
                or existing_credit_record["record_phase"] != "post_reward_before_reset"
                or len(record["physics_substeps"]) != 4):
            raise RuntimeError("Execution observer endpoint/physics/generation source differs")
        data = self.asset.data
        feet = self.raw._stair_foot_ids
        record["original_credit_endpoint"] = existing_credit_record
        record["original_pre_reset_state"] = dict(
            root_pos_w=_value(data.root_pos_w[0]), root_quat_w=_value(data.root_quat_w[0]),
            root_link_lin_vel_w=_value(data.root_link_lin_vel_w[0]), root_ang_vel_w=_value(data.root_ang_vel_w[0]),
            foot_body_pos_w=_value(data.body_pos_w[0, feet]),
            foot_body_quat_w=_value(data.body_quat_w[0, feet]),
            foot_body_lin_vel_w=_value(data.body_lin_vel_w[0, feet]),
            foot_body_ang_vel_w=_value(data.body_ang_vel_w[0, feet]),
            termination_terms={name: bool(self.raw.termination_manager.get_term(name)[0])
                               for name in self.raw.termination_manager.active_terms})
        if not finite_json(record):
            raise RuntimeError("Nonfinite actual execution observer record")
        self.stream.write(json.dumps(record, allow_nan=False) + "\n");self.stream.flush()
        self.recorded_controls += 1
        self.recorded_physics_substeps += 4
        self.first_recorded_control = self.first_recorded_control or record["control"]
        self.last_recorded_control = record["control"]
        self.pending = None

    def snapshot(self):
        return dict(contract=copy.deepcopy(self.contract), controls_seen=self.controls_seen,
            recorded_controls=self.recorded_controls, recorded_physics_substeps=self.recorded_physics_substeps,
            first_recorded_control=self.first_recorded_control, last_recorded_control=self.last_recorded_control,
            pending_control=self.pending["control"] if self.pending is not None else None,
            action_joint_ids=list(self.joint_ids), action_joint_names=list(self.joint_names),
            original_native_write_bound=self.asset.write_data_to_sim is self.observed_write,
            original_native_write_source=dict(module=self.original_write.__func__.__module__,
                                              qualname=self.original_write.__func__.__qualname__),
            read_only=True, true_PhysX_PD_drive_torque_available=False)

    def _write_manifest(self):
        value = self.snapshot()
        if self.pending is None and execution_snapshot_errors(self.contract, value):
            raise RuntimeError("Execution observer live manifest differs from its contract")
        path = self.directory / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def close(self):
        if self.closed:
            return
        try:
            if self.pending is not None:
                if finite_json(self.pending):
                    (self.directory / "partial_control.json").write_text(json.dumps(self.pending, indent=2, allow_nan=False) + "\n")
                else:
                    (self.directory / "nonfinite_partial_control.repr.txt").write_text(repr(self.pending) + "\n")
            self._write_manifest()
        finally:
            self.stream.close()
            if self.asset.write_data_to_sim is self.observed_write:
                if self.had_instance_write:
                    self.asset.write_data_to_sim = self.original_instance_write
                else:
                    del self.asset.write_data_to_sim
            self.closed = True
