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


def _finite_copy(value):
    result = _value(value)
    return result if finite_json(result) else None


def _existing_terrain_input(raw, transition, asset, control, generation):
    """Copy actual preaction inputs; never evaluate a sensor/model/normalizer.

    The action c consumes observations produced at the previous control epoch.
    Cached reconstruction is a separate, unnormalized root-yaw-frame quantity.
    Missing diagnostic values remain null and never create a learner stop.
    """
    epoch = int(raw.common_step_counter)
    result = dict(action_control=control,observation_producer_epoch=epoch,
        expected_observation_epoch=control-1,episode_reset_generation=generation,
        status="unavailable",normalized_actor_terrain=None,reconstructed_height_m=None,
        reconstruction_cache_epoch=None,ray_cache=None,grid=None,latest_cached_sensor_pose=None,
        existing_reconstruction_hit_mask=None,
        preaction_root_position_w=_finite_copy(asset.data.root_pos_w[0]),
        preaction_root_quaternion_wxyz=_finite_copy(asset.data.root_quat_w[0]),
        env_origin_w=_finite_copy(raw.stair_progress.origins[0]),
        actual_riser_height_m=_finite_copy(raw.stair_progress.heights[0]),
        normalized_actor_slice=[468,1029],
        height_semantics="existing_reconstruction_yaw_grid_Z_relative_to_current_root_Z_before_ObsTerm_clip_scale",
        actor_semantics="actual_PPO_transition_input_after_own_frozen_normalization",
        issues=[])
    obs = getattr(transition,"observations",None)
    if obs is not None and tuple(obs.shape)==(1024,1034):
        result["normalized_actor_terrain"]=_finite_copy(obs[0,468:1029])
    if result["normalized_actor_terrain"] is None:
        result["issues"].append("actual_normalized_actor_terrain_missing_or_nonfinite")
    cache = getattr(raw,"_h2_lidar_recon_cache",None)
    if (isinstance(cache,tuple) and len(cache)==2 and type(cache[0]) is int
            and tuple(getattr(cache[1],"shape",()))==(1024,561)):
        result["reconstruction_cache_epoch"]=cache[0]
        result["reconstructed_height_m"]=_finite_copy(cache[1][0])
        if cache[0]!=epoch:
            result["issues"].append("reconstruction_cache_epoch_differs")
    if result["reconstructed_height_m"] is None:
        result["issues"].append("existing_reconstruction_missing_or_nonfinite")
    rays = getattr(raw,"_h2_lidar_ray_cache",None)
    if (isinstance(rays,tuple) and len(rays)==5 and type(rays[0]) is int
            and len(getattr(rays[3],"shape",()))==2 and rays[3].shape[0]==1024
            and tuple(getattr(rays[4],"shape",()))==tuple(rays[3].shape)
            and tuple(getattr(rays[1],"shape",()))==(1024,)
            and tuple(getattr(rays[2],"shape",()))==(1024,3)):
        ranges,mask=rays[3][0],rays[4][0]
        valid=mask>0.5;count=int(valid.sum());bins=int(mask.shape[0])
        result["ray_cache"]=dict(epoch=rays[0],range_bins=bins,valid_bins=count,
            valid_fraction=count/bins if bins else None,
            valid_range_min_m=_finite_copy(ranges[valid].min()) if count else None,
            valid_range_max_m=_finite_copy(ranges[valid].max()) if count else None,
            terrain_type=_finite_copy(rays[1][0]),cached_root_position_w=_finite_copy(rays[2][0]))
        if rays[0]!=epoch:
            result["issues"].append("ray_cache_epoch_differs")
    else:
        result["issues"].append("existing_ray_cache_missing_or_shape_differs")
    model=getattr(raw,"_h2_lidar_reconstructor",None)
    if model is not None:
        names=("hb_nx","hb_ny","hb_x0","hb_y0","hb_res")
        if all(hasattr(model,name) for name in names):
            result["grid"]={name:_finite_copy(getattr(model,name)) for name in names}
            result["grid"]["xy_frame"]="current_root_yaw_aligned; flatten_y_then_x"
            if (result["grid"]["hb_nx"]!=33 or result["grid"]["hb_ny"]!=17
                    or any(result["grid"][name] is None for name in names)):
                result["issues"].append("existing_grid_registration_differs_or_nonfinite")
        hit=getattr(model,"_last_hit",None)
        if hit is not None and tuple(hit.shape)==(1024,561):
            result["existing_reconstruction_hit_mask"]=_finite_copy(hit[0])
        ptr=getattr(model,"_pose_ptr",None);fill=getattr(model,"_pose_fill",None)
        pos=getattr(model,"_pose_pos",None);quat=getattr(model,"_pose_quat",None)
        if (ptr is not None and fill is not None and pos is not None and quat is not None
                and tuple(ptr.shape)==(1024,) and tuple(fill.shape)==(1024,)
                and len(pos.shape)==3 and pos.shape[0]==1024 and pos.shape[-1]==3
                and tuple(quat.shape)==(1024,pos.shape[1],4)
                and int(fill[0])>0 and pos.shape[1]>0):
            index=(int(ptr[0])-1)%int(pos.shape[1])
            result["latest_cached_sensor_pose"]=dict(source="existing_pose_ring_latest_item",
                ring_index=index,position_w=_finite_copy(pos[0,index]),
                quaternion_wxyz=_finite_copy(quat[0,index]))
    if result["grid"] is None:
        result["issues"].append("existing_grid_registration_missing")
    if any(result[name] is None for name in ("preaction_root_position_w",
            "preaction_root_quaternion_wxyz","env_origin_w","actual_riser_height_m")):
        result["issues"].append("existing_preaction_registration_nonfinite")
    if epoch!=control-1:
        result["issues"].append("observation_epoch_not_action_control_minus1")
    if not result["issues"]:
        result["status"]="captured"
    return result


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
        self.terrain_input_records = self.terrain_input_unavailable_records = 0
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
            bridge = getattr(self.raw, "stair_contact_bridge", None)
            if bridge is not None:
                bridge.received_action()
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
            existing_terrain_input=_existing_terrain_input(self.raw,transition,self.asset,control,
                int(self.credit.pre_generation[0])),
            physics_counter_before=int(self.raw._sim_step_counter), physics_substeps=[],
            pending_postphysics_reset_flush_writes=0)

    def _copy_after_native_write(self):
        record = self.pending
        step = int(self.raw._sim_step_counter)
        phase = len(record["physics_substeps"]) + 1
        control = int(self.raw.common_step_counter)
        # SDK increments the control counter after all four physics steps.
        # Its later subset-reset flush writes native data without another step.
        # Env0 can still await its endpoint when other environments reset.
        if control == record["control"]:
            if (phase == 5 and step == record["physics_counter_before"] + 4
                    and record["pending_postphysics_reset_flush_writes"] == 0):
                record["pending_postphysics_reset_flush_writes"] = 1
                return
            raise RuntimeError("Execution observer postphysics reset flush differs")
        if (control != record["control"] - 1 or phase > 4
                or step != record["physics_counter_before"] + phase):
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
        if record["existing_terrain_input"]["status"]=="captured":
            self.terrain_input_records += 1
        else:
            self.terrain_input_unavailable_records += 1
        self.first_recorded_control = self.first_recorded_control or record["control"]
        self.last_recorded_control = record["control"]
        self.pending = None

    def snapshot(self):
        return dict(contract=copy.deepcopy(self.contract), controls_seen=self.controls_seen,
            recorded_controls=self.recorded_controls, recorded_physics_substeps=self.recorded_physics_substeps,
            terrain_input_records=self.terrain_input_records,
            terrain_input_unavailable_records=self.terrain_input_unavailable_records,
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
