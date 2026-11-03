"""Read-only same-control reward coverage, captured before terminal reset."""
import json
from pathlib import Path
import torch


class StairCreditProbe:
    def __init__(self, raw, runner, directory, video_binding):
        self.raw, self.runner = raw, runner
        self.reward_names = list(raw.reward_manager.active_terms)
        if len(self.reward_names) != 19 or raw.num_envs != 1024:
            raise RuntimeError("Credit observer requires actual19/1024")
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.trace = (self.directory / "env0_pre_reset_controls.jsonl").open("x")
        self.windows = video_binding["windows"]
        self.trace_controls = [0, 0, 0]
        self.controls = 0
        self.pending = False
        self.closed = False
        self.captured = torch.zeros(1024, dtype=torch.bool, device=raw.device)
        self.labels = ["foot_controls", "near", "both_near_same_riser", "own_unloaded", "other70",
            "both_force_valid", "episode_force_eligible", "tilt_lt0p5", "derived_overspeed_weight_nonzero",
            "raw_rising", "actual_positive_eligible", "actually_paid_positive"]
        self.counts = torch.zeros(6, len(self.labels), dtype=torch.long, device=raw.device)
        self.reward_integrals = torch.zeros(6, 19, dtype=torch.float64, device=raw.device)
        self.delta_integrals = torch.zeros(6, 4, dtype=torch.float64, device=raw.device)
        self.last_note_counter = -1
        self.video = None
        self.fixed_input_height_counts = torch.zeros(3, dtype=torch.long, device=raw.device)
        self.current_input_height_counts = torch.zeros_like(self.fixed_input_height_counts)
        self.current_input_update = -1

    def record_selected_rows(self, ids, control_step, completed_updates, rollout_step, *, kind):
        if int(self.raw.common_step_counter) != control_step:
            raise RuntimeError("Input height metadata is not from its actual preaction epoch")
        if kind == "current":
            if rollout_step == 0:
                self.current_input_height_counts.zero_()
                self.current_input_update = completed_updates
            if self.current_input_update != completed_updates:
                raise RuntimeError("Input height metadata missed its rollout boundary")
            destination = self.current_input_height_counts
        elif kind == "fixed" and completed_updates == 0 and rollout_step < 24:
            destination = self.fixed_input_height_counts
        else:
            raise RuntimeError("Undeclared input height metadata request")
        actual_rows = self.raw.scene.terrain.terrain_levels[ids]
        destination += torch.bincount(actual_rows, minlength=3)

    def begin(self):
        if self.pending or int(self.raw.common_step_counter) != self.controls:
            raise RuntimeError("Credit observer control lifecycle mismatch")
        self.pending = True
        self.expected_counter = self.controls + 1
        self.captured.zero_()
        self.pre_generation = self.raw.initialization_reset_generation.clone()

    def note_clearance(self, eligible, weight, previous_level, previous_near, raw_delta, applied_delta):
        """Copy actual branch inputs/results; never modify reward-owned tensors."""
        if not self.pending or int(self.raw.common_step_counter) != self.expected_counter:
            raise RuntimeError("Credit observer received an unrelated clearance epoch")
        self.actual_eligible = eligible.clone()
        self.actual_weight = weight.clone()
        self.previous_level = previous_level.clone()
        self.previous_near = previous_near.clone()
        self.actual_raw = raw_delta.clone()
        self.actual_applied = applied_delta.clone()
        self.last_note_counter = self.expected_counter

    def capture(self, ids):
        if not self.pending:
            return  # initial, pre-learning reset has no rewarded transition
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.raw.device)
        ids = ids[~self.captured[ids]]
        if not ids.numel():
            return
        if self.last_note_counter != self.expected_counter:
            raise RuntimeError("Credit observer has no actual clearance transition")
        env = self.raw
        p = env.stair_progress
        fz = p.ground_force_end[..., 2]
        tilt = torch.acos((-env.asset.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        same = (p.clearance_level == self.previous_level) & p.clearance_near & self.previous_near
        both_force = p.current_force_valid.all(-1)[:, None].expand(-1, 2)
        speed_u = ((env.asset.data.root_link_lin_vel_w[:, 0] - 0.3) / 0.3).clamp(0, 1)
        speed_weight = (1-speed_u).square()*(1+2*speed_u)
        flags = torch.stack((torch.ones_like(same), p.clearance_near, same,
            fz < p.minimum_load[:, None], fz.flip(-1) >= 0.7*p.weight[:, None], both_force,
            p.force_credit_eligible[:, None].expand(-1, 2),
            (tilt < 0.5)[:, None].expand(-1, 2),
            (speed_weight > 0)[:, None].expand(-1, 2), self.actual_raw > 0,
            self.actual_eligible, self.actual_applied > 0), -1)
        # The manager stores weighted rates. Multiplying once bydt yields the
        # very contributions in its actual reward buffer, without reevaluation.
        contributions = env.reward_manager._step_reward[ids] * env.step_dt
        if not bool(torch.isfinite(contributions).all()):
            raise RuntimeError("Nonfinite actual reward contribution")
        group = (ids >= 256).long()*3 + p.terrain_level[ids]
        delta = torch.stack((self.actual_raw[ids].clamp_min(0).sum(-1),
            self.actual_raw[ids].clamp_max(0).sum(-1),
            self.actual_applied[ids].clamp_min(0).sum(-1),
            self.actual_applied[ids].clamp_max(0).sum(-1)), -1)
        for index in range(6):
            selected = group == index
            self.counts[index] += flags[ids][selected].sum(dim=(0, 1))
            self.reward_integrals[index] += contributions[selected].double().sum(0)
            self.delta_integrals[index] += delta[selected].double().sum(0)
        if bool((ids == 0).any()):
            self._trace_env0(contributions[(ids == 0).nonzero().flatten()[0]], tilt[0], flags[0])
        self.captured[ids] = True

    def _trace_env0(self, contributions, tilt, flags):
        c = int(self.runner.alg.completed_updates)
        window = next((i for i, w in enumerate(self.windows)
            if w[0] <= c <= w[1] and self.trace_controls[i] < w[2]), None)
        if window is None:
            return
        self.trace_controls[window] += 1
        env, p = self.raw, self.raw.stair_progress
        soles, rotations = env._stair_foot_pose()
        local = soles[:1] - p.origins[:1, None]
        corners = torch.einsum("nfij,fkj->nfki", rotations[:1], p.corners) + local[:, :, None]
        term = env.action_manager.get_term("JointPositionAction")
        leg_ids = term._joint_ids
        upper_ids = list(env.action_manager.held_joint_ids)
        def values(tensor):
            return tensor.detach().cpu().tolist()
        value = dict(control=self.expected_counter, completed_updates=c, env_id=0,
            episode_reset_generation=int(self.pre_generation[0]), actual_row=int(p.terrain_level[0]),
            initial_partial_episode=bool(p.partial[0]), actual_elapsed_s=float(p.duration[0]),
            actual_height_m=float(p.heights[0]), record_phase="post_reward_before_reset",
            terminated=bool(env.termination_manager.terminated[0]), timed_out=bool(env.termination_manager.time_outs[0]),
            root_link_velocity=values(env.asset.data.root_link_lin_vel_w[0]), root_tilt=float(tilt),
            whole_sole_front=values(corners[0, ..., 0].amax(-1)),
            whole_sole_rear=values(corners[0, ..., 0].amin(-1)),
            whole_sole_bottom=values(corners[0, ..., 2].amin(-1)),
            selected_riser=values(p.clearance_level[0]), ground_normal_force=values(p.ground_force_end[0]),
            raw_normal_force=values(p.raw_force_end[0]), qualification_flags=values(flags),
            qualification_flag_names=self.labels, actual_positive_weight=float(self.actual_weight[0]),
            actual_raw_clearance_delta=values(self.actual_raw[0]), actual_paid_clearance_delta=values(self.actual_applied[0]),
            weighted_actual_reward_contributions=values(contributions), reward_names=self.reward_names,
            actual_total_reward=float(env.reward_buf[0]), actual_verified_feet=int(p.verified[0].sum()),
            force_credit_eligible=bool(p.force_credit_eligible[0]), ever_invalid=bool(p.ever_invalid[0]),
            raw_leg_actions=values(env.action_manager.action[0]),
            bounded_leg_targets=values(term._processed_actions[0]),
            actual_leg_positions=values(env.asset.data.joint_pos[0, leg_ids]),
            actual_leg_velocities=values(env.asset.data.joint_vel[0, leg_ids]),
            upper_default_target_error=values(env.asset.data.joint_pos[0, upper_ids]
                - env.asset.data.default_joint_pos[0, upper_ids]),
            original_physical_acceptance_unchanged=True, training_not_formal=True)
        self.trace.write(json.dumps(value, allow_nan=False) + "\n")
        self.trace.flush()

    def finish(self):
        self.capture((~self.captured).nonzero().flatten())
        if not self.pending or not bool(self.captured.all()):
            raise RuntimeError("Credit observer missed a terminal or active environment")
        self.pending = False
        self.controls += 1
        if self.video is not None:
            self.video.observe()

    def snapshot(self):
        return dict(kind="same_control_pre_reset_credit_coverage_v1", controls=self.controls,
            group_order=["target_row0", "target_row1", "target_row2", "course_row0", "course_row1", "course_row2"],
            flag_names=self.labels, counts=self.counts.cpu().tolist(),
            reward_names=self.reward_names, actual_reward_integrals=self.reward_integrals.cpu().tolist(),
            raw_positive_negative_paid_positive_negative_delta_integrals=self.delta_integrals.cpu().tolist(),
            trace_controls=list(self.trace_controls), initial_partial_in_training_coverage=True,
            fixed_input_height_counts=self.fixed_input_height_counts.cpu().tolist(),
            current_input_height_counts=self.current_input_height_counts.cpu().tolist(),
            current_input_completed_update=self.current_input_update,
            terminal_before_reset=True, new_actor_observation=False, extra_NN_calls=0)

    def close(self):
        if not self.closed:
            self.closed = True
            self.trace.close()
