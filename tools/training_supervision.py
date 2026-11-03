"""Physical episode telemetry and an iteration-boundary stagnation guard."""

import json
import math
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class NoImprovementStop(RuntimeError):
    pass


_EFFECT_COMMON_THRESHOLDS = (
    "survival20s_fraction", "survival_and_progress_fraction", "locomotion_success_fraction",
)
_EFFECT_STAIR_THRESHOLDS = (
    "mean_completed_levels", "first_level_bilateral_fraction", "all_levels_bilateral_fraction",
    "stair_success_fraction", "mean_verified_feet", "mean_transferred_levels",
)
_EFFECT_CANONICAL_FIELDS = (
    "window_episode_count", "completed_episode_count", "mean_episode_s", "median_episode_s",
    "survival20s_fraction", "survival_horizon_fraction", "early_termination_fraction",
    "command_progress_ratio", "bilateral_landings_fraction", "survival_and_progress_fraction",
    "locomotion_success_fraction",
)
_EFFECT_FLAT_FIELDS = _EFFECT_CANONICAL_FIELDS + (
    "raw_mean_episode_s", "mean_xy_velocity_error_m_s", "mean_stance_slip_m_s",
    "physical_invalid_fraction", "contact_evidence_valid_fraction", "mean_net_world_x_m",
    "mean_raw_net_world_x_m", "mean_left_valid_landings", "mean_right_valid_landings",
    "moving_episode_count",
)
_EFFECT_STAIR_FIELDS = _EFFECT_CANONICAL_FIELDS + _EFFECT_STAIR_THRESHOLDS + (
    "mean_valid_upward_root_rise_m", "invalid_physical_episode_fraction", "physical_valid_episode_count",
)


def training_effect_profile(task_name, guarded_flat, has_flat_tracker, has_stair_tracker):
    """Bind the metric source to the declared task, never to available keys."""
    if has_flat_tracker and has_stair_tracker:
        raise RuntimeError("Effect monitoring cannot combine flat and stair trackers")
    if task_name == "Unitree-H2-Flat-Balance-Policy":
        if not guarded_flat or not has_flat_tracker or has_stair_tracker:
            raise RuntimeError("Guarded-flat effect monitoring requires its flat tracker")
        return "guarded_flat"
    if task_name == "Unitree-H2-Stair-Ascent-Policy":
        if guarded_flat or has_flat_tracker or not has_stair_tracker:
            raise RuntimeError("Stair effect monitoring requires its stair tracker")
        return "stairs"
    if task_name in ("Unitree-H2-Velocity-Flat", "Unitree-H2-Velocity-Rough", "Unitree-H2-Velocity-Stairs"):
        if guarded_flat or has_flat_tracker or has_stair_tracker:
            raise RuntimeError("Velocity walking effect monitoring requires MotionProbe")
        return "walking"
    raise ValueError("Effect monitoring requires an explicit supported training task")


def validate_training_effect_metrics(profile, summary, terrain_level, thresholds):
    """Validate the selected source before it can be hidden by a merged probe."""
    if (profile not in ("guarded_flat", "stairs", "walking") or not isinstance(summary, dict)
            or not isinstance(thresholds, dict) or any(not isinstance(name, str) for name in thresholds)):
        raise RuntimeError("Invalid training effect profile or source summary")
    known = set(_EFFECT_COMMON_THRESHOLDS + _EFFECT_STAIR_THRESHOLDS + ("terrain_level",))
    unknown = set(thresholds) - known
    if unknown:
        raise RuntimeError(f"Unknown training effect thresholds: {sorted(unknown)}")
    names = _EFFECT_COMMON_THRESHOLDS
    if terrain_level is not None:
        if not math.isfinite(float(terrain_level)):
            raise RuntimeError("Non-finite scene terrain level for effect monitoring")
        if "terrain_level" in thresholds:
            names += ("terrain_level",)
    if profile == "stairs":
        names += _EFFECT_STAIR_THRESHOLDS
    missing = set(names) - set(thresholds)
    if missing:
        raise RuntimeError(f"Missing required training effect thresholds: {sorted(missing)}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in thresholds.values()):
        raise RuntimeError("Training effect thresholds must be positive and finite")
    # Preserve the policy's comparison order and all its threshold values.
    names = tuple(name for name in thresholds if name in names)
    if not summary and profile == "walking":
        count = 0
    else:
        missing = {"window_episode_count", "completed_episode_count"} - set(summary)
        if missing:
            raise RuntimeError(f"Missing {profile} effect counters: {sorted(missing)}")
        if any(isinstance(summary[name], bool) or not isinstance(summary[name], (int, float))
               for name in ("window_episode_count", "completed_episode_count")):
            raise RuntimeError("Effect episode counters must be numeric counts")
        count = float(summary["window_episode_count"])
        completed = float(summary["completed_episode_count"])
        if (not math.isfinite(count) or not math.isfinite(completed) or count < 0
                or completed < count or not count.is_integer() or not completed.is_integer()):
            raise RuntimeError("Invalid complete-episode counters for effect monitoring")
        if not all(math.isfinite(float(value)) for value in summary.values()):
            raise RuntimeError("Non-finite authoritative training effect summary")
        if count:
            required = (_EFFECT_FLAT_FIELDS if profile == "guarded_flat" else
                        _EFFECT_STAIR_FIELDS if profile == "stairs" else _EFFECT_CANONICAL_FIELDS)
            missing = set(required) - set(summary)
            if missing:
                raise RuntimeError(f"Missing authoritative {profile} effect metrics: {sorted(missing)}")
            if any(isinstance(summary[name], bool) or not isinstance(summary[name], (int, float))
                   for name in required):
                raise RuntimeError("Authoritative training effect metrics must be numeric")
    return {"profile": profile, "threshold_metrics": list(names),
            "not_applicable_threshold_metrics": [name for name in thresholds if name not in names],
            "scene_terrain_level_available": terrain_level is not None,
            "source": {"guarded_flat": "flat_balance_progress", "stairs": "stair_progress",
                       "walking": "motion_probe"}[profile],
            "complete_episode_evidence": count > 0,
            "evidence_status": "complete_episode_window" if count else "no_complete_episode_evidence"}


def configure_stair_learning_rates(algorithm, ppo_learning_rate, rnd_learning_rate, *, resumed=False):
    """Keep fresh PPO independent of RSL2.3.3's local RND-LR shadowing.

    A resumed optimizer owns its saved group LRs. Only restore the adaptive-PPO
    scalar from that readback; neither optimizer group is overwritten on resume.
    """
    requested = float(ppo_learning_rate)
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError("Requested PPO learning rate must be positive and finite")
    before = [float(group["lr"]) for group in algorithm.optimizer.param_groups]
    rnd_optimizer = getattr(algorithm, "rnd_optimizer", None)
    rnd_before = [] if rnd_optimizer is None else [float(group["lr"]) for group in rnd_optimizer.param_groups]
    if not before or any(not math.isfinite(value) or value <= 0 for value in before + rnd_before):
        raise ValueError("Optimizer learning-rate readback is empty, non-finite or non-positive")
    if not resumed:
        for group in algorithm.optimizer.param_groups:
            group["lr"] = requested
    actual = [float(group["lr"]) for group in algorithm.optimizer.param_groups]
    algorithm.learning_rate = actual[0]
    rnd_actual = [] if rnd_optimizer is None else [float(group["lr"]) for group in rnd_optimizer.param_groups]
    if rnd_actual != rnd_before:
        raise RuntimeError("PPO initialization changed the independent RND optimizer LR")
    if not resumed:
        if actual != [requested] * len(actual) or float(algorithm.learning_rate) != requested:
            raise RuntimeError("Fresh PPO learning-rate write/readback differs from the request")
        if rnd_actual and (rnd_learning_rate is None or rnd_actual != [float(rnd_learning_rate)] * len(rnd_actual)):
            raise RuntimeError("Fresh RND optimizer LR differs from its independent configuration")
    return {"mode": "resume_saved_optimizer_lrs" if resumed else "fresh_explicit_ppo_lr",
            "requested_ppo_learning_rate": requested, "requested_rnd_learning_rate": rnd_learning_rate,
            "ppo_optimizer_group_lrs_before": before, "ppo_optimizer_group_lrs": actual,
            "ppo_algorithm_learning_rate": float(algorithm.learning_rate),
            "rnd_optimizer_group_lrs": rnd_actual, "rnd_lr_unchanged": True,
            "resume_optimizer_lrs_preserved": bool(resumed),
            "adaptive_kl_schedule_remains_enabled": algorithm.schedule == "adaptive"}


class MotionProbe:
    """Measure complete episodes before automatic scene resets teleport the robot."""

    def __init__(self, env):
        from h2_task.locomotion_progress import FootfallTracker
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.feet, _ = self.sensor.find_bodies(env.cfg.feet_body_names, preserve_order=True)
        self.robot_feet, _ = self.robot.find_bodies(env.cfg.feet_body_names, preserve_order=True)
        n, d = env.num_envs, env.device
        self.pending = torch.zeros(n, dtype=torch.bool, device=d)
        # Initial episodes have artificially randomized lengths in RSL-RL; exclude them.
        self.complete_episode = torch.zeros(n, dtype=torch.bool, device=d)
        self.age = torch.zeros(n, device=d)
        self.target_distance = torch.zeros(n, device=d)
        self.projected_distance = torch.zeros(n, device=d)
        self.error_integral = torch.zeros(n, device=d)
        self.slip_integral = torch.zeros(n, device=d)
        self.footfalls = FootfallTracker(n, d)
        self.air_time = self.footfalls.air_time
        self.landings = self.footfalls.landings
        self.pre_xy = torch.zeros((n, 2), device=d)
        self.desired_xy = torch.zeros((n, 2), device=d)
        self.pre_feet = torch.zeros((n, 2, 2), device=d)
        self.rows = deque(maxlen=5000)
        self.completed_episodes = 0
        original_reset = env._reset_idx

        def before_reset(raw, env_ids):
            self.finish(env_ids)
            self.end_episodes(env_ids)
            original_reset(env_ids)

        env._reset_idx = types.MethodType(before_reset, env)

    def begin(self):
        self.pre_xy.copy_(self.robot.data.root_pos_w[:, :2])
        self.pre_feet.copy_(self.robot.data.body_pos_w[:, self.robot_feet, :2])
        command = self.env.command_manager.get_command("base_velocity")
        q = self.robot.data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        self.desired_xy[:, 0] = cosine * command[:, 0] - sine * command[:, 1]
        self.desired_xy[:, 1] = sine * command[:, 0] + cosine * command[:, 1]
        self.pending[:] = True

    def finish(self, env_ids=None):
        ids = self.pending.nonzero(as_tuple=False).flatten() if env_ids is None else env_ids[self.pending[env_ids]]
        if not ids.numel():
            return
        delta = self.robot.data.root_pos_w[ids, :2] - self.pre_xy[ids]
        if not torch.isfinite(delta).all() or not torch.isfinite(self.robot.data.root_lin_vel_w[ids]).all():
            raise RuntimeError("Non-finite physical motion telemetry")
        dt = self.env.step_dt
        speed = torch.linalg.vector_norm(self.desired_xy[ids], dim=-1)
        active = speed >= 0.1
        direction = self.desired_xy[ids] / torch.clamp(speed[:, None], min=1e-6)
        self.age[ids] += dt
        self.target_distance[ids] += torch.where(active, speed * dt, 0.0)
        self.projected_distance[ids] += torch.where(active, (delta * direction).sum(-1), 0.0)
        self.error_integral[ids] += torch.linalg.vector_norm(
            self.robot.data.root_lin_vel_w[ids, :2] - self.desired_xy[ids], dim=-1) * dt
        all_contact = torch.linalg.vector_norm(self.sensor.data.net_forces_w[:, self.feet, :], dim=-1) > 5.0
        contact = all_contact[ids]
        self.footfalls.update(all_contact, dt, ids)
        foot_speed = torch.linalg.vector_norm(
            self.robot.data.body_pos_w[ids][:, self.robot_feet, :2] - self.pre_feet[ids], dim=-1) / dt
        self.slip_integral[ids] += (foot_speed * contact).sum(-1) / torch.clamp(contact.sum(-1), min=1) * dt
        self.pending[ids] = False

    def end_episodes(self, env_ids):
        ids = env_ids[self.complete_episode[env_ids] & (self.age[env_ids] > 0)]
        if ids.numel():
            failed = self.env.reset_terminated[ids]
            survived = (self.age[ids] >= 0.98 * self.env.max_episode_length_s) & ~failed
            progress = self.projected_distance[ids] / torch.clamp(self.target_distance[ids], min=1e-6)
            active = self.target_distance[ids] >= 0.1
            bilateral = (self.landings[ids] >= 2).all(-1)
            success = survived & (self.target_distance[ids] >= 0.5) & (progress >= 0.8) & (progress <= 1.2)
            values = torch.stack((self.age[ids], survived.float(), failed.float(),
                                  progress.clamp(-2, 2), active.float(), bilateral.float(),
                                  success.float(), (success & bilateral).float(),
                                  self.error_integral[ids] / self.age[ids],
                                  self.slip_integral[ids] / self.age[ids],
                                  (self.age[ids] >= 20.0-1e-4).float()), dim=1).cpu().numpy()
            self.rows.extend(values)
            self.completed_episodes += len(values)
        for value in (self.age, self.target_distance, self.projected_distance,
                      self.error_integral, self.slip_integral, self.air_time, self.landings):
            value[env_ids] = 0.0
        self.complete_episode[env_ids] = True
        self.footfalls.reset(env_ids)

    def summary(self):
        if not self.rows:
            return {}
        rows = np.asarray(self.rows)
        active = rows[:, 4] > 0
        return {
            "window_episode_count": len(rows), "completed_episode_count": self.completed_episodes,
            "mean_episode_s": float(rows[:, 0].mean()), "median_episode_s": float(np.median(rows[:, 0])),
            "survival20s_fraction": float(rows[:, 10].mean()),
            "survival_horizon_fraction": float(rows[:, 1].mean()), "early_termination_fraction": float(rows[:, 2].mean()),
            "moving_episode_count": int(active.sum()),
            "command_progress_ratio": float(rows[active, 3].mean()) if active.any() else 0.0,
            "bilateral_landings_fraction": float(rows[:, 5].mean()),
            "survival_and_progress_fraction": float(rows[:, 6].mean()),
            "locomotion_success_fraction": float(rows[:, 7].mean()),
            "mean_xy_velocity_error_m_s": float(rows[:, 8].mean()),
            "mean_stance_slip_m_s": float(rows[:, 9].mean()),
        }


class ProbedVecEnv(RslRlVecEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_probe = MotionProbe(self.unwrapped)

    def get_observations(self):
        result = super().get_observations()
        callback = getattr(self, "actual_raw_guard_callback", None)
        if callback is not None:
            callback(result[0])
        return result

    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("Non-finite training actions")
        self.motion_probe.begin()
        result = super().step(actions)
        self.motion_probe.finish()
        if not torch.isfinite(result[0]).all() or not torch.isfinite(result[1]).all():
            raise RuntimeError("Non-finite training observations or rewards")
        callback = getattr(self, "actual_raw_guard_callback", None)
        if callback is not None:
            callback(result[0])
        return result


class SupervisedRunner(OnPolicyRunner):
    def __init__(self, *args, supervision_path, policy, effect_task_name, normalize_initial_observations=False, guarded_flat=False, guarded_stair=False, **kwargs):
        self.guarded_flat = bool(guarded_flat)
        self.guarded_stair = bool(guarded_stair)
        if self.guarded_flat and self.guarded_stair:
            raise ValueError("A run must declare exactly one physical task")
        self.guarded_split = self.guarded_flat or self.guarded_stair
        self.effect_task_name = effect_task_name
        effect_env = (args[0] if args else kwargs["env"]).unwrapped
        self.effect_metric_profile = training_effect_profile(
            effect_task_name, self.guarded_flat,
            getattr(effect_env, "flat_balance_progress", None) is not None,
            getattr(effect_env, "stair_progress", None) is not None)
        if self.guarded_split:
            # Installed RSL only recognizes the literal PPO training type. Keep
            # that constructor contract local and restore its module immediately.
            import copy
            import rsl_rl.runners.on_policy_runner as runner_module
            from guarded_flat_ppo import H2GuardedPPO
            if self.guarded_stair:
                from guarded_stair_transfer import H2GuardedStairPPO
                algorithm_type = H2GuardedStairPPO
            else:
                algorithm_type = H2GuardedPPO
            local_args = list(args)
            cfg = copy.deepcopy(local_args[1] if len(local_args)>1 else kwargs["train_cfg"])
            if cfg["algorithm"]["class_name"] != algorithm_type.__name__:
                raise ValueError("Guarded runner requires its declared split algorithm")
            cfg["algorithm"]["class_name"] = "PPO"
            if len(local_args)>1: local_args[1]=cfg
            else: kwargs["train_cfg"]=cfg
            original = runner_module.PPO
            runner_module.PPO = algorithm_type
            try:
                super().__init__(*local_args, **kwargs)
            finally:
                runner_module.PPO = original
        else:
            super().__init__(*args, **kwargs)
        self.supervision_path = Path(supervision_path)
        self.stop_policy = policy
        self.supervision = json.loads(self.supervision_path.read_text())
        self.telemetry_path = Path(self.log_dir) / "motion_metrics.jsonl"
        self.assessments_path = Path(self.log_dir) / "effect_assessments.jsonl"
        self.best = None
        self.last_improvement_completed = None
        self.last_improvement_epoch = None
        self.stagnant_assessments = 0
        self.initial_observation_preprocessing = {
            "enabled": bool(normalize_initial_observations), "learn_calls": 0,
            "normalized_initial_batches": 0,
            "scope": "Explicit direct-policy task entry only; general VecEnv observations remain raw",
            "method": "One actor/critic normalization immediately before each learn call's first PPO act",
            "normalizer_statistics_continue_updating": not self.guarded_split,
            "subsequent_batches": "Unmodified RSL step normalization",
            "rnd_observations_changed": False,
        }

    def train_mode(self):
        super().train_mode()
        if self.guarded_split:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def learn(self, *args, **kwargs):
        preprocessing = self.initial_observation_preprocessing
        if not preprocessing["enabled"]:
            return super().learn(*args, **kwargs)
        if self.training_type != "rl" or not self.empirical_normalization:
            raise RuntimeError("Stair initial-batch normalization requires PPO with empirical dual normalization")
        preprocessing["learn_calls"] += 1
        original_act = self.alg.act
        had_override = "act" in vars(self.alg)
        previous_override = vars(self.alg).get("act")
        restored = False

        def restore_act():
            nonlocal restored
            if not restored:
                if had_override:
                    self.alg.act = previous_override
                else:
                    del self.alg.act
                restored = True

        def initial_act(obs, privileged_obs, *act_args, **act_kwargs):
            # The original learn loop has already moved these tensors to its
            # device and entered train/inference modes. Do not touch env/extras,
            # RND state or the later step normalization path.
            restore_act()
            normalizers = {"actor": self.obs_normalizer, "critic": self.privileged_obs_normalizer}
            context=getattr(self.alg,"trial_protocol",None) in ("explicit_task_context_v1","explicit_task_context_active_support_v1","explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1")
            if context:
                preprocessing["last_normalizer_prefix_counts_before"]={
                    name:int(normalizer.prefix_count) for name,normalizer in normalizers.items()}
            else:
                preprocessing["last_normalizer_counts_before"] = {
                    name: int(normalizer.count.item()) for name, normalizer in normalizers.items()}
            obs = self.obs_normalizer(obs)
            privileged_obs = (self.privileged_obs_normalizer(privileged_obs)
                              if self.privileged_obs_type is not None else obs)
            if context:
                preprocessing["last_normalizer_prefix_counts_after"]={
                    name:int(normalizer.prefix_count) for name,normalizer in normalizers.items()}
            else:
                preprocessing["last_normalizer_counts_after"] = {
                    name: int(normalizer.count.item()) for name, normalizer in normalizers.items()}
            preprocessing["normalized_initial_batches"] += 1
            preprocessing["last_actor_shape"] = list(obs.shape)
            preprocessing["last_critic_shape"] = list(privileged_obs.shape)
            self.update_state(initial_observation_preprocessing=dict(preprocessing))
            if self.log_dir is not None and (Path(self.log_dir)/"run.json").exists():
                metadata = json.loads((Path(self.log_dir)/"run.json").read_text())
                metadata["initial_observation_preprocessing"] = dict(preprocessing)
                write_json(Path(self.log_dir)/"run.json", metadata)
            return original_act(obs, privileged_obs, *act_args, **act_kwargs)

        self.alg.act = initial_act
        try:
            return super().learn(*args, **kwargs)
        finally:
            restore_act()

    def update_state(self, **fields):
        self.supervision.update(fields)
        write_json(self.supervision_path, self.supervision)

    def save(self, path, infos=None):
        if self.guarded_split:
            from guarded_flat_ppo import save_packet
            save_packet(path, self.alg.packet(self.current_learning_iteration, infos))
        else:
            super().save(path, infos)
        checkpoint = Path(path)
        self.update_state(latest_checkpoint=str(checkpoint),
                          latest_checkpoint_iteration=int(self.current_learning_iteration),
                          latest_checkpoint_size=checkpoint.stat().st_size)

    def log(self, locs, *args, **kwargs):
        if not all(math.isfinite(float(value)) for value in locs["loss_dict"].values()):
            raise RuntimeError("Non-finite optimization losses")
        if not torch.isfinite(self.alg.policy.action_std).all():
            raise RuntimeError("Non-finite policy action noise")
        exploration_budget = getattr(self, "stair_exploration_budget", None)
        if exploration_budget is not None:
            exploration_budget.assert_valid()
        super().log(locs, *args, **kwargs)
        iteration = int(locs["it"])
        completed = iteration + 1
        if iteration != 0 and completed % 100:
            return
        motion = self.env.motion_probe.summary()
        effect_source = motion
        stair_progress = getattr(self.env.unwrapped, "stair_progress", None)
        flat_progress = getattr(self.env.unwrapped, "flat_balance_progress", None)
        effect_profile = training_effect_profile(
            self.effect_task_name, self.guarded_flat, flat_progress is not None, stair_progress is not None)
        if effect_profile != self.effect_metric_profile:
            raise RuntimeError("Training effect metric profile changed during the run")
        if effect_profile == "stairs":
            effect_source = stair_progress.summary()
        elif effect_profile == "guarded_flat":
            effect_source = flat_progress.summary()
        motion = dict(motion)
        motion.update(effect_source)
        terrain = self.env.unwrapped.scene.terrain
        terrain_level = None
        if getattr(terrain, "terrain_levels", None) is not None:
            terrain_level = float(terrain.terrain_levels.float().mean())
            motion["terrain_level"] = terrain_level
        effect_schema = validate_training_effect_metrics(
            effect_profile, effect_source, terrain_level,
            self.stop_policy["absolute_improvement_thresholds"])
        motion["command_upper_m_s"] = float(self.env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x[1])
        motion["walking_curriculum_stage"] = int(getattr(self.env.unwrapped, "_h2_walking_curriculum_stage", 0))
        if not all(math.isfinite(float(value)) for value in motion.values()):
            raise RuntimeError("Non-finite physical episode metrics")
        point = {"iteration": iteration, "completed_iterations": completed, "checked_at_epoch": time.time(),
                 "losses": {name: float(value) for name, value in locs["loss_dict"].items()}, "motion": motion,
                 "effect_metric_schema": effect_schema}
        if exploration_budget is not None:
            point["exploration"] = exploration_budget.snapshot()
        if self.guarded_stair:
            temporal_credit = self.alg.temporal_credit_snapshot()
            if temporal_credit is not None:
                point["temporal_credit"] = temporal_credit
            rollout_collection = self.alg.rollout_collection_snapshot()
            if rollout_collection is not None:
                point["rollout_collection"] = rollout_collection
            clearance_progress = self.alg.clearance_progress_snapshot()
            if clearance_progress is not None:
                point["clearance_progress_potential"] = clearance_progress
            height_profile = self.alg.height_profile_snapshot()
            if height_profile is not None:
                point["height_profile"] = height_profile
                course = self.env.unwrapped.stair_height_course
                point["height_course"] = height_profile["height_course"]
                point["target15_progress"] = course.target_summary()
                point["per_height_training_progress"] = course.per_height_summary()
                point["mixed_motion_is_not_target15_success"] = True
            execution = self.alg.execution_chain_snapshot()
            if execution is not None:
                point["execution_chain"] = execution
        with self.telemetry_path.open("a") as stream:
            stream.write(json.dumps(point) + "\n")
        for name, value in motion.items():
            self.writer.add_scalar("Motion/" + name, value, iteration)
        self.writer.flush()
        self.update_state(status="training", checked_at_epoch=time.time(), iteration=iteration,
                          completed_iterations=completed, motion=motion,
                          telemetry=str(self.telemetry_path), effect_assessments=str(self.assessments_path))
        if exploration_budget is not None:
            self.update_state(exploration=point["exploration"])
        if completed % self.stop_policy["assessment_interval_iterations"] or completed < self.stop_policy["warmup_iterations"]:
            return
        if motion.get("window_episode_count", 0) < 1000:
            raise RuntimeError("Insufficient complete episodes for the training effect monitor")
        effect_names = effect_schema["threshold_metrics"]
        effect_vector = {name: float(motion[name])
                         for name in [*effect_names, "median_episode_s", "command_progress_ratio"]}
        if self.best is not None:
            if (set(self.best) != set(effect_vector)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in self.best.values())):
                raise RuntimeError("Invalid or changed training effect metric vector")
        now = time.time()
        improvements = []
        if self.best is None:
            improvements = ["warmup_baseline"]
            self.best = dict(effect_vector)
        else:
            for name in effect_names:
                threshold = self.stop_policy["absolute_improvement_thresholds"][name]
                if effect_vector[name] >= self.best[name] + threshold:
                    improvements.append(name)
            if effect_vector["median_episode_s"] >= self.best["median_episode_s"] + max(1.0, self.best["median_episode_s"] * 0.1):
                improvements.append("median_episode_s")
            # Progress gains must not be obtained by sacrificing episode stability.
            if (effect_vector["command_progress_ratio"] >= self.best["command_progress_ratio"] + 0.1
                    and effect_vector["median_episode_s"] >= self.best["median_episode_s"] * 0.9):
                improvements.append("command_progress_ratio")
        if improvements:
            self.last_improvement_completed = completed
            self.last_improvement_epoch = now
            self.stagnant_assessments = 0
            for name in effect_vector:
                self.best[name] = max(self.best[name], effect_vector[name])
        else:
            self.stagnant_assessments += 1
        elapsed_iterations = completed - self.last_improvement_completed
        elapsed_seconds = now - self.last_improvement_epoch
        stagnated = (self.stagnant_assessments >= 3
                     and elapsed_iterations >= self.stop_policy["patience_iterations"]
                     and elapsed_seconds >= self.stop_policy["patience_seconds"])
        assessment = {**point, "meaningful_improvements": improvements,
                      "best_motion": self.best, "stagnant_assessments": self.stagnant_assessments,
                      "iterations_without_improvement": elapsed_iterations,
                      "seconds_without_improvement": elapsed_seconds, "stop_required": stagnated}
        with self.assessments_path.open("a") as stream:
            stream.write(json.dumps(assessment) + "\n")
        self.update_state(last_effect_assessment=assessment)
        print("[H2 EFFECT] " + json.dumps(assessment), flush=True)
        if stagnated:
            checkpoint = Path(self.log_dir) / f"model_{iteration}_early_stop.pt"
            self.save(str(checkpoint))
            analysis = {"status": "stopped_no_improvement", "reason_zh": "学习期后，实际运动指标连续至少3000 iter且两小时没有达到实质改善门槛。",
                        "assessment": assessment, "checkpoint": str(checkpoint),
                        "acceptance": "Training episode telemetry is not locomotion acceptance. Perform a frozen-policy Isaac Sim replay before attributing the failure.",
                        "next_analysis": ["Compare complete motion windows and termination causes", "Inspect frozen-policy gait, support and body collapse", "Check main's frozen terrain reconstruction and upper-body masking without changing them"],
                        "automatic_restart_allowed": False}
            write_json(Path(self.log_dir) / "early_stop_analysis.json", analysis)
            self.update_state(status="stopped_no_improvement", stop_reason=analysis["reason_zh"],
                              analysis=str(Path(self.log_dir) / "early_stop_analysis.json"))
            raise NoImprovementStop(analysis["reason_zh"])
