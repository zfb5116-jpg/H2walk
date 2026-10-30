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
                                  self.slip_integral[ids] / self.age[ids]), dim=1).cpu().numpy()
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
            "survival20s_fraction": float(rows[:, 1].mean()), "early_termination_fraction": float(rows[:, 2].mean()),
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

    def step(self, actions):
        if not torch.isfinite(actions).all():
            raise RuntimeError("Non-finite training actions")
        self.motion_probe.begin()
        result = super().step(actions)
        self.motion_probe.finish()
        if not torch.isfinite(result[0]).all() or not torch.isfinite(result[1]).all():
            raise RuntimeError("Non-finite training observations or rewards")
        return result


class SupervisedRunner(OnPolicyRunner):
    def __init__(self, *args, supervision_path, policy, **kwargs):
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

    def update_state(self, **fields):
        self.supervision.update(fields)
        write_json(self.supervision_path, self.supervision)

    def save(self, path, infos=None):
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
        super().log(locs, *args, **kwargs)
        iteration = int(locs["it"])
        completed = iteration + 1
        if iteration != 0 and completed % 100:
            return
        motion = self.env.motion_probe.summary()
        terrain = self.env.unwrapped.scene.terrain
        if getattr(terrain, "terrain_levels", None) is not None:
            motion["terrain_level"] = float(terrain.terrain_levels.float().mean())
        motion["command_upper_m_s"] = float(self.env.unwrapped.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x[1])
        motion["walking_curriculum_stage"] = int(getattr(self.env.unwrapped, "_h2_walking_curriculum_stage", 0))
        if not all(math.isfinite(float(value)) for value in motion.values()):
            raise RuntimeError("Non-finite physical episode metrics")
        point = {"iteration": iteration, "completed_iterations": completed, "checked_at_epoch": time.time(),
                 "losses": {name: float(value) for name, value in locs["loss_dict"].items()}, "motion": motion}
        with self.telemetry_path.open("a") as stream:
            stream.write(json.dumps(point) + "\n")
        for name, value in motion.items():
            self.writer.add_scalar("Motion/" + name, value, iteration)
        self.writer.flush()
        self.update_state(status="training", checked_at_epoch=time.time(), iteration=iteration,
                          completed_iterations=completed, motion=motion,
                          telemetry=str(self.telemetry_path), effect_assessments=str(self.assessments_path))
        if completed % self.stop_policy["assessment_interval_iterations"] or completed < self.stop_policy["warmup_iterations"]:
            return
        if motion.get("window_episode_count", 0) < 1000:
            raise RuntimeError("Insufficient complete episodes for the training effect monitor")
        now = time.time()
        improvements = []
        if self.best is None:
            improvements = ["warmup_baseline"]
            self.best = dict(motion)
        else:
            for name, threshold in self.stop_policy["absolute_improvement_thresholds"].items():
                if motion[name] >= self.best[name] + threshold:
                    improvements.append(name)
            if motion["median_episode_s"] >= self.best["median_episode_s"] + max(1.0, self.best["median_episode_s"] * 0.1):
                improvements.append("median_episode_s")
            # Progress gains must not be obtained by sacrificing episode stability.
            if (motion["command_progress_ratio"] >= self.best["command_progress_ratio"] + 0.1
                    and motion["median_episode_s"] >= self.best["median_episode_s"] * 0.9):
                improvements.append("command_progress_ratio")
        if improvements:
            self.last_improvement_completed = completed
            self.last_improvement_epoch = now
            self.stagnant_assessments = 0
            for name in [*self.stop_policy["absolute_improvement_thresholds"], "median_episode_s", "command_progress_ratio"]:
                self.best[name] = max(self.best[name], motion[name])
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
