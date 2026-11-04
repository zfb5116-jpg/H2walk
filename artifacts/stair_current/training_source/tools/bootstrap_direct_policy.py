"""Distill the retained flat controller into the unchanged12-action policy.

This is one bounded teacher-data/initialization job, not a flat acceptance
retest or PPO continuation. Teacher execution is untouched. Student history
and frozen perception have separate action semantics and temporal state.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_checkpoint", type=Path, required=True)
parser.add_argument("--source_root", type=Path, default=ROOT/"source")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--state", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--updates_per_step", type=int, default=4)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules.normalizer import EmpiricalNormalization

sys.path.insert(0, str(args.source_root.resolve()/"CtrlLab/CtrlLab/tasks/manager_based"))
import h2_task
from h2_task.assets import H2_JOINT_NAMES
from h2_task.networks import H2ActorCriticOur
from h2_bootstrap_contract import DirectActionHistory, teacher_position_label, assert_observation_layout


def write(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2)+"\n")
    temporary.replace(path)


def update(**values):
    old = json.loads(args.state.read_text())
    old.update(values, checked_at_epoch=time.time())
    write(args.state, old)


def config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


class ShadowRaw:
    """Same real scene/rays, separate frozen reconstructor and temporal cache."""
    def __init__(self, raw, history):
        from h2_task.networks.lidar_terrain_reconstructor import LidarTerrainReconstructor
        self._raw, self._history = raw, history
        term = raw.action_manager.get_term("JointPositionAction")
        self.action_manager = types.SimpleNamespace(action=history.latest(), get_term=lambda _: term)
        pcfg = raw.cfg.observations.policy.height_scanner
        checkpoint = Path(pcfg.params["checkpoint"])
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._h2_lidar_reconstructor = LidarTerrainReconstructor(
            n_ch=int(blob["n_ch"]), n_az=int(blob["n_az"]), n_frames=int(blob["n_frames"]),
            proprio_dim=int(blob["proprio_dim"]), gru_delay_steps=int(blob["gru_delay_steps"]),
            use_height_splat=bool(blob["use_height_splat"]))
        self._h2_lidar_reconstructor.load_state_dict(blob["model_state_dict"], strict=True)
        self._h2_lidar_reconstructor.to(raw.device).eval()
        for p in self._h2_lidar_reconstructor.parameters():
            p.requires_grad_(False)
        self._h2_lidar_recon_cache = None
        self._h2_lidar_recon_hidden = None
        self.unwrapped = self

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def scan(self):
        self.action_manager.action = self._history.latest()
        cfg = self._raw.cfg.observations.policy.height_scanner
        result = cfg.func(self, **cfg.params).clone()
        if cfg.clip is not None:
            result.clamp_(*cfg.clip)
        if cfg.scale is not None:
            result.mul_(cfg.scale)
        return result


def main():
    torch.manual_seed(args.seed)
    spec = gym.spec("Unitree-H2-Velocity-Stairs")
    cfg, agent = config(spec.kwargs["env_cfg_entry_point"]), config(spec.kwargs["rsl_rl_cfg_entry_point"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.episode_length_s = 20.
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.height_scanner.params["side_noise_probability"] = 0.
    cfg.events.push_robot = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (.3, .3)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.resampling_time_range = (100000., 100000.)
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0., 0.)
    dump_yaml(str(args.output/"teacher_environment.yaml"), cfg)
    env = gym.make(spec.id, cfg=cfg)
    raw = env.unwrapped
    # Apply each environment's task command at resampling, BEFORE observation
    # and frozen-perception generation. Do not label a0.3-command observation
    # using a command that was secretly changed to0 after the fact.
    command_term = raw.command_manager.get_term("base_velocity")
    resample = command_term._resample_command

    def resample_teacher(term_object, env_ids):
        resample(env_ids)
        ids = torch.as_tensor(env_ids, device=raw.device, dtype=torch.long)
        stationary = ids[ids % 4 == 0]
        term_object.command[stationary, 0] = 0.
    command_term._resample_command = types.MethodType(resample_teacher, command_term)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.)
    try:
        obs, extras = wrapped.get_observations()
        assert tuple(obs.shape) == (args.num_envs, 1029)
        assert extras["observations"]["critic"].shape[-1] == 866
        assert extras["observations"]["rnd_state"].shape[-1] == 73 and wrapped.num_actions == 12
        assert raw.cfg.stair_reference is None and raw.cfg.contact_gait_reference is not None
        term = raw.action_manager.get_term("JointPositionAction")
        layout = assert_observation_layout(raw.observation_manager, term)
        robot, device = raw.scene["robot"], raw.device
        assert not term._support_priority_groups
        base_ratio = (robot.data.joint_damping[:, term._joint_ids]
                      /robot.data.joint_stiffness[:, term._joint_ids]).clone()
        initial_K = robot.root_physx_view.get_dof_stiffnesses().to(device)[:, term._joint_ids]
        initial_D = robot.root_physx_view.get_dof_dampings().to(device)[:, term._joint_ids]
        assert torch.allclose(initial_K, robot.data.joint_stiffness[:, term._joint_ids], atol=1e-3)
        assert torch.allclose(initial_D, robot.data.joint_damping[:, term._joint_ids], atol=1e-3)
        pcfg = agent.policy.to_dict()
        pcfg.pop("class_name")
        teacher = H2ActorCriticOur(1029, 866, 12, **pcfg).to(device).eval()
        saved = torch.load(args.teacher_checkpoint, map_location=device, weights_only=True)
        teacher.load_state_dict(saved["model_state_dict"], strict=True)
        student = H2ActorCriticOur(1029, 866, 12, **pcfg).to(device)
        student.actor.load_state_dict(teacher.actor.state_dict(), strict=True)
        tnorm = EmpiricalNormalization([1029]).to(device).eval()
        tnorm.load_state_dict(saved["obs_norm_state_dict"], strict=True)
        snorm = copy.deepcopy(tnorm).eval()
        cnorm = EmpiricalNormalization([866]).to(device).eval()
        cnorm.load_state_dict(saved["privileged_obs_norm_state_dict"], strict=True)
        for normalizer, sl in ((snorm, slice(323, 468)), (cnorm, slice(70, 99))):
            normalizer._mean[:, sl] = 0.
            normalizer._var[:, sl] = 1.
            normalizer._std[:, sl] = 1.
        # The new task uses the same frozen LiDAR perception in actor/critic.
        # Rebase the critic's old privileged-height statistics accordingly.
        for key in ("_mean", "_var", "_std"):
            getattr(cnorm, key)[:, 305:866] = getattr(snorm, key)[:, 468:1029]
        lower = robot.data.soft_joint_pos_limits[:, term._joint_ids, 0].clone()
        upper = robot.data.soft_joint_pos_limits[:, term._joint_ids, 1].clone()
        mid, half = (lower+upper)/2, (upper-lower)/2
        history = DirectActionHistory(args.num_envs, device, list(term._joint_names), H2_JOINT_NAMES)
        shadow = ShadowRaw(raw, history)
        k = robot.data.joint_stiffness[:, term._joint_ids].clone()
        caps = robot.data.joint_effort_limits[:, term._joint_ids].clone()
        initial_std = (.10*caps/(k*half).clamp_min(1e-6)).median(0).values.clamp(.01, .20)
        with torch.no_grad():
            student.std.copy_(initial_std)
        optimizer = torch.optim.Adam(student.actor.parameters(), lr=1.e-3)
        label = {}
        original = term.process_actions

        def capture(action_term, actions):
            original(actions)
            K = robot.data.joint_stiffness[:, term._joint_ids]
            D = robot.data.joint_damping[:, term._joint_ids]
            if not torch.equal(D/K, base_ratio):
                raise RuntimeError("Teacher D/K changed across the delayed position-only equivalence")
            if bool((robot.data.joint_effort_target[:, term._joint_ids] != 0).any()):
                raise RuntimeError("Teacher leg external effort violates the zero-force label contract")
            velocity = action_term.gait_reference.qdot.reshape(args.num_envs, 12)
            label.clear()
            label.update(teacher_position_label(action_term._processed_actions.clone(),
                velocity.clone(), K, D, lower, upper))
            label["weight"] = K/caps
        term.process_actions = types.MethodType(capture, term)
        capacity = 65536
        rb_obs = torch.empty(capacity, 1029, dtype=torch.float16, pin_memory=True)
        rb_target = torch.empty(capacity, 12, dtype=torch.float32, pin_memory=True)
        rb_weight = torch.empty_like(rb_target, pin_memory=True)
        # Keep a deterministic, stratified sample from EVERY collection step.
        # The last simulation frame must not certify an early best checkpoint.
        validation_capacity = args.steps*16
        validation_obs = torch.empty(validation_capacity, 1029, dtype=torch.float16)
        validation_target = torch.empty(validation_capacity, 12)
        validation_weight = torch.empty_like(validation_target)
        validation_standing = torch.empty(validation_capacity, dtype=torch.bool)
        validation_count = 0
        pointer, available = 0, 0
        indices = torch.arange(args.num_envs, device=device)
        holdout = indices % 40 < 4  # one standing and three walking lanes per block
        standing = indices % 4 == 0
        best, clipped_total, labeled_total = float("inf"), 0, 0
        started = time.time()
        update(status="collecting_teacher_and_distilling", policy_dimensions=[1029, 866, 73, 12],
               source_root=str(args.source_root), teacher_checkpoint=str(args.teacher_checkpoint),
               action_semantics="full soft-range tanh joint positions; native velocity target0", steps=0,
               student_shadow_perception_independent=True, ppo_updates=0, observation_layout=layout)
        for step in range(args.steps):
            with torch.no_grad():
                sobs, _ = history.rewrite(obs)
                sobs[:, 468:1029] = shadow.scan()
                normalized = snorm(sobs).detach()
                teacher_actions = teacher.act_inference(tnorm(obs))
                next_obs, _, dones, extras = wrapped.step(teacher_actions)
                target = label["position"].detach()
                weight = label["weight"].detach()
                valid = torch.isfinite(normalized).all(-1) & torch.isfinite(target).all(-1)
                selected = []
                for group, count in ((holdout & standing, 4), (holdout & ~standing, 12)):
                    pool = (valid & group).nonzero().flatten()
                    if len(pool):
                        selected.append(pool[(torch.arange(min(count, len(pool)), device=device)
                                             +step*count) % len(pool)])
                validation_ids = torch.cat(selected)
                end = validation_count+len(validation_ids)
                validation_obs[validation_count:end] = normalized[validation_ids].to("cpu", dtype=torch.float16)
                validation_target[validation_count:end] = target[validation_ids].cpu()
                validation_weight[validation_count:end] = weight[validation_ids].cpu()
                validation_standing[validation_count:end] = standing[validation_ids].cpu()
                validation_count = end
                ids = (valid & ~holdout).nonzero().flatten()
                n = len(ids)
                slots = (torch.arange(n)+pointer) % capacity
                rb_obs[slots] = normalized[ids].to("cpu", dtype=torch.float16)
                rb_target[slots] = target[ids].cpu()
                rb_weight[slots] = weight[ids].cpu()
                pointer, available = (pointer+n) % capacity, min(capacity, available+n)
                clipped_total += int(label["clipped_mask"].any(-1).sum())
                labeled_total += args.num_envs
                history.append(label["raw"], dones.bool())
                obs = next_obs
            if available >= args.batch:
                for _ in range(args.updates_per_step):
                    take = torch.randint(available, (args.batch,))
                    x = rb_obs[take].to(device=device, dtype=torch.float32, non_blocking=True)
                    target_batch = rb_target[take].to(device, non_blocking=True)
                    wb = rb_weight[take].to(device, non_blocking=True)
                    predicted = mid[0]+half[0]*torch.tanh(student.act_inference(x))
                    loss = ((predicted-target_batch)*wb).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite behavior-distillation objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.actor.parameters(), 1.)
                    optimizer.step()
            if (step+1) % 50 == 0:
                with torch.no_grad():
                    predicted = mid+half*torch.tanh(student.act_inference(normalized))
                    error = ((predicted-target)*weight).abs()[valid & holdout].flatten()
                    mean = float(error.mean())
                    p95 = float(torch.quantile(error, .95))
                    group_error = ((predicted-target)*weight).abs()
                    standing_mean = float(group_error[valid & holdout & standing].mean())
                    walking_mean = float(group_error[valid & holdout & ~standing].mean())
                point = dict(steps=step+1, mean_normalized_torque_error=mean,
                    p95_normalized_torque_error=p95, teacher_label_clipped_fraction=clipped_total/max(1,labeled_total),
                    standing_mean_normalized_error=standing_mean, walking_mean_normalized_error=walking_mean,
                    replay_samples=available, wall_seconds=time.time()-started)
                with (args.output/"distillation_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(point)+"\n")
                print("[H2 BOOTSTRAP] "+json.dumps(point), flush=True)
                update(**point)
                score = max(standing_mean, walking_mean)
                if score < best:
                    best = score
                    payload = dict(format="h2_direct_policy_initialization_v1", ppo_completed_iterations=0,
                        model_state_dict=student.state_dict(), obs_norm_state_dict=snorm.state_dict(),
                        privileged_obs_norm_state_dict=cnorm.state_dict(), action_semantics="full_soft_tanh_position_v1",
                        teacher_checkpoint_sha256=hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest(),
                        policy_dimensions=[1029,866,73,12], metadata=point)
                    temporary = args.output/"initial_policy.pt.tmp"
                    torch.save(payload, temporary)
                    temporary.replace(args.output/"initial_policy.pt")
        early_best = torch.load(args.output/"initial_policy.pt", map_location="cpu", weights_only=True)
        final_candidate = dict(early_best, model_state_dict={k: v.detach().cpu().clone()
            for k, v in student.state_dict().items()}, metadata=point)
        candidates = []
        for name, candidate in (("online_best", early_best), ("final", final_candidate)):
            student.load_state_dict(candidate["model_state_dict"], strict=True)
            errors = []
            with torch.no_grad():
                for start in range(0, validation_count, args.batch):
                    end = min(start+args.batch, validation_count)
                    x = validation_obs[start:end].to(device=device, dtype=torch.float32)
                    predicted = mid[0]+half[0]*torch.tanh(student.act_inference(x))
                    err = ((predicted.cpu()-validation_target[start:end])
                           *validation_weight[start:end]).abs()
                    errors.append(err)
            error = torch.cat(errors)
            mask = validation_standing[:validation_count]
            quality = dict(validation_samples=validation_count, coverage_steps=args.steps,
                mean_normalized_torque_error=float(error.mean()),
                p95_normalized_torque_error=float(torch.quantile(error.flatten(), .95)),
                standing_mean_normalized_error=float(error[mask].mean()),
                walking_mean_normalized_error=float(error[~mask].mean()),
                teacher_label_clipped_fraction=clipped_total/max(1,labeled_total),
                selected_candidate=name, selected_training_step=candidate["metadata"]["steps"])
            candidates.append((max(quality["standing_mean_normalized_error"],
                                   quality["walking_mean_normalized_error"]), candidate, quality))
        _, initial, quality = min(candidates, key=lambda item: item[0])
        initial["metadata"] = quality
        temporary = args.output/"initial_policy.pt.tmp"
        torch.save(initial, temporary)
        temporary.replace(args.output/"initial_policy.pt")
        budget_ready = (quality["mean_normalized_torque_error"] < .05
                        and quality["standing_mean_normalized_error"] < .05
                        and quality["walking_mean_normalized_error"] < .05
                        and quality["p95_normalized_torque_error"] < .15
                        and quality["teacher_label_clipped_fraction"] < .05)
        report = dict(status="completed", initialization_ready=budget_ready,
            steps=args.steps, teacher_transitions=labeled_total, ppo_updates=0,
            initial_policy=str(args.output/"initial_policy.pt"), initialization_quality=quality,
            best_normalized_error=best, validation_candidates=[c[2] for c in candidates],
            formal_walking_acceptance=False, stair_acceptance=False,
            source_root=str(args.source_root), teacher_checkpoint=str(args.teacher_checkpoint),
            system_scope="Teacher system drives physics; shadow student action/perception histories only initialize direct policy.")
        write(args.output/"result.json", report)
        update(**report)
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        app.close()
