"""Train the H2 tasks using this checkout's task and robot assets."""

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-H2-Velocity-Stairs",
                    choices=[f"Unitree-H2-Velocity-{name}" for name in ("Flat", "Rough", "Stairs")])
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Number of additional learning iterations in this run.")
parser.add_argument("--save_interval", type=int, default=None,
                    help="Checkpoint interval in global learning iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--expected_completed_iterations", type=int, default=None,
                    help="Require the checkpoint to contain this many completed iterations.")
parser.add_argument("--supervision_state", type=Path, required=True)
parser.add_argument("--stop_policy", type=Path, required=True)
parser.add_argument("--expected_git_commit", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or (args.max_iterations is not None and args.max_iterations < 1):
    parser.error("num_envs and max_iterations must be positive")
if args.save_interval is not None and args.save_interval < 1:
    parser.error("save_interval must be positive")
if args.expected_completed_iterations is not None and (
    args.checkpoint is None or args.expected_completed_iterations < 1
):
    parser.error("expected_completed_iterations requires a checkpoint and must be positive")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_pickle, dump_yaml
from training_supervision import NoImprovementStop, ProbedVecEnv, SupervisedRunner, write_json

# Import only the H2 task from this repository, with no CtrlLab installation.
TASK_PARENT = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based"
sys.path.insert(0, str(TASK_PARENT))
import h2_task
from h2_task.networks import H2ActorCriticOur
import rsl_rl.runners.on_policy_runner as rsl_runner_module

# RSL-RL 2.3.3 resolves custom names in this module's globals.
rsl_runner_module.H2ActorCriticOur = H2ActorCriticOur


def load_config(entry_point):
    module, name = entry_point.split(":")
    return getattr(importlib.import_module(module), name)()


def main():
    current_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if current_commit != args.expected_git_commit:
        raise RuntimeError(f"Expected main source {args.expected_git_commit}, found {current_commit}")
    stop_policy = json.loads(args.stop_policy.read_text())
    state = json.loads(args.supervision_state.read_text())
    spec = gym.spec(args.task)
    env_cfg = load_config(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = load_config(spec.kwargs["rsl_rl_cfg_entry_point"])
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    agent_cfg.experiment_name = f"h2_main_{args.task.rsplit('-', 1)[-1].lower()}_manager_based"
    agent_cfg.run_name = args.run_name
    agent_cfg.logger = "tensorboard"
    agent_cfg.resume = args.checkpoint is not None
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    completed_iterations = 0
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required = {"iter", "model_state_dict", "optimizer_state_dict"}
        if agent_cfg.empirical_normalization:
            required.update({"obs_norm_state_dict", "privileged_obs_norm_state_dict"})
        if agent_cfg.algorithm.rnd_cfg is not None:
            required.update({"rnd_state_dict", "rnd_optimizer_state_dict"})
        if not required.issubset(saved):
            raise ValueError(f"Incomplete resume checkpoint: missing {sorted(required - saved.keys())}")
        completed_iterations = int(saved["iter"]) + 1
        if args.expected_completed_iterations is not None and (
            completed_iterations != args.expected_completed_iterations
        ):
            raise ValueError(f"Expected {args.expected_completed_iterations} completed iterations, "
                             f"checkpoint contains {completed_iterations}")
        del saved
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        run_name += "_" + args.run_name
    log_dir = ROOT / "logs/rsl_rl" / agent_cfg.experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    source_dir = log_dir / "training_source"
    source_dir.mkdir()
    for name in ("train.py", "train.sh", "training_supervision.py"):
        shutil.copy2(ROOT / "tools" / name, source_dir / name)
    shutil.copy2(args.stop_policy, source_dir / "stop_policy.json")
    # Include the new task modules as well as tracked files in the reproducible snapshot.
    shutil.copytree(ROOT / "source", source_dir / "source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT / "source").rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    (source_dir / "main_source_sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    source_patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"])
    (source_dir / "main_runtime_fix.patch").write_bytes(source_patch)
    env_cfg.io_descriptors_output_dir = str(log_dir)
    dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
    dump_pickle(str(log_dir / "params/env.pkl"), env_cfg)
    dump_pickle(str(log_dir / "params/agent.pkl"), agent_cfg)
    metadata = {
        "task": args.task, "task_source": h2_task.__file__,
        "asset": env_cfg.scene.robot.spawn.usd_path,
        "num_envs": args.num_envs, "max_iterations": agent_cfg.max_iterations,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval, "seed": args.seed,
        "device": agent_cfg.device, "checkpoint": str(checkpoint) if checkpoint else None,
        "starting_iteration": completed_iterations,
        "target_total_iterations": completed_iterations + agent_cfg.max_iterations,
        "source_branch": "main", "motion_supervision": True,
        "control_settings": {
            "stance_preload_torques": getattr(env_cfg, "stance_preload_torques", None),
            "stance_orientation_feedback": getattr(env_cfg, "stance_orientation_feedback", None),
            "lateral_com_feedback": getattr(env_cfg, "lateral_com_feedback", None),
            "contact_gait_reference": getattr(env_cfg, "contact_gait_reference", None),
            "progressive_walking_curriculum": getattr(env_cfg, "progressive_walking_curriculum", False),
            "action_scale": env_cfg.actions.JointPositionAction.scale,
            "init_noise_std": agent_cfg.policy.init_noise_std,
        },
        "initialization_mode": "checkpoint_resume" if checkpoint else "from_scratch",
        "source_snapshot": str(source_dir / "source"),
        "local_source_patch_sha256": hashlib.sha256(source_patch).hexdigest(),
        "local_source_patch": str(source_dir / "main_runtime_fix.patch"),
        "stop_policy": stop_policy,
        "status": "initializing",
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    }
    (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    state.update(status="initializing", run_dir=str(log_dir), metadata=str(log_dir / "run.json"),
                 git_commit=current_commit, source_hashes=source_hashes,
                 runtime_settings={"lock_upper_body": bool(getattr(env_cfg, "lock_upper_body_during_training", False)),
                                   "terrain_reconstruction": env_cfg.observations.policy.height_scanner.func.__name__})
    write_json(args.supervision_state, state)
    print(f"[H2] Log directory: {log_dir}", flush=True)
    print(f"[H2] Configuration: {json.dumps(metadata)}", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env = ProbedVecEnv(env, clip_actions=agent_cfg.clip_actions)
        obs, extras = env.get_observations()
        metadata["observation_shapes"] = {
            name: list(value.shape) for name, value in extras["observations"].items()
        }
        metadata["policy_shape"] = list(obs.shape)
        metadata["num_actions"] = env.num_actions
        action_manager = env.unwrapped.action_manager
        action_term = action_manager.get_term("JointPositionAction")
        metadata["action_contract"] = {
            "class": type(action_term).__name__,
            "joint_names": list(action_term._joint_names),
            "articulation_joint_ids": list(action_term._joint_ids),
            "held_upper_body_joint_ids": list(getattr(action_manager, "held_joint_ids", [])),
        }
        for name, value in {"policy": obs, **extras["observations"]}.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite initial observations in {name}")
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[H2] Runtime dimensions: {json.dumps(metadata['observation_shapes'])}; "
              f"actions={env.num_actions}", flush=True)
        runner = SupervisedRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device,
                                  supervision_path=args.supervision_state, policy=stop_policy)
        runner.add_git_repo_to_log(__file__)
        if checkpoint:
            runner.load(str(checkpoint), load_optimizer=True)
            # RSL-RL 2.3.3 stores the last completed zero-based iteration and
            # restarts its loop at that same index unless advanced here.
            if int(runner.current_learning_iteration) + 1 != completed_iterations:
                raise ValueError("Loaded runner iteration disagrees with checkpoint")
            runner.current_learning_iteration = completed_iterations
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            runner.tot_timesteps = completed_iterations * agent_cfg.num_steps_per_env * args.num_envs
        metadata["initial_learning_state"] = {
            "checkpoint_loaded": checkpoint is not None,
            "iteration": int(runner.current_learning_iteration),
            "policy_optimizer_state_entries": len(runner.alg.optimizer.state),
            "rnd_optimizer_state_entries": len(runner.alg.rnd_optimizer.state) if runner.alg.rnd else None,
            "policy_std": runner.alg.policy.std.detach().cpu().tolist(),
            "normalizer_counts": {
                label: {key: float(value.item()) for key, value in normalizer.state_dict().items()
                        if "count" in key and value.numel() == 1}
                for label, normalizer in [("actor", runner.obs_normalizer),
                                          ("critic", runner.privileged_obs_normalizer)]
            } if runner.empirical_normalization else {},
        }
        if checkpoint is None:
            if (metadata["initial_learning_state"]["iteration"] != 0
                    or metadata["initial_learning_state"]["policy_optimizer_state_entries"] != 0
                    or metadata["initial_learning_state"]["rnd_optimizer_state_entries"] not in (None, 0)):
                raise RuntimeError("Fresh training unexpectedly contains previous optimizer state")
        metadata["status"] = "training"
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[H2] Learning iterations: {completed_iterations} -> "
              f"{metadata['target_total_iterations']} "
              f"({agent_cfg.max_iterations} additional)", flush=True)
        runner.update_state(status="training", observation_shapes=metadata["observation_shapes"],
                            num_actions=env.num_actions, policy_shape=metadata["policy_shape"],
                            training_started_at_epoch=time.time(),
                            initialization_mode=metadata["initialization_mode"],
                            action_contract=metadata["action_contract"],
                            locked_action_ids=env.unwrapped.action_manager._locked_action_ids)
        try:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        except NoImprovementStop:
            metadata.update(status="stopped_no_improvement",
                            completed_iterations=int(runner.current_learning_iteration) + 1,
                            final_checkpoint=runner.supervision["latest_checkpoint"],
                            analysis=runner.supervision["analysis"])
            write_json(log_dir / "run.json", metadata)
            print("[H2] Stopped because physical movement metrics stagnated", flush=True)
            return
        metadata["status"] = "completed"
        metadata["completed_iterations"] = int(runner.current_learning_iteration) + 1
        metadata["final_checkpoint"] = str(log_dir / f"model_{runner.current_learning_iteration}.pt")
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        runner.update_state(status="completed", completed_iterations=metadata["completed_iterations"],
                            iteration=int(runner.current_learning_iteration), completed_at_epoch=time.time())
        print(f"[H2] Completed {metadata['completed_iterations']} total iterations", flush=True)
    finally:
        if "runner" in locals() and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if args.supervision_state.exists():
            state = json.loads(args.supervision_state.read_text())
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                         failed_at_epoch=time.time(), traceback=traceback.format_exc())
            write_json(args.supervision_state, state)
        raise
    finally:
        simulation_app.close()
