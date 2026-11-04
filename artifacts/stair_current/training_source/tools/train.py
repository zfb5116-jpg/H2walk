"""Train the H2 tasks using this checkout's task and robot assets."""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
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
                    choices=[f"Unitree-H2-Velocity-{name}" for name in ("Flat", "Rough", "Stairs")]
                            + ["Unitree-H2-Stair-Ascent-Policy", "Unitree-H2-Flat-Balance-Policy"])
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Number of additional learning iterations in this run.")
parser.add_argument("--save_interval", type=int, default=None,
                    help="Checkpoint interval in global learning iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--initial_policy", type=Path, default=None,
                    help="Reviewed direct-policy model/own-normalizer initialization; optimization starts fresh.")
parser.add_argument("--guarded_flat_plan", type=Path, default=None,
                    help="Immutable reviewed protocol for the separate bounded guarded-flat trial.")
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
if args.initial_policy is not None and (args.checkpoint is not None or args.task not in (
        "Unitree-H2-Stair-Ascent-Policy", "Unitree-H2-Flat-Balance-Policy")):
    parser.error("initial_policy is only for a fresh direct-policy run, not checkpoint resume")
if args.task == "Unitree-H2-Flat-Balance-Policy" and (args.initial_policy is None
        or args.checkpoint is not None or args.guarded_flat_plan is None):
    parser.error("This guarded-flat trial requires its reviewed plan and fresh initial_policy, without resume")
if args.guarded_flat_plan is not None and args.task != "Unitree-H2-Flat-Balance-Policy":
    parser.error("guarded_flat_plan only applies to the separate flat repair task")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_pickle, dump_yaml
from training_supervision import (NoImprovementStop, ProbedVecEnv, SupervisedRunner,
                                  configure_stair_learning_rates, write_json)

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
        raise RuntimeError(f"Expected source {args.expected_git_commit}, found {current_commit}")
    stop_policy = json.loads(args.stop_policy.read_text())
    state = json.loads(args.supervision_state.read_text())
    guarded_flat = args.task == "Unitree-H2-Flat-Balance-Policy"
    guarded_plan = None
    typed_flat_transfer = False
    context_flat = False
    if guarded_flat:
        guarded_plan = json.loads(args.guarded_flat_plan.resolve(strict=True).read_text())
        trial_protocol = guarded_plan.get("trial_protocol", "guarded_flat_balance_v1")
        active_support_flat = trial_protocol in ("explicit_task_context_active_support_v1","explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1")
        context_flat = trial_protocol in ("explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1")
        typed_flat_transfer = trial_protocol in ("overspeed_credit_action_margin_v1", "simultaneous_output_proposal_v1", "reserved_simultaneous_output_proposal_v1", "resource_bounded_reserved_output_proposal_v1", "strict_primal_resource_output_proposal_v1", "command_scaled_velocity_cost_v1", "outward_tilt_growth_v1", "episode_clean_force_eligibility_v1", "explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1")
        expected_initial_sha = {
            "guarded_flat_balance_v1":"ce822132385de7b9c70315087bbcc4da1739d0da8460cf656c39e14aa1424688",
            "overspeed_credit_action_margin_v1":"01e1584898c5313ae996ae3c3c57ce23d4cc06ba51d7ab8decbae01c59e88a68",
            "simultaneous_output_proposal_v1":"319ff350be8c8e6359be978313f484262e2e1f9ddef03d675b7bf0a6ac7968c5",
            "reserved_simultaneous_output_proposal_v1":"ffcec4bbe62f63f77fcdfdac53d322fe19ef355dda40b7a2b1ee394015f81ed0",
            "resource_bounded_reserved_output_proposal_v1":"e96a9fd29cc0d7457db742133345c0224f2dbcdde522692b52eba88ebdd6977d",
            "strict_primal_resource_output_proposal_v1":"4cecd1fc24d618502402c1bd3ed5bcb0fdc7f44a0981a3bfe5855895715176b2",
            "command_scaled_velocity_cost_v1":"fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa",
            "outward_tilt_growth_v1":"fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa",
            "episode_clean_force_eligibility_v1":"fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa",
            "explicit_task_context_v1":"fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa",
            "explicit_task_context_active_support_v1":"b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce",
            "explicit_task_context_full_outward_growth_v1":"b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce",
            "explicit_task_context_forward_excess_cost_v1":"f27165c7c3ed0f1033e2ef02b1cfbd4d6120af3d4eac070d8fda7c720d53eb83",
            "explicit_task_context_longer_budget_v1":"756d623374ec9770ab3472e82587af8106f157129c8ab9a29e2a1f58f2714b32",
        }.get(trial_protocol)
        if (guarded_plan["task"] != args.task or guarded_plan["network_dimensions"] != ([1034,871,73,12] if context_flat else [1029,866,73,12])
                or guarded_plan["initial_policy_sha256"] != expected_initial_sha
                or expected_initial_sha is None
                or args.num_envs != 1024 or args.seed != 42):
            raise ValueError("Guarded-flat reviewed initialization/task/layout differs")
        if trial_protocol in ("episode_clean_force_eligibility_v1", "explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1"):
            from evaluate_flat_balance_policy import (validate_episode_force_reward_plan,
                                                     validate_episode_force_reward_metadata)
            validate_episode_force_reward_plan(guarded_plan)
        if trial_protocol == "explicit_task_context_longer_budget_v1":
            from v11_parent_custody import (COMPLETE_V13_PROTECTED_FILES, COMPLETE_V13_RECEIPT_SHA,
                                             validate_complete_v13_protected_source_records)
            canonical_parent = ROOT / "logs/rsl_rl/h2_main_policy_manager_based/2026-11-01_21-08-43_guarded_flat_balance_7093dca0/model_999.pt"
            if (args.initial_policy.resolve(strict=True) != canonical_parent
                    or Path(guarded_plan["initial_policy"]).resolve(strict=True) != canonical_parent
                    or guarded_plan.get("status") != "reviewed_ready"
                    or set(guarded_plan.get("protected_network_input_native_source_sha256", {})) != set(COMPLETE_V13_PROTECTED_FILES)):
                raise ValueError("New longer-budget type requires its readyplan/canonical parent/exact14 keys before model loading")
            parent_receipt_path = canonical_parent.with_suffix(".validation.json")
            if (Path(guarded_plan["initial_policy_CPU_receipt"]).resolve(strict=True) != parent_receipt_path
                    or guarded_plan["initial_policy_CPU_receipt_sha256"] != COMPLETE_V13_RECEIPT_SHA
                    or hashlib.sha256(parent_receipt_path.read_bytes()).hexdigest() != COMPLETE_V13_RECEIPT_SHA):
                raise ValueError("New longer-budget type requires its already-issued canonical parent CPU metadata")
            parent_receipt = json.loads(parent_receipt_path.read_text())
            validate_complete_v13_protected_source_records(
                guarded_plan["protected_network_input_native_source_sha256"], parent_receipt["source_sha256"],
                {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in COMPLETE_V13_PROTECTED_FILES})
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
    if guarded_flat:
        from task_context_metadata import (CONTEXT_PROTOCOLS, LONGER_BUDGET_CONTEXT_PROTOCOL,
                                           context_learning_budget)
        budget = context_learning_budget(trial_protocol) if trial_protocol in CONTEXT_PROTOCOLS else 1000
        if (agent_cfg.max_iterations != budget or guarded_plan["max_iterations"] != budget
                or agent_cfg.save_interval != 500 or agent_cfg.num_steps_per_env != 24
                or agent_cfg.algorithm.class_name != "H2GuardedPPO"):
            raise ValueError("Guarded-flat typed budget/save500/24-step split-PPO protocol changed")
        if trial_protocol == LONGER_BUDGET_CONTEXT_PROTOCOL:
            # Opt in for this reviewed type only; the task's default remains v13/1000.
            env_cfg.flat_trial_protocol = trial_protocol
            agent_cfg.algorithm.trial_protocol = trial_protocol
    if guarded_flat and getattr(agent_cfg.algorithm,"trial_protocol","guarded_flat_balance_v1") != trial_protocol:
        raise ValueError("Guarded-flat saved config and reviewed trial profile differ")
    agent_cfg.experiment_name = f"h2_main_{args.task.rsplit('-', 1)[-1].lower()}_manager_based"
    agent_cfg.run_name = args.run_name
    agent_cfg.logger = "tensorboard"
    agent_cfg.resume = args.checkpoint is not None
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    initial_policy = args.initial_policy.resolve(strict=True) if args.initial_policy else None
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
    training_tool_names = ["train.py", "train.sh", "training_supervision.py"]
    if args.task in ("Unitree-H2-Stair-Ascent-Policy", "Unitree-H2-Flat-Balance-Policy"):
        training_tool_names.append("stair_exploration.py")
    if guarded_flat:
        training_tool_names.append("guarded_flat_ppo.py")
        if trial_protocol in ("simultaneous_output_proposal_v1", "reserved_simultaneous_output_proposal_v1", "resource_bounded_reserved_output_proposal_v1", "strict_primal_resource_output_proposal_v1", "command_scaled_velocity_cost_v1", "outward_tilt_growth_v1", "episode_clean_force_eligibility_v1", "explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1"):
            training_tool_names.append("functional_proposal.py")
        if trial_protocol in ("episode_clean_force_eligibility_v1", "explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1"):
            training_tool_names.extend(["evaluate_flat_balance_policy.py", "evaluate_direct_initialization.py"])
        if context_flat:
            training_tool_names.extend(["task_context_normalization.py", "task_context_metadata.py"])
        if active_support_flat:
            training_tool_names.append("v11_parent_custody.py")
    for name in training_tool_names:
        shutil.copy2(ROOT / "tools" / name, source_dir / name)
    shutil.copy2(args.stop_policy, source_dir / "stop_policy.json")
    if guarded_flat:
        shutil.copy2(args.guarded_flat_plan, source_dir / "guarded_flat_plan.json")
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
    dependency_paths = [Path(env_cfg.scene.robot.spawn.usd_path)]
    lidar_checkpoint = env_cfg.observations.policy.height_scanner.params.get("checkpoint")
    if lidar_checkpoint:
        dependency_paths.append(Path(lidar_checkpoint))
    generator = env_cfg.scene.terrain.terrain_generator
    task_geometry = None if generator is None else {
        "rows": generator.num_rows, "columns": generator.num_cols,
        "size_m": list(generator.size), "curriculum": generator.curriculum,
        "sub_terrains": {
            name: {"function": getattr(term.function, "__module__", "")+"."
                               +getattr(term.function, "__qualname__", type(term.function).__name__),
                   **{key: getattr(term, key) for key in
                      ("num_steps", "minimum_riser_height", "maximum_riser_height", "height_levels",
                       "riser_height", "tread_depth", "stair_width", "approach_distance")
                      if hasattr(term, key)}}
            for name, term in generator.sub_terrains.items()
        },
    }
    metadata = {
        "task": args.task, "task_source": h2_task.__file__,
        "asset": env_cfg.scene.robot.spawn.usd_path,
        "num_envs": args.num_envs, "max_iterations": agent_cfg.max_iterations,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval, "seed": args.seed,
        "device": agent_cfg.device, "checkpoint": str(checkpoint) if checkpoint else None,
        "task_geometry": task_geometry,
        "starting_iteration": completed_iterations,
        "target_total_iterations": completed_iterations + agent_cfg.max_iterations,
        "source_branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(), "motion_supervision": True,
        "control_settings": {
            "stance_preload_torques": getattr(env_cfg, "stance_preload_torques", None),
            "stance_orientation_feedback": getattr(env_cfg, "stance_orientation_feedback", None),
            "lateral_com_feedback": getattr(env_cfg, "lateral_com_feedback", None),
            "contact_gait_reference": getattr(env_cfg, "contact_gait_reference", None),
            "stair_reference": getattr(env_cfg, "stair_reference", None),
            "progressive_walking_curriculum": getattr(env_cfg, "progressive_walking_curriculum", False),
            "action_scale": env_cfg.actions.JointPositionAction.scale,
            "init_noise_std": agent_cfg.policy.init_noise_std,
        },
        "algorithm_settings": {
            "name": agent_cfg.algorithm.class_name,
            "gamma": agent_cfg.algorithm.gamma,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "rnd_learning_rate": (agent_cfg.algorithm.rnd_cfg.learning_rate
                                  if agent_cfg.algorithm.rnd_cfg is not None else None),
            "desired_kl": agent_cfg.algorithm.desired_kl,
        },
        "initialization_mode": (guarded_plan["initialization_mode"] if typed_flat_transfer
                                else "checkpoint_resume" if checkpoint else "behavior_cloned_direct_policy" if initial_policy else "from_scratch"),
        "initial_policy": str(initial_policy) if initial_policy else None,
        "external_dependency_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in dependency_paths
        },
        "training_tool_sha256": {
            name: hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest()
            for name in training_tool_names
        },
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
        if hasattr(env.unwrapped, "stair_progress"):
            raw = env.unwrapped
            metadata["runtime_stair_geometry"] = {
                "terrain_rows": raw.scene.terrain.cfg.terrain_generator.num_rows,
                "riser_min_m": raw._stair_minimum_height,
                "riser_max_m": raw._stair_maximum_height,
                "current_riser_min_m": float(raw._stair_heights().min()),
                "current_riser_max_m": float(raw._stair_heights().max()),
                "levels": raw.stair_progress.steps, "depth_m": raw.stair_progress.depth,
                "width_m": raw.stair_progress.width, "approach_m": raw.stair_progress.approach,
                "scripted_gait_installed": action_term.gait_reference is not None,
                "domain_bounds_local": list(raw._stair_domain_bounds_local),
                "below_floor_tolerance_m": raw.cfg.stair_below_floor_tolerance_m,
                "termination_terms": list(raw.termination_manager.active_terms),
                "episode_evidence_directory": str(raw.stair_episode_evidence_directory),
            }
        for name, value in {"policy": obs, **extras["observations"]}.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite initial observations in {name}")
        if guarded_flat:
            metadata["runtime_flat_geometry"] = env.unwrapped.runtime_contract
            if trial_protocol in ("episode_clean_force_eligibility_v1", "explicit_task_context_v1", "explicit_task_context_active_support_v1", "explicit_task_context_full_outward_growth_v1", "explicit_task_context_forward_excess_cost_v1", "explicit_task_context_longer_budget_v1"):
                validate_episode_force_reward_metadata(metadata["runtime_flat_geometry"], guarded_plan)
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[H2] Runtime dimensions: {json.dumps(metadata['observation_shapes'])}; "
              f"actions={env.num_actions}", flush=True)
        runner = SupervisedRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device,
                                  supervision_path=args.supervision_state, policy=stop_policy,
                                  normalize_initial_observations=args.task in ("Unitree-H2-Stair-Ascent-Policy", "Unitree-H2-Flat-Balance-Policy"),
                                  guarded_flat=guarded_flat, effect_task_name=args.task)
        runner.add_git_repo_to_log(__file__)
        if context_flat:
            from task_context_normalization import make_task_context_normalizers, normalization_metadata
            context_normalizers=make_task_context_normalizers(runner.device)
            runner.obs_normalizer=context_normalizers["actor"]
            runner.privileged_obs_normalizer=context_normalizers["critic"]
        if initial_policy:
            boot = torch.load(initial_policy, map_location="cpu" if typed_flat_transfer else agent_cfg.device, weights_only=True)
            if typed_flat_transfer:
                from guarded_flat_ppo import validate_typed_policy_transfer, apply_typed_policy_transfer
                transfer = validate_typed_policy_transfer(boot, initial_policy, guarded_plan, ROOT)
                transfer = apply_typed_policy_transfer(runner.alg,
                    {"actor":runner.obs_normalizer,"critic":runner.privileged_obs_normalizer},boot,transfer)
                metadata["initial_policy_transfer"] = transfer
                parent_source_hashes = transfer["parent_source_sha256"]
            else:
                if (boot.get("format") != "h2_direct_policy_initialization_v1"
                        or boot.get("ppo_completed_iterations") != 0
                        or boot.get("action_semantics") != "full_soft_tanh_position_v1"
                        or boot.get("policy_dimensions") != [1029, 866, 73, 12]):
                    raise ValueError("Initial policy does not match the direct-action contract")
                receipt = json.loads((initial_policy.parent / "result.json").read_text())
                if not guarded_flat and not receipt.get("initialization_ready", False):
                    raise ValueError("Behavior initialization did not pass its declared quality gate")
                runner.alg.policy.load_state_dict(boot["model_state_dict"], strict=True)
                runner.obs_normalizer.load_state_dict(boot["obs_norm_state_dict"], strict=True)
                runner.privileged_obs_normalizer.load_state_dict(boot["privileged_obs_norm_state_dict"], strict=True)
                parent_source_hashes = boot["source_sha256"]
                metadata["behavior_initialization_quality"] = boot["metadata"]
            if guarded_flat:
                digest = hashlib.sha256(initial_policy.read_bytes()).hexdigest()
                if (initial_policy != Path(guarded_plan["initial_policy"]).resolve(strict=True)
                        or digest != guarded_plan["initial_policy_sha256"]):
                    raise ValueError("Guarded-flat is not initializing its exact reviewed policy parent")
                for name in ("obs_norm_state_dict", "privileged_obs_norm_state_dict"):
                    if int(boot[name]["prefix.count" if active_support_flat else "count"]) != 24576000:
                        raise ValueError("Guarded-flat inherited normalization count changed")
                # Only this distinct repair trial permits the already measured
                # unsuccessful initializer; it is not a declaration of readiness.
                metadata["guarded_flat_initial_policy_not_ready"] = True
            metadata["initial_policy_sha256"] = hashlib.sha256(initial_policy.read_bytes()).hexdigest()
            if guarded_flat:
                if active_support_flat:
                    if not all(transfer.get(key) is True for key in ("actor_and_critic_loaded_bitwise","dualnorm_loaded_bitwise","same_width_context_transfer")):
                        raise RuntimeError("v11 must transfer the complete same-width model and own norms")
                    metadata["initial_actor_weights_match_parent_exactly"] = True
                    metadata["initial_critic_weights_match_parent_exactly"] = True
                    for path in (ROOT/"source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks").rglob("*"):
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix!=".pyc":
                            if hashlib.sha256(path.read_bytes()).hexdigest()!=parent_source_hashes.get(str(path.relative_to(ROOT))):
                                raise RuntimeError("v11 network must retain the exact parent source")
                    metadata["initial_network_source_matches_parent_exactly"] = True
                elif context_flat:
                    if not all(transfer.get(key) is True for key in (
                        "actor_and_critic_prefix_loaded_bitwise","dualnorm_prefix_loaded_bitwise","new_context_columns_zero")):
                        raise RuntimeError("Initial context transfer did not verify inherited prefixes and zero columns")
                    metadata["initial_actor_prefix_and_zero_columns_verified"]=True
                    metadata["initial_critic_prefix_and_zero_columns_verified"]=True
                    deltas=guarded_plan["reported_source_delta_sha256"]
                    for path in (ROOT/"source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks").rglob("*"):
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix!=".pyc":
                            key=str(path.relative_to(ROOT))
                            if hashlib.sha256(path.read_bytes()).hexdigest()!=deltas.get(key,parent_source_hashes.get(key)):
                                raise RuntimeError("Context network source is not the exact reviewed adapter")
                    metadata["initial_context_source_matches_reviewed_delta"]=True
                else:
                    if any(not torch.equal(value.detach().cpu(), boot["model_state_dict"]["actor."+name].detach().cpu())
                           for name,value in runner.alg.policy.actor.state_dict().items()):
                        raise RuntimeError("Initial guarded-flat actor is not bitwise the reviewed candidate")
                    metadata["initial_actor_weights_match_parent_exactly"] = True
                    metadata["initial_critic_weights_match_parent_exactly"] = all(
                        torch.equal(value.detach().cpu(),boot["model_state_dict"]["critic."+name].detach().cpu())
                        for name,value in runner.alg.policy.critic.state_dict().items())
                    if not metadata["initial_critic_weights_match_parent_exactly"]:
                        raise RuntimeError("Initial guarded-flat critic differs from the reviewed transferred model")
                    for path in (ROOT/"source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks").rglob("*"):
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                            key = str(path.relative_to(ROOT))
                            if hashlib.sha256(path.read_bytes()).hexdigest() != parent_source_hashes.get(key):
                                raise RuntimeError("Guarded-flat initialization network source differs from its parent")
                    metadata["initial_network_source_matches_parent_exactly"] = True
                    if not typed_flat_transfer:
                        metadata["initial_actor_weights_match_ce822_exactly"] = True
                        metadata["initial_network_source_matches_ce822_exactly"] = True
            del boot
        if checkpoint:
            runner.load(str(checkpoint), load_optimizer=True)
            # RSL-RL 2.3.3 stores the last completed zero-based iteration and
            # restarts its loop at that same index unless advanced here.
            if int(runner.current_learning_iteration) + 1 != completed_iterations:
                raise ValueError("Loaded runner iteration disagrees with checkpoint")
            runner.current_learning_iteration = completed_iterations
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            runner.tot_timesteps = completed_iterations * agent_cfg.num_steps_per_env * args.num_envs
        if args.task == "Unitree-H2-Stair-Ascent-Policy":
            metadata["learning_rate_wiring"] = configure_stair_learning_rates(
                runner.alg, metadata["algorithm_settings"]["learning_rate"],
                metadata["algorithm_settings"]["rnd_learning_rate"], resumed=checkpoint is not None)
            metadata["initial_observation_preprocessing"] = runner.initial_observation_preprocessing
        budget_cfg = getattr(agent_cfg, "stair_exploration_budget", None)
        if budget_cfg is not None:
            from stair_exploration import StairExplorationBudget
            ids = action_term._joint_ids
            robot = env.unwrapped.scene["robot"]
            runner.stair_exploration_budget = StairExplorationBudget(
                runner.alg.policy, runner.alg.optimizer,
                robot.data.soft_joint_pos_limits[:, ids],
                initialize=checkpoint is None,
                stiffness=robot.data.joint_stiffness[:, ids],
                effort_limits=robot.data.joint_effort_limits[:, ids], **budget_cfg)
            metadata["stair_exploration_budget"] = runner.stair_exploration_budget.snapshot()
            metadata["variance_initialization_overrides_parent_std"] = initial_policy is not None
            if not typed_flat_transfer:
                metadata["variance_initialization_overrides_BC_std"] = initial_policy is not None
            if guarded_flat:
                metadata["std_reset_to_half_budget"] = torch.equal(runner.alg.policy.std.detach(),
                    runner.stair_exploration_budget.initial)
                if not metadata["std_reset_to_half_budget"]:
                    raise RuntimeError("New guarded-flat optimization did not reset its declared half noise budget")
        if guarded_flat:
            from guarded_flat_ppo import create_fixed_real_bank
            guard_dir = log_dir / "functional_guard"
            fixed_bank = guard_dir / "fixed_real_bank.npz"
            if context_flat:
                fixed_bank=None
                bank_receipt=dict(status="collecting_first_rollout",rows=0,target_rows=4096,sealed=False,
                                  kind="new_run_first_rollout_actual_preaction_v1")
            else:
                audit_dataset = Path(guarded_plan["fixed_real_audit_dataset"]).resolve(strict=True) if typed_flat_transfer else initial_policy.parent/"dataset"
                bank_receipt = create_fixed_real_bank(audit_dataset, fixed_bank, seed=args.seed)
            runner.obs_normalizer.eval();runner.privileged_obs_normalizer.eval()
            attachment={}
            if context_flat:
                case=args.supervision_state.resolve(strict=True).parent
                request_path=case/"request.json"
                request_sha=hashlib.sha256(request_path.read_bytes()).hexdigest()
                plan_sha=hashlib.sha256(args.guarded_flat_plan.read_bytes()).hexdigest()
                def bank_sealed(receipt):
                    # Only this declared acquisition artifact is created while source/parent inputs stay frozen.
                    if hashlib.sha256(request_path.read_bytes()).hexdigest()!=request_sha:
                        raise RuntimeError("Acquisition request identity changed")
                    if receipt.get("status")!="sealed" or receipt.get("rows")!=4096:
                        raise RuntimeError("Acquisition callback requires a complete real4096 bank")
                    for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),("acquisition_log_path","acquisition_log_sha256")):
                        path=Path(receipt[key]).resolve(strict=True)
                        if not path.is_relative_to(guard_dir) or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt[sha_key]:
                            raise RuntimeError("Actual acquisition artifact binding differs: "+key)
                    binding=dict(case=str(case),request_sha256=request_sha,plan_sha256=plan_sha,
                                 fixed_real_bank=receipt,scope="One real first24-control bank, sealed before update0; no acceptance claim")
                    # Share the monitor lock and publish its complete seal last:
                    # a monitor must never observe half of collecting -> sealed.
                    with (case/"monitor.write.lock").open("a") as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        metadata["guarded_flat_learning"]["fixed_real_bank"]=receipt
                        write_json(log_dir/"run.json",metadata)
                        runner.update_state(guarded_flat_learning=metadata["guarded_flat_learning"],
                                            fixed_bank_acquisition_status="sealed",actual_bank_seal=str(case/"actual_bank.sealed.json"))
                        temporary=case/"actual_bank.sealed.pending"
                        with temporary.open("x") as stream:
                            json.dump(binding,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
                            stream.flush();os.fsync(stream.fileno())
                        os.link(temporary,case/"actual_bank.sealed.json")
                        temporary.unlink()
                spec=dict(guarded_plan["audit_bank_source"],wrapper_clip_actions=float(agent_cfg.clip_actions),
                    bindings=dict(case=str(case),request_sha256=request_sha,source_commit=metadata["git_commit"],
                                  source_sha256=source_hashes,training_tool_sha256=metadata["training_tool_sha256"],
                                  plan_path=str(args.guarded_flat_plan.resolve(strict=True)),plan_sha256=plan_sha,
                                  initial_policy_sha256=metadata["initial_policy_sha256"]))
                attachment=dict(acquisition_spec=spec,on_bank_sealed=bank_sealed)
            runner.alg.attach_guard({"actor":runner.obs_normalizer,"critic":runner.privileged_obs_normalizer},
                runner.stair_exploration_budget,fixed_bank,guard_dir,metadata["initial_policy_sha256"],**attachment)
            robot = env.unwrapped.scene["robot"]
            action_ids = action_term._joint_ids
            def capture_actual_raw(observation):
                limits = robot.data.soft_joint_pos_limits[:,action_ids]
                runner.alg.guard.prepare_actual_raw(observation,limits[...,0],limits[...,1],
                    robot.data.joint_stiffness[:,action_ids],robot.data.joint_effort_limits[:,action_ids],
                    int(env.unwrapped.common_step_counter),
                    **(dict(reset_generation=env.unwrapped.initialization_reset_generation,
                            episode_length_buf=env.unwrapped.episode_length_buf) if context_flat else {}))
            env.actual_raw_guard_callback = capture_actual_raw
            metadata["guarded_flat_learning"] = {
                "plan_path":str(args.guarded_flat_plan.resolve(strict=True)),
                "plan_sha256":hashlib.sha256(args.guarded_flat_plan.read_bytes()).hexdigest(),
                "initial_policy_sha256":metadata["initial_policy_sha256"],
                "fixed_real_bank":bank_receipt,"current_rollout_real_rows":1024,"total_guard_rows":5120,
                "normalizers_frozen":True,
                **({"normalizer_prefix_counts":{name:int(norm.prefix_count) for name,norm in context_normalizers.items()},
                    "normalization_abi":normalization_metadata(context_normalizers)} if context_flat else
                   {"normalizer_counts":{"actor":int(runner.obs_normalizer.count),"critic":int(runner.privileged_obs_normalizer.count)}}),
                "optimizer_format":runner.alg.checkpoint_format,"trial_protocol":runner.alg.trial_protocol,
                "state":runner.alg.learning_state(),
                "action_margin":runner.alg.learning_state()["action_margin"],
                "proposal_projection":runner.alg.learning_state().get("proposal_projection"),
                "snapshot_policy":"RAM before every whole actor update; first persisted once; veto permanently preserves baseline/candidate/restored",
                "physical_acceptance_claimed":False,
            }
            metadata["initial_observation_preprocessing"] = runner.initial_observation_preprocessing
        metadata["initial_learning_state"] = {
            "checkpoint_loaded": checkpoint is not None,
            "behavior_policy_initialized": initial_policy is not None,
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
        if context_flat:
            metadata["initial_learning_state"].pop("normalizer_counts",None)
            metadata["initial_learning_state"].update(
                normalizer_prefix_counts={name:int(norm.prefix_count) for name,norm in context_normalizers.items()},
                normalization_abi=normalization_metadata(context_normalizers))
        if guarded_flat:
            metadata["initialization_mode"] = (guarded_plan["initialization_mode"]
                if typed_flat_transfer else "guarded_flat_reward_repair_fresh_split_PPO")
            metadata["initial_learning_state"].update(
                actor_optimizer_state_entries=len(runner.alg.actor_optimizer.state),
                critic_optimizer_state_entries=len(runner.alg.critic_optimizer.state),
                imitation_optimizer_loaded=False,guarded_ppo_state=runner.alg.learning_state())
        if checkpoint is None:
            if (metadata["initial_learning_state"]["iteration"] != 0
                    or metadata["initial_learning_state"]["policy_optimizer_state_entries"] != 0
                    or metadata["initial_learning_state"]["rnd_optimizer_state_entries"] not in (None, 0)):
                raise RuntimeError("Fresh training unexpectedly contains previous optimizer state")
            if guarded_flat and len(runner.alg.critic_optimizer.state) != 0:
                raise RuntimeError("New guarded-flat critic optimizer was not fresh")
        metadata["status"] = "training"
        (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"[H2] Learning iterations: {completed_iterations} -> "
              f"{metadata['target_total_iterations']} "
              f"({agent_cfg.max_iterations} additional)", flush=True)
        runner.update_state(status="training", observation_shapes=metadata["observation_shapes"],
                            num_actions=env.num_actions, policy_shape=metadata["policy_shape"],
                            training_started_at_epoch=time.time(),
                            initialization_mode=metadata["initialization_mode"],
                            **({"learning_rate_wiring": metadata["learning_rate_wiring"],
                                "initial_observation_preprocessing": metadata["initial_observation_preprocessing"]}
                               if args.task == "Unitree-H2-Stair-Ascent-Policy" else {}),
                            action_contract=metadata["action_contract"],
                            locked_action_ids=env.unwrapped.action_manager._locked_action_ids)
        if guarded_flat:
            runner.update_state(guarded_flat_learning=metadata["guarded_flat_learning"],
                runtime_flat_geometry=metadata["runtime_flat_geometry"],
                initial_observation_preprocessing=metadata["initial_observation_preprocessing"])
        try:
            functional_stop_type = ()
            if guarded_flat:
                from guarded_flat_ppo import FunctionalUpdateRejected
                functional_stop_type = FunctionalUpdateRejected
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        except functional_stop_type as rejected:
            metadata.update(status="stopped_functional_guard",completed_iterations=runner.alg.completed_updates,
                total_timesteps=int(runner.tot_timesteps),final_checkpoint=rejected.restored_checkpoint,
                actual_sampled_environment_steps=int(runner.tot_timesteps)+rejected.discarded_environment_steps,
                discarded_uncommitted_rollout_steps=rejected.discarded_environment_steps,
                functional_guard_evidence=rejected.evidence,guarded_ppo_state=runner.alg.learning_state())
            write_json(log_dir/"run.json",metadata)
            runner.update_state(status="stopped_functional_guard",completed_iterations=runner.alg.completed_updates,
                iteration=runner.alg.completed_updates-1,latest_checkpoint=rejected.restored_checkpoint,
                final_checkpoint=rejected.restored_checkpoint,functional_guard_evidence=rejected.evidence,
                actual_sampled_environment_steps=metadata["actual_sampled_environment_steps"],
                discarded_uncommitted_rollout_steps=rejected.discarded_environment_steps,
                guarded_ppo_state=runner.alg.learning_state(),no_automatic_retry=True)
            print("[H2] Whole PPO update restored and stopped by the functional guard",flush=True)
            return
        except NoImprovementStop:
            metadata.update(status="stopped_no_improvement",
                            completed_iterations=int(runner.current_learning_iteration) + 1,
                            final_checkpoint=runner.supervision["latest_checkpoint"],
                            analysis=runner.supervision["analysis"])
            write_json(log_dir / "run.json", metadata)
            print("[H2] Stopped because physical movement metrics stagnated", flush=True)
            return
        except BaseException as error:
            if context_flat:
                acquisition=runner.alg.guard.acquisition_state()
                before_optimization=(runner.alg.completed_updates==0 and acquisition.get("status")!="sealed")
                metadata.update(status="failed",completed_iterations=runner.alg.completed_updates,
                    error_type=type(error).__name__,error=str(error),
                    failure_scope="preoptimization_bank_acquisition" if before_optimization else "runtime_error",
                    fixed_bank_acquisition=acquisition,
                    fully_processed_rollout_transitions=int(runner.alg.storage.step),
                    partial_physical_step_count_unknown=True,
                    guarded_ppo_state=runner.alg.learning_state())
                write_json(log_dir/"run.json",metadata)
                runner.update_state(status="failed",error_type=type(error).__name__,error=str(error),
                    failure_scope=metadata["failure_scope"],fixed_bank_acquisition=acquisition,
                    completed_iterations=runner.alg.completed_updates,iteration=runner.alg.completed_updates-1)
            raise
        metadata["status"] = "completed"
        metadata["completed_iterations"] = int(runner.current_learning_iteration) + 1
        metadata["total_timesteps"] = int(runner.tot_timesteps)
        if guarded_flat:
            metadata["actual_sampled_environment_steps"] = int(runner.tot_timesteps)
            metadata["discarded_uncommitted_rollout_steps"] = 0
            metadata["guarded_ppo_state"] = runner.alg.learning_state()
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
