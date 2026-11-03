"""One fresh, bounded stair run initialized from the independently accepted flat policy."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--supervision_state", type=Path, required=True)
parser.add_argument("--stop_policy", type=Path, required=True)
parser.add_argument("--expected_git_commit", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.io import dump_pickle, dump_yaml
from training_supervision import (NoImprovementStop, ProbedVecEnv, SupervisedRunner, write_json,
                                  configure_stair_learning_rates)
from guarded_stair_transfer import (TASK, PROTOCOL, FORMAT, BUDGET, parent_admission, apply_stair_transfer)
from guarded_flat_ppo import FunctionalUpdateRejected
from task_context_normalization import make_task_context_normalizers, normalization_metadata
from stair_exploration import StairExplorationBudget
from stair_initialization_binding import (initialization_config_errors, exploration_snapshot_errors,
    temporal_credit_config_snapshot, temporal_credit_config_errors, temporal_credit_snapshot_errors,
    ROLLOUT_STEPS_BY_PROTOCOL, rollout_credit_collection_errors, rollout_credit_budget_errors,
    rollout_credit_seal_errors, clearance_progress_config_errors, clearance_progress_snapshot_errors)

sys.path.insert(0, str(ROOT/"source/CtrlLab/CtrlLab/tasks/manager_based"))
import h2_task
from h2_task.networks import H2ActorCriticOur
import rsl_rl.runners.on_policy_runner as rsl_runner_module
rsl_runner_module.H2ActorCriticOur = H2ActorCriticOur


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.expected_git_commit:
        raise RuntimeError("Source commit changed after admission")
    plan = json.loads(args.plan.read_text())
    policy = json.loads(args.stop_policy.read_text())
    state = json.loads(args.supervision_state.read_text())
    provenance = parent_admission(plan, ROOT)
    protocol = plan["trial_protocol"]
    rollout_steps = ROLLOUT_STEPS_BY_PROTOCOL[protocol]
    spec = gym.spec(TASK)
    cfg = load_config(spec.kwargs["env_cfg_entry_point"])
    agent = load_config(spec.kwargs["rsl_rl_cfg_entry_point"])
    cfg.stair_clearance_progress_potential = True
    cfg.scene.num_envs = 1024
    cfg.seed = agent.seed = 42
    cfg.sim.device = agent.device = args.device or "cuda:0"
    agent.resume = False
    if (agent.max_iterations != BUDGET or agent.save_interval != 500 or agent.num_steps_per_env != rollout_steps
            or agent.algorithm.class_name != "H2GuardedStairPPO" or agent.algorithm.trial_protocol != PROTOCOL):
        raise RuntimeError("Stair configuration differs from the reviewed budget/type")
    request_path = args.supervision_state.resolve().parent/"request.json"
    request = json.loads(request_path.read_text())
    initialization = plan.get("exploration_initialization")
    config_errors = initialization_config_errors(initialization, agent.stair_exploration_budget)
    temporal_credit = plan.get("temporal_credit")
    rollout_collection = plan.get("rollout_collection")
    runner_config = agent.to_dict()
    resolved_control_dt_s = cfg.sim.dt * cfg.decimation
    temporal_errors = temporal_credit_config_errors(
        temporal_credit, runner_config, protocol=protocol, control_dt_s=resolved_control_dt_s)
    if (config_errors or plan.get("trial_protocol") != PROTOCOL or plan.get("checkpoint_format") != FORMAT
            or request.get("exploration_initialization") != initialization
            or request.get("plan_sha256") != sha(args.plan) or request.get("trial_protocol") != PROTOCOL
            or request.get("checkpoint_format") != FORMAT):
        raise RuntimeError("Fresh exploration plan/request/resolved config binding failed: " + "; ".join(config_errors))
    if temporal_errors or request.get("temporal_credit") != temporal_credit:
        raise RuntimeError("Temporal credit plan/request/resolved config binding failed: " + "; ".join(temporal_errors))
    if (rollout_credit_collection_errors(rollout_collection, protocol=protocol)
            or rollout_credit_collection_errors(request.get("rollout_collection"), protocol=request.get("trial_protocol"))
            or request.get("rollout_collection") != rollout_collection
            or rollout_credit_budget_errors(request, protocol=request.get("trial_protocol"))):
        raise RuntimeError("Rollout credit plan/request48-current/24-fixed binding failed")
    clearance_progress = plan.get("clearance_progress_potential")
    geometry = cfg.scene.terrain.terrain_generator.sub_terrains["ascending_stairway"]
    clearance_config = dict(enabled=cfg.stair_clearance_progress_potential, gamma=float(cfg.stair_shaping_gamma),
        outer_potential_weight=float(cfg.rewards.potential.weight), strict_episode_force=cfg.stair_strict_episode_force,
        approach_m=float(geometry.approach_distance), depth_m=float(geometry.tread_depth),
        real_riser_count=int(geometry.num_steps))
    if (clearance_progress_config_errors(clearance_progress, clearance_config, protocol=protocol)
            or request.get("clearance_progress_potential") != clearance_progress):
        raise RuntimeError("Clearance progress plan/request/resolved reward configuration disagrees")
    log_dir = ROOT/"logs/rsl_rl/h2_guarded_stair_transfer"/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_"+args.supervision_state.parent.name[-8:])
    log_dir.mkdir(parents=True, exist_ok=False)
    cfg.io_descriptors_output_dir = str(log_dir)
    snapshot = log_dir/"training_source"
    snapshot.mkdir()
    shutil.copytree(ROOT/"source", snapshot/"source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tool_names = ["train_stair_transfer.py", "guarded_stair_transfer.py", "guarded_flat_ppo.py",
                  "functional_proposal.py", "selected_row_block.py", "stair_exploration.py", "stair_initialization_binding.py", "training_supervision.py",
                  "task_context_normalization.py", "task_context_metadata.py", "v11_parent_custody.py",
                  "evaluate_stair_policy.py", "evaluate_stair_policy.sh", "validate_stair_transfer.py", "launch_stair_transfer.py",
                  "launch_stair_learning.py", "launch_stair_evaluation.py", "launch_residual_learning.py"]
    for name in tool_names:
        shutil.copy2(ROOT/"tools"/name, snapshot/name)
    shutil.copy2(args.plan, snapshot/"stair_transfer_plan.json")
    shutil.copy2(args.stop_policy, snapshot/"stop_policy.json")
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"source").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    write_json(snapshot/"main_source_sha256.json", source_hashes)
    dump_yaml(str(log_dir/"params/env.yaml"), cfg)
    dump_yaml(str(log_dir/"params/agent.yaml"), agent)
    dump_pickle(str(log_dir/"params/env.pkl"), cfg)
    dump_pickle(str(log_dir/"params/agent.pkl"), agent)
    dependencies = [Path(cfg.scene.robot.spawn.usd_path), Path(cfg.observations.policy.height_scanner.params["checkpoint"])]
    metadata = dict(task=TASK, trial_protocol=PROTOCOL, checkpoint_format=FORMAT, asset=str(cfg.scene.robot.spawn.usd_path),
        case=str(args.supervision_state.resolve().parent),
        git_commit=commit, source_branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        source_snapshot=str(snapshot/"source"), source_sha256=source_hashes,
        training_tool_sha256={name:sha(ROOT/"tools"/name) for name in tool_names},
        external_dependency_sha256={str(p.resolve(strict=True)):sha(p) for p in dependencies},
        num_envs=1024, num_steps_per_env=rollout_steps, max_iterations=BUDGET, target_total_iterations=BUDGET,
        expected_total_timesteps=BUDGET*1024*rollout_steps,
        starting_iteration=0, save_interval=500, seed=42, device=agent.device, checkpoint=None,
        initialization_mode="accepted_flat_same_width_model_and_own_norm_only_fresh_stair_PPO",
        initial_policy=plan["initial_policy"], initial_policy_sha256=plan["initial_policy_sha256"],
        plan_path=str(args.plan.resolve()), plan_sha256=sha(args.plan), stop_policy=policy,
        exploration_initialization=initialization,
        runtime_exploration_config=dict(agent.stair_exploration_budget),
        temporal_credit=temporal_credit,
        rollout_collection=rollout_collection,
        clearance_progress_potential=clearance_progress, runtime_clearance_progress_config=clearance_config,
        runtime_temporal_credit_config=temporal_credit_config_snapshot(
            runner_config, control_dt_s=resolved_control_dt_s),
        inherited_numeric_execution_profile_scope="numeric_executor_only",
        ppo_temporal_credit_override=dict(parameter="gae_lambda", value=agent.algorithm.lam,
            scope="stair_transfer_only", exact_binding="temporal_credit"),
        early_progress_policy=plan["early_progress_policy"],
        early_progress_policy_sha256=plan["early_progress_policy_sha256"],
        policy_dimensions=[1034,871,73,12], inherited_numeric_execution_profile="explicit_task_context_longer_budget_v1",
        status="initializing", physical_acceptance_passed=False, stairs_acceptance_passed=False)
    from selected_row_block import selected_row_block_contract, selected_row_backend_metadata
    metadata["selected_row_evaluation"] = selected_row_block_contract()
    metadata["selected_row_arithmetic_context"] = selected_row_backend_metadata()
    write_json(log_dir/"run.json", metadata)
    state.update(status="initializing", run_dir=str(log_dir), metadata=str(log_dir/"run.json"), source_hashes=source_hashes)
    write_json(args.supervision_state, state)
    print("[H2] Stair log directory: "+str(log_dir), flush=True)
    env = gym.make(TASK, cfg=cfg)
    runner = None
    try:
        env = ProbedVecEnv(env, clip_actions=agent.clip_actions)
        raw = env.unwrapped
        temporal_errors = temporal_credit_config_errors(temporal_credit, runner_config, protocol=protocol,
                                                       control_dt_s=raw.step_dt)
        if temporal_errors:
            raise RuntimeError("Actual environment temporal configuration differs before policy construction: " + "; ".join(temporal_errors))
        obs, extras = env.get_observations()
        shapes = {k:list(v.shape) for k,v in extras["observations"].items()}
        if (obs.shape != (1024,1034) or tuple(shapes["critic"]) != (1024,871)
                or tuple(shapes["rnd_state"]) != (1024,73) or env.num_actions != 12):
            raise RuntimeError("Actual same-width stair ABI differs")
        action = raw.action_manager.get_term("JointPositionAction")
        command = raw.command_manager.get_term("base_velocity")
        if type(command).__name__ != "WorldStairVelocityCommand":
            raise RuntimeError("Target-height training requires its fixed world stair command")
        metadata["world_stair_command"] = command.runtime_contract()
        from h2_task.stair_clearance_rewards import full_footprint_clearance_contract
        metadata["stair_clearance_target"] = full_footprint_clearance_contract(
            approach=raw.stair_progress.approach, depth=raw.stair_progress.depth,
            steps=raw.stair_progress.steps)
        if (type(action).__name__ != "DirectLegPositionAction" or action.gait_reference is not None
                or len(raw.action_manager.held_joint_ids) != 17 or not raw.stair_progress.strict_episode_force):
            raise RuntimeError("Stair action, upper hold or force contract changed")
        metadata.update(observation_shapes=shapes, policy_shape=list(obs.shape), num_actions=12,
            runtime_stair_geometry=dict(terrain_rows=raw.scene.terrain.cfg.terrain_generator.num_rows,
                riser_min_m=raw._stair_minimum_height, riser_max_m=raw._stair_maximum_height,
                current_riser_min_m=float(raw._stair_heights().min()), current_riser_max_m=float(raw._stair_heights().max()),
                levels=raw.stair_progress.steps, depth_m=raw.stair_progress.depth, width_m=raw.stair_progress.width,
                approach_m=raw.stair_progress.approach, episode_length_s=float(cfg.episode_length_s),
                domain_bounds_local=list(raw._stair_domain_bounds_local), scripted_gait_installed=False,
                termination_terms=list(raw.termination_manager.active_terms)),
            task_context=raw.stair_task_context_contract,
            action_contract=dict(class_name=type(action).__name__, joint_names=list(action._joint_names),
                articulation_joint_ids=list(action._joint_ids), held_upper_body_joint_ids=list(raw.action_manager.held_joint_ids)),
            reward_contract={name:dict(weight=float(raw.reward_manager.get_term_cfg(name).weight),
                function=raw.reward_manager.get_term_cfg(name).func.__name__)
                for name in raw.reward_manager.active_terms})
        runner = SupervisedRunner(env, runner_config, log_dir=str(log_dir), device=agent.device,
            supervision_path=args.supervision_state, policy=policy, effect_task_name=TASK,
            normalize_initial_observations=True, guarded_stair=True)
        metadata["runtime_temporal_credit"] = runner.alg.bind_temporal_credit(runner, raw, temporal_credit)
        metadata["runtime_clearance_progress"] = runner.alg.bind_clearance_progress(raw, clearance_progress)
        metadata["runtime_rollout_collection"] = runner.alg.rollout_collection_snapshot()
        metadata["ppo_temporal_credit_override"]["value"] = metadata["runtime_temporal_credit"]["gae_lambda"]
        metadata["learning_rate_wiring"] = configure_stair_learning_rates(
            runner.alg,agent.algorithm.actor_learning_rate,agent.algorithm.rnd_cfg.learning_rate,resumed=False)
        normalizers = make_task_context_normalizers(runner.device)
        runner.obs_normalizer = normalizers["actor"]
        runner.privileged_obs_normalizer = normalizers["critic"]
        # The initialization scope has exactly one parent load, after all input admission.
        saved = torch.load(plan["initial_policy"], map_location="cpu", weights_only=True)
        metadata["initial_policy_transfer"] = apply_stair_transfer(runner.alg, normalizers, saved, provenance)
        del saved
        robot = raw.scene["robot"]
        ids = action._joint_ids
        runner.stair_exploration_budget = StairExplorationBudget(runner.alg.policy, runner.alg.actor_optimizer,
            robot.data.soft_joint_pos_limits[:,ids], initialize=True,
            stiffness=robot.data.joint_stiffness[:,ids], effort_limits=robot.data.joint_effort_limits[:,ids],
            **agent.stair_exploration_budget)
        if not torch.equal(runner.alg.policy.std, runner.stair_exploration_budget.initial):
            raise RuntimeError("Fresh stair exploration was not reset to its reviewed configured initial fraction")
        metadata["stair_exploration_budget"] = runner.stair_exploration_budget.snapshot()
        metadata["initial_learning_state"] = dict(iteration=int(runner.current_learning_iteration),
            actor_optimizer_state_entries=len(runner.alg.actor_optimizer.state),
            critic_optimizer_state_entries=len(runner.alg.critic_optimizer.state),
            rnd_optimizer_state_entries=len(runner.alg.rnd_optimizer.state), parent_packet_loads=1,
            parent_optimizer_or_RNG_loaded=False, normalization_abi=normalization_metadata(normalizers),
            policy_std=runner.alg.policy.std.detach().cpu().tolist(),
            temporal_credit=runner.alg.temporal_credit_snapshot(),
            rollout_collection=runner.alg.rollout_collection_snapshot(),
            clearance_progress_potential=runner.alg.clearance_progress_snapshot())
        initialization_errors = exploration_snapshot_errors(initialization, metadata["stair_exploration_budget"],
            initial_learning_state=metadata["initial_learning_state"], at_initialization=True)
        if initialization_errors:
            raise RuntimeError("Actual fresh exploration initialization binding failed: " + "; ".join(initialization_errors))
        temporal_errors = temporal_credit_snapshot_errors(temporal_credit, metadata["runtime_temporal_credit"],
                                                         protocol=protocol)
        if temporal_errors:
            raise RuntimeError("Actual temporal credit initialization binding failed: " + "; ".join(temporal_errors))
        case = args.supervision_state.resolve().parent
        request_path = case/"request.json"
        request_sha, plan_sha = sha(request_path), sha(args.plan)
        guard_dir = log_dir/"functional_guard"
        bank_receipt = dict(status="collecting_first_rollout", rows=0, target_rows=4096, sealed=False,
                            kind="new_run_first_rollout_actual_preaction_v1")
        metadata["guarded_stair_learning"] = dict(optimizer_format=FORMAT, trial_protocol=PROTOCOL,
            fixed_real_bank=bank_receipt, current_rollout_real_rows=1024, total_guard_rows=5120,
            rollout_collection=runner.alg.rollout_collection_snapshot(),
            clearance_progress_potential=runner.alg.clearance_progress_snapshot(),
            normalization_abi=normalization_metadata(normalizers), normalizers_frozen=True)
        def bank_sealed(receipt):
            if (sha(request_path) != request_sha or sha(args.plan) != plan_sha
                    or receipt["status"] != "sealed" or receipt["rows"] != 4096
                    or receipt.get("bindings") != acquisition["bindings"]
                    or runner.alg.storage.step != receipt.get("seal_storage_step")
                    or int(runner.alg.rnd.update_counter) != receipt.get("rnd_rollout_update_counter")
                    or runner.alg.completed_updates != 0
                    or runner.alg.rollout_collection_snapshot() != rollout_collection
                    or rollout_credit_seal_errors(receipt, protocol=runner.alg.stair_trial_protocol)):
                raise RuntimeError("First-rollout acquisition binding changed")
            for key, skey in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),("acquisition_log_path","acquisition_log_sha256")):
                p = Path(receipt[key]).resolve(strict=True)
                if not p.is_relative_to(guard_dir) or sha(p) != receipt[skey]:
                    raise RuntimeError("Actual acquisition artifact binding changed")
            with (case/"monitor.write.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                metadata["guarded_stair_learning"]["fixed_real_bank"] = receipt
                write_json(log_dir/"run.json", metadata)
                runner.update_state(guarded_stair_learning=metadata["guarded_stair_learning"], fixed_bank_acquisition_status="sealed")
                with (case/"actual_bank.sealed.json").open("x") as stream:
                    json.dump(dict(case=str(case),request_sha256=request_sha,plan_sha256=plan_sha,
                                   trial_protocol=PROTOCOL,checkpoint_format=FORMAT,
                                   temporal_credit=runner.alg.temporal_credit_snapshot(),
                                   rollout_collection=runner.alg.rollout_collection_snapshot(),
                                   clearance_progress_potential=runner.alg.clearance_progress_snapshot(),
                                   actual_seal_storage_step=runner.alg.storage.step,
                                   actual_seal_RND_counter=int(runner.alg.rnd.update_counter),
                                   actual_current_rows_at_seal=receipt["current_rows_at_seal"],
                                   fixed_real_bank=receipt),stream,indent=2,allow_nan=False)
                    stream.flush(); os.fsync(stream.fileno())
        acquisition = dict(plan["audit_bank_source"], wrapper_clip_actions=float(agent.clip_actions),
            trial_protocol=PROTOCOL,temporal_credit=temporal_credit,rollout_collection=rollout_collection,
            clearance_progress_potential=clearance_progress,
            bindings=dict(case=str(case),request_sha256=request_sha,source_commit=commit,
                source_sha256=source_hashes,training_tool_sha256=metadata["training_tool_sha256"],
                trial_protocol=PROTOCOL,temporal_credit=temporal_credit,rollout_collection=rollout_collection,
                clearance_progress_potential=clearance_progress,
                plan_path=str(args.plan.resolve()),plan_sha256=plan_sha,initial_policy_sha256=plan["initial_policy_sha256"]))
        runner.alg.attach_guard(normalizers, runner.stair_exploration_budget, None, guard_dir,
            plan["initial_policy_sha256"], acquisition_spec=acquisition, on_bank_sealed=bank_sealed)
        metadata["runtime_rollout_collection"] = runner.alg.rollout_collection_snapshot()
        def capture(observation):
            limits = robot.data.soft_joint_pos_limits[:,ids]
            runner.alg.guard.prepare_actual_raw(observation, limits[...,0], limits[...,1],
                robot.data.joint_stiffness[:,ids], robot.data.joint_effort_limits[:,ids], int(raw.common_step_counter),
                reset_generation=raw.initialization_reset_generation, episode_length_buf=raw.episode_length_buf)
        env.actual_raw_guard_callback = capture
        metadata["initial_observation_preprocessing"] = runner.initial_observation_preprocessing
        metadata["status"] = "training"
        write_json(log_dir/"run.json", metadata)
        runner.update_state(status="training", training_started_at_epoch=time.time(),
            observation_shapes=shapes, num_actions=12, policy_shape=list(obs.shape),
            initial_policy_transfer=metadata["initial_policy_transfer"],
            guarded_stair_learning=metadata["guarded_stair_learning"], runtime_stair_geometry=metadata["runtime_stair_geometry"],
            temporal_credit=metadata["runtime_temporal_credit"],
            rollout_collection=metadata["runtime_rollout_collection"],
            clearance_progress_potential=runner.alg.clearance_progress_snapshot())
        try:
            runner.learn(num_learning_iterations=BUDGET, init_at_random_ep_len=True)
        except FunctionalUpdateRejected as error:
            metadata.update(status="stopped_functional_guard", completed_iterations=runner.alg.completed_updates,
                final_checkpoint=error.restored_checkpoint, functional_guard_evidence=error.evidence,
                total_timesteps=int(runner.tot_timesteps), actual_sampled_environment_steps=int(runner.tot_timesteps)+error.discarded_environment_steps,
                discarded_uncommitted_rollout_steps=error.discarded_environment_steps)
        except NoImprovementStop:
            metadata.update(status="stopped_no_improvement",completed_iterations=runner.alg.completed_updates,
                final_checkpoint=runner.supervision["latest_checkpoint"], analysis=runner.supervision["analysis"])
        else:
            metadata.update(status="completed",completed_iterations=runner.alg.completed_updates,
                total_timesteps=int(runner.tot_timesteps),actual_sampled_environment_steps=int(runner.tot_timesteps),
                discarded_uncommitted_rollout_steps=0,final_checkpoint=str(log_dir/f"model_{BUDGET-1}.pt"))
        metadata["guarded_ppo_state"] = runner.alg.learning_state()
        write_json(log_dir/"run.json", metadata)
        runner.update_state(status=metadata["status"], completed_iterations=runner.alg.completed_updates,
            iteration=runner.alg.completed_updates-1, final_checkpoint=metadata["final_checkpoint"],
            guarded_ppo_state=metadata["guarded_ppo_state"], automatic_restart_allowed=False)
    except BaseException as error:
        metadata.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        if runner is not None:
            try:
                metadata["guarded_ppo_state"] = runner.alg.learning_state()
            except BaseException as state_error:
                metadata["guarded_ppo_state_error"] = str(state_error)
        write_json(log_dir/"run.json",metadata)
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.flush(); runner.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        state = json.loads(args.supervision_state.read_text())
        state.update(status="failed",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_json(args.supervision_state,state)
        raise
    finally:
        app.close()
