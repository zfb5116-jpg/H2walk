"""Frozen, policy-owned H2 stair acceptance in Isaac Sim.

The checkpoint's preserved training source supplies the environment and policy.
The existing StairAcceptance/FootfallTracker judge first episodes independently
of the training reward/progress counters. This program never trains a model or
installs a walking reference. Importing it does not start Isaac Sim.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]
TASK = "Unitree-H2-Stair-Ascent-Policy"
GEOMETRY = dict(height=0.15, depth=0.35, steps=6, width=2.4, approach=1.4)
DURATION = 90.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(source):
    return {"source/" + str(path.relative_to(source)): sha256(path)
            for path in sorted(source.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def class_ast_hash(path, name):
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.ClassDef) and item.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def prepare_source(checkpoint, output, requested_source=None):
    """Copy and bind a training snapshot, never silently fall back to current ROOT."""
    run_dir = checkpoint.parent
    run_file = run_dir / "run.json"
    run = json.loads(run_file.read_text())
    if run.get("task") not in (TASK, TASK + "-Play"):
        raise ValueError("Checkpoint run is not the policy-owned stair task")
    source = Path(run.get("source_snapshot", run_dir / "training_source/source")).resolve(strict=True)
    if requested_source is not None and requested_source.resolve(strict=True) != source:
        raise ValueError("--source_root must be the checkpoint run's recorded training snapshot")
    training = source.parent
    expected = json.loads((training / "main_source_sha256.json").read_text())
    actual = source_hashes(source)
    if expected != actual:
        raise RuntimeError("Training source no longer matches main_source_sha256.json")
    package = Path("CtrlLab/CtrlLab/tasks/manager_based/h2_task")
    independent = ROOT / "source" / package
    if sha256(source / package / "stair_acceptance.py") != sha256(independent / "stair_acceptance.py"):
        raise RuntimeError("The preserved independent StairAcceptance definition changed")
    if class_ast_hash(source / package / "locomotion_progress.py", "FootfallTracker") != class_ast_hash(
            independent / "locomotion_progress.py", "FootfallTracker"):
        raise RuntimeError("The original 80–600ms FootfallTracker definition changed")
    shutil.copytree(training, output / "training_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = output / "training_source/source"
    if source_hashes(frozen) != actual:
        raise RuntimeError("Copied training source is incomplete")
    shutil.copy2(run_file, output / "checkpoint_run.json")
    for name in ("agent.yaml", "env.yaml"):
        shutil.copy2(run_dir / "params" / name, output / ("training_" + name))
    (output / "tools_snapshot").mkdir()
    for name in ("evaluate_stair_policy.py", "evaluate_stair_policy.sh"):
        shutil.copy2(Path(__file__).parent / name, output / "tools_snapshot" / name)
    write_json(output / "source_sha256.json", actual)
    return run, frozen, actual


def load_config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def yaml_scalar(document, key_path):
    """Read a saved config scalar without constructing its Python-tagged objects."""
    import yaml
    node = yaml.compose(document, Loader=yaml.SafeLoader)
    for key in key_path:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Saved environment has no scalar at {key_path}")
        entries = {item.value: value for item, value in node.value}
        node = entries[key]
    if not isinstance(node, yaml.ScalarNode):
        raise ValueError(f"Saved environment config value is not scalar: {key_path}")
    return node.value


def finite_checkpoint(torch, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stair_transfer = saved.get("format") == "h2_guarded_stair_ppo_v1"
    required = {"iter", "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
                "privileged_obs_norm_state_dict", "rnd_state_dict", "rnd_optimizer_state_dict"}
    if stair_transfer:
        required.remove("optimizer_state_dict")
        required.update(("actor_optimizer_state_dict", "critic_optimizer_state_dict", "normalization_abi"))
        receipt_path = checkpoint.with_suffix(".validation.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "passed" or receipt.get("scope") != "natural_final"
                or receipt.get("checkpoint_sha256") != sha256(checkpoint)
                or saved.get("trial_protocol") != "accepted_flat_context_stair_transfer_v1"
                or saved.get("iter") != 4999 or saved["guarded_ppo_state"]["completed_updates"] != 5000):
            raise ValueError("Typed stair replay requires its new complete natural-final CPU receipt")
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Require a complete PPO checkpoint, including both normalizers and RND state")
    if int(saved["iter"]) < 0:
        raise ValueError("Invalid checkpoint iteration")
    for key in ("obs_norm_state_dict", "privileged_obs_norm_state_dict"):
        norm = saved[key]
        count_key, std_key = ("prefix.count", "prefix._std") if stair_transfer else ("count", "_std")
        if int(norm[count_key]) <= 0 or not bool((norm[std_key] >= 0).all()):
            raise ValueError("Checkpoint normalization was not populated or contains an invalid scale")
    tensors = 0

    def visit(value):
        nonlocal tensors
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Checkpoint contains non-finite tensors")
            tensors += 1
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(saved)
    return saved, tensors


def states_equal(torch, module, saved):
    state = module.state_dict()
    return state.keys() == saved.keys() and all(torch.equal(state[key].cpu(), saved[key]) for key in state)


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, help="Must equal this checkpoint's recorded training snapshot")
    parser.add_argument("--task", choices=(TASK, TASK + "-Play"), default=TASK + "-Play")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=True,
                        help="env0 video is always saved for this formal evaluator")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1024:
        parser.error("Formal stair acceptance requires at least 1024 environments")
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.output.resolve()
    args.enable_cameras = True
    return args, AppLauncher


def evaluate(args, app, metadata, source, expected_hashes):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import yaml
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    # assets/robot.py derives its default paths from __file__; preserved source
    # therefore needs the original external asset path, never a rewritten USD.
    run = json.loads((args.output / "checkpoint_run.json").read_text())
    asset = Path(run["asset"])
    if not asset.is_absolute():
        asset = ROOT / asset
    asset = asset.resolve(strict=True)
    os.environ["CTRLLAB_H2_USD_PATH"] = str(asset)
    recon_path = yaml_scalar((args.output / "training_env.yaml").read_text(),
                            ("observations", "policy", "height_scanner", "params", "checkpoint"))
    recon = Path(recon_path)
    recon = (ROOT / recon).resolve(strict=True) if not recon.is_absolute() else recon.resolve(strict=True)
    os.environ["CTRLLAB_H2_LIDAR_RECON_CHECKPOINT"] = str(recon)
    training_dependencies = run.get("external_dependency_sha256")
    if training_dependencies is not None:
        if not {str(asset), str(recon)}.issubset(training_dependencies):
            raise RuntimeError("Training dependency manifest omits its USD or LiDAR checkpoint")
        for path, expected in training_dependencies.items():
            if sha256(Path(path).resolve(strict=True)) != expected:
                raise RuntimeError(f"External dependency no longer matches its training fingerprint: {path}")
    metadata["training_external_dependency_hash_bound"] = training_dependencies is not None
    metadata["training_external_dependency_sha256"] = training_dependencies
    metadata["external_dependency_hash_scope"] = "training manifest files; USD referenced layers/meshes only covered if listed"
    sys.path.insert(0, str(source / "CtrlLab/CtrlLab/tasks/manager_based"))
    import h2_task
    from h2_task.assets import H2_LEG_JOINT_NAMES, H2_JOINT_NAMES
    from h2_task.direct_leg_action import DirectLegPositionAction, soft_range_targets
    from h2_task.locomotion_progress import FootfallTracker
    from h2_task.networks import H2ActorCriticOur
    from h2_task.stair_acceptance import StairAcceptance
    from h2_task.stair_ascent_terrain import ascent_stair_generator

    task_path = Path(h2_task.__file__).resolve()
    if not task_path.is_relative_to(source):
        raise RuntimeError("Imported h2_task is not the frozen training source")
    saved, tensor_count = finite_checkpoint(torch, args.checkpoint)
    agent = yaml.safe_load((args.output / "training_agent.yaml").read_text())
    if not agent.get("empirical_normalization") or agent["policy"]["class_name"] != "H2ActorCriticOur":
        raise ValueError("Require the unchanged H2 actor with saved empirical dual normalization")
    if saved.get("action_semantics", "full_soft_tanh_position_v1") != "full_soft_tanh_position_v1":
        raise ValueError("Checkpoint action semantics disagree with direct position control")
    spec = gym.spec(args.task)
    cfg = load_config(spec.kwargs["env_cfg_entry_point"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.episode_length_s = DURATION
    cfg.stair_minimum_height = cfg.stair_maximum_height = GEOMETRY["height"]
    cfg.stair_height_levels = 2
    cfg.stair_tread_depth, cfg.stair_steps = GEOMETRY["depth"], GEOMETRY["steps"]
    cfg.stair_width, cfg.stair_approach_distance = GEOMETRY["width"], GEOMETRY["approach"]
    cfg.scene.terrain.terrain_generator = ascent_stair_generator(
        levels=2, minimum=GEOMETRY["height"], maximum=GEOMETRY["height"],
        depth=GEOMETRY["depth"], steps=GEOMETRY["steps"],
        width=GEOMETRY["width"], approach=GEOMETRY["approach"])
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.height_scanner.params["side_noise_probability"] = 0.0
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    command = cfg.commands.base_velocity
    for ranges in (command.ranges, command.limit_ranges):
        ranges.lin_vel_x = (0.3, 0.3)
        ranges.lin_vel_y = ranges.ang_vel_z = ranges.heading = (0.0, 0.0)
    command.heading_command, command.rel_standing_envs, command.debug_vis = False, 0.0, False
    command.resampling_time_range = (1000000.0, 1000000.0)
    if any(getattr(cfg, name, None) is not None for name in
           ("contact_gait_reference", "stair_reference", "stance_orientation_feedback", "lateral_com_feedback")):
        raise RuntimeError("Formal policy evaluation must not install scripted control or legacy feedback")
    if cfg.stance_preload_torques:
        raise RuntimeError("Direct policy evaluation must not add reference preload")
    if cfg.actions.JointPositionAction.class_type is not DirectLegPositionAction:
        raise RuntimeError("New stair task is not configured for direct policy position control")
    cfg.viewer.origin_type, cfg.viewer.asset_name, cfg.viewer.env_index = "asset_root", "robot", 0
    cfg.viewer.eye, cfg.viewer.lookat, cfg.viewer.resolution = (-2.5, 3.8, 1.8), (0.8, 0.0, 0.6), (768, 576)
    cfg.io_descriptors_output_dir = str(args.output)
    dump_yaml(str(args.output / "environment.yaml"), cfg)
    metadata.update(checkpoint_iteration=int(saved["iter"]), completed_training_iterations=int(saved["iter"])+1,
                    finite_checkpoint_tensors=tensor_count, task_source=str(task_path), asset=str(asset),
                    asset_sha256=sha256(asset), policy_mode="frozen deterministic twelve-action direct position policy",
                    action_semantics="full_soft_tanh_position_v1", optimizer_loaded=False,
                    rnd_used_for_control=False, observation_corruption=False,
                    evaluation_changes=["fixed 15cm staircase", "no curriculum updates", "no observation corruption",
                                        "no external pushes", "zero reset yaw", "fixed forward command 0.3m/s"],
                    original_termination_terms_retained=True,
                    rendering="env0 visible; other 1023+ clones hidden only from rendering; physics stays active")
    for group in (cfg.observations.policy, cfg.observations.critic):
        if (group.height_scanner.func.__name__ != "reconstructed_lidar_height_scan"
                or Path(group.height_scanner.params["checkpoint"]).resolve() != recon):
            raise RuntimeError("Actor/critic perception does not use the recorded frozen LiDAR reconstructor")
    metadata["frozen_lidar_checkpoint"] = dict(path=str(recon), sha256=sha256(recon), provenance=(
        "training-time dependency SHA verified" if training_dependencies is not None else
        "recorded training config path; current fingerprint, historical content not hash-bound"))
    env, video, raw, tracker = None, None, None, None
    trajectory, event_chunks = [], []
    runtime_error = None
    simulated_steps = 0
    try:
        env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions"))
        raw = env.unwrapped
        robot, device, n = raw.scene["robot"], raw.device, args.num_envs
        term = raw.action_manager.get_term("JointPositionAction")
        if type(term) is not DirectLegPositionAction or term.gait_reference is not None:
            raise RuntimeError("The runtime action term is not the unmodified direct-position action")
        if list(term._joint_names) != list(H2_LEG_JOINT_NAMES) or len(H2_JOINT_NAMES) != 29:
            raise RuntimeError("Leg/observation joint-name contract changed")
        if raw.action_manager.total_action_dim != 12 or raw.action_manager._locked_action_ids:
            raise RuntimeError("Policy actions have been masked or their dimension changed")
        held_names = [robot.joint_names[i] for i in raw.action_manager.held_joint_ids]
        if set(held_names) != set(H2_JOINT_NAMES) - set(H2_LEG_JOINT_NAMES):
            raise RuntimeError("The original17 upper-body default targets are not retained")
        controlled_ids = {int(i) for i in term._joint_ids}
        covered_leg_ids = set()
        for actuator in robot.actuators.values():
            indices = actuator.joint_indices
            ids = (list(range(robot.num_joints))[indices] if isinstance(indices, slice)
                   else [int(i) for i in indices])
            owns_legs = bool(controlled_ids.intersection(ids))
            if owns_legs:
                if not actuator.is_implicit_model or not set(ids).issubset(controlled_ids):
                    raise RuntimeError("Policy leg joints are not isolated native implicit drive groups")
                covered_leg_ids.update(ids)
            # Existing explicit upper-body PD is retained. It is not a new
            # action/reference and must not be mistaken for an implicit drive.
            if getattr(actuator, "support_priority_buffer", None) is not None:
                raise RuntimeError("A scripted effort-priority actuator is attached")
        if covered_leg_ids != controlled_ids:
            raise RuntimeError("Some direct policy leg joints have no native implicit drive")
        obs, extras = wrapped.get_observations()
        shapes = {key: list(value.shape) for key, value in extras["observations"].items()}
        stair_transfer = saved.get("format") == "h2_guarded_stair_ppo_v1"
        actor_width, critic_width = (1034,871) if stair_transfer else (1029,866)
        if shapes != {"policy": [n, actor_width], "critic": [n, critic_width], "rnd_state": [n, 73]}:
            raise RuntimeError(f"Network observation dimensions changed: {shapes}")
        if bool((raw.episode_length_buf != 0).any()):
            raise RuntimeError("Formal evaluation must start with 1024 complete zero-length first episodes")
        if not torch.allclose(raw._stair_heights(), torch.full((n,), GEOMETRY["height"], device=device)):
            raise RuntimeError("Actual terrain row heights are not uniformly15cm")
        # Instantiate inference modules only: no PPO optimizer/storage/update loop
        # is needed or allocated. The RSL inference formula is norm -> actor.
        policy_cfg = dict(agent["policy"])
        policy_cfg.pop("class_name")
        actor = H2ActorCriticOur(actor_width, critic_width, 12, **policy_cfg).to(device)
        actor.load_state_dict(saved["model_state_dict"], strict=True)
        if stair_transfer:
            from task_context_normalization import make_task_context_normalizers
            blocks = make_task_context_normalizers(device)
            actor_norm, critic_norm = blocks["actor"], blocks["critic"]
        else:
            actor_norm = EmpiricalNormalization([1029], until=1.0e8).to(device)
            critic_norm = EmpiricalNormalization([866], until=1.0e8).to(device)
        actor_norm.load_state_dict(saved["obs_norm_state_dict"], strict=True)
        critic_norm.load_state_dict(saved["privileged_obs_norm_state_dict"], strict=True)
        for module in (actor, actor_norm, critic_norm):
            module.eval().requires_grad_(False)
        torch.save({"obs_norm_state_dict": saved["obs_norm_state_dict"],
                    "privileged_obs_norm_state_dict": saved["privileged_obs_norm_state_dict"]},
                   args.output / "saved_observation_normalizers.pt")
        metadata["saved_observation_normalizers_sha256"] = sha256(args.output / "saved_observation_normalizers.pt")
        count_key = "prefix.count" if stair_transfer else "count"
        metadata["normalizer_prefix_counts" if stair_transfer else "normalizer_counts"] = {
            key: int(saved[key][count_key]) for key in ("obs_norm_state_dict", "privileged_obs_norm_state_dict")}
        if stair_transfer:
            metadata["normalization_abi"] = saved["normalization_abi"]
            metadata["normalization_implementation_sha256"] = sha256(Path(__file__).parent/"task_context_normalization.py")
        metadata["runtime_joint_mapping"] = dict(action_joint_names=list(term._joint_names),
            action_articulation_ids=list(term._joint_ids), observed_joint_names=list(H2_JOINT_NAMES),
            articulation_joint_names=list(robot.joint_names), held_joint_ids=list(raw.action_manager.held_joint_ids))
        foot_ids, foot_names = raw.scene["contact_forces"].find_bodies(raw.cfg.feet_body_names, preserve_order=True)
        if list(foot_names) != ["left_ankle_pitch_link", "right_ankle_pitch_link"]:
            raise RuntimeError("Ground/unfiltered feet do not have the required left/right order")
        filtered_sensors = [raw.scene["stair_ascent_left_ground"], raw.scene["stair_ascent_right_ground"]]
        for item in filtered_sensors:
            if list(item.cfg.filter_prim_paths_expr) != ["/World/ground/terrain/mesh"]:
                raise RuntimeError("Ground-filtered foot sensor points at a different collision surface")
        sensor = raw.scene["contact_forces"]
        origins = raw.scene.terrain.env_origins.clone()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        failed, timed_out = torch.zeros_like(alive), torch.zeros_like(alive)
        sample_count = torch.zeros(n, dtype=torch.long, device=device)
        sampled_epoch = torch.full_like(sample_count, -1)
        reasons = {key: torch.zeros_like(alive) for key in raw.termination_manager.active_terms}
        start_pos = robot.data.root_pos_w.clone()
        tracker = StairAcceptance(n, device, origins, raw.masses,
                                 height=GEOMETRY["height"], depth=GEOMETRY["depth"],
                                 steps=GEOMETRY["steps"], approach=GEOMETRY["approach"])
        tracker.root_start = start_pos.clone()
        footfalls = FootfallTracker(n, device)
        positive_seen = torch.zeros(n, 2, dtype=torch.bool, device=device)
        mismatch_steps = torch.zeros(n, 2, dtype=torch.long, device=device)
        min_root_height = start_pos[:, 2] - origins[:, 2]
        max_tilt = torch.zeros(n, device=device)
        max_actual_soft_excess = torch.zeros(n, device=device)
        upper_target_error = torch.zeros(n, device=device)
        held = list(raw.action_manager.held_joint_ids)
        qids = list(term._joint_ids)
        endpoint = {}

        def physical_state():
            sole, rotation = raw._stair_foot_pose()
            forces = []
            for item in filtered_sensors:
                matrix = item.data.force_matrix_w
                if matrix is None or tuple(matrix.shape) != (n, 1, 1, 3):
                    raise RuntimeError("Ground-filtered foot forces have invalid shape or no native filter data")
                forces.append(matrix[:, 0, 0])
            return dict(root_position=robot.data.root_pos_w.clone(), root_quaternion=robot.data.root_quat_w.clone(),
                        root_velocity=robot.data.root_lin_vel_w.clone(), sole_position=sole.clone(),
                        foot_quaternion=robot.data.body_quat_w[:, raw._stair_foot_ids].clone(),
                        foot_rotation=rotation.clone(), filtered_force=torch.stack(forces, 1),
                        unfiltered_force=sensor.data.net_forces_w[:, foot_ids].clone(),
                        joint_position=robot.data.joint_pos[:, qids].clone(),
                        joint_velocity=robot.data.joint_vel[:, qids].clone(),
                        joint_target=robot.data.joint_pos_target[:, qids].clone(),
                        raw_action=raw.action_manager.action.clone(),
                        control_step=torch.full((n,), int(raw.common_step_counter), dtype=torch.long, device=device),
                        physics_step=torch.full((n,), int(raw._sim_step_counter), dtype=torch.long, device=device),
                        root_tilt=torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))

        endpoint.update({key: value.clone() for key, value in physical_state().items()})
        np.savez_compressed(args.output / "runtime_physical_parameters.npz",
            masses=raw.masses.cpu().numpy(), joint_names=np.array(robot.joint_names),
            stiffness=robot.data.joint_stiffness.cpu().numpy(), damping=robot.data.joint_damping.cpu().numpy(),
            effort_limits=robot.data.joint_effort_limits.cpu().numpy(),
            soft_joint_limits=robot.data.soft_joint_pos_limits.cpu().numpy(),
            armature=robot.data.joint_armature.cpu().numpy(), origins=origins.cpu().numpy())
        solver_k = robot.root_physx_view.get_dof_stiffnesses().to(device)
        solver_d = robot.root_physx_view.get_dof_dampings().to(device)
        solver_cap = robot.root_physx_view.get_dof_max_forces().to(device)
        actuator_certificate = {}
        for name, actuator in robot.actuators.items():
            ids = actuator.joint_indices
            resolved = (list(range(robot.num_joints))[ids] if isinstance(ids, slice)
                        else [int(i) for i in ids])
            owns_legs = bool(controlled_ids.intersection(resolved))
            implicit = bool(actuator.is_implicit_model)
            k_error = float((solver_k[:, ids] - actuator.stiffness).abs().max()) if implicit else None
            d_error = float((solver_d[:, ids] - actuator.damping).abs().max()) if implicit else None
            cap_error = float((solver_cap[:, ids] - actuator.effort_limit).abs().max()) if implicit else None
            if owns_legs and max(k_error, d_error, cap_error) > 1e-3:
                raise RuntimeError("Native implicit gain/cap readback disagrees with the actuator")
            lags = getattr(getattr(actuator, "positions_delay_buffer", None), "time_lags", None)
            actuator_certificate[name] = dict(class_name=type(actuator).__name__, native_implicit=implicit,
                controls_policy_leg_joints=owns_legs, articulation_joint_ids=resolved,
                upper_body_explicit_PD_retained=not implicit and not owns_legs,
                native_stiffness_max_error=k_error, native_damping_max_error=d_error,
                native_effort_cap_max_error=cap_error,
                sampled_delay_min=None if lags is None else int(lags.min()),
                sampled_delay_max=None if lags is None else int(lags.max()))
        metadata["native_actuator_certificate"] = actuator_certificate

        def capture(ids):
            """One same-post-physics measurement per alive environment/epoch."""
            ids = ids[alive[ids] & (sampled_epoch[ids] != int(raw.common_step_counter))]
            if not ids.numel():
                return
            state = physical_state()
            if not all(bool(torch.isfinite(value[ids]).all()) for value in state.values()):
                raise RuntimeError("Non-finite first-episode physical state")
            force, unfiltered = state["filtered_force"], state["unfiltered_force"]
            # Full-vector same-foot positive control, including sign. Mismatches
            # are preserved and disqualify contact evidence; they never create
            # artificial air by substituting zero forces into FootfallTracker.
            tolerance = 1.0 + 0.02 * unfiltered.norm(dim=-1)
            agreement = (force - unfiltered).norm(dim=-1) <= tolerance
            relevant = (force.norm(dim=-1) > 5.0) | (unfiltered.norm(dim=-1) > 5.0)
            mismatch_steps[ids] += (~agreement[ids] & relevant[ids]).long()
            local_z = state["sole_position"][..., 2] - origins[:, None, 2]
            on_measured_level = (local_z / GEOMETRY["height"] - (local_z / GEOMETRY["height"]).round()).abs() <= 0.20
            proof = (agreement & (force[..., 2] > tracker.threshold[:, None])
                     & (unfiltered[..., 2] > tracker.threshold[:, None]) & on_measured_level
                     & (state["foot_rotation"][..., 2, 2] > 0.975))
            positive_seen[ids] |= proof[ids]
            contact = force.norm(dim=-1) > 5.0
            previous_air = footfalls.air_time[ids].clone()
            previous_landings = footfalls.landings[ids].clone()
            landed = contact[ids] & ~footfalls.previous_contact[ids]
            footfalls.update(contact, raw.step_dt, ids)
            active = torch.zeros_like(alive)
            active[ids] = True
            tracker.update(state["root_position"], state["sole_position"], state["foot_rotation"], force,
                           footfalls.landings, raw.step_dt, active, state["root_tilt"])
            events = landed.nonzero()
            if events.numel():
                eid, side = ids[events[:, 0]], events[:, 1]
                rows = dict(env_id=eid, foot=side,
                    time_s=(sample_count[eid] + 1) * raw.step_dt,
                    flight_s=previous_air[events[:, 0], side],
                    valid_alternating=(footfalls.landings[eid, side] > previous_landings[events[:, 0], side]),
                    sole_position=state["sole_position"][eid, side], filtered_force=force[eid, side],
                    unfiltered_force=unfiltered[eid, side], force_agreement=agreement[eid, side])
                event_chunks.append({key: value.cpu().numpy().copy() for key, value in rows.items()})
            limits = robot.data.soft_joint_pos_limits[:, qids]
            expected = soft_range_targets(state["raw_action"], limits)
            if float((state["joint_target"][ids] - expected[ids]).abs().max()) > 2e-6:
                raise RuntimeError("Published leg position is being overridden after the policy's direct mapping")
            if bool((robot.data.joint_vel_target[ids][:, qids] != 0).any()) or bool(
                    (robot.data.joint_effort_target[ids][:, qids] != 0).any()):
                raise RuntimeError("Direct policy leg velocity/extra effort target is not zero")
            excess = torch.maximum(limits[..., 0] - state["joint_position"],
                                   state["joint_position"] - limits[..., 1]).clamp_min(0).amax(-1)
            max_actual_soft_excess[ids] = torch.maximum(max_actual_soft_excess[ids], excess[ids])
            if held:
                error = (robot.data.joint_pos_target[ids][:, held]
                         - robot.data.default_joint_pos[ids][:, held]).abs().amax(-1)
                upper_target_error[ids] = torch.maximum(upper_target_error[ids], error)
                if bool((error > 2e-6).any()):
                    raise RuntimeError("The original upper-body default targets changed")
            min_root_height[ids] = torch.minimum(min_root_height[ids], state["root_position"][ids, 2] - origins[ids, 2])
            max_tilt[ids] = torch.maximum(max_tilt[ids], state["root_tilt"][ids])
            for key, value in state.items():
                endpoint[key][ids] = value[ids]
            sample_count[ids] += 1
            sampled_epoch[ids] = int(raw.common_step_counter)

        original_reset = raw._reset_idx

        def before_reset(self, env_ids):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
            ids = ids[alive[ids]]
            if ids.numel():
                capture(ids)
                failed[ids] = self.termination_manager.terminated[ids]
                timed_out[ids] = self.termination_manager.time_outs[ids]
                for name, value in reasons.items():
                    value[ids] = self.termination_manager.get_term(name)[ids]
                alive[ids] = False
            original_reset(env_ids)

        raw._reset_idx = types.MethodType(before_reset, raw)
        from pxr import UsdGeom
        for env_index in range(1, n):
            UsdGeom.Imageable(raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}")).MakeInvisible()
        video = imageio.get_writer(str(args.output / "env0.mp4"), fps=10, codec="libx264", quality=7)
        metadata.update(status="evaluating", observation_shapes=shapes, step_dt_s=raw.step_dt,
                        physics_dt_s=raw.physics_dt, actions=12, first_episode_capture="pre-reset, every20ms; one update per epoch",
                        contact_positive_control="same-foot filtered/unfiltered full-vector agreement within1N+2%; both feet loaded",
                        force_semantics="native ContactSensor normal-force vectors; no frictional wrench or motor-torque claim",
                        contact_mismatch_policy="kept in event timing; final contact-evidence gate false, never fabricated air",
                        required_successful_envs=max(922, math.ceil(0.9*n)))
        write_json(args.output / "run.json", metadata)
        max_steps = round(DURATION / raw.step_dt)
        if abs(max_steps * raw.step_dt - DURATION) > 1e-6:
            raise RuntimeError("90s must be an integer number of control intervals")
        sample_every = max(1, round(0.1 / raw.step_dt))
        try:
            for step in range(max_steps):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before formal evaluation finished")
                with torch.inference_mode():
                    if not bool(torch.isfinite(obs).all()):
                        raise RuntimeError("Non-finite policy observation")
                    action = actor.act_inference(actor_norm(obs))
                    if tuple(action.shape) != (n, 12) or not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Invalid frozen policy actions")
                    obs, _, _, _ = wrapped.step(action)
                    capture(alive.nonzero().flatten())
                simulated_steps = step + 1
                if step % sample_every == 0 or step + 1 == max_steps or not bool(alive.any()):
                    # endpoint has terminal copies for ended first episodes; no
                    # post-reset pose is ever substituted into these trajectories.
                    frame = {key: value.cpu().numpy().copy() for key, value in endpoint.items() if key != "foot_rotation"}
                    frame.update(time_s=np.array((step+1)*raw.step_dt), first_episode_active=alive.cpu().numpy(),
                                 first_episode_time_s=(sample_count*raw.step_dt).cpu().numpy(),
                                 completed_levels=tracker.completed.cpu().numpy(),
                                 valid_alternating_landings=footfalls.landings.cpu().numpy())
                    trajectory.append(frame)
                    image = Image.fromarray(raw.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, image.width, 44), fill="black")
                    draw.text((8, 4), f"H2 frozen direct policy | six15cm stairs | t={(step+1)*raw.step_dt:.2f}s", fill="white")
                    draw.text((8, 23), f"First episodes active {int(alive.sum())}/{n}; env0 "
                              + ("active" if bool(alive[0]) else "ended (reset visible, excluded)"), fill="white")
                    video.append_data(np.asarray(image))
                    if step == 0:
                        image.save(args.output / "env0_start.png")
                if (step + 1) % max(1, round(2.0/raw.step_dt)) == 0:
                    print(f"[STAIR_POLICY] t={(step+1)*raw.step_dt:.2f}s first_active={int(alive.sum())}/{n} "
                          f"six_levels={int(tracker.verified.all(dim=(1,2)).sum())}", flush=True)
                if not bool(alive.any()):
                    break
        except BaseException as error:
            runtime_error = error
        duration = sample_count * raw.step_dt
        survived = ~failed & (sample_count >= max_steps)
        physical_success, acceptance = tracker.result(endpoint["root_position"], survived)
        force_valid = positive_seen.all(-1) & (mismatch_steps.sum(-1) == 0)
        success = physical_success & force_valid
        unchanged = (states_equal(torch, actor, saved["model_state_dict"])
                     and states_equal(torch, actor_norm, saved["obs_norm_state_dict"])
                     and states_equal(torch, critic_norm, saved["privileged_obs_norm_state_dict"]))
        if not unchanged and runtime_error is None:
            runtime_error = RuntimeError("Frozen policy/normalizer state changed during evaluation")
        checkpoint_unchanged = sha256(args.checkpoint) == metadata["checkpoint_sha256"]
        source_unchanged = source_hashes(source) == expected_hashes
        external_unchanged = (sha256(asset) == metadata["asset_sha256"]
                              and sha256(recon) == metadata["frozen_lidar_checkpoint"]["sha256"])
        if not (checkpoint_unchanged and source_unchanged and external_unchanged) and runtime_error is None:
            runtime_error = RuntimeError("Checkpoint/source/external asset changed during evaluation")
        required = metadata["required_successful_envs"]
        summary = dict(acceptance)
        summary.update(required_successful_envs=required, independent_physical_successful_envs=int(physical_success.sum()),
            successful_envs=int(success.sum()), success_percent=float(success.float().mean()*100),
            foot_contact_positive_control_envs=int(positive_seen.all(-1).sum()),
            contact_evidence_valid_envs=int(force_valid.sum()), force_mismatch_envs=int((mismatch_steps.sum(-1)>0).sum()),
            survived90s_envs=int(survived.sum()), terminated_envs=int(failed.sum()),
            termination_counts={name: int(value.sum()) for name, value in reasons.items()})
        terminal = dict(start_position=start_pos, end_position=endpoint["root_position"], duration_s=duration,
                        survived90s=survived, terminated=failed, timed_out=timed_out, success=success,
                        independent_physical_success=physical_success, completed_levels=tracker.completed,
                        verified_feet=tracker.verified, verified_time_s=tracker.verified_time,
                        verified_position=tracker.verified_position, platform_dwell_s=tracker.platform_max_time,
                        valid_alternating_landings=footfalls.landings, foot_positive_control=positive_seen,
                        force_mismatch_steps=mismatch_steps, contact_evidence_valid=force_valid,
                        maximum_tilt_rad=max_tilt, minimum_root_height_m=min_root_height,
                        maximum_actual_soft_limit_excess_rad=max_actual_soft_excess,
                        maximum_upper_body_target_error_rad=upper_target_error, origins=origins, **endpoint)
        np.savez_compressed(args.output / "stairs_events.npz", **{key:value.cpu().numpy() for key,value in terminal.items()})
        if trajectory:
            np.savez_compressed(args.output / "trajectories.npz",
                                **{key:np.stack([item[key] for item in trajectory]) for key in trajectory[0]})
        if event_chunks:
            np.savez_compressed(args.output / "footfall_events.npz",
                                **{key:np.concatenate([item[key] for item in event_chunks]) for key in event_chunks[0]})
        else:
            np.savez_compressed(args.output / "footfall_events.npz", env_id=np.zeros(0, dtype=np.int64))
        cpu = {key: value.cpu().numpy() for key, value in terminal.items()}
        rows = []
        for i in range(n):
            row = dict(env_id=i, duration_s=float(cpu["duration_s"][i]), survived90s=bool(cpu["survived90s"][i]),
                terminated=bool(cpu["terminated"][i]), timed_out=bool(cpu["timed_out"][i]),
                success=bool(cpu["success"][i]), independent_physical_success=bool(cpu["independent_physical_success"][i]),
                completed_levels=int(cpu["completed_levels"][i]), platform_dwell_s=float(cpu["platform_dwell_s"][i]),
                root_rise_m=float(cpu["end_position"][i,2]-cpu["start_position"][i,2]),
                contact_evidence_valid=bool(cpu["contact_evidence_valid"][i]),
                maximum_tilt_rad=float(cpu["maximum_tilt_rad"][i]),
                maximum_actual_soft_limit_excess_rad=float(cpu["maximum_actual_soft_limit_excess_rad"][i]),
                maximum_upper_body_target_error_rad=float(cpu["maximum_upper_body_target_error_rad"][i]),
                termination_reasons="|".join(name for name,value in reasons.items() if bool(value[i])))
            for side, label in enumerate(("left", "right")):
                row[label+"_valid_alternating_landings"] = int(cpu["valid_alternating_landings"][i,side])
                row[label+"_foot_positive_control"] = bool(cpu["foot_positive_control"][i,side])
                row[label+"_force_mismatch_steps"] = int(cpu["force_mismatch_steps"][i,side])
                for level in range(GEOMETRY["steps"]):
                    row[f"level{level+1}_{label}_verified"] = bool(cpu["verified_feet"][i,level,side])
                    row[f"level{level+1}_{label}_time_s"] = float(cpu["verified_time_s"][i,level,side])
            for field in ("start_position", "end_position", "root_velocity", "root_quaternion"):
                for axis, value in enumerate(cpu[field][i]):
                    row[f"{field}_{axis}"] = float(value)
            rows.append(row)
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata.update(status="failed" if runtime_error else "completed", finished_at_epoch=time.time(),
            simulated_steps=simulated_steps, simulated_duration_s=simulated_steps*raw.step_dt,
            checkpoint_unchanged=checkpoint_unchanged, frozen_source_unchanged=source_unchanged,
            external_asset_unchanged=external_unchanged, policy_and_both_normalizers_unchanged=unchanged,
            acceptance=summary, passed=bool(int(success.sum()) >= required and runtime_error is None),
            formal_acceptance_passed=bool(int(success.sum()) >= required and runtime_error is None),
            artifacts=dict(episodes="episodes.csv", events="stairs_events.npz", footfalls="footfall_events.npz",
                           trajectories="trajectories.npz", video="env0.mp4", source="training_source/source",
                           normalizers="saved_observation_normalizers.pt"))
        if runtime_error:
            metadata["error"] = type(runtime_error).__name__ + ": " + str(runtime_error)
        write_json(args.output / "result.json", metadata)
        write_json(args.output / "run.json", metadata)
        print("[STAIR_POLICY] RESULT " + json.dumps(summary), flush=True)
        if runtime_error:
            raise runtime_error
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()


def main():
    args, AppLauncher = parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="initializing", started_at_epoch=time.time(), checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint), task=args.task, num_envs=args.num_envs,
        duration_s=DURATION, staircase=GEOMETRY, seed=args.seed, training_updates=0,
        diagnostic_only=False, no_reference_controller=True, formal_acceptance_passed=False,
        direct_policy_owns_all_twelve_leg_targets=True, upper_body="original17 default PD; original native head drives retained",
        evaluator_sha256=sha256(__file__), required_successful_envs=max(922, math.ceil(0.9*args.num_envs)))
    metadata["evaluator_tools_sha256"] = {name: sha256(Path(__file__).parent / name) for name in
                                        ("evaluate_stair_policy.py", "evaluate_stair_policy.sh")}
    app = None
    try:
        _, source, hashes = prepare_source(args.checkpoint, args.output, args.source_root)
        metadata.update(training_source=str(source), original_training_source=str(
            json.loads((args.output / "checkpoint_run.json").read_text())["source_snapshot"]),
            source_sha256_manifest="source_sha256.json",
            independent_stair_acceptance_sha256=hashes[
                "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/stair_acceptance.py"],
            independent_footfall_class_sha256=class_ast_hash(source/
                "CtrlLab/CtrlLab/tasks/manager_based/h2_task/locomotion_progress.py", "FootfallTracker"))
        write_json(args.output / "run.json", metadata)
        app = AppLauncher(args).app
        evaluate(args, app, metadata, source, hashes)
    except BaseException as error:
        metadata.update(status="failed", finished_at_epoch=time.time(), formal_acceptance_passed=False,
                        passed=False, error=type(error).__name__ + ": " + str(error))
        write_json(args.output / "run.json", metadata)
        write_json(args.output / "result.json", metadata)
        (args.output / "failure_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
