"""One deterministic, first-episode BC or post-update deployment diagnostic.

Run the two arms in separate, sequential Isaac processes. Each uses its own
saved actor and BOTH saved normalizers, the same preserved training source and
the original 3 cm initial stair curriculum. This is neither training nor formal
stair acceptance. Importing this module does not start Isaac Sim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]
TASK = "Unitree-H2-Stair-Ascent-Policy"
DEFAULT_RUN = ROOT / ("logs/rsl_rl/h2_main_policy_manager_based/"
    "2026-10-30_23-40-28_direct_stair_domain_forward_budget_eeaecc74")


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


def yaml_scalar(document, keys):
    import yaml
    node = yaml.compose(document, Loader=yaml.SafeLoader)
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Saved configuration has no scalar at {keys}")
        node = {item.value: value for item, value in node.value}[key]
    if not isinstance(node, yaml.ScalarNode):
        raise ValueError(f"Saved configuration value is not scalar: {keys}")
    return node.value


def prepare(args, metadata):
    run = json.loads((args.training_run / "run.json").read_text())
    if run["task"] != TASK or run.get("status") != "completed":
        raise ValueError("Require the completed, policy-owned stair training run")
    source = Path(run["source_snapshot"]).resolve(strict=True)
    if args.source_root != source or source != args.training_run / "training_source/source":
        raise ValueError("--source_root must be this run's preserved training_source/source")
    expected = json.loads((source.parent / "main_source_sha256.json").read_text())
    if source_hashes(source) != expected:
        raise RuntimeError("Preserved training source fingerprint mismatch")
    original_checkpoint = (Path(run["initial_policy"]) if args.arm == "bc"
                           else args.training_run / "model_0.pt").resolve(strict=True)
    expected_checkpoint = sha256(original_checkpoint)
    if sha256(args.checkpoint) != expected_checkpoint:
        raise ValueError("Checkpoint is not the requested run's BC initializer or post-update model_0")
    if args.arm == "bc" and expected_checkpoint != run["initial_policy_sha256"]:
        raise RuntimeError("Training-time BC initializer fingerprint mismatch")
    if args.seed != int(run["seed"]) or args.num_envs != int(run["num_envs"]):
        raise ValueError("Diagnostic seed/environment count must match the training run")
    shutil.copytree(source.parent, args.output / "training_source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = args.output / "training_source/source"
    if source_hashes(frozen) != expected:
        raise RuntimeError("Copied training source is incomplete")
    for name in ("agent.yaml", "env.yaml"):
        shutil.copy2(args.training_run / "params" / name, args.output / ("training_" + name))
    shutil.copy2(args.training_run / "run.json", args.output / "training_run.json")
    snapshots = args.output / "tools_snapshot"
    snapshots.mkdir()
    for name in ("diagnose_stair_initialization.py", "diagnose_stair_initialization.sh"):
        shutil.copy2(Path(__file__).parent / name, snapshots / name)
    dependencies = run.get("external_dependency_sha256")
    if not dependencies:
        raise RuntimeError("Require training-time USD/LiDAR dependency fingerprints")
    for path, expected_sha in dependencies.items():
        if sha256(Path(path).resolve(strict=True)) != expected_sha:
            raise RuntimeError(f"Training dependency changed: {path}")
    asset = Path(run["asset"])
    asset = (ROOT / asset).resolve(strict=True) if not asset.is_absolute() else asset.resolve(strict=True)
    recon = Path(yaml_scalar((args.output / "training_env.yaml").read_text(),
        ("observations", "policy", "height_scanner", "params", "checkpoint")))
    recon = (ROOT / recon).resolve(strict=True) if not recon.is_absolute() else recon.resolve(strict=True)
    if not {str(asset), str(recon)}.issubset(dependencies):
        raise RuntimeError("USD or LiDAR checkpoint is absent from the training dependency manifest")
    os.environ["CTRLLAB_H2_USD_PATH"] = str(asset)
    os.environ["CTRLLAB_H2_LIDAR_RECON_CHECKPOINT"] = str(recon)
    metadata.update(source_root=str(source), imported_source_root=str(frozen),
        source_commit=run.get("git_commit"), checkpoint_sha256=expected_checkpoint,
        external_dependency_sha256=dependencies, original_arm_checkpoint=str(original_checkpoint),
        deployment_comparison="BC versus first PPO-update deployment, each with its own saved dual normalization",
        checkpoint_semantics=("behavior-cloned initialization, zero PPO updates" if args.arm == "bc" else
            "RSL model_0 saved after the first rollout and first PPO update; not a pristine initializer"))
    write_json(args.output / "source_sha256.json", expected)
    return run, frozen, expected


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("bc", "pt0"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--training_run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1024 or not 0 < args.duration <= 3.0:
        parser.error("This diagnostic is exactly1024 environments and at most3 simulated seconds")
    for key in ("checkpoint", "source_root", "training_run"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    args.enable_cameras = False
    return args, AppLauncher


def diagnose(args, app, metadata, source, expected_hashes):
    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    def cpu(value):
        return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()

    def fingerprint(value):
        array = np.ascontiguousarray(value)
        return dict(shape=list(array.shape), dtype=str(array.dtype),
                    sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())

    def save_arrays(name, arrays):
        arrays = {key: cpu(value) for key, value in arrays.items()}
        for key, value in arrays.items():
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite diagnostic evidence {name}:{key}")
        np.savez_compressed(args.output / (name + ".npz"), **arrays)
        return {key: fingerprint(value) for key, value in arrays.items()}

    def unchanged(module, original):
        current = module.state_dict()
        return current.keys() == original.keys() and all(
            torch.equal(value.detach().cpu(), original[key]) for key, value in current.items())

    sys.path.insert(0, str(source / "CtrlLab/CtrlLab/tasks/manager_based"))
    import h2_task
    from h2_task.assets import H2_JOINT_NAMES, H2_LEG_JOINT_NAMES
    from h2_task.direct_leg_action import DirectLegPositionAction, soft_range_targets
    from h2_task.networks import H2ActorCriticOur
    if not Path(h2_task.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Imported task is not the copied training snapshot")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    required = ("model_state_dict", "obs_norm_state_dict", "privileged_obs_norm_state_dict")
    if not isinstance(saved, dict) or not set(required).issubset(saved):
        raise ValueError("Both arms require an actor/critic model and both saved normalizers")
    if args.arm == "bc":
        if saved.get("format") != "h2_direct_policy_initialization_v1" or saved.get("ppo_completed_iterations") != 0:
            raise ValueError("BC arm is not the original zero-PPO initialization")
        if saved.get("action_semantics") != "full_soft_tanh_position_v1":
            raise ValueError("BC direct-action semantics mismatch")
    elif int(saved.get("iter", -1)) != 0:
        raise ValueError("PT0 arm requires the first RSL model_0 checkpoint")
    for key in required:
        if any(not bool(torch.isfinite(value).all()) for value in saved[key].values()
               if isinstance(value, torch.Tensor)):
            raise ValueError(f"Non-finite checkpoint state: {key}")
    for key in required[1:]:
        if int(saved[key]["count"]) <= 0 or bool((saved[key]["_std"] < 0).any()):
            raise ValueError("Saved normalization is unpopulated or invalid")
    agent = yaml.safe_load((args.output / "training_agent.yaml").read_text())
    if not agent["empirical_normalization"] or agent["policy"]["class_name"] != "H2ActorCriticOur":
        raise ValueError("The saved unchanged H2 architecture and empirical normalization are required")
    entry = gym.spec(TASK).kwargs["env_cfg_entry_point"]
    module, name = entry.split(":")
    cfg = getattr(importlib.import_module(module), name)()
    if cfg.observations.policy.enable_corruption is not False:
        raise RuntimeError("Frozen training config must already disable artificial policy corruption")
    if yaml_scalar((args.output / "training_env.yaml").read_text(),
                   ("observations", "policy", "enable_corruption")) != "false":
        raise RuntimeError("Saved training observation corruption does not match this diagnostic")
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device or "cuda:0"
    cfg.io_descriptors_output_dir = str(args.output)
    # Never replace the terrain, reset distribution, material/gain randomization,
    # command sampler or native terminations. First episodes all start on row0.
    # Curriculum updates after a FIRST episode ends cannot alter another alive
    # first episode; disabling only these updates keeps the original initial row.
    cfg.curriculum.terrain_levels = None
    if cfg.episode_length_s != 90.0 or cfg.scene.terrain.max_init_terrain_level != 0:
        raise RuntimeError("The original90s episode/initial row contract changed")
    geometry = cfg.scene.terrain.terrain_generator
    stair = geometry.sub_terrains["ascending_stairway"]
    if (geometry.num_rows, geometry.num_cols, tuple(geometry.size), stair.num_steps,
        stair.minimum_riser_height, stair.maximum_riser_height, stair.tread_depth,
        stair.stair_width, stair.approach_distance) != (8, 4, (12.0, 6.0), 6, .03, .15, .35, 2.4, 1.4):
        raise RuntimeError("Diagnostic must preserve the training3cm initial curriculum geometry")
    if any(getattr(cfg, key, None) is not None for key in
           ("contact_gait_reference", "stair_reference", "stance_orientation_feedback", "lateral_com_feedback")):
        raise RuntimeError("No teacher/reference/legacy controller may be attached")
    if cfg.stance_preload_torques or cfg.actions.JointPositionAction.class_type is not DirectLegPositionAction:
        raise RuntimeError("Direct position control must have no preload/reference override")
    for group in (cfg.observations.policy, cfg.observations.critic):
        recon = str(Path(group.height_scanner.params["checkpoint"]).resolve())
        if recon != os.environ["CTRLLAB_H2_LIDAR_RECON_CHECKPOINT"]:
            raise RuntimeError("Perception does not use the training-time frozen reconstructor")
    dump_yaml(str(args.output / "environment.yaml"), cfg)
    metadata.update(status="initializing_environment", task=TASK, source_task_file=str(h2_task.__file__),
        initial_riser_height_m=.03, observation_corruption=False,
        observation_corruption_already_disabled_in_training=True, environment_timeout_s=90.0,
        deterministic_policy_mean=True, first_frame_preprocessing="own saved actor norm -> actor.act_inference",
        optimizer_loaded=False, rnd_used_for_control=False, no_reference_controller=True,
        configuration_changes=["num_envs/device/seed bound to run", "output directory", "disable post-episode terrain curriculum updates"],
        force_semantics="ContactSensor normal-force vectors; neither full frictional GRF nor isolated motor torque")
    write_json(args.output / "run.json", metadata)
    env, wrapped = None, None
    try:
        env = gym.make(TASK, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions"))
        raw, n = env.unwrapped, args.num_envs
        robot, device = raw.scene["robot"], raw.device
        term = raw.action_manager.get_term("JointPositionAction")
        if type(term) is not DirectLegPositionAction or term.gait_reference is not None:
            raise RuntimeError("Runtime does not use policy-owned direct position actions")
        qids = [int(i) for i in term._joint_ids]
        held = [int(i) for i in raw.action_manager.held_joint_ids]
        if (list(term._joint_names) != list(H2_LEG_JOINT_NAMES) or len(qids) != 12
                or raw.action_manager.total_action_dim != 12 or raw.action_manager._locked_action_ids):
            raise RuntimeError("Twelve-action joint mapping or actor authority changed")
        if set(robot.joint_names[i] for i in held) != set(H2_JOINT_NAMES) - set(H2_LEG_JOINT_NAMES):
            raise RuntimeError("Original17 upper-body held joints changed")
        if bool((raw.episode_length_buf != 0).any()) or bool((raw.action_manager.action != 0).any()):
            raise RuntimeError("First episodes must begin at zero age with reset action history")
        if not torch.allclose(raw._stair_heights(), torch.full((n,), .03, device=device)):
            raise RuntimeError("Actual first-episode stairs are not uniformly3cm")
        obs, extras = wrapped.get_observations()
        shapes = {key: list(value.shape) for key, value in extras["observations"].items()}
        if shapes != {"policy": [n, 1029], "critic": [n, 866], "rnd_state": [n, 73]}:
            raise RuntimeError(f"Unchanged observation layout required: {shapes}")
        if bool((obs[:, 323:468] != 0).any()) or bool((extras["observations"]["critic"][:, 70:99] != 0).any()):
            raise RuntimeError("Initial policy/critic padded action histories are not reset to zero")
        policy_cfg = dict(agent["policy"])
        policy_cfg.pop("class_name")
        actor = H2ActorCriticOur(1029, 866, 12, **policy_cfg).to(device)
        actor_norm = EmpiricalNormalization([1029], until=1.0e8).to(device)
        critic_norm = EmpiricalNormalization([866], until=1.0e8).to(device)
        for module_, key in zip((actor, actor_norm, critic_norm), required):
            module_.load_state_dict(saved[key], strict=True)
            module_.eval().requires_grad_(False)
        torch.save({key: saved[key] for key in required[1:]}, args.output / "saved_observation_normalizers.pt")
        metadata["normalizer_counts"] = {key: int(saved[key]["count"]) for key in required[1:]}
        metadata["runtime_joint_mapping"] = dict(articulation_joint_names=list(robot.joint_names),
            body_names=list(robot.body_names), action_joint_names=list(term._joint_names),
            action_articulation_ids=qids, held_upper_body_joint_ids=held, observed_joint_names=list(H2_JOINT_NAMES))
        metadata["initial_state_velocity_semantics"] = (
            "root_state_w has actor-root pose with COM linear velocity; root_link_state_w uses link linear velocity")
        sensors = [raw.scene["stair_ascent_left_ground"], raw.scene["stair_ascent_right_ground"]]
        contact_sensor = raw.scene["contact_forces"]
        foot_ids, foot_names = contact_sensor.find_bodies(raw.cfg.feet_body_names, preserve_order=True)
        if list(foot_names) != ["left_ankle_pitch_link", "right_ankle_pitch_link"]:
            raise RuntimeError("Ground/unfiltered sensor left/right foot order changed")
        for item in sensors:
            if list(item.cfg.filter_prim_paths_expr) != ["/World/ground/terrain/mesh"]:
                raise RuntimeError("Ground-filtered sensor surface changed")
        origins = raw.scene.terrain.env_origins.clone()
        initial = dict(root_state_w=robot.data.root_state_w.clone(),
            root_link_state_w=robot.data.root_link_state_w.clone(),
            joint_position=robot.data.joint_pos.clone(), joint_velocity=robot.data.joint_vel.clone(),
            default_joint_position=robot.data.default_joint_pos.clone(),
            default_joint_velocity=robot.data.default_joint_vel.clone(),
            joint_position_target=robot.data.joint_pos_target.clone(),
            joint_velocity_target=robot.data.joint_vel_target.clone(),
            joint_effort_target=robot.data.joint_effort_target.clone(),
            body_position=robot.data.body_pos_w.clone(), body_quaternion=robot.data.body_quat_w.clone(),
            origins=origins, scene_origins=raw.scene.env_origins.clone(),
            terrain_levels=raw.scene.terrain.terrain_levels.clone(), terrain_types=raw.scene.terrain.terrain_types.clone(),
            actual_riser_height=raw._stair_heights().clone(),
            command=raw.command_manager.get_command("base_velocity").clone(),
            action=raw.action_manager.action.clone(), previous_action=raw.action_manager.prev_action.clone(),
            previous_previous_action=raw.action_manager.prev_prev_action.clone(),
            episode_age=raw.episode_length_buf.clone(),
            control_counter=np.array(int(raw.common_step_counter), dtype=np.int64),
            physics_counter=np.array(int(raw._sim_step_counter), dtype=np.int64))
        initial_fingerprint = save_arrays("initial_state", initial)
        view = robot.root_physx_view
        physical = dict(native_mass=view.get_masses(), native_com_pose=view.get_coms(),
            native_inertia=view.get_inertias(), native_stiffness=view.get_dof_stiffnesses(),
            native_damping=view.get_dof_dampings(), native_effort_cap=view.get_dof_max_forces(),
            native_velocity_cap=view.get_dof_max_velocities(), native_armature=view.get_dof_armatures(),
            native_friction_properties=view.get_dof_friction_properties(), native_material_properties=view.get_material_properties(),
            data_stiffness=robot.data.joint_stiffness, data_damping=robot.data.joint_damping,
            data_effort_limits=robot.data.joint_effort_limits, data_velocity_limits=robot.data.joint_vel_limits,
            hard_joint_limits=robot.data.joint_pos_limits, soft_joint_limits=robot.data.soft_joint_pos_limits,
            joint_names=np.asarray(robot.joint_names), body_names=np.asarray(robot.body_names),
            physics_dt_s=np.array(raw.physics_dt), control_dt_s=np.array(raw.step_dt))
        certificate, covered = {}, set()
        for name, actuator in robot.actuators.items():
            ids = (list(range(robot.num_joints))[actuator.joint_indices] if isinstance(actuator.joint_indices, slice)
                   else [int(i) for i in actuator.joint_indices])
            owns_leg = bool(set(ids).intersection(qids))
            if owns_leg and (not actuator.is_implicit_model or not set(ids).issubset(qids)):
                raise RuntimeError("Direct leg native implicit groups changed")
            if owns_leg:
                covered.update(ids)
            if getattr(actuator, "support_priority_buffer", None) is not None:
                raise RuntimeError("Reference effort-priority channel unexpectedly attached")
            lag_fields = {}
            for kind in ("positions", "velocities", "efforts"):
                buffer = getattr(actuator, kind + "_delay_buffer", None)
                if buffer is None:
                    raise RuntimeError(f"Required original transport-delay buffer missing: {name}/{kind}")
                lag = buffer.time_lags.clone()
                physical[f"actuator_{name}_{kind}_delay_steps"] = lag
                lag_fields[kind] = lag
            if not (torch.equal(lag_fields["positions"], lag_fields["velocities"])
                    and torch.equal(lag_fields["positions"], lag_fields["efforts"])):
                raise RuntimeError("Native position/velocity/effort transport lags disagree")
            if owns_leg and bool(((lag_fields["positions"] < 0) | (lag_fields["positions"] > 6)).any()):
                raise RuntimeError("Original0..6 physics-step leg delay changed")
            for field in ("stiffness", "damping", "effort_limit", "velocity_limit"):
                physical[f"actuator_{name}_{field}"] = getattr(actuator, field).clone()
            errors = {}
            if actuator.is_implicit_model:
                for native_key, field in (("native_stiffness", "stiffness"), ("native_damping", "damping"),
                                           ("native_effort_cap", "effort_limit")):
                    errors[field] = float((physical[native_key].to(device)[:, ids] - getattr(actuator, field)).abs().max())
                if owns_leg and max(errors.values()) > 1e-3:
                    raise RuntimeError("Actual native leg K/D/cap differs from actuator readback")
            certificate[name] = dict(class_name=type(actuator).__name__, joint_ids=ids,
                native_implicit=bool(actuator.is_implicit_model), policy_leg_group=owns_leg,
                min_delay=int(lag_fields["positions"].min()), max_delay=int(lag_fields["positions"].max()),
                implicit_readback_errors=errors)
        if covered != set(qids):
            raise RuntimeError("Some leg joints have no original implicit actuator group")
        metadata["native_actuator_certificate"] = certificate
        metadata["native_friction_semantics"] = "Raw installed SDK DOF properties, not assumed Coulomb coefficient times spatial joint force"
        physical_fingerprint = save_arrays("initial_physical_parameters", physical)
        raw_critic = extras["observations"]["critic"]
        with torch.inference_mode():
            first_normalized, first_critic_normalized = actor_norm(obs), critic_norm(raw_critic)
            first_action = actor.act_inference(first_normalized)
        first = dict(raw_policy=obs.clone(), raw_critic=raw_critic.clone(),
            normalized_policy=first_normalized.clone(), normalized_critic=first_critic_normalized.clone(),
            proposed_raw_action=first_action.clone(), normalizer_policy_count=saved[required[1]]["count"],
            normalizer_critic_count=saved[required[2]]["count"])
        first_fingerprint = save_arrays("first_inference", first)
        pairing = dict(schema_version=1, arm=args.arm, seed=args.seed, num_envs=n,
            source_root=metadata["source_root"], source_sha256=expected_hashes,
            external_dependency_sha256=metadata["external_dependency_sha256"],
            required_groups=dict(initial_state=initial_fingerprint, initial_physical_parameters=physical_fingerprint,
                first_raw_observations={key: first_fingerprint[key] for key in ("raw_policy", "raw_critic")}),
            excluded_from_pairing=["checkpoint content", "own saved normalizers", "normalized observations", "policy actions/targets"],
            pairing_status="pending_manager_comparison", fingerprint_semantics="SHA256 of contiguous array bytes, shape and dtype; not NPZ archive hash",
            limits="Exact saved-array agreement is a required pairing check, not proof of deterministic PhysX trajectories")
        write_json(args.output / "pairing_manifest.json", pairing)
        alive = torch.ones(n, device=device, dtype=torch.bool)
        terminated, timed_out = torch.zeros_like(alive), torch.zeros_like(alive)
        reasons = {key: torch.zeros_like(alive) for key in raw.termination_manager.active_terms}
        sample_count = torch.zeros(n, device=device, dtype=torch.long)
        sampled_epoch = torch.full_like(sample_count, -1)
        positive = torch.zeros(n, 2, device=device, dtype=torch.bool)
        mismatches = torch.zeros(n, 2, device=device, dtype=torch.long)
        max_tilt, max_soft_excess = torch.zeros(n, device=device), torch.zeros(n, device=device)
        start_pos = robot.data.root_pos_w.clone()

        def physical_state():
            sole, rotation = raw._stair_foot_pose()
            forces = []
            for sensor in sensors:
                matrix = sensor.data.force_matrix_w
                if matrix is None or tuple(matrix.shape) != (n, 1, 1, 3):
                    raise RuntimeError("Ground-filtered foot normal-force layout is invalid")
                forces.append(matrix[:, 0, 0].clone())
            return dict(root_position=robot.data.root_pos_w.clone(), root_quaternion=robot.data.root_quat_w.clone(),
                root_link_velocity=robot.data.root_link_lin_vel_w.clone(), root_com_velocity=robot.data.root_lin_vel_w.clone(),
                root_angular_velocity=robot.data.root_ang_vel_w.clone(),
                root_tilt=torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)),
                joint_position=robot.data.joint_pos.clone(), joint_velocity=robot.data.joint_vel.clone(),
                joint_target=robot.data.joint_pos_target.clone(), velocity_target=robot.data.joint_vel_target.clone(),
                extra_effort_target=robot.data.joint_effort_target.clone(), sole_position=sole.clone(), foot_rotation=rotation.clone(),
                filtered_normal_force=torch.stack(forces, 1),
                unfiltered_foot_normal_force=contact_sensor.data.net_forces_w[:, foot_ids].clone(),
                all_body_normal_force=contact_sensor.data.net_forces_w.clone(),
                raw_action=raw.action_manager.action.clone(),
                control_step=torch.full((n,), int(raw.common_step_counter), device=device, dtype=torch.long),
                physics_step=torch.full((n,), int(raw._sim_step_counter), device=device, dtype=torch.long))

        endpoint = physical_state()
        trajectory = []

        def append_frame(time_s):
            frame = {key: cpu(value) for key, value in endpoint.items()}
            frame.update(time_s=np.array(time_s), first_episode_active=cpu(alive),
                         first_episode_time_s=cpu(sample_count * raw.step_dt),
                         first_episode_terminated=cpu(terminated), first_episode_timed_out=cpu(timed_out))
            trajectory.append(frame)

        append_frame(0.0)

        def capture(ids):
            ids = ids[alive[ids] & (sampled_epoch[ids] != int(raw.common_step_counter))]
            if not ids.numel():
                return
            state = physical_state()
            if any(not bool(torch.isfinite(value[ids]).all()) for value in state.values()):
                raise RuntimeError("Non-finite first-episode physical state")
            expected = soft_range_targets(state["raw_action"], robot.data.soft_joint_pos_limits[:, qids])
            if float((state["joint_target"][ids][:, qids] - expected[ids]).abs().max()) > 2e-6:
                raise RuntimeError("Direct policy position targets have been overridden")
            if bool((state["velocity_target"][ids][:, qids] != 0).any()) or bool(
                    (state["extra_effort_target"][ids][:, qids] != 0).any()):
                raise RuntimeError("Direct policy native leg vtarget/extra effort changed from zero")
            if held and float((state["joint_target"][ids][:, held]
                              - robot.data.default_joint_pos[ids][:, held]).abs().max()) > 2e-6:
                raise RuntimeError("Original upper-body default position targets changed")
            filtered, unfiltered = state["filtered_normal_force"], state["unfiltered_foot_normal_force"]
            agree = (filtered - unfiltered).norm(dim=-1) <= 1.0 + .02 * unfiltered.norm(dim=-1)
            relevant = (filtered.norm(dim=-1) > 5.0) | (unfiltered.norm(dim=-1) > 5.0)
            mismatches[ids] += (~agree[ids] & relevant[ids]).long()
            positive[ids] |= (agree & (filtered[..., 2] > 5.0) & (unfiltered[..., 2] > 5.0))[ids]
            max_tilt[ids] = torch.maximum(max_tilt[ids], state["root_tilt"][ids])
            limits = robot.data.soft_joint_pos_limits
            excess = torch.maximum(limits[..., 0] - state["joint_position"],
                                   state["joint_position"] - limits[..., 1]).clamp_min(0).amax(-1)
            max_soft_excess[ids] = torch.maximum(max_soft_excess[ids], excess[ids])
            for key in endpoint:
                endpoint[key][ids] = state[key][ids]
            sample_count[ids] += 1
            sampled_epoch[ids] = int(raw.common_step_counter)

        original_reset = raw._reset_idx

        def before_reset(self, env_ids):
            ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
            ids = ids[alive[ids]]
            if ids.numel():
                capture(ids)
                terminated[ids] = self.termination_manager.terminated[ids]
                timed_out[ids] = self.termination_manager.time_outs[ids]
                for name, bits in reasons.items():
                    bits[ids] = self.termination_manager.get_term(name)[ids]
                alive[ids] = False
            original_reset(env_ids)

        raw._reset_idx = types.MethodType(before_reset, raw)
        original_apply = raw.action_manager.apply_action

        def record_first_published(self):
            original_apply()
            if "published_position_target" not in first:
                first.update(applied_raw_action=self.action.clone(), processed_target=term._processed_actions.clone(),
                    published_position_target=robot.data.joint_pos_target.clone(),
                    published_velocity_target=robot.data.joint_vel_target.clone(),
                    published_extra_effort_target=robot.data.joint_effort_target.clone(),
                    published_control_step=np.array(int(raw.common_step_counter)),
                    first_receiving_physics_step=np.array(int(raw._sim_step_counter)))
                save_arrays("first_inference", first)

        raw.action_manager.apply_action = types.MethodType(record_first_published, raw.action_manager)
        max_steps = round(args.duration / raw.step_dt)
        if abs(max_steps * raw.step_dt - args.duration) > 1e-6:
            raise ValueError("Diagnostic duration must be an integer number of original control intervals")
        metadata.update(status="evaluating", step_dt_s=raw.step_dt, physics_dt_s=raw.physics_dt,
            observation_shapes=shapes, no_settlement_or_warmup_steps=True,
            published_target_semantics="ActionManager output before original transport buffers; not received motor effort",
            paired_causal_eligibility="pending_manager_comparison", saved_norms="saved_observation_normalizers.pt")
        write_json(args.output / "run.json", metadata)
        print(f"[STAIR_INIT] arm={args.arm} t=0.00s first_active={n}/{n} native_failed=0", flush=True)
        runtime_error, steps = None, 0
        try:
            for step in range(max_steps):
                if not app.is_running():
                    raise RuntimeError("Simulator closed before the bounded diagnostic ended")
                with torch.inference_mode():
                    # First action is the saved, correctly normalized initial
                    # inference. Every following action uses the same own norm.
                    action = first_action if step == 0 else actor.act_inference(actor_norm(obs))
                    if tuple(action.shape) != (n, 12) or not bool(torch.isfinite(action).all()):
                        raise RuntimeError("Non-finite or wrong-shaped deterministic policy action")
                    obs, _, _, _ = wrapped.step(action)
                    capture(alive.nonzero().flatten())
                steps = step + 1
                append_frame(steps * raw.step_dt)
                if steps % max(1, round(.5 / raw.step_dt)) == 0 or not bool(alive.any()) or steps == max_steps:
                    print(f"[STAIR_INIT] arm={args.arm} t={steps*raw.step_dt:.2f}s "
                          f"first_active={int(alive.sum())}/{n} native_failed={int(terminated.sum())}", flush=True)
                if not bool(alive.any()):
                    break
        except BaseException as error:
            runtime_error = error
        if "published_position_target" not in first and runtime_error is None:
            runtime_error = RuntimeError("No original pre-physics action publication was captured")
        duration = sample_count * raw.step_dt
        horizon = alive & (sample_count >= max_steps)
        terminal = dict(**endpoint, origins=origins, start_position=start_pos,
            duration_s=duration, terminated=terminated, timed_out=timed_out,
            reached_diagnostic_horizon=horizon, censored_at_horizon=horizon,
            first_episode_active=alive, maximum_tilt_rad=max_tilt,
            maximum_actual_soft_limit_excess_rad=max_soft_excess,
            foot_positive_control=positive, force_mismatch_steps=mismatches,
            **{"termination_" + name: value for name, value in reasons.items()})
        save_arrays("first_episode_terminal", terminal)
        np.savez_compressed(args.output / "trajectories.npz",
            **{key: np.stack([item[key] for item in trajectory]) for key in trajectory[0]})
        terminal_cpu = {key: cpu(value) for key, value in terminal.items()}
        rows = []
        for i in range(n):
            row = dict(env_id=i, duration_s=float(terminal_cpu["duration_s"][i]),
                terminated=bool(terminal_cpu["terminated"][i]), timed_out=bool(terminal_cpu["timed_out"][i]),
                reached_diagnostic_horizon=bool(terminal_cpu["reached_diagnostic_horizon"][i]),
                censored_at_horizon=bool(terminal_cpu["censored_at_horizon"][i]),
                termination_reasons="|".join(name for name, bits in reasons.items() if bool(bits[i])),
                net_world_forward_m=float(terminal_cpu["root_position"][i, 0] - terminal_cpu["start_position"][i, 0]),
                root_rise_m=float(terminal_cpu["root_position"][i, 2] - terminal_cpu["start_position"][i, 2]),
                maximum_tilt_rad=float(terminal_cpu["maximum_tilt_rad"][i]),
                maximum_actual_soft_limit_excess_rad=float(terminal_cpu["maximum_actual_soft_limit_excess_rad"][i]),
                left_positive_control=bool(terminal_cpu["foot_positive_control"][i, 0]),
                right_positive_control=bool(terminal_cpu["foot_positive_control"][i, 1]),
                left_force_mismatch_steps=int(terminal_cpu["force_mismatch_steps"][i, 0]),
                right_force_mismatch_steps=int(terminal_cpu["force_mismatch_steps"][i, 1]))
            for label in ("root_position", "root_quaternion", "root_link_velocity", "root_com_velocity"):
                for axis, value in enumerate(terminal_cpu[label][i]):
                    row[f"{label}_{axis}"] = float(value)
            rows.append(row)
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        immutable = all(unchanged(module_, saved[key]) for module_, key in zip((actor, actor_norm, critic_norm), required))
        checkpoint_unchanged = sha256(args.checkpoint) == metadata["checkpoint_sha256"]
        source_unchanged = (source_hashes(source) == expected_hashes
                            and source_hashes(args.source_root) == expected_hashes)
        external_unchanged = all(sha256(Path(path)) == digest
                                 for path, digest in metadata["external_dependency_sha256"].items())
        if not (immutable and checkpoint_unchanged and source_unchanged and external_unchanged) and runtime_error is None:
            runtime_error = RuntimeError("Frozen policy/normalizers/checkpoint/source/external dependency changed")
        metadata.update(status="failed" if runtime_error else "completed", finished_at_epoch=time.time(),
            simulated_steps=steps, simulated_duration_s=steps * raw.step_dt,
            first_episode_horizon_survivors=int(horizon.sum()), native_terminated_envs=int(terminated.sum()),
            native_timeout_envs=int(timed_out.sum()), mean_first_duration_s=float(duration.mean()),
            median_first_duration_s=float(duration.median()), maximum_first_duration_s=float(duration.max()),
            termination_counts={name: int(bits.sum()) for name, bits in reasons.items()},
            ground_foot_positive_control_envs=int(positive.all(-1).sum()),
            foot_force_mismatch_envs=int((mismatches.sum(-1) > 0).sum()),
            checkpoint_unchanged=checkpoint_unchanged, frozen_source_unchanged=source_unchanged,
            external_dependencies_unchanged=external_unchanged,
            policy_and_both_normalizers_unchanged=immutable,
            formal_acceptance_passed=False, diagnostic_only=True, training_updates=0,
            endpoint_semantics="First physical terminal copied before original reset; ended lanes frozen in trajectories",
            limitations=["3s censored deterministic deployment diagnostic, not90s formal acceptance",
                "BC and PT0 each use own saved dualnorm; actor-only causal attribution is invalid",
                "Pairing requires manager comparison of saved initial physical arrays and raw observations",
                "Matching initial arrays does not certify deterministic physics evolution",
                "Normal-force/contact and published PD targets do not establish full GRF or measured motor torque"],
            artifacts=dict(initial_state="initial_state.npz", native_parameters="initial_physical_parameters.npz",
                initial_inference="first_inference.npz", pairing="pairing_manifest.json", episodes="episodes.csv",
                terminal="first_episode_terminal.npz", trajectories="trajectories.npz",
                normalizers="saved_observation_normalizers.pt", source="training_source/source"))
        if runtime_error:
            metadata["error"] = type(runtime_error).__name__ + ": " + str(runtime_error)
        write_json(args.output / "result.json", metadata)
        write_json(args.output / "run.json", metadata)
        print("[STAIR_INIT] RESULT " + json.dumps(dict(arm=args.arm, status=metadata["status"],
            first_episode_horizon_survivors=int(horizon.sum()), termination_counts=metadata["termination_counts"])), flush=True)
        if runtime_error:
            raise runtime_error
    finally:
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()


def main():
    args, AppLauncher = parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status="initializing", diagnostic_only=True, formal_acceptance_passed=False,
        training_updates=0, arm=args.arm, checkpoint=str(args.checkpoint), training_run=str(args.training_run),
        seed=args.seed, num_envs=args.num_envs, duration_limit_s=args.duration, started_at_epoch=time.time(),
        network_dimensions=[1029, 866, 73, 12], action_semantics="full_soft_tanh_position_v1",
        tool_sha256=sha256(__file__), upper_body="Original17 default-target PD and original native head drives retained")
    app = None
    try:
        _, source, hashes = prepare(args, metadata)
        write_json(args.output / "run.json", metadata)
        app = AppLauncher(args).app
        diagnose(args, app, metadata, source, hashes)
    except BaseException as error:
        metadata.update(status="failed", finished_at_epoch=time.time(), diagnostic_only=True,
            formal_acceptance_passed=False, error=type(error).__name__ + ": " + str(error))
        write_json(args.output / "run.json", metadata)
        write_json(args.output / "result.json", metadata)
        (args.output / "failure_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
