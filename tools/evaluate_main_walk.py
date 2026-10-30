"""Evaluate a frozen H2 checkpoint in Isaac Sim without updating the policy."""

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
import traceback
import types
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--terrain", choices=["flat", "mixed", "stairs"], default="flat")
parser.add_argument("--stair_height", type=float, default=0.15)
parser.add_argument("--stair_depth", type=float, default=0.35)
parser.add_argument("--stair_steps", type=int, default=6)
parser.add_argument("--stair_approach", type=float, default=1.4)
parser.add_argument("--early_reject_failures", type=int, default=0,
                    help="Development-only early rejection after this many failed first episodes; never grants acceptance.")
parser.add_argument("--video", action="store_true")
parser.add_argument("--stochastic_actions", action="store_true",
                    help="Frozen-policy control with checkpoint exploration noise; no learning.")
parser.add_argument("--hold_default_pose", action="store_true",
                    help="Diagnostic only: zero policy residual tests the configured physical controller.")
parser.add_argument("--source_root", type=Path, default=ROOT / "source",
                    help="Use a preserved source snapshot for a controlled baseline replay.")
parser.add_argument("--diagnose_contacts", action="store_true",
                    help="Diagnostic replay only: replace unfiltered contact terminations with height/orientation checks.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
args.checkpoint = args.checkpoint.resolve(strict=True)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.math import quat_apply, matrix_from_quat
import isaaclab.envs.mdp as common_mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(args.source_root.resolve() / "CtrlLab/CtrlLab/tasks/manager_based"))
import h2_task
from h2_task.networks import H2ActorCriticOur
from h2_task import locomotion_progress as progress_module
import rsl_rl.runners.on_policy_runner as rsl_runner_module

rsl_runner_module.H2ActorCriticOur = H2ActorCriticOur


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    temporary.replace(path)


def config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def main():
    started = time.time()
    spec = gym.spec("Unitree-H2-Velocity-Stairs")
    cfg = config(spec.kwargs["env_cfg_entry_point"])
    agent = config(spec.kwargs["rsl_rl_cfg_entry_point"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed = args.seed
    cfg.sim.device = agent.device = args.device or "cuda:0"
    cfg.episode_length_s = args.duration
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.observations.policy.height_scanner.params["side_noise_probability"] = 0.0
    if args.diagnose_contacts:
        cfg.terminations.base_contact = None
        cfg.terminations.fall_body_contact = None
        cfg.terminations.contact_out_of_time = None
        cfg.terminations.low_root_height = TerminationTermCfg(
            func=common_mdp.root_height_below_minimum, params={"minimum_height": 0.55})
        for label, body in [("pelvis_ground", "pelvis"), ("left_ground", "left_ankle_pitch_link"),
                            ("right_ground", "right_ankle_pitch_link")]:
            setattr(cfg.scene, label, ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + body, history_length=3,
                update_period=cfg.sim.dt, filter_prim_paths_expr=["/World/ground/terrain/mesh"]))
    if args.terrain == "flat":
        # Retain generated-terrain interfaces and the trained scan layout on a flat tile.
        from h2_task.config.terrain_cfg import FLAT_ROUGH_TERRAINS_CFG
        cfg.scene.terrain.terrain_generator = deepcopy(FLAT_ROUGH_TERRAINS_CFG)
        cfg.scene.terrain.terrain_generator.num_rows = 1
        cfg.scene.terrain.terrain_generator.num_cols = 1
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.max_init_terrain_level = 0
    elif args.terrain == "stairs":
        from h2_task.stair_terrain import straight_stair_generator
        cfg.scene.terrain.terrain_generator = straight_stair_generator(
            height=args.stair_height, depth=args.stair_depth,
            steps=args.stair_steps, approach=args.stair_approach)
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.stair_reference = {
            "requested_levels": args.stair_steps, "maximum_rise": .225,
            "maximum_first_stride": .47, "pelvis_height": .95,
            "brake_duration": 1.1, "transfer_duration": 3.0,
            "flight_duration": .50, "clearance": .08, "map_wait_timeout": 2.0,
            "preparation_clearance": .04,
            "maximum_joint_rate": 8.0, "cross_duration": .24,
            "descent_duration": .10, "swing_toe_up_rad": 0.0,
            "swing_toe_down_rad": .055,
            "top_pelvis_height": 1.0,
            "maximum_vertical_acceleration": 11.5,
            "moving_inertia_feedforward": True,
            "kinematic_preview_dt": .01,
            "stair_drive_frequency_rad_s": 28.0,
            "stair_drive_handover_s": .45,
            "entry_stride_search": .65,
            "support_medial_offset": .025,
            "joint_kinematic_reserve": .002,
            "swing_residual_scale": 0.0,
            "support_effort_priority_axes": [4],
        }
        # Each filtered view has exactly one rigid body per environment.
        # These sensors observe contacts; original termination terms stay on.
        for label,body in [("stair_left_ground","left_ankle_pitch_link"),
                           ("stair_right_ground","right_ankle_pitch_link")]:
            setattr(cfg.scene,label,ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/"+body,history_length=3,
                update_period=cfg.sim.dt,filter_prim_paths_expr=["/World/ground/terrain/mesh"],
                track_contact_points=True,max_contact_data_count_per_prim=64))
        for body in ["pelvis"]+[side+joint+"_link" for side in ("left_","right_")
                               for joint in ("hip_pitch","hip_roll","hip_yaw","knee")]:
            setattr(cfg.scene,"stair_ground_"+body,ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/"+body,history_length=3,
                update_period=cfg.sim.dt,filter_prim_paths_expr=["/World/ground/terrain/mesh"]))
    else:
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.scene.terrain.terrain_generator.curriculum = True
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1000000.0, 1000000.0)
    command.debug_vis = False
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (3.3, 3.3, 1.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    cfg.viewer.resolution = (768, 576)
    if args.terrain == "stairs":
        cfg.viewer.eye = (-2.5, 3.8, 1.8)
        cfg.viewer.lookat = (.8, 0, .6)
    cfg.io_descriptors_output_dir = str(args.output)
    dump_yaml(str(args.output / "environment.yaml"), cfg)
    dump_yaml(str(args.output / "agent.yaml"), agent)
    metadata = {
        "status": "initializing", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "num_envs": args.num_envs,
        "terrain": args.terrain, "duration_s": args.duration,
        "command_m_s": [args.speed, 0.0, 0.0], "seed": args.seed,
        "evaluation_changes": ["fixed forward velocity", "zero initial yaw", "no observation corruption",
                               "no external pushes", "no curriculum updates"],
        "startup_domain_randomization": "same as training",
        "policy_mode": ("stochastic frozen policy" if args.stochastic_actions else "deterministic inference")
                       + " with saved observation normalization",
        "training_updates": 0,
        "lock_upper_body": bool(cfg.lock_upper_body_during_training),
        "controller": {"action_class": cfg.actions.JointPositionAction.class_type.__name__,
                       "stance_preload_torques": getattr(cfg, "stance_preload_torques", None),
                       "stance_orientation_feedback": getattr(cfg, "stance_orientation_feedback", None),
                       "lateral_com_feedback": getattr(cfg, "lateral_com_feedback", None),
                       "contact_gait_reference": getattr(cfg, "contact_gait_reference", None)},
        "lateral_map_corruption": False,
        "forward_reward_function": cfg.rewards.tracking_lin_vel_x.func.__name__,
        "forward_reward_requires_progress": (
            cfg.rewards.tracking_lin_vel_x.func.__name__ == "net_forward_progress"
            or cfg.rewards.tracking_lin_vel_x.params.get("require_forward_progress", False)),
        "positive_progress_credit_capped_at_command": (
            cfg.rewards.tracking_lin_vel_x.func.__name__ == "net_forward_progress"),
        "lateral_command_limits": list(cfg.commands.base_velocity.limit_ranges.lin_vel_y),
        "diagnostic_only": args.diagnose_contacts or args.hold_default_pose,
        "control_mode": (("reference_controller_zero_residual_diagnostic" if args.hold_default_pose
                          else "reference_controller_frozen_policy_residual")
                         if getattr(cfg, "contact_gait_reference", None) is not None else
                         "default_pose_pd_diagnostic" if args.hold_default_pose else "frozen_policy"),
        "trajectory_rule": "first episode only; terminal state captured before automatic reset",
        "movement_screen": {"survive_full_horizon": True, "minimum_forward_progress_ratio": 0.8,
                            "maximum_forward_progress_ratio": 1.2,
                            "minimum_valid_alternating_landings_each_foot": 2},
        "source": str(Path(__file__).resolve()), "source_sha256": sha256(__file__),
        "task_source": str(Path(h2_task.__file__).resolve()),
        "rendering": "env0 visible; all other simulated environments hidden from video only",
        "replicate_physics": cfg.scene.replicate_physics,
    }
    if args.terrain == "stairs":
        metadata["movement_screen"] = {
            "survive_full_horizon": True, "each_level_both_feet_loaded": True,
            "levels": args.stair_steps, "minimum_platform_dwell_s": 2.0,
            "expected_net_root_rise_m": args.stair_steps * args.stair_height,
            "root_rise_tolerance_m": .10, "minimum_successful_envs": 922,
        }
        metadata["stair_geometry"] = {"height_m": args.stair_height, "depth_m": args.stair_depth,
                                       "steps": args.stair_steps, "approach_m": args.stair_approach}
        metadata["controller_perception"] = "raw finite LiDAR plus separately labelled actual loaded-foot contact polygons; at most one bounded25mm LiDAR gap model per patch; no configured stair geometry or privileged height scanner enters controller"
        metadata["controller_changed_since_flat_checkpoint_training"] = True
        metadata["stair_reference_options"] = cfg.stair_reference
        metadata["stair_route_control"] = {
            "entry":"current loaded feet plus observed higher pair; bounded whole-leg/shared-COM route screening, no invented preparation-floor cells",
            "landing_coverage":"all cells intersecting true oriented sole plus8mm; raw/tactile/existing bounded single-cell provenance, no unknown fill",
            "swing":"raw-return bounded pre-riser advance, modest low lift with bounded toe-down, actual whole-foot clearance before crossing, level before landing",
            "impedance":"post-approach smooth native hip/knee bandwidth handover and ankle-roll preload headroom; residual and hip feedback nominal torque preserved",
            "support":"2D load allocation on split-position contacts; full10%body-weight80ms contact before transfer",
            "single_support_pressure":"unboxed feedback request, then actual rotated sole8mm inset and loaded native joint-effort projection; no extra symmetric world-axis pressure limit",
            "body_height":"total COM XY with explicit pelvis Z reference; measured mass-shape offset reconstructs displayed COM Z and velocity",
            "vertical_feedforward_scope":"root-reference acceleration only; moving-base and full leg-shape vertical inertia are not compensated",
            "feedback_ownership":"flat approach retains its original IMU/hip assists; active stair reference exclusively owns body-pose IK, COM feedback, G/GRF and native PD; frozen residual retained",
            "control_pose_samples":"pre-control physical pose with explicit control_sample_step and control_sample_valid; no zero-filled approach sample is treated as a measured pose",
            "system_scope":"explicit complete reference controller plus unchanged frozen residual; PT alone is not claimed to climb",
        }
        metadata["moving_inertia_feedforward_scope"] = {
            "model":"current native moving-leg6x6 generalized mass block times kinematic-preview acceleration plus native C",
            "static_gravity_and_contact_preload":"unchanged position-bias path, added once",
            "dynamic_injection":"separate native velocity-target increment normalized by D+K*physics_dt; not an exact measured torque or TGS equivalence claim",
            "limits":"unchanged native total drive-force caps,8rad/s geometric rate,30rad/s final target cap",
            "omitted":"root and other-joint acceleration coupling; not full-body inverse dynamics",
        }
    if args.hold_default_pose:
        metadata["policy_mode"] = "zero learned residual diagnostic; checkpoint loaded for shape/provenance only"
    elif getattr(cfg, "contact_gait_reference", None) is not None:
        metadata["policy_mode"] = "explicit reference gait controller plus frozen learned residual with saved normalization"
    status_path = args.output / "result.json"
    write_status(status_path,metadata)
    print("[EVAL] Configuration: " + json.dumps(metadata), flush=True)
    env = gym.make("Unitree-H2-Velocity-Stairs", cfg=cfg, render_mode="rgb_array" if args.video else None)
    video = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        raw = env.unwrapped
        reference_controller = getattr(raw.action_manager.get_term("JointPositionAction"), "gait_reference", None)
        obs, extra = wrapped.get_observations()
        shapes = {key: list(value.shape) for key, value in extra["observations"].items()}
        assert obs.shape == (args.num_envs, 1029), obs.shape
        assert wrapped.num_actions in (12, 29), wrapped.num_actions
        assert shapes["critic"][-1] == 866 and shapes["rnd_state"][-1] == 73, shapes
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=raw.device)
        assert not runner.alg.policy.training and not runner.obs_normalizer.training
        robot = raw.scene["robot"]
        if args.video:
            # Terrain environments may share an origin. Collision filtering isolates
            # their physics; hide the other clones only to obtain an env0 video.
            from pxr import UsdGeom
            for env_index in range(1, args.num_envs):
                prim = raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}")
                UsdGeom.Imageable(prim).MakeInvisible()
        sensor = raw.scene["contact_forces"]
        from h2_task.assets import H2_LEG_JOINT_NAMES
        action_term = raw.action_manager.get_term("JointPositionAction")
        action_names = list(action_term._joint_names)
        leg_joint_ids = robot.find_joints(H2_LEG_JOINT_NAMES, preserve_order=True)[0]
        metadata["runtime_joint_action_mapping"] = {
            "articulation_joint_names": list(robot.joint_names),
            "action_joint_names": action_names,
            "action_to_articulation_ids": list(action_term._joint_ids),
            "leg_articulation_ids": list(leg_joint_ids),
            "leg_action_ids_by_name": [action_names.index(name) for name in H2_LEG_JOINT_NAMES],
            "locked_action_ids": list(raw.action_manager._locked_action_ids),
            "held_joint_ids": list(getattr(raw.action_manager, "held_joint_ids", [])),
        }
        held_ids = metadata["runtime_joint_action_mapping"]["held_joint_ids"]
        metadata["maximum_upper_body_target_error_rad"] = 0.0
        metadata["policy_std"] = runner.alg.policy.std.detach().cpu().tolist()
        drive_stiffness = robot.root_physx_view.get_dof_stiffnesses().to(raw.device)
        drive_damping = robot.root_physx_view.get_dof_dampings().to(raw.device)
        actuator_contract = {}
        for name, actuator in robot.actuators.items():
            ids = actuator.joint_indices
            implicit = bool(actuator.is_implicit_model)
            actuator_contract[name] = {
                "class": type(actuator).__name__, "solver_integrated": implicit,
                "joint_names": [robot.joint_names[i] for i in
                                (range(robot.num_joints) if isinstance(ids, slice) else ids)],
                "stiffness_median": actuator.stiffness.median(0).values.cpu().tolist(),
                "damping_median": actuator.damping.median(0).values.cpu().tolist(),
                "effort_limits": actuator.effort_limit[0].cpu().tolist(),
                "configured_command_delay_physics_steps": [getattr(actuator.cfg, "min_delay", 0),
                                                            getattr(actuator.cfg, "max_delay", 0)],
            }
            if implicit:
                k_error = float((drive_stiffness[:, ids] - actuator.stiffness).abs().max())
                d_error = float((drive_damping[:, ids] - actuator.damping).abs().max())
                actuator_contract[name].update(solver_stiffness_error=k_error, solver_damping_error=d_error)
                assert k_error < 1e-3 and d_error < 1e-3, actuator_contract[name]
            if hasattr(actuator, "positions_delay_buffer"):
                actuator_contract[name]["sampled_delay_range"] = [
                    int(actuator.positions_delay_buffer.time_lags.min()),
                    int(actuator.positions_delay_buffer.time_lags.max())]
        metadata["runtime_actuator_contract"] = actuator_contract
        mass_matrices = robot.root_physx_view.get_generalized_mass_matrices()
        np.savez_compressed(args.output / "initial_articulation_mass_matrices.npz",
                            indices=np.array([0, args.num_envs // 2, args.num_envs - 1]),
                            mass_matrix=mass_matrices[[0, args.num_envs // 2, args.num_envs - 1]].cpu().numpy(),
                            joint_names=np.array(robot.joint_names),
                            armature=robot.data.joint_armature[0].cpu().numpy())
        feet, foot_names = sensor.find_bodies(raw.cfg.feet_body_names, preserve_order=True)
        robot_feet, _ = robot.find_bodies(raw.cfg.feet_body_names, preserve_order=True)
        pelvis_ids, _ = sensor.find_bodies("pelvis")
        n, device = args.num_envs, raw.device
        alive = torch.ones(n, dtype=torch.bool, device=device)
        elapsed = torch.zeros(n, device=device)
        start_pos = robot.data.root_pos_w.clone()
        end_pos = start_pos.clone()
        end_vel = robot.data.root_lin_vel_w.clone()
        end_quat = robot.data.root_quat_w.clone()
        end_feet = robot.data.body_pos_w[:, robot_feet, :].clone()
        end_foot_quat = robot.data.body_quat_w[:, robot_feet, :].clone()
        end_joints = robot.data.joint_pos.clone()
        end_joint_target = robot.data.joint_pos_target.clone()
        end_contact = sensor.data.net_forces_w[:, feet, :].clone()
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        timeout = torch.zeros_like(failed)
        reasons = {name: torch.zeros_like(failed) for name in raw.termination_manager.active_terms}
        min_height = torch.full((n,), float("inf"), device=device)
        max_tilt = torch.zeros(n, device=device)
        velocity_error_sum = torch.zeros(n, device=device)
        samples = torch.zeros(n, device=device)
        touchdowns = torch.zeros((n, 2), dtype=torch.int32, device=device)
        if not hasattr(progress_module, "FootfallTracker"):
            # Old checkpoints retain their physical source. Apply the current
            # acceptance-event definition, recorded independently of that source.
            helper_path = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/locomotion_progress.py"
            helper_spec = importlib.util.spec_from_file_location("_h2_acceptance_events", helper_path)
            helper_module = importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(helper_module)
        else:
            helper_module = progress_module
        footfalls = helper_module.FootfallTracker(n, device)
        stair_tracker = None
        terminal_safe_cells = torch.zeros(n, device=device, dtype=torch.long)
        terminal_real_hits = torch.zeros_like(terminal_safe_cells)
        if args.terrain == "stairs":
            from h2_task.stair_acceptance import StairAcceptance
            stair_tracker = StairAcceptance(
                n, device, raw.scene.env_origins, raw.masses,
                height=args.stair_height, depth=args.stair_depth,
                steps=args.stair_steps, approach=args.stair_approach)
            stair_tracker.root_start = start_pos.clone()
        metadata["landing_metric_source"] = {"path": str(Path(helper_module.__file__).resolve()),
                                              "sha256": sha256(helper_module.__file__),
                                              "rule": "alternating, 80-600ms flight, other foot already supports, initial contact and hops excluded"}
        last_contact = torch.linalg.vector_norm(sensor.data.net_forces_w[:, feet, :], dim=-1) > 5.0
        history = []
        controller_history = []
        footfall_event_chunks = []
        controller_keys = ("state", "steps", "failure", "ik_error", "reference_com",
                           "reference_velocity", "load", "support_fraction", "capture_error",
                           "actual_com", "actual_velocity", "reference_cop", "contact_points",
                           "feedforward_torque", "plan_violation", "foot_target", "time", "stance",
                           "swing_time", "swing_target", "q", "qdot", "touchdown_count", "landing_margin")
        if stair_tracker is not None:
            controller_keys += ("levels_completed", "anchors", "pair_targets", "approach", "leading", "bottom_aligned", "preparing",
                                "clearance_passed", "cross_elapsed", "descend_time", "ik_solver_error", "rate_limit_count")
            controller_keys += ("descent_elapsed", "processed_residual")
            controller_keys += ("foot_yaw", "swing_yaw_goal", "yaw_checked_time")
            controller_keys += (("contact_age","contact_pitch")
                                +reference_controller.CONTACT_HANDOVER_FIELDS+reference_controller.IK_COM_X_FIELDS+reference_controller.FAILURE_DIAGNOSTIC_FIELDS)
            controller_keys += ("preparation_world", "first_pair_world", "entry_plan_diagnostics",
                                "approach_speed", "approach_obstacle_x", "approach_block_reason")
            controller_keys += ("pressure_centers","pressure_radii","feedforward_effort_ratio","swing_ik_diagnostics",
                                "segment_c2","projected_contact_feasible","pressure_projection_shift")
            controller_keys += ("requested_cop","requested_acceleration_local","pressure_seed_box_applied")
            controller_keys += ("body_z_offset","body_plan_position","body_plan_velocity","body_plan_acceleration")
            controller_keys += ("legacy_feedback_support_force", "control_root_position", "control_root_quaternion",
                                "control_root_link_velocity", "control_root_angular_velocity", "control_joint_position",
                                "control_joint_velocity", "control_soles", "control_foot_quaternion", "control_sample_step",
                                "control_sample_valid", "control_frame_origin", "control_frame_heading")
            controller_keys += ("dynamic_qddot","dynamic_qdot_preview","dynamic_torque_raw","dynamic_torque_applied",
                                "dynamic_inertia_torque","dynamic_coriolis_torque","dynamic_velocity_bias",
                                "dynamic_velocity_target","dynamic_mass_diagonal","dynamic_valid","dynamic_diagnostics",
                                "dynamic_peak_raw","dynamic_peak_applied","dynamic_applied_steps","dynamic_clip_steps",
                                "drive_velocity_target","cross_advancing","descent_advancing")
            controller_keys += ("drive_stiffness","drive_damping","drive_gain_blend")
            controller_keys += ("support_drive_role", "support_pose_pd_torque", "support_effort_requested",
                                "support_effort_target", "support_effort_expression", "support_drive_velocity",
                                "support_position_clipped", "support_velocity_clipped", "support_effort_clipped")
            controller_keys += (reference_controller.RECEIVER_FLOAT_FIELDS+
                                reference_controller.RECEIVER_BOOL_FIELDS+reference_controller.RECEIVER_INT_FIELDS)
            controller_keys += ("entry_direct","entry_leading","entry_root_height","entry_foot_yaw","entry_kinematics")
            controller_keys += ("support_backshift","lift_target","lift_boundary","lift_boundary_observed","lift_source_evidence")
        controller_terminal = ({key: getattr(reference_controller, key).clone() for key in controller_keys}
                               if reference_controller is not None else {})
        contact_diagnostic = {"unfiltered_pelvis_over_1N_steps": torch.zeros(n, device=device),
                              "max_unfiltered_pelvis_force_N": torch.zeros(n, device=device)}
        filtered_ground_labels = ([label for label in raw.scene.sensors if label.startswith("stair_ground_")]
                                  if stair_tracker is not None else [])
        metadata["filtered_nonfoot_ground_sensor_bodies"] = [label.removeprefix("stair_ground_") for label in filtered_ground_labels]
        metadata["unfiltered_contact_body_names"] = sensor.body_names
        metadata["measured_joint_effort_scope"] = "PhysX projected net active-axis transmission, includes drive/friction/limits; not isolated motor drive torque"
        if stair_tracker is not None:
            registration = {}
            for label in ["stair_left_ground","stair_right_ground"]+filtered_ground_labels:
                view = raw.scene[label].contact_physx_view
                registration[label] = {
                    "sensor_count":view.sensor_count,"filter_count":view.filter_count,
                    "sensor_paths_first_last":[view.sensor_paths[0],view.sensor_paths[-1]],
                    "filter_paths":view.filter_paths,
                }
            (args.output/"contact_sensor_registration.json").write_text(json.dumps(registration,indent=2)+'\n')
            metadata["pressure_allocation"] = "Two-dimensional real-foot load distribution, consistent bounded COM/CoP references and exact actual-load/actual-joint torque plus actual-sole8mm pressure projection. Native torque caps unchanged."
        contact_body_max = torch.zeros(n,len(sensor.body_names),device=device)
        contact_body_duration = torch.zeros_like(contact_body_max)
        ground_body_max = torch.zeros(n,len(filtered_ground_labels),device=device)
        end_net_effort = torch.zeros_like(robot.data.joint_pos)
        end_body_force = torch.zeros_like(sensor.data.net_forces_w)
        end_ground_force = torch.zeros_like(ground_body_max)
        tactile_terminal = ({key:value.clone() for key,value in reference_controller.map.tactile.latest.items()}
                            if stair_tracker is not None else {})
        tactile_stats_terminal = ({key:value.clone() for key,value in reference_controller.map.tactile.stats.items()}
                                  if stair_tracker is not None else {})
        if stair_tracker is not None:
            np.savez_compressed(args.output/"runtime_physical_parameters.npz",
                body_mass=robot.root_physx_view.get_masses().cpu().numpy(),
                joint_stiffness=robot.data.joint_stiffness.cpu().numpy(),
                joint_damping=robot.data.joint_damping.cpu().numpy(),
                joint_effort_limit=robot.data.joint_effort_limits.cpu().numpy(),
                joint_velocity_limit=robot.data.joint_vel_limits.cpu().numpy(),
                joint_friction_coeff=robot.data.joint_friction_coeff.cpu().numpy(),
                joint_dynamic_friction_coeff=robot.data.joint_dynamic_friction_coeff.cpu().numpy(),
                joint_viscous_friction_coeff=robot.data.joint_viscous_friction_coeff.cpu().numpy(),
                joint_names=np.asarray(robot.joint_names),body_names=np.asarray(robot.body_names),
                leg_joint_ids=np.asarray(leg_joint_ids),sensor_body_names=np.asarray(sensor.body_names))
        if args.diagnose_contacts:
            for label in ["pelvis_ground", "left_ground", "right_ground"]:
                contact_diagnostic["max_" + label + "_force_N"] = torch.zeros(n, device=device)
        feet_min_z = torch.full((n, 2), float("inf"), device=device)
        feet_max_z = torch.full((n, 2), -float("inf"), device=device)
        entry_evidence = None
        wrench_reader = None
        wrench_history = []
        wrench_capture_errors = []
        last_wrench_control_step = -1
        wrench_start_control_step = int(raw.common_step_counter)
        wrench_max_steps = math.ceil(args.duration / raw.step_dt)
        if stair_tracker is not None:
            from stair_entry_evidence import install_entry_evidence
            from stair_wrench_evidence import build_foot_wrench_reader
            entry_evidence = install_entry_evidence(reference_controller,raw,robot)
            wrench_reader = build_foot_wrench_reader(raw, robot)
            incoming_body_names = ["pelvis"]+[robot.joint_names[i].replace("_joint", "_link") for i in leg_joint_ids]
            incoming_body_ids, incoming_body_names = robot.find_bodies(incoming_body_names, preserve_order=True)
            if len(incoming_body_ids) != 13:
                raise ValueError("Need pelvis and all12 leg links for incoming-wrench diagnostics")
            incoming_probe = robot.root_physx_view.get_link_incoming_joint_force()
            if incoming_probe.shape != (n, len(robot.body_names), 6) or not bool(torch.isfinite(incoming_probe).all()):
                raise ValueError("Native incoming joint-wrench getter failed startup layout/finite check")
            metadata["native_support_effort_receiver"] = {
                "mode": "original delayed effort-role packet; disjoint native implicit PD and explicit force budgets",
                "axes": ["loaded ankle roll"], "original_delay_physics_steps": [0, 6],
                "external_force_port_duplicates_priority_effort": False,
                "motor_budget": "native drive cap=(1-r_eff)*original limit; explicit force<=r_eff*original limit; sum<=original19Nm on loaded roll; ordinary drive-only cap semantics confirmed for this USD",
                "native_gains": "nominal planning/cache K/D unchanged; solver gains=(1-r_eff)*nominal, written and read back before physics; posture remains implicit",
                "soft_boundary_guard": "actual or one5ms-step-predicted crossing; guard/release/recovery2/4/6mrad inside original soft interval; implicit full-gain recovery with external0, latched until inward release",
                "receiver_counter_scope": "physics_step names the receiving step; current_q/qd precede it; wrench data measures its endpoint",
                "PD_expression_scope": "effective native-gain request only; explicit force command is separate; neither native PD expression nor transmission is a measured isolated motor drive output",
                "incoming_joint_wrench_body_names": incoming_body_names,
                "incoming_joint_wrench_startup_layout_verified": True,
                "incoming_joint_wrench_scope": "raw get_link_incoming_joint_force; Tensor API calls its frame child-joint while ArticulationData doc says parent-body; no transform/isolated-motor interpretation validated; nonzero TGS velocity iterations have a known reporting limitation",
                "old_host_encoding_fields": "role and base effort are command payloads; old host position/velocity encoding fields are unused",
                "ordinary_leg_receiver_recording": "all12 lower-body joints record their original delayed position/velocity/effort ports and inputq/qd; ordinary K/D are nominal caches separately checked by the impedance writer, not a fresh per-sample native getter",
                "publication_epoch": "a read-only integer buffer shares original sampled lag/call count, without new RNG or command changes; native_command_published_control_step is the actual process_actions epoch, invalid=-1 after reset. Only when native_command_pre_physics is true and publication>=0: receiver-input age=(receiving_physics_step-1-decimation*publication_control_step)*physics_dt. Receiver common_step_counter is also recorded; reset/forward extra writes are marked nonphysics and have no exact age inferred from this formula",
            }
            metadata["ik_pelvis_x_admittance"] = {
                "scope": "virtual pelvis IK-X reference only; nominal DCM p/v/a, projected CoP, GRF, swing-foot world trajectory and policy input layout unchanged",
                "measurement": "current physically measured loaded-foot normal CoP on its measured plane; established foot positive control and current full-force/finite/plane/source checks; another foot full normal-force norm below5N; never interpreted as a noncoplanar all-contact ZMP",
                "observer_not_map": "one or two loaded points may define measured pressure but do not establish a support polygon or any map/acceptance evidence; original support and stair acceptance gates unchanged",
                "state": "critical-damped exact held-input update; omega_com=sqrt(sqrt(vertical/height)*existing28rad_s_native_reference); <=30mm offset; unavailable signal/nonSS supplies zero input with smooth leakage",
                "feasibility": "proposal and previous offset checked with actual rate-limited IK, original soft/preload margins and rotation; per-control-counter commit; no extra velocity injection; bound and CPU checks are not physical safety or stair acceptance proof",
                "network_structure_changed": False,
            }
            metadata["foot_wrench_evidence"] = {
                "source_sha256": sha256(ROOT / "tools/stair_wrench_evidence.py"),
                "sample_period_s": 5 * raw.step_dt,
                "force_window_s": raw.cfg.sim.dt,
                "reset_alignment": "capture before any reset; otherwise immediately after the completed control step",
                "friction_physics_validated": False,
                "scope": "diagnostic only; independent original stair acceptance is unchanged",
            }

        def capture_wrench_if_due(stage):
            nonlocal last_wrench_control_step
            if wrench_reader is None:
                return
            control_step = int(raw.common_step_counter)
            relative_step = control_step - wrench_start_control_step
            if control_step == last_wrench_control_step or relative_step <= 0:
                return
            if (relative_step - 1) % 5 != 0 and relative_step != wrench_max_steps:
                return
            # One global sample before the first reset in this control step
            # retains the genuine terminal contact/pose of every environment.
            last_wrench_control_step = control_step
            foot_quat = robot.data.body_quat_w[:, robot_feet].clone()
            sole = robot.data.body_pos_w[:, robot_feet].clone() + quat_apply(
                foot_quat, reference_controller.kin.sole_offset.expand(n, -1, -1))
            root_pos = robot.data.root_pos_w.clone()
            root_quat = robot.data.root_quat_w.clone()
            try:
                sample = wrench_reader.capture(
                    sole, foot_quat, physics_step_index=raw._sim_step_counter,
                    control_step_index=control_step,
                    root_quat_w=root_quat,
                    no_push_window_confirmed=raw.cfg.events.push_robot is None)
                sample.update(time_s=np.float64(relative_step * raw.step_dt),
                              first_episode_active=alive.cpu().numpy().copy(),
                              root_pos_w_m=root_pos.cpu().numpy().copy(),
                              root_quat_w_wxyz=root_quat.cpu().numpy().copy(),
                              root_link_velocity_w_m_s=robot.data.root_link_lin_vel_w.cpu().numpy().copy(),
                              root_angular_velocity_w_rad_s=robot.data.root_ang_vel_w.cpu().numpy().copy(),
                              joint_position12=robot.data.joint_pos[:, leg_joint_ids].cpu().numpy().copy(),
                              joint_velocity12=robot.data.joint_vel[:, leg_joint_ids].cpu().numpy().copy(),
                              capture_stage=np.asarray(stage))
                incoming = robot.root_physx_view.get_link_incoming_joint_force().clone()
                if incoming.shape != (n, len(robot.body_names), 6):
                    raise ValueError("Unexpected native incoming joint wrench layout")
                sample.update(
                    incoming_joint_wrench_native_frame=incoming[:,incoming_body_ids].cpu().numpy().copy(),
                    incoming_joint_wrench_finite=torch.isfinite(incoming[:,incoming_body_ids]).all(dim=(-1,-2)).cpu().numpy(),
                    incoming_body_pos_w_m=robot.data.body_pos_w[:,incoming_body_ids].cpu().numpy().copy(),
                    incoming_body_quat_w_wxyz=robot.data.body_quat_w[:,incoming_body_ids].cpu().numpy().copy())
                for key in (reference_controller.RECEIVER_FLOAT_FIELDS+reference_controller.RECEIVER_BOOL_FIELDS
                            +reference_controller.RECEIVER_INT_FIELDS+reference_controller.CONTACT_HANDOVER_FIELDS+reference_controller.IK_COM_X_FIELDS+reference_controller.FAILURE_DIAGNOSTIC_FIELDS):
                    sample[key] = getattr(reference_controller,key).cpu().numpy().copy()
                wrench_history.append(sample)
            except Exception as error:
                from stair_wrench_evidence import _raise_if_fatal
                _raise_if_fatal(error)
                wrench_capture_errors.append({"control_step": control_step,
                                              "reason": type(error).__name__ + ": " + str(error)})
        original_reset = raw._reset_idx

        def capture_before_reset(self, env_ids):
            capture_wrench_if_due("before_reset")
            ids = env_ids[alive[env_ids]]
            if ids.numel():
                if reference_controller is not None:
                    for key in controller_keys:
                        controller_terminal[key][ids] = getattr(reference_controller, key)[ids]
                end_pos[ids] = robot.data.root_pos_w[ids]
                end_vel[ids] = robot.data.root_lin_vel_w[ids]
                end_quat[ids] = robot.data.root_quat_w[ids]
                end_feet[ids] = robot.data.body_pos_w[ids][:, robot_feet, :]
                end_foot_quat[ids] = robot.data.body_quat_w[ids][:, robot_feet, :]
                end_joints[ids] = robot.data.joint_pos[ids]
                end_joint_target[ids] = robot.data.joint_pos_target[ids]
                end_contact[ids] = sensor.data.net_forces_w[ids][:, feet, :]
                if stair_tracker is not None:
                    for key,value in reference_controller.map.tactile.latest.items():
                        tactile_terminal[key][ids] = value[ids]
                    for key,value in reference_controller.map.tactile.stats.items():
                        tactile_stats_terminal[key][ids] = value[ids]
                    end_net_effort[ids] = robot.root_physx_view.get_dof_projected_joint_forces()[ids]
                    end_body_force[ids] = sensor.data.net_forces_w[ids]
                    end_ground_force[ids] = torch.stack([self.scene[label].data.force_matrix_w.norm(dim=-1).flatten(1).sum(1)
                                                        for label in filtered_ground_labels],-1)[ids]
                    contact_body_max[ids] = torch.maximum(contact_body_max[ids],end_body_force[ids].norm(dim=-1))
                    contact_body_duration[ids] = torch.maximum(contact_body_duration[ids],sensor.data.current_contact_time[ids])
                    ground_body_max[ids] = torch.maximum(ground_body_max[ids],end_ground_force[ids])
                    terminal_safe_cells[ids] = reference_controller.map.safe[ids].sum(-1)
                    terminal_real_hits[ids] = reference_controller.map.real_hit_count[ids]
                    if (ids == 0).any():
                        mapping = reference_controller.map
                        np.savez_compressed(args.output / "perception_env0_first_terminal.npz",
                                            minimum=mapping.minimum[0].cpu().numpy(), maximum=mapping.maximum[0].cpu().numpy(),
                                            safe=mapping.safe[0].cpu().numpy(), x=mapping.x.cpu().numpy(), y=mapping.y.cpu().numpy(),
                                            plane_x=mapping.plane_x.cpu().numpy(), first_seen=mapping.first_seen[0].cpu().numpy(),
                                            last_seen=mapping.last_seen[0].cpu().numpy(), clock=mapping.clock,
                                            modelled=mapping.modelled[0].cpu().numpy(),model_height=mapping.model_height[0].cpu().numpy(),
                                            tactile_known=mapping.tactile_known[0].cpu().numpy(),
                                            tactile_height=mapping.tactile_height[0].cpu().numpy(),
                                            tactile_invalid=mapping.tactile_invalid[0].cpu().numpy(),
                                            nx=mapping.nx, ny=mapping.ny, env_origin=self.scene.env_origins[0].cpu().numpy())
                elapsed[ids] = self.episode_length_buf[ids] * self.step_dt
                failed[ids] = self.reset_terminated[ids]
                timeout[ids] = self.reset_time_outs[ids]
                min_height[ids] = torch.minimum(min_height[ids], robot.data.root_pos_w[ids, 2] - self.scene.env_origins[ids, 2])
                terminal_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[ids, 2], -1.0, 1.0))
                max_tilt[ids] = torch.maximum(max_tilt[ids], terminal_tilt)
                for name, values in reasons.items():
                    values[ids] = self.termination_manager.get_term(name)[ids]
                alive[ids] = False
            original_reset(env_ids)

        raw._reset_idx = types.MethodType(capture_before_reset, raw)
        metadata.update(status="evaluating", observation_shapes=shapes,
                        actions=wrapped.num_actions, step_dt_s=raw.step_dt,
                        checkpoint_iteration=int(runner.current_learning_iteration), foot_bodies=foot_names)
        write_status(status_path,metadata)
        print("[EVAL] JOINT_ACTION_MAPPING: " + json.dumps(metadata["runtime_joint_action_mapping"]), flush=True)
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw
            video = imageio.get_writer(str(args.output / "env0.mp4"), fps=10, codec="libx264", quality=7)
        max_steps = math.ceil(args.duration / raw.step_dt)
        for step in range(max_steps):
            if not simulation_app.is_running():
                raise RuntimeError("Simulator stopped before evaluation completed")
            was_alive = alive.clone()
            with torch.inference_mode():
                actions = (torch.zeros((args.num_envs, wrapped.num_actions), device=raw.device)
                           if args.hold_default_pose else
                           runner.alg.policy.act(runner.obs_normalizer(obs))
                           if args.stochastic_actions else policy(obs))
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite policy actions")
                obs, reward, done, info = wrapped.step(actions)
                capture_wrench_if_due("post_step_no_reset")
                if stair_tracker is not None and not metadata.get("stair_impedance_runtime_verified",False):
                    proof = reference_controller.impedance.first_verification
                    if proof is not None:
                        metadata["stair_impedance_runtime_verified"] = True
                        metadata["stair_impedance_runtime"] = proof
                        (args.output/"stair_impedance_runtime_verification.json").write_text(
                            json.dumps(proof,indent=2)+'\n')
                        write_status(status_path,metadata)
                        print("[EVAL] STAIR_IMPEDANCE_RUNTIME: "+json.dumps(proof),flush=True)
                if stair_tracker is not None and not metadata.get("moving_dynamics_runtime_verified",False):
                    active_dynamic = (reference_controller.dynamic_valid&alive[:,None]).nonzero()
                    if active_dynamic.numel():
                        env_id,side = map(int,active_dynamic[0].cpu().tolist())
                        gait = reference_controller
                        metadata["moving_dynamics_runtime_verified"] = True
                        metadata["first_moving_dynamics"] = {
                            "env_id":env_id,"side":side,"time_s":(step+1)*raw.step_dt,
                            **{key:getattr(gait,key)[env_id,side].cpu().tolist() for key in
                               ("dynamic_mass_diagonal","dynamic_qddot","dynamic_coriolis_torque",
                                "dynamic_torque_raw","dynamic_torque_applied","dynamic_velocity_bias",
                                "qdot","drive_velocity_target","feedforward_torque")}}
                        write_status(status_path,metadata)
                        print("[EVAL] MOVING_DYNAMICS_RUNTIME: "+json.dumps(metadata["first_moving_dynamics"]),flush=True)
                if stair_tracker is not None and not metadata.get("foot_contact_positive_control_confirmed",False):
                    proof = reference_controller.map.tactile.first_foot_positive_control
                    if all(item is not None for item in proof):
                        metadata["foot_contact_positive_control_confirmed"] = True
                        metadata["first_foot_positive_control"] = proof
                        write_status(status_path,metadata)
                        print("[EVAL] FOOT_CONTACT_POSITIVE_CONTROL: "+json.dumps(proof),flush=True)
                if held_ids:
                    hold_error = float((robot.data.joint_pos_target[:, held_ids]
                                        - robot.data.default_joint_pos[:, held_ids]).abs().max())
                    metadata["maximum_upper_body_target_error_rad"] = max(
                        metadata["maximum_upper_body_target_error_rad"], hold_error)
                    if hold_error > 1e-6:
                        raise RuntimeError("Upper-body position targets differ from the locked ready pose")
                if not torch.isfinite(obs).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite simulation observations or rewards")
                position = robot.data.root_pos_w.clone()
                velocity = robot.data.root_lin_vel_w.clone()
                quaternion = robot.data.root_quat_w.clone()
                feet_position = robot.data.body_pos_w[:, robot_feet, :].clone()
                feet_quaternion = robot.data.body_quat_w[:, robot_feet, :].clone()
                joint_position = robot.data.joint_pos.clone()
                joint_target = robot.data.joint_pos_target.clone()
                contact_force = sensor.data.net_forces_w[:, feet, :].clone()
                if stair_tracker is not None:
                    body_force = sensor.data.net_forces_w.norm(dim=-1)
                    contact_body_max[alive] = torch.maximum(contact_body_max[alive],body_force[alive])
                    contact_body_duration[alive] = torch.maximum(contact_body_duration[alive],sensor.data.current_contact_time[alive])
                    ground_forces = torch.stack([raw.scene[label].data.force_matrix_w.norm(dim=-1).flatten(1).sum(1)
                                                 for label in filtered_ground_labels],-1)
                    ground_body_max[alive] = torch.maximum(ground_body_max[alive],ground_forces[alive])
                just_done = was_alive & ~alive
                position[~alive] = end_pos[~alive]
                velocity[~alive] = end_vel[~alive]
                quaternion[~alive] = end_quat[~alive]
                feet_position[~alive] = end_feet[~alive]
                feet_quaternion[~alive] = end_foot_quat[~alive]
                joint_position[~alive] = end_joints[~alive]
                joint_target[~alive] = end_joint_target[~alive]
                contact_force[~alive] = end_contact[~alive]
                end_pos[alive] = position[alive]
                end_vel[alive] = velocity[alive]
                elapsed[alive] = (step + 1) * raw.step_dt
                min_height[was_alive] = torch.minimum(min_height[was_alive], position[was_alive, 2] - raw.scene.env_origins[was_alive, 2])
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
                max_tilt[alive] = torch.maximum(max_tilt[alive], tilt[alive])
                velocity_error_sum[was_alive] += torch.linalg.vector_norm(velocity[was_alive, :2] - torch.tensor([args.speed, 0.0], device=device), dim=-1)
                samples[was_alive] += 1
                contact = torch.linalg.vector_norm(contact_force, dim=-1) > 5.0
                touchdowns[alive] += (contact[alive] & ~last_contact[alive]).int()
                last_contact = contact
                if stair_tracker is not None:
                    previous_air = footfalls.air_time.clone()
                    previous_foot_contact = footfalls.previous_contact.clone()
                    previous_seen = footfalls.seen_contact.clone()
                    previous_landing = footfalls.last_landing.clone()
                    previous_counts = footfalls.landings.clone()
                footfalls.update(contact, raw.step_dt, was_alive.nonzero(as_tuple=False).flatten())
                if stair_tracker is not None:
                    soles = feet_position + quat_apply(
                        feet_quaternion, reference_controller.kin.sole_offset.expand(n, -1, -1))
                    touched = contact & ~previous_foot_contact & was_alive[:,None]
                    event_ids,event_feet = touched.nonzero(as_tuple=True)
                    if event_ids.numel():
                        other_support = contact.flip(-1)&previous_foot_contact.flip(-1)
                        short = previous_air < .08-1e-6
                        long = previous_air > .60+1e-6
                        same = previous_landing[:,None]==torch.arange(2,device=device)[None]
                        reason = ((~previous_seen).int()+2*(~other_support).int()
                                  +4*short.int()+8*long.int()+16*same.int())
                        accepted = footfalls.landings>previous_counts
                        footfall_event_chunks.append({
                            "env_id":event_ids.cpu().numpy(),"foot":event_feet.cpu().numpy(),
                            "time_s":np.full(event_ids.numel(),(step+1)*raw.step_dt),
                            "tracker_air_duration_s":previous_air[event_ids,event_feet].cpu().numpy(),
                            "valid_alternating_landing":accepted[event_ids,event_feet].cpu().numpy(),
                            "rejection_bits":reason[event_ids,event_feet].cpu().numpy(),
                            "sole_position_w":soles[event_ids,event_feet].cpu().numpy(),
                            "contact_force_w":contact_force[event_ids,event_feet].cpu().numpy(),
                            "controller_state":torch.where(alive,reference_controller.state,controller_terminal["state"])[event_ids].cpu().numpy(),
                            "controller_phase_time_s":torch.where(alive,reference_controller.time,controller_terminal["time"])[event_ids].cpu().numpy()})
                    root_tilt = torch.acos(matrix_from_quat(quaternion)[:, 2, 2].clamp(-1, 1))
                    stair_tracker.update(position, soles, matrix_from_quat(feet_quaternion), contact_force,
                                         footfalls.landings, raw.step_dt, was_alive, root_tilt)
                pelvis_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, :, pelvis_ids, :], dim=-1).amax(dim=(1, 2))
                contact_diagnostic["unfiltered_pelvis_over_1N_steps"][alive] += (pelvis_force[alive] > 1.0).float()
                contact_diagnostic["max_unfiltered_pelvis_force_N"][alive] = torch.maximum(
                    contact_diagnostic["max_unfiltered_pelvis_force_N"][alive], pelvis_force[alive])
                if args.diagnose_contacts:
                    for label in ["pelvis_ground", "left_ground", "right_ground"]:
                        force = torch.linalg.vector_norm(raw.scene[label].data.force_matrix_w_history, dim=-1).flatten(1).amax(dim=1)
                        contact_diagnostic["max_" + label + "_force_N"][alive] = torch.maximum(
                            contact_diagnostic["max_" + label + "_force_N"][alive], force[alive])
                feet_z = robot.data.body_pos_w[:, robot_feet, 2] - raw.scene.env_origins[:, None, 2]
                feet_min_z[alive] = torch.minimum(feet_min_z[alive], feet_z[alive])
                feet_max_z[alive] = torch.maximum(feet_max_z[alive], feet_z[alive])
            if step % 5 == 0 or step == max_steps - 1:
                if reference_controller is not None:
                    controller_point = {"time_s": (step + 1) * raw.step_dt}
                    for key in controller_keys:
                        value = getattr(reference_controller, key).clone()
                        value[~alive] = controller_terminal[key][~alive]
                        controller_point[key] = value.cpu().numpy()
                    controller_history.append(controller_point)
                history.append({"time_s": (step + 1) * raw.step_dt,
                                "position": position.cpu().numpy(), "velocity": velocity.cpu().numpy(),
                                "quaternion": quaternion.cpu().numpy(),
                                "joint_position": joint_position.cpu().numpy(),
                                "joint_target": joint_target.cpu().numpy(), "policy_action": actions.cpu().numpy(),
                                "feet_position": feet_position.cpu().numpy(),
                                "feet_quaternion": feet_quaternion.cpu().numpy(),
                                "feet_contact_force": contact_force.cpu().numpy(),
                                "feet_ground_forces": (torch.stack([torch.linalg.vector_norm(raw.scene[label].data.force_matrix_w, dim=-1).flatten(1).sum(1)
                                                                     for label in ["left_ground", "right_ground"]], dim=1).cpu().numpy()
                                                       if args.diagnose_contacts else np.zeros((n, 2))),
                                "first_episode_active": was_alive.cpu().numpy()})
                if stair_tracker is not None:
                    net_effort = robot.root_physx_view.get_dof_projected_joint_forces().clone()
                    body_force = sensor.data.net_forces_w.clone()
                    ground_forces = ground_forces.clone()
                    net_effort[~alive],body_force[~alive],ground_forces[~alive] = end_net_effort[~alive],end_body_force[~alive],end_ground_force[~alive]
                    history[-1]["projected_net_joint_force"] = net_effort.cpu().numpy()
                    history[-1]["unfiltered_body_contact_force"] = body_force.cpu().numpy()
                    history[-1]["ground_filtered_body_force"] = ground_forces.cpu().numpy()
                    history[-1]["tactile_cell_count"] = reference_controller.map.tactile_known.sum(-1).cpu().numpy()
                    history[-1]["tactile_contact_age"] = reference_controller.map.tactile.age.cpu().numpy()
                    for key,value in reference_controller.map.tactile.latest.items():
                        value = value.clone()
                        value[~alive] = tactile_terminal[key][~alive]
                        history[-1]["tactile_"+key] = value.cpu().numpy()
                if video is not None:
                    frame = Image.fromarray(raw.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 768, 44), fill=(0, 0, 0))
                    draw.text((8, 5), f"H2 frozen checkpoint | {args.terrain} | command {args.speed:.2f} m/s | t={(step+1)*raw.step_dt:.2f}s", fill="white")
                    draw.text((8, 23), f"First episode active: {int(alive.sum())}/{n} | env0 {'active' if bool(alive[0]) else 'ended (auto-reset visible)'}", fill="white")
                    if args.diagnose_contacts or args.hold_default_pose:
                        draw.rectangle((0, 548, 768, 576), fill=(0, 0, 0))
                        label = ("DIAGNOSTIC ONLY: zero learned residual, physical controller"
                                 if args.hold_default_pose else
                                 "DIAGNOSTIC ONLY: unfiltered contact terminations disabled")
                        draw.text((8, 554), label, fill="yellow")
                    video.append_data(np.asarray(frame))
                    if step in (0, 50, 100, 150, 250, 500, 750, 995):
                        frame.save(args.output / f"frame_{step:04d}.png")
            if (step + 1) % 100 == 0:
                print(f"[EVAL] t={(step+1)*raw.step_dt:.2f}s first_episode_active={int(alive.sum())}/{n} failed={int(failed.sum())}", flush=True)
                if stair_tracker is not None:
                    print("[STAIRS] " + json.dumps({
                        "completed_each_level_both_feet": stair_tracker.verified.all(-1).sum(0).cpu().tolist(),
                        "platform_dwell2s": int((stair_tracker.platform_max_time >= 2 - 1e-5).sum()),
                        "active_controller_phases": {str(i): int(((reference_controller.state == i) & alive).sum())
                                                     for i in range(9)},
                        "bottom_aligned_first_episode": int((reference_controller.bottom_aligned & alive).sum())}), flush=True)
            if not alive.any():
                break
            if args.early_reject_failures and int(failed.sum()) >= args.early_reject_failures:
                metadata["development_early_rejection"] = {"failed_first_episodes": int(failed.sum()),
                                                           "time_s": (step + 1) * raw.step_dt,
                                                           "acceptance_allowed": False}
                print("[EVAL] EARLY_REJECT: first-episode failures already preclude acceptance", flush=True)
                break
            if stair_tracker is not None and args.early_reject_failures:
                state = reference_controller.state
                no_ascent_remaining = ((~alive) | (state == reference_controller.HOLD)
                                       | (state == reference_controller.TOP_STAND)).all()
                potential = alive & (state == reference_controller.TOP_STAND)
                if no_ascent_remaining and int(potential.sum()) < 922:
                    metadata["development_early_rejection"] = {
                        "reason": "All remaining first-episode controllers are held or finished; fewer than922 can reach the platform.",
                        "failed_first_episodes": int(failed.sum()), "time_s": (step + 1) * raw.step_dt,
                        "held_survivors": int((alive & (state == reference_controller.HOLD)).sum()),
                        "top_survivors": int(potential.sum()), "acceptance_allowed": False}
                    print("[EVAL] EARLY_REJECT: no first-episode ascent remains possible", flush=True)
                    break
        horizon_survived = (~failed) & (elapsed >= args.duration - raw.step_dt * 1.1)
        if stair_tracker is not None:
            horizon_survived = (~failed) & (elapsed >= args.duration)
        displacement = end_pos - start_pos
        progress = (displacement[:, 0] / (args.speed * args.duration)
                    if abs(args.speed) > 1e-6 else torch.zeros_like(displacement[:, 0]))
        screen = horizon_survived & (abs(args.speed) >= 0.1) & (progress >= 0.8) & (progress <= 1.2)
        gait_screen = screen & (footfalls.landings >= 2).all(-1)
        if stair_tracker is not None:
            gait_screen, stair_result = stair_tracker.result(end_pos, horizon_survived)
            metadata["stairs_results"] = stair_result
            metadata["stairs_acceptance_passed"] = int(gait_screen.sum()) >= 922
            entry_evidence.save(args.output/"entry_windows.npz")
            for key,value in reference_controller.map.tactile.stats.items():
                tactile_stats_terminal[key][alive] = value[alive]
            metadata["tactile_first_episode_diagnostics"] = {
                "normal_sign":reference_controller.map.tactile.normal_sign,
                "first_foot_positive_control":reference_controller.map.tactile.first_foot_positive_control,
                "counts_by_foot":{key:value.sum(0).cpu().tolist() for key,value in tactile_stats_terminal.items()},
                "contact_polygon_interior_is_inferred_support_not_direct_point_measurements":True,
                "point_force_positive_control_required":True,
            }
            np.savez_compressed(args.output/"contact_and_effort_evidence.npz",
                time_s=np.asarray([point["time_s"] for point in history]),
                **{key:np.stack([point[key] for point in history]) for key in
                   ("projected_net_joint_force","unfiltered_body_contact_force","ground_filtered_body_force",
                    "tactile_cell_count","tactile_contact_age")},
                **{"tactile_"+key:np.stack([point["tactile_"+key] for point in history])
                   for key in reference_controller.map.tactile.latest},
                **{"tactile_count_"+key:value.cpu().numpy() for key,value in tactile_stats_terminal.items()},
                unfiltered_body_names=np.asarray(sensor.body_names),
                filtered_body_names=np.asarray(metadata["filtered_nonfoot_ground_sensor_bodies"]),
                max_unfiltered_force_N=contact_body_max.cpu().numpy(),
                max_unfiltered_continuous_contact_s=contact_body_duration.cpu().numpy(),
                max_filtered_ground_force_N=ground_body_max.cpu().numpy())
            if footfall_event_chunks:
                np.savez_compressed(args.output/"footfall_events.npz",**{
                    key:np.concatenate([chunk[key] for chunk in footfall_event_chunks]) for key in footfall_event_chunks[0]})
            metadata["footfall_event_log"] = {"path":"footfall_events.npz","sample_dt_s":raw.step_dt,
                "rejection_bits":{"1":"no previous contact","2":"other foot not already supporting",
                                  "4":"air shorter than80ms","8":"air longer than600ms","16":"same foot as last valid landing"},
                "authoritative_validity":"original FootfallTracker count increment, thresholds unchanged"}
            terminal_safe_cells[alive] = reference_controller.map.safe[alive].sum(-1)
            terminal_real_hits[alive] = reference_controller.map.real_hit_count[alive]
            if bool(alive[0]):
                mapping = reference_controller.map
                np.savez_compressed(args.output/"perception_env0_last_first_episode.npz",
                    **{key:getattr(mapping,key)[0].cpu().numpy() for key in
                       ("minimum","maximum","safe","first_seen","last_seen","modelled","model_height",
                        "tactile_known","tactile_height","tactile_invalid")},
                    x=mapping.x.cpu().numpy(),y=mapping.y.cpu().numpy(),plane_x=mapping.plane_x.cpu().numpy(),
                    clock=mapping.clock,nx=mapping.nx,ny=mapping.ny,env_origin=raw.scene.env_origins[0].cpu().numpy())
            np.savez_compressed(args.output / "stairs_events.npz",
                                verified=stair_tracker.verified.cpu().numpy(),
                                event_time_s=stair_tracker.verified_time.cpu().numpy(),
                                event_position_w=stair_tracker.verified_position.cpu().numpy(),
                                completed_levels=stair_tracker.completed.cpu().numpy(),
                                perception_safe_cells_terminal=terminal_safe_cells.cpu().numpy(),
                                lidar_real_hits_terminal=terminal_real_hits.cpu().numpy(),
                                platform_max_time_s=stair_tracker.platform_max_time.cpu().numpy(),
                                root_start=start_pos.cpu().numpy(), root_end=end_pos.cpu().numpy(),
                                success=gait_screen.cpu().numpy(), env_origins=raw.scene.env_origins.cpu().numpy())
        mean_error = velocity_error_sum / torch.clamp(samples, min=1)
        if reference_controller is not None:
            for key in controller_keys:
                controller_terminal[key][alive] = getattr(reference_controller, key)[alive]
        records = []
        for i in range(n):
            records.append({"env_id": i, "first_episode_duration_s": float(elapsed[i]),
                            "failed": bool(failed[i]), "timed_out": bool(timeout[i]),
                            "survived_horizon": bool(horizon_survived[i]),
                            "forward_displacement_m": float(displacement[i, 0]),
                            "lateral_displacement_m": float(displacement[i, 1]),
                            "forward_progress_ratio": float(progress[i]),
                            "mean_xy_velocity_error_m_s": float(mean_error[i]),
                            "min_root_height_m": float(min_height[i]),
                            "max_tilt_deg": float(max_tilt[i] * 180 / math.pi),
                            "left_touchdowns": int(touchdowns[i, 0]), "right_touchdowns": int(touchdowns[i, 1]),
                            "left_valid_alternating_landings": int(footfalls.landings[i, 0]),
                            "right_valid_alternating_landings": int(footfalls.landings[i, 1]),
                            "survival_and_progress_screen": bool(screen[i]),
                            "screen_with_bilateral_touchdowns": bool(gait_screen[i]),
                            **({"stair_levels_completed": int(stair_tracker.completed[i]),
                                 "platform_dwell_max_s": float(stair_tracker.platform_max_time[i]),
                                 "net_root_rise_m": float(displacement[i, 2]),
                                 "stairs_joint_success": bool(gait_screen[i])} if stair_tracker is not None else {}),
                            "left_foot_z_range_m": float(feet_max_z[i, 0] - feet_min_z[i, 0]),
                            "right_foot_z_range_m": float(feet_max_z[i, 1] - feet_min_z[i, 1]),
                            **{name: float(values[i]) for name, values in contact_diagnostic.items()},
                            **{"termination_" + name: bool(values[i]) for name, values in reasons.items()}})
        with (args.output / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        np.savez_compressed(args.output / "trajectories.npz", time_s=np.array([x["time_s"] for x in history]),
                            position=np.stack([x["position"] for x in history]),
                            velocity=np.stack([x["velocity"] for x in history]),
                            feet_position=np.stack([x["feet_position"] for x in history]),
                            feet_quaternion=np.stack([x["feet_quaternion"] for x in history]),
                            quaternion=np.stack([x["quaternion"] for x in history]),
                            joint_position=np.stack([x["joint_position"] for x in history]),
                            joint_target=np.stack([x["joint_target"] for x in history]),
                            policy_action=np.stack([x["policy_action"] for x in history]),
                            feet_contact_force=np.stack([x["feet_contact_force"] for x in history]),
                            feet_ground_forces=np.stack([x["feet_ground_forces"] for x in history]),
                            first_episode_active=np.stack([x["first_episode_active"] for x in history]),
                            start_position=start_pos.cpu().numpy())
        if controller_history:
            np.savez_compressed(args.output / "controller_trace.npz",
                                **{key: np.stack([x[key] for x in controller_history])
                                   for key in controller_history[0]})
            if stair_tracker is not None:
                owned = ~np.stack([point["approach"] for point in controller_history])
                legacy_force = np.stack([point["legacy_feedback_support_force"] for point in controller_history])
                peak = float(np.abs(legacy_force[owned]).max()) if owned.any() else 0.0
                if peak != 0.0:
                    raise RuntimeError("Legacy feedback remained active after stair ownership transfer")
                metadata["stair_feedback_ownership_runtime"] = {
                    "owned_samples": int(owned.sum()), "legacy_support_force_peak_N": peak,
                    "exclusive_ownership_verified": bool(owned.any() and peak == 0.0)}
                metadata["native_support_budget_runtime"] = {
                    name:actuator.support_native_verification for name,actuator in robot.actuators.items()
                    if getattr(actuator,"support_priority_buffer",None) is not None}
            terminal_steps = controller_terminal["steps"]
            terminal_failure = controller_terminal["failure"]
            metadata["reference_controller_summary"] = {
                "reference_steps_mean": float(terminal_steps.float().mean()),
                "reference_steps_min": int(terminal_steps.min()),
                "reference_steps_max": int(terminal_steps.max()),
                "unload_timeout_envs": int((terminal_failure == 1).sum()),
                "landing_timeout_envs": int((terminal_failure == 2).sum()),
                "infeasible_plan_envs": int((terminal_failure == 3).sum()),
                "terminal_ik_error_max_m": float(controller_terminal["ik_error"].max())}
            if stair_tracker is not None:
                metadata["stairs_controller_summary"] = reference_controller.summary()
                failure = controller_terminal["failure"]
                metadata["stairs_controller_summary"]["first_episode_terminal_failure_counts"] = {
                    str(int(code)): int((failure == code).sum()) for code in failure.unique()}
                direct=controller_terminal["entry_direct"]
                metadata["stairs_controller_summary"]["first_episode_direct_entry_envs"]=int(direct.sum())
                metadata["stairs_controller_summary"]["first_episode_lift_raw_boundary_witness_envs"]=int(
                    controller_terminal["lift_boundary_observed"].sum())
                if direct.any():
                    metadata["stairs_controller_summary"]["direct_entry_root_height_min_median_max_m"]=[
                        float(f(controller_terminal["entry_root_height"][direct])) for f in (torch.min,torch.median,torch.max)]
                for label, key in (("first_episode_terminal_state_counts", "state"),
                                   ("first_episode_terminal_level_counts", "levels_completed")):
                    value = controller_terminal[key]
                    metadata["stairs_controller_summary"][label] = {
                        str(int(code)): int((value == code).sum()) for code in value.unique()}
        if wrench_reader is not None:
            if wrench_history:
                np.savez_compressed(args.output / "foot_wrench_evidence.npz", **{
                    key: np.stack([sample[key] for sample in wrench_history]) for key in wrench_history[0]})
            metadata["foot_wrench_evidence"].update(
                summary=wrench_reader.summary(), capture_errors=wrench_capture_errors,
                frames=len(wrench_history), path="foot_wrench_evidence.npz")
        metadata.update(status="completed", elapsed_wall_s=time.time() - started,
                        simulated_steps=step + 1, checkpoint_unchanged=sha256(args.checkpoint) == metadata["checkpoint_sha256"],
                        results={"survived_20s_count": int((elapsed >= 20).sum()),
                                 "survived_horizon_count": int(horizon_survived.sum()), "survival_horizon_s": args.duration,
                                 "failed_count": int(failed.sum()),
                                 "survival_and_progress_count": int(screen.sum()), "bilateral_touchdown_screen_count": int(gait_screen.sum()),
                                 "mean_first_episode_duration_s": float(elapsed.mean()),
                                 "median_first_episode_duration_s": float(elapsed.median()),
                                 "mean_forward_displacement_m": float(displacement[:, 0].mean()),
                                 "median_forward_displacement_m": float(displacement[:, 0].median()),
                                 "mean_xy_velocity_error_m_s": float(mean_error.mean()),
                                 "termination_counts": {name: int(values.sum()) for name, values in reasons.items()},
                                 "contact_diagnostics": {name: {"max": float(values.max()), "mean": float(values.mean()), "nonzero_env_count": int((values > 0).sum())}
                                                         for name, values in contact_diagnostic.items()},
                                 "surviving_at_2s": int((elapsed >= 2.0).sum()), "surviving_at_5s": int((elapsed >= 5.0).sum()),
                                 "surviving_at_10s": int((elapsed >= 10.0).sum())},
                        artifacts={"per_environment": "episodes.csv", "trajectories": "trajectories.npz",
                                   "video": "env0.mp4" if args.video else None})
        write_status(status_path,metadata)
        print("[EVAL] RESULT: " + json.dumps(metadata["results"]), flush=True)
    except BaseException:
        primary_error = traceback.format_exc()
        try:
            if locals().get("entry_evidence") is not None:
                entry_evidence.save(args.output/"entry_windows.npz")
            if locals().get("controller_history"):
                np.savez_compressed(args.output/"partial_controller_trace.npz",
                    **{key:np.stack([point[key] for point in controller_history]) for key in controller_history[0]})
            if locals().get("wrench_history"):
                np.savez_compressed(args.output / "partial_foot_wrench_evidence.npz", **{
                    key: np.stack([sample[key] for sample in wrench_history]) for key in wrench_history[0]})
        except BaseException:
            metadata["partial_evidence_write_error"] = traceback.format_exc()
        metadata.update(status="failed", error=primary_error)
        write_status(status_path,metadata)
        print("[EVAL] FAILURE: " + metadata["error"], flush=True)
        raise
    finally:
        if video is not None:
            video.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
