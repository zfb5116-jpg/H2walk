import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ..robot_env import RobotEnv
from ..contact_events import stumble_from_force_history
from ..foot_geometry import lowest_sole_height_world

"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std ** 2) * contacts.any(dim=-1)


def track_ang_vel_z_exp(
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std ** 2) * contacts.any(dim=-1)


def track_forward_velocity(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float = 0.5,
        command_threshold: float = 0.15,
) -> torch.Tensor:
    """Track forward velocity in the same base frame as the velocity command."""
    if not 0.0 < std < float("inf"):
        raise ValueError("std must be finite and positive.")
    if not 0.0 <= command_threshold < float("inf"):
        raise ValueError("command_threshold must be finite and non-negative.")
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    forward = (command[:, 0] > command_threshold) & (torch.abs(command[:, 1]) < command_threshold)
    error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    return torch.exp(-torch.square(error / std)) * forward


def track_forward_world_velocity(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float = 0.5,
        command_threshold: float = 0.15,
) -> torch.Tensor:
    """Historical name retained for configs; uses corrected base-frame tracking.

    The old world-X comparison conflicted with base-frame commands after yaw
    rotations. This alias preserves imports, not that reward behavior.
    """
    return track_forward_velocity(env, command_name, std, command_threshold)


"""
base penalty
"""


def base_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    base_body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else asset_cfg.body_ids
    base_lin_acc_w = asset.data.body_lin_acc_w[:, base_body_id]
    base_ang_acc_w = asset.data.body_ang_acc_w[:, base_body_id]
    base_lin_acc_b = quat_apply_inverse(base_quat, base_lin_acc_w)
    base_ang_acc_b = quat_apply_inverse(base_quat, base_ang_acc_w)
    reward = torch.square(base_lin_acc_b[:, 2]) + 0.02 * torch.sum(torch.square(base_ang_acc_b[:, :2]), dim=1)
    return reward


def base_lateral_velocity_l2(
        env: ManagerBasedRLEnv,
        command_name: str,
        command_threshold: float = 0.15,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral tracking error while walking primarily forward.

    The small-lateral-command gate is retained. Subtracting that command avoids
    penalizing correct tracking of lateral commands inside the gate.
    """
    if not 0.0 <= command_threshold < float("inf"):
        raise ValueError("command_threshold must be finite and non-negative.")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    forward_mode = torch.abs(command[:, 1]) < command_threshold
    return torch.square(asset.data.root_lin_vel_b[:, 1] - command[:, 1]) * forward_mode


def base_lateral_acc_l2(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize rapid left-right pelvis acceleration that causes visible sway."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_body_id = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, list) else asset_cfg.body_ids
    base_lin_acc_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.body_lin_acc_w[:, base_body_id])
    return torch.square(base_lin_acc_b[:, 1])


def base_height_l2(
        env: RobotEnv,
        target_height: float,
) -> torch.Tensor:
    env.update_feet_state()
    z_vals = env.feet_pos_b[:, :, 2]
    base_height = -torch.min(z_vals, dim=1).values
    return torch.square(base_height - target_height)


def base_height_l1(
        env: RobotEnv,
        target_height: float,
) -> torch.Tensor:
    env.update_feet_state()
    z_vals = env.feet_pos_b[:, :, 2]
    base_height = -torch.min(z_vals, dim=1).values
    return torch.abs(base_height - target_height)


"""
joint
"""


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def joint_power_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_tor = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_vel) * torch.abs(joint_tor), dim=-1)


def joint_power_var(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_tor = asset.data.applied_torque[:, asset_cfg.joint_ids]
    power = torch.abs(joint_vel * joint_tor)
    power_var = torch.var(power, dim=-1, unbiased=False)
    return power_var


def joint_tor_limits(env: ManagerBasedRLEnv,
                     soft_ratio: float,
                     effort_limit: float | None = None,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                     ) -> torch.Tensor:
    """Penalize applied torque above each joint's soft simulation effort limit.

    An explicit scalar ``effort_limit`` retains the previous common-limit
    baseline. Excess remains in N m, capped at 1 per joint as before.
    """
    if not 0.0 < soft_ratio <= 1.0:
        raise ValueError("soft_ratio must satisfy 0 < soft_ratio <= 1.")
    if effort_limit is not None and not 0.0 < effort_limit < float("inf"):
        raise ValueError("effort_limit must be finite and positive when supplied.")
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids] if effort_limit is None else effort_limit
    out_of_limits = (
            torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - limits * soft_ratio
    )
    # Preserve the existing cap of 1 N m excess per joint.
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
action
"""


def action_smooth_l2(env: RobotEnv) -> torch.Tensor:
    rate = env.action_manager.action + env.action_manager.prev_prev_action - 2 * env.action_manager.prev_action
    return torch.sum(torch.square(rate), dim=1)


"""
feet
"""


def feet_air_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(-torch.abs(last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.01
    return reward


def feet_contact_time(
        env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum(-torch.abs(last_contact_time - threshold) * first_air, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.01
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    air_time_var = torch.var(torch.clip(last_air_time, max=0.5), dim=1, unbiased=False)
    contact_time_var = torch.var(torch.clip(last_contact_time, max=0.5), dim=1, unbiased=False)
    air_contact_var = 0.25 * torch.mean(torch.square(last_air_time - last_contact_time), dim=1)
    reward = air_time_var + contact_time_var + air_contact_var
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.01
    return reward


def feet_air_vel_norm(env, sensor_cfg: SceneEntityCfg,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * (~contacts), dim=1)
    return reward


def feet_swing_height(
        env: RobotEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        target_feet_height: float = -0.67,
        height_std: float = 0.10,
) -> torch.Tensor:
    """Reward lifting each swinging foot to a useful stair-clearing height."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    env.update_feet_state()
    feet_height_b = env.feet_pos_b[:, :, 2]
    height_score = torch.exp(-torch.square((feet_height_b - target_feet_height) / height_std))
    swing = ~contacts
    swing_count = swing.sum(dim=1).clamp(min=1)
    reward = torch.sum(height_score * swing, dim=1) / swing_count
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.15
    return reward * moving


def feet_stair_clearance(
        env: RobotEnv,
        command_name: str,
        foot_sensor_cfg: SceneEntityCfg,
        terrain_sensor_cfg: SceneEntityCfg,
        clearance: float = 0.04,
        min_rise: float = 0.04,
        max_rise: float = 0.24,
        height_std: float = 0.12,
        target_mode: str = "highest",
        height_reference: str = "link",
        sole_points_b=None,
) -> torch.Tensor:
    """Reward swing clearance using the existing ground-truth height scanner.

    ``nearest`` queries each foot's lane in the horizontal heading frame and
    targets the first raised scan column, rather than a more distant high step.
    This is a reward-shaping hypothesis, not a footstep planner. ``highest``
    retains the shared body-lane target for ablations, with invalid-ray fixes.
    Both modes require valid ground and front samples. Missing rays never
    establish a world-z=0 floor. Descending stairs and lateral/backward commands
    are outside the nearest-target experiment.

    ``link`` keeps the previous ankle-link height reference. ``sole`` uses the
    lowest world-Z point of each foot's oriented sole surface, with optional
    custom local vertices [foot, point, 3]. It requires body world quaternions.
    This surface-height correction does not measure actual support coverage.
    """
    if target_mode not in {"nearest", "highest"}:
        raise ValueError("target_mode must be 'nearest' or 'highest'")
    if height_reference not in {"link", "sole"}:
        raise ValueError("height_reference must be 'link' or 'sole'")
    if height_reference == "link" and sole_points_b is not None:
        raise ValueError("sole_points_b requires height_reference='sole'")
    if not (0.0 <= clearance < float("inf") and 0.0 < min_rise < max_rise < float("inf")
            and 0.0 < height_std < float("inf")):
        raise ValueError("Require clearance >= 0, 0 < min_rise < max_rise, height_std > 0")
    contacts_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    contacts = contacts_sensor.data.net_forces_w_history[:, :, foot_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1).values > 1.0

    terrain_sensor = env.scene[terrain_sensor_cfg.name]
    hits_w = terrain_sensor.data.ray_hits_w
    root_pos_w = env.asset.data.root_pos_w
    root_quat_w = yaw_quat(env.asset.data.root_quat_w)
    ray_count = hits_w.shape[1]
    ray_quat = root_quat_w.unsqueeze(1).expand(-1, ray_count, -1)
    valid = torch.isfinite(hits_w).all(dim=-1)
    safe_hits = torch.where(valid.unsqueeze(-1), hits_w, root_pos_w.unsqueeze(1))
    hits_b = quat_apply_inverse(ray_quat.reshape(-1, 4), (safe_hits - root_pos_w.unsqueeze(1)).reshape(-1, 3)).view_as(hits_w)
    feet_w = env.asset.data.body_pos_w[:, env.feet_id_b, :]
    foot_quat = root_quat_w.unsqueeze(1).expand(-1, feet_w.shape[1], -1)
    feet_b = quat_apply_inverse(foot_quat.reshape(-1, 4), (feet_w - root_pos_w.unsqueeze(1)).reshape(-1, 3)).view_as(feet_w)
    # [environment, foot, ray]. The scanner has 5 cm spacing; a 2.5 cm
    # tolerance groups one nearest longitudinal column, including rotated grids.
    if target_mode == "nearest":
        relative = hits_b.unsqueeze(1) - feet_b.unsqueeze(2)
        lane = torch.abs(relative[..., 1]) < 0.10
        under_min = -0.12
    else:
        relative = hits_b.unsqueeze(1).expand(-1, feet_w.shape[1], -1, -1)
        lane = torch.abs(relative[..., 1]) < 0.25
        under_min = -0.30
    front = valid.unsqueeze(1) & lane & (relative[..., 0] > 0.08) & (relative[..., 0] < 0.42)
    under = valid.unsqueeze(1) & lane & (relative[..., 0] > under_min) & (relative[..., 0] < 0.02)
    z = safe_hits[:, None, :, 2].expand_as(front)
    ground = torch.where(under, z, torch.nan).nanmedian(dim=-1).values
    # A finite placeholder permits safe arithmetic; usable below excludes it.
    ground = torch.where(under.any(dim=-1), ground, root_pos_w[:, None, 2])
    if target_mode == "nearest":
        raised = front & (z - ground.unsqueeze(-1) >= min_rise - 1e-6)
        nearest_x = torch.where(raised, relative[..., 0], torch.inf).amin(dim=-1)
        first_column = front & (torch.abs(relative[..., 0] - nearest_x.unsqueeze(-1)) < 0.025)
        front_height = torch.where(first_column, z, torch.nan).nanmedian(dim=-1).values
        has_front = first_column.any(dim=-1)
    else:
        front_height = torch.where(front, z, -torch.inf).amax(dim=-1)
        has_front = front.any(dim=-1)
    front_height = torch.where(has_front, front_height, ground)
    rise = front_height - ground
    usable = under.any(dim=-1) & has_front & torch.isfinite(feet_w).all(dim=-1)
    if target_mode == "nearest":
        # Do not skip an unreachably high near step in favor of a farther one.
        usable &= (rise >= min_rise - 1e-6) & (rise <= max_rise + 1e-6)
        # Nonzero shaping at min_rise also covers the 2 cm curriculum start.
        rise_weight = (rise / max_rise).clamp(0.0, 1.0)
    else:
        rise_weight = ((rise - min_rise) / (max_rise - min_rise)).clamp(0.0, 1.0)
    target_z = front_height + clearance
    if height_reference == "sole":
        body_quat_w = getattr(env.asset.data, "body_quat_w", None)
        if body_quat_w is None:
            raise ValueError("height_reference='sole' requires asset.data.body_quat_w")
        sole_z_w = lowest_sole_height_world(
            feet_w, body_quat_w[:, env.feet_id_b, :], sole_points_b=sole_points_b,
        )
        usable &= torch.isfinite(sole_z_w)
        feet_z_w = torch.nan_to_num(sole_z_w)
    else:
        feet_z_w = torch.nan_to_num(feet_w[..., 2])
    swing = ~contacts
    swing_count = swing.sum(dim=1).clamp(min=1)
    height_score = torch.exp(-torch.square((feet_z_w - target_z) / height_std))
    command = env.command_manager.get_command(command_name)
    moving_forward = command[:, 0] > 0.15
    if target_mode == "nearest":
        moving_forward &= (torch.abs(command[:, 1]) < 0.15) & (torch.abs(command[:, 2]) < 0.3)
    score = torch.sum(height_score * rise_weight * usable * swing, dim=1) / swing_count
    return score * moving_forward


def feet_acc_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_acc = asset.data.body_lin_acc_w[:, asset_cfg.body_ids]
    return torch.sum(feet_acc.norm(dim=-1), dim=1)


def feet_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    return torch.sum(feet_vel.norm(dim=-1), dim=1)


def feet_stumble_norm(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        force_ratio: float = 5.0,
        temporal_mode: str = "per_sample",
) -> torch.Tensor:
    """Detect lateral-dominant foot contact using time-aligned force samples.

    Set ``temporal_mode='independent_max'`` for the original history reduction.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return stumble_from_force_history(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :],
        force_ratio=force_ratio,
        temporal_mode=temporal_mode,
    )


def feet_clearance_l2(
        env: RobotEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        target_feet_height=-0.23,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    env.update_feet_state()
    feet_vel_xy_b = torch.norm(env.feet_vel_b[:, :, :2], dim=-1, p=0.5)
    feet_height_b = env.feet_pos_b[:, :, 2]
    height_error = torch.square(torch.clip(feet_height_b - target_feet_height, max=0.0))
    reward = torch.sum(~contacts * height_error * feet_vel_xy_b, dim=-1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.01
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


"""
dynamic floating base
"""


def dynamic_floating_base_l2(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize floating base torques"""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    base_body_id = asset_cfg.body_ids[0]
    feet_body_id = asset_cfg.body_ids[1:]

    mat_A = asset.root_physx_view.get_generalized_mass_matrices()
    vec_C = asset.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
    vec_G = asset.root_physx_view.get_gravity_compensation_forces()

    mat_J = asset.root_physx_view.get_jacobians()[:, feet_body_id, :3]
    mat_J = mat_J.reshape(mat_J.shape[0], -1, mat_J.shape[-1])
    mat_J_trans = mat_J.transpose(1, 2)

    feet_forces = torch.mean(contact_sensor.data.net_forces_w_history, dim=1)[:, sensor_cfg.body_ids]
    feet_forces = feet_forces.reshape(feet_forces.shape[0], -1).unsqueeze(2)

    base_acc = asset.data.body_acc_w[:, base_body_id].squeeze(1)

    ddq = torch.cat([base_acc, asset.data.joint_acc], dim=-1).unsqueeze(2)

    tau = (
            torch.bmm(mat_A, ddq)
            + vec_C.unsqueeze(2)
            + vec_G.unsqueeze(2)
            - torch.bmm(mat_J_trans, feet_forces)
    ).squeeze(2)
    reward_base = torch.sum(torch.square(tau[:, :6]), dim=1)
    reward_joint = torch.sum(torch.square(tau[:, 6:]), dim=1)
    return reward_base + 0.1 * reward_joint


def support_force_z_axis_norm(
        env: RobotEnv,
        sensor_cfg: SceneEntityCfg,
        gravity=9.81,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_id = sensor_cfg.body_ids

    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_forces = torch.mean(net_contact_forces, dim=1)[:, feet_id]
    total_z_force = torch.sum(feet_forces[:, :, 2], dim=-1)

    reward = torch.any(contacts, dim=1).to(torch.float32) * torch.abs(total_z_force - env.masses * gravity)
    return reward


"""
stand still
"""


def stand_still_joint_deviation_l1(
        env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command, dim=1) < command_threshold)
