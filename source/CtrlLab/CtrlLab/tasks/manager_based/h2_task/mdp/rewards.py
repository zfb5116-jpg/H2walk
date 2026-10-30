import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from ..robot_env import RobotEnv

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


def tracking_lin_vel_x(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float = 0.5,
        command_threshold: float = 0.15,
        require_forward_progress: bool = False,
) -> torch.Tensor:
    """Track the commanded forward velocity in the robot base frame.

    The command is expressed in the base frame. Comparing it with world-frame
    velocity makes a yawed robot appear to move in the wrong direction and
    encourages diagonal/turning behavior on stairs.
    """
    if not 0.0 < std < float("inf"):
        raise ValueError("std must be finite and positive.")
    if not 0.0 <= command_threshold < float("inf"):
        raise ValueError("command_threshold must be finite and non-negative.")
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    error = forward_velocity - command[:, 0]
    score = torch.exp(-torch.square(error / std))
    if require_forward_progress:
        # A moving command earns no forward-tracking reward by standing still
        # or travelling backwards. Full credit still requires matching speed.
        progress_fraction = (forward_velocity * command[:, 0].sign()
                             / command[:, 0].abs().clamp_min(1e-6)).clamp(0.0, 1.0)
        score *= progress_fraction
    return score * moving


def heading_tracking(
        env: ManagerBasedRLEnv,
        command_name: str,
        std: float = 0.5,
        command_threshold: float = 0.05,
) -> torch.Tensor:
    """Track the heading controller's yaw-rate command in the base frame.

    A zero yaw command therefore rewards zero yaw rate, which prevents the
    policy from turning while it is trying to climb straight ahead.
    """
    if not 0.0 < std < float("inf"):
        raise ValueError("std must be finite and positive.")
    if not 0.0 <= command_threshold < float("inf"):
        raise ValueError("command_threshold must be finite and non-negative.")
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_ang_vel_b[:, 2] - command[:, 2]
    return torch.exp(-torch.square(error / std))


def lin_vel_y_l2(
        env: ManagerBasedRLEnv,
        command_name: str,
        command_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral base velocity during commanded locomotion.

    Stair climbing should remain aligned with the target direction. The term
    intentionally penalizes actual lateral motion instead of rewarding a
    lateral command match, so small command noise cannot produce diagonal
    walking.
    """
    if not 0.0 <= command_threshold < float("inf"):
        raise ValueError("command_threshold must be finite and non-negative.")
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * moving


# Compatibility aliases for older configs and checkpoints.
track_forward_velocity = tracking_lin_vel_x
track_forward_world_velocity = tracking_lin_vel_x


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
    """Penalize sideways pelvis velocity while walking primarily forward.

    Sideways commands are exempted so this term suppresses unwanted sway without
    making a commanded lateral walk impossible.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    forward_mode = torch.abs(command[:, 1]) < command_threshold
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * forward_mode


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
                     effort_limit: float,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                     ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
            torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - effort_limit * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
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
) -> torch.Tensor:
    """Reward swing feet that clear the nearest raised terrain in the forward lane."""
    contacts_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
    contacts = contacts_sensor.data.net_forces_w_history[:, :, foot_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1).values > 1.0

    terrain_sensor = env.scene[terrain_sensor_cfg.name]
    hits_w = terrain_sensor.data.ray_hits_w
    root_pos_w = env.asset.data.root_pos_w
    root_quat_w = env.asset.data.root_quat_w
    ray_count = hits_w.shape[1]
    ray_quat = root_quat_w.unsqueeze(1).expand(-1, ray_count, -1)
    hits_b = quat_apply_inverse(ray_quat.reshape(-1, 4), (hits_w - root_pos_w.unsqueeze(1)).reshape(-1, 3)).view_as(hits_w)

    valid = torch.isfinite(hits_w).all(dim=-1)
    lane = torch.abs(hits_b[:, :, 1]) < 0.25
    front = valid & lane & (hits_b[:, :, 0] > 0.08) & (hits_b[:, :, 0] < 0.42)
    under = valid & lane & (hits_b[:, :, 0] > -0.30) & (hits_b[:, :, 0] < 0.02)
    nan = torch.full_like(hits_w[:, :, 2], float("nan"))
    front_height = torch.nan_to_num(torch.where(front, hits_w[:, :, 2], -torch.inf).amax(dim=1), neginf=0.0)
    ground_height = torch.nan_to_num(torch.where(under, hits_w[:, :, 2], nan).nanmedian(dim=1).values, nan=0.0)
    rise = torch.clamp(front_height - ground_height, min=0.0, max=max_rise)
    target_z = front_height + clearance

    env.update_feet_state()
    feet_z_w = env.asset.data.body_pos_w[:, env.feet_id_b, 2]
    swing = ~contacts
    swing_count = swing.sum(dim=1).clamp(min=1)
    height_score = torch.exp(-torch.square((feet_z_w - target_z.unsqueeze(1)) / height_std))
    rise_weight = ((rise - min_rise) / (max_rise - min_rise)).clamp(0.0, 1.0)
    moving_forward = env.command_manager.get_command(command_name)[:, 0] > 0.15
    score = torch.sum(height_score * swing, dim=1) / swing_count
    return score * rise_weight * moving_forward


def feet_edge_penalty(
        env: ManagerBasedRLEnv,
        terrain_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        edge_height_threshold: float = 0.035,
        edge_radius: float = 0.05,
        tolerance: float = 1e-4,
        max_edges: int = 128,
        grid_size: tuple[int, int] = (33, 17),
        foot_x_range: tuple[float, float] = (-0.025, 0.12),
        foot_y_range: tuple[float, float] = (-0.03, 0.03),
        foot_z_range: tuple[float, float] = (-0.04, 0.0),
        foot_grid_size: tuple[int, int, int] = (5, 3, 2),
) -> torch.Tensor:
    """Penalize fast volume points that penetrate detected terrain edges.

    This is the H2-compatible form of the *Hiking in the Wild* foothold
    penalty. The upstream implementation detects sharp mesh edges and queries
    a volume-points sensor against edge cylinders. H2 already has a dense
    height scanner, so adjacent scanner hits are used as the edge detector and
    the same volume-point/cylinder query is performed in the robot base frame.
    The returned value is positive and should be configured with a negative
    reward weight:

    ``sum_i 1[d_i > tolerance] * (||v_i|| + 1e-6) * d_i``.
    """
    if edge_height_threshold < 0.0 or edge_radius <= 0.0 or max_edges <= 0:
        raise ValueError("edge_height_threshold and edge_radius must be non-negative and max_edges must be positive.")
    if len(grid_size) != 2 or min(grid_size) < 2:
        raise ValueError("grid_size must contain two dimensions of at least two rays.")
    if len(foot_grid_size) != 3 or min(foot_grid_size) < 1:
        raise ValueError("foot_grid_size must contain three positive dimensions.")

    asset: Articulation = env.scene[asset_cfg.name]
    terrain_sensor = env.scene[terrain_sensor_cfg.name]
    hits_w = terrain_sensor.data.ray_hits_w
    num_envs = hits_w.shape[0]
    ray_count = hits_w.shape[1]
    grid_x, grid_y = grid_size
    if ray_count != grid_x * grid_y:
        # A mismatched scanner should not make training fail. It simply has
        # no edge evidence until the configured scanner resolution is fixed.
        return torch.zeros(num_envs, device=hits_w.device, dtype=hits_w.dtype)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ray_quat = root_quat_w.unsqueeze(1).expand(-1, ray_count, -1)
    hits_b = quat_apply_inverse(
        ray_quat.reshape(-1, 4),
        (hits_w - root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).view_as(hits_w).view(num_envs, grid_x, grid_y, 3)
    valid = torch.isfinite(hits_b).all(dim=-1)

    # Every adjacent pair of valid rays is a short terrain-edge candidate.
    x_start, x_end = hits_b[:, :-1, :, :].reshape(num_envs, -1, 3), hits_b[:, 1:, :, :].reshape(num_envs, -1, 3)
    x_valid = valid[:, :-1, :] & valid[:, 1:, :]
    x_strength = torch.abs(x_end[..., 2] - x_start[..., 2])
    x_strength = torch.where(x_valid.reshape(num_envs, -1) & (x_strength > edge_height_threshold), x_strength, -torch.inf)

    y_start, y_end = hits_b[:, :, :-1, :].reshape(num_envs, -1, 3), hits_b[:, :, 1:, :].reshape(num_envs, -1, 3)
    y_valid = valid[:, :, :-1] & valid[:, :, 1:]
    y_strength = torch.abs(y_end[..., 2] - y_start[..., 2])
    y_strength = torch.where(y_valid.reshape(num_envs, -1) & (y_strength > edge_height_threshold), y_strength, -torch.inf)

    edge_start = torch.cat((x_start, y_start), dim=1)
    edge_end = torch.cat((x_end, y_end), dim=1)
    edge_strength = torch.cat((x_strength, y_strength), dim=1)
    edge_count = edge_strength.shape[1]
    select_count = min(max_edges, edge_count)
    strength, edge_ids = torch.topk(edge_strength, k=select_count, dim=1)
    batch_ids = torch.arange(num_envs, device=hits_b.device).unsqueeze(1)
    edge_start = edge_start[batch_ids, edge_ids]
    edge_end = edge_end[batch_ids, edge_ids]
    edge_valid = torch.isfinite(strength)
    # Do not let NaN/Inf coordinates from invalid scanner rays contaminate the
    # distance calculation before the validity mask is applied.
    edge_start = torch.where(edge_valid.unsqueeze(-1), edge_start, torch.zeros_like(edge_start))
    edge_end = torch.where(edge_valid.unsqueeze(-1), edge_end, torch.zeros_like(edge_end))

    # Match the official G1 volume-point footprint while keeping the H2 term
    # independent of a custom sensor class.
    x = torch.linspace(*foot_x_range, foot_grid_size[0], device=hits_b.device, dtype=hits_b.dtype)
    y = torch.linspace(*foot_y_range, foot_grid_size[1], device=hits_b.device, dtype=hits_b.dtype)
    z = torch.linspace(*foot_z_range, foot_grid_size[2], device=hits_b.device, dtype=hits_b.dtype)
    local_points = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1).reshape(-1, 3)
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
    num_bodies = body_pos_w.shape[1]
    point_count = local_points.shape[0]
    local_points = local_points.view(1, 1, point_count, 3).expand(num_envs, num_bodies, -1, -1)
    body_quat_w = body_quat_w.unsqueeze(2).expand(-1, -1, point_count, -1)
    rotated_offset_w = quat_apply(body_quat_w, local_points)
    points_w = body_pos_w.unsqueeze(2) + rotated_offset_w
    points_vel_w = body_lin_vel_w.unsqueeze(2) + torch.linalg.cross(
        body_ang_vel_w.unsqueeze(2), rotated_offset_w, dim=-1
    )
    points_b = quat_apply_inverse(
        root_quat_w.unsqueeze(1).unsqueeze(2).expand_as(body_quat_w),
        points_w - root_pos_w.unsqueeze(1).unsqueeze(2),
    ).flatten(1, 2)
    points_speed = torch.linalg.vector_norm(points_vel_w, dim=-1).flatten(1, 2)

    # Distance from every foot volume point to the nearest selected edge
    # segment, equivalent to the edge-cylinder penetration query.
    segment = edge_end - edge_start
    segment_norm_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1e-8)
    point_to_start = points_b.unsqueeze(2) - edge_start.unsqueeze(1)
    projection = torch.sum(point_to_start * segment.unsqueeze(1), dim=-1) / segment_norm_sq.unsqueeze(1)
    projection = projection.clamp(0.0, 1.0)
    nearest = edge_start.unsqueeze(1) + projection.unsqueeze(-1) * segment.unsqueeze(1)
    distance = torch.linalg.vector_norm(points_b.unsqueeze(2) - nearest, dim=-1)
    penetration = (edge_radius - distance).clamp_min(0.0) * edge_valid.unsqueeze(1)
    penetration = penetration.max(dim=-1).values
    active = (penetration > tolerance).to(points_speed.dtype)
    return torch.sum(active * (points_speed + 1e-6) * penetration, dim=1)


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
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 水平方向力 (X, Y)
    horizontal_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1).max(dim=1).values
    # 垂直方向力 (Z)
    vertical_force = torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2]).max(dim=1).values
    # 如果水平力比垂直力大 5 倍，则视为“绊倒”
    stumble_condition = horizontal_force > 5.0 * vertical_force
    # 判断每个环境中是否有任意一个脚部满足这个条件
    stumble = torch.any(stumble_condition, dim=-1)  # [num_envs], dtype: bool
    return stumble


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
