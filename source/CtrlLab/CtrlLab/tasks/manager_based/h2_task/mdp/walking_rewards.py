"""Contact-driven biped objectives, with no unobserved gait clock."""

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def horizontal_forward_velocity(env):
    robot = env.scene["robot"]
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[:, 0]


def upright_score(env, tilt_std=0.5):
    gravity_xy = env.scene["robot"].data.projected_gravity_b[:, :2]
    return torch.exp(-gravity_xy.square().sum(-1) / tilt_std**2)


def stable_forward_tracking(env, command_name, std=0.5, command_threshold=0.05, tilt_std=0.5):
    """Credit horizontal commanded motion, smoothly reduced by body tilt."""
    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = horizontal_forward_velocity(env)
    progress = (velocity * command.sign() / command.abs().clamp_min(1e-6)).clamp(0.0, 1.0)
    tracking = torch.exp(-((velocity - command) / std).square())
    return tracking * progress * upright_score(env, tilt_std) * (command.abs() > command_threshold)


def net_forward_progress(env, command_name, velocity_error_weight=0.15, command_threshold=0.1):
    """Signed physical displacement: a closed rocking cycle cannot earn progress.

    Negative displacement is never masked by tilt/contact/phase. The extra
    Positive credit saturates at the commanded rate, so overspeed cannot earn
    more than correct tracking. The velocity cost preserves backwards debt.
    LocomotionProgress captures the pre-step pose before any reset teleport.
    """
    tracker = env.locomotion_progress
    desired = tracker.desired_xy
    speed = desired.norm(dim=-1)
    direction = desired / speed[:, None].clamp_min(1e-6)
    delta = env.scene["robot"].data.root_pos_w[:, :2] - tracker.pre_xy
    measured_speed = (delta * direction).sum(-1) / env.step_dt
    rate = measured_speed / speed.clamp_min(command_threshold)
    return (rate.clamp(max=1.0) - velocity_error_weight * (rate - 1.0).square()) * (speed >= command_threshold)


def feet_lateral_separation(env, asset_cfg: SceneEntityCfg, minimum=0.10):
    """Penalize narrow/crossed feet; forward staggering cannot hide narrowing."""
    robot = env.scene[asset_cfg.name]
    delta_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]] - robot.data.body_pos_w[:, asset_cfg.body_ids[1]]
    delta_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)
    return ((minimum - delta_yaw[:, 1]).clamp(min=0.0) / minimum).square()


def feet_support_clearance(env, command_name, sensor_cfg: SceneEntityCfg,
                           asset_cfg: SceneEntityCfg, target_clearance=0.06, height_std=0.04):
    """Reward a small swing clearance above the actual supporting foot, in world Z."""
    sensor = env.scene[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    single_support = contact.sum(-1) == 1
    foot_z = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2]
    support_z = torch.where(contact, foot_z, 0.0).sum(-1)
    swing_z = torch.where(~contact, foot_z, 0.0).sum(-1)
    clearance = swing_z - support_z
    score = torch.exp(-((clearance - target_clearance) / height_std).square())
    command = env.command_manager.get_command(command_name)[:, 0]
    progress = (horizontal_forward_velocity(env) * command.sign()
                / command.abs().clamp_min(0.1)).clamp(0.0, 1.0)
    return score * single_support * upright_score(env) * progress * (command.abs() > 0.1)


class feet_support_transition(ManagerTermBase):
    """Positive bounded single support plus genuine alternating touchdown events.

    Initial landings, contact chatter, simultaneous hopping and repeated
    touchdowns by the same foot do not earn the event bonus. Long one-legged
    balancing loses the support reward. All state clears on an environment reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.seen_contact = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.last_landing = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.seen_contact[ids] = False
        self.last_landing[ids] = -1

    def __call__(self, env, command_name, sensor_cfg: SceneEntityCfg,
                 minimum_air_time=0.08, maximum_air_time=0.6,
                 support_weight=0.75, touchdown_bonus=0.20):
        sensor = env.scene[sensor_cfg.name]
        ids = sensor_cfg.body_ids
        air = sensor.data.current_air_time[:, ids]
        stance = sensor.data.current_contact_time[:, ids]
        contact = stance > 0.0
        first_contact = sensor.compute_first_contact(env.step_dt)[:, ids]
        last_air = sensor.data.last_air_time[:, ids]
        single_support = contact.sum(-1) == 1
        in_mode_time = torch.where(contact, stance, air).amin(-1)
        swing_time = air.amax(-1)
        fade = ((maximum_air_time - swing_time) / 0.15).clamp(0.0, 1.0)
        support = (in_mode_time / 0.3).clamp(0.0, 1.0) * single_support * fade

        # The other foot must already support the robot, rather than land in a hop.
        other_support = contact.flip(-1) & ~first_contact.flip(-1)
        valid = (first_contact & self.seen_contact & other_support
                 & (last_air >= minimum_air_time - 1e-6) & (last_air <= maximum_air_time + 1e-6))
        foot = valid.long().argmax(-1)
        alternating = valid.sum(-1).eq(1) & (foot != self.last_landing)
        self.last_landing[alternating] = foot[alternating]
        self.seen_contact |= contact

        command = env.command_manager.get_command(command_name)[:, 0]
        progress = (horizontal_forward_velocity(env) * command.sign()
                    / command.abs().clamp_min(0.1)).clamp(0.0, 1.0)
        # A weak onset signal can initiate a step; full bonuses require motion.
        forward_gate = 0.2 + 0.8 * progress
        moving = command.abs() > 0.1
        # RewardManager multiplies rates by dt. The event earns touchdown_bonus
        # per valid step, independent of the control period.
        return ((support_weight * support + touchdown_bonus * alternating / env.step_dt)
                * forward_gate * upright_score(env) * moving)


def joint_torque_saturation(env, asset_cfg: SceneEntityCfg, soft_ratio=0.75):
    """Use each joint's actual effort limit, including the lower ankle limits."""
    robot = env.scene[asset_cfg.name]
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1e-6)
    requested = robot.data.computed_torque[:, asset_cfg.joint_ids].abs()
    return (requested / limits - soft_ratio).clamp(min=0.0, max=5.0).square().sum(-1)
