"""Task rewards for real ascent, without a fixed pose, gait phase or speed target."""

import torch


def stair_event(env, name):
    env._update_stair_progress()
    # RewardManager multiplies by dt. Potential differences and one-off events
    # retain their documented per-transition value at the control frequency.
    return env.stair_progress.events[name] / env.step_dt


def stair_failure(env):
    return env.termination_manager.terminated.float() / env.step_dt


def stair_stance_slip(env):
    env._update_stair_progress()
    return env.stair_progress.current_slip.square()


def stair_tilt_margin(env, free_angle=0.5):
    tilt = torch.acos((-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(-1, 1))
    return (tilt-free_angle).clamp_min(0).square()


def stair_drive_request(env):
    """A small request regularizer; the implicit PD estimate is not measured motor torque."""
    term = env.action_manager.get_term("JointPositionAction")
    robot = env.scene["robot"]
    ids = term._joint_ids
    limit = robot.data.joint_effort_limits[:, ids].clamp_min(1e-6)
    return (robot.data.computed_torque[:, ids] / limit).clamp(-5, 5).square().mean(-1)


def stair_target_change(env):
    # Normalize in the policy's bounded coordinates, rather than penalizing
    # unused raw magnitudes near tanh saturation or imposing a new rate cap.
    return (env.action_manager.action.tanh()-env.action_manager.prev_action.tanh()).square().mean(-1)


def stair_native_overspeed(env):
    term = env.action_manager.get_term("JointPositionAction")
    robot = env.scene["robot"]
    limit = robot.data.joint_vel_limits[:, term._joint_ids].clamp_min(1e-6)
    return (robot.data.joint_vel[:, term._joint_ids].abs()/limit-1).clamp_min(0).square().sum(-1)


def stair_soft_limit_excess(env):
    term = env.action_manager.get_term("JointPositionAction")
    robot = env.scene["robot"]
    q = robot.data.joint_pos[:, term._joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, term._joint_ids]
    return ((limits[..., 0]-q).clamp_min(0)+(q-limits[..., 1]).clamp_min(0)).sum(-1)


def stair_height_curriculum(env, env_ids):
    """Advance complete physical ascents, never reward, standing or tile distance alone."""
    if not hasattr(env, "stair_progress"):
        return 0.0
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    episode = env.stair_progress.last_episode
    terrain = env.scene.terrain
    row = terrain.terrain_levels[ids]
    highest = terrain.cfg.terrain_generator.num_rows-1
    up = episode["valid"][ids] & episode["success"][ids].bool() & (row < highest)
    down = (episode["valid"][ids] & episode["terminated"][ids].bool()
            & (episode["completed_levels"][ids] == 0) & (episode["duration_s"][ids] < 10.0)
            & (row > 0))
    terrain.update_env_origins(ids, up, down)
    return terrain.terrain_levels.float().mean()

