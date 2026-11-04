from typing import Sequence

import torch
from isaaclab.assets import Articulation

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from ..locomotion_progress import curriculum_ready


def terrain_levels_lifetime(
        env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # compute robot lifetime
    life_time = env.episode_length_buf[env_ids] / env.max_episode_length
    move_up_lifetime = life_time > 0.98
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up_walk = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked far enough progress to harder terrains
    if bool(getattr(env.cfg, "progressive_walking_curriculum", False)):
        progress = env.locomotion_progress
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        ratio = progress.projected_distance[ids] / progress.target_distance[ids].clamp_min(1e-6)
        walk = ((progress.landings[ids] >= 2).all(-1)
                & (progress.target_distance[ids] >= 0.5) & (ratio >= 0.8) & (ratio <= 1.2))
        move_up = (move_up_lifetime & walk & progress.valid_episode[ids]
                   & (progress.episode_generation[ids] == progress.generation)
                   & ~env.termination_manager.terminated[ids]
                   & (getattr(env, "_h2_walking_curriculum_stage", 0) >= 1))
    else:
        move_up = move_up_lifetime & move_up_walk
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = life_time < 0.6
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "tracking_lin_vel_x",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Retain the argument for configuration compatibility; rewards alone cannot
    # distinguish slow-command tracking by standing from successful walking.
    del env_ids, reward_term_name
    last_check = getattr(env, "_h2_command_curriculum_last_check", 0)
    if env.common_step_counter - last_check >= env.max_episode_length:
        env._h2_command_curriculum_last_check = env.common_step_counter
        metrics = env.locomotion_progress.summary()
        env._h2_command_curriculum_metrics = metrics
        if curriculum_ready(metrics):
            if bool(getattr(env.cfg, "progressive_walking_curriculum", False)):
                stage = getattr(env, "_h2_walking_curriculum_stage", 0)
                env._h2_walking_curriculum_stage = min(stage + 1, 2)
                command_term.cfg.heading_command = True
                ranges.heading = (-0.5, 0.5) if stage == 0 else limit_ranges.heading
                yaw_limit = 0.1 if stage == 0 else limit_ranges.ang_vel_z[1]
                ranges.ang_vel_z = (-yaw_limit, yaw_limit)
                print(f"[H2 CURRICULUM] physical walking gate passed: stage={env._h2_walking_curriculum_stage}, "
                      f"metrics={metrics}", flush=True)
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()
            # In-flight episodes from the easier command generation cannot be
            # reused to authorize another expansion at the next check.
            env.locomotion_progress.advance_generation()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "heading_tracking",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
