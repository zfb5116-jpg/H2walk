from typing import Sequence

import torch
from isaaclab.assets import Articulation

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from ..progress_curriculum import ProgressCurriculum, expanded_x_range


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
    move_up = move_up_lifetime & move_up_walk
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = life_time < 0.6
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels_legacy(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
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

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def lin_vel_cmd_levels(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "base_velocity",
        min_episodes: int = 32,
        evaluation_interval_steps: int | None = None,
        success_rate: float = 0.8,
        min_progress_ratio: float = 0.8,
        max_progress_ratio: float = 1.2,
        min_episode_fraction: float = 0.98,
        min_active_time: float = 2.0,
        min_target_distance: float = 0.5,
        expansion_step: float = 0.1,
        grouping: str = "global",
        min_group_episodes: int = 32,
) -> torch.Tensor:
    """Expand X speed only after completed episodes make commanded progress.

    This replaces the reward-only gate; ``lin_vel_cmd_levels_legacy`` is the
    explicit baseline. The progress tracker is wired through RobotEnv's actual
    step/reset path, and missing integration is an error rather than fallback.
    """
    tracker = getattr(env, "command_progress_tracker", None)
    if tracker is None:
        raise RuntimeError("Progress command curriculum requires RobotEnv.command_progress_tracker")
    if grouping not in ("global", "terrain_columns"):
        raise ValueError("grouping must be 'global' or 'terrain_columns'")
    group_ids = None
    required_groups = None
    if grouping == "terrain_columns":
        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        if generator is None or not getattr(generator, "curriculum", False):
            raise ValueError("terrain_columns requires a curriculum terrain generator with fixed columns")
        columns = getattr(generator, "num_cols", None)
        if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
            raise ValueError("terrain generator num_cols must be a positive integer")
        required_groups = tuple(range(columns))
        group_ids = getattr(terrain, "terrain_types", None)
        if (not isinstance(group_ids, torch.Tensor) or group_ids.dtype != torch.long
                or group_ids.shape != tracker.elapsed_time.shape
                or group_ids.device != tracker.elapsed_time.device):
            raise ValueError("terrain_types must be a full-environment long tensor on the tracker device")
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    next_x = expanded_x_range(ranges.lin_vel_x, command_term.cfg.limit_ranges.lin_vel_x, expansion_step)
    interval = env.max_episode_length if evaluation_interval_steps is None else evaluation_interval_steps
    settings = (command_name, interval, min_episodes, success_rate, min_progress_ratio,
                max_progress_ratio, min_episode_fraction, min_active_time,
                min_target_distance, expansion_step, grouping, required_groups, min_group_episodes)
    evaluator = getattr(env, "_command_progress_curriculum", None)
    if evaluator is None:
        if group_ids is not None:
            # Validate full expected topology once, never infer the required
            # families from a partial reset batch. Fixed-column importer
            # curricula only change rows, not these assignments.
            if tuple(group_ids.unique().cpu().tolist()) != required_groups:
                raise ValueError("terrain_types must cover every expected column; use global for undersized debug scenes")
            env._command_progress_terrain_groups = group_ids
        evaluator = ProgressCurriculum(
            interval_steps=interval, min_episodes=min_episodes, success_rate=success_rate,
            min_progress_ratio=min_progress_ratio, max_progress_ratio=max_progress_ratio,
            min_episode_fraction=min_episode_fraction, min_active_time=min_active_time,
            min_target_distance=min_target_distance,
            generation=getattr(env, "_command_progress_generation", 0),
            required_group_ids=required_groups, min_group_episodes=min_group_episodes,
        )
        env._command_progress_curriculum = evaluator
        env._command_progress_curriculum_settings = settings
    elif env._command_progress_curriculum_settings != settings:
        raise ValueError("Do not change progress curriculum settings within a running environment")
    if group_ids is not None and group_ids is not env._command_progress_terrain_groups:
        raise ValueError("Do not replace fixed terrain_types within a running curriculum")
    if evaluator.generation != getattr(env, "_command_progress_generation", 0):
        raise RuntimeError("Command progress generation is inconsistent")
    result = evaluator.observe(
        tracker, env_ids,
        terminated=env.termination_manager.terminated,
        time_outs=env.termination_manager.time_outs,
        max_episode_length_s=env.max_episode_length_s,
        current_step=env.common_step_counter,
        group_ids=group_ids,
    )
    changed = False
    if result is not None:
        changed = result["passed"] and next_x != tuple(ranges.lin_vel_x)
        if changed:
            ranges.lin_vel_x = next_x
            env._command_progress_generation = evaluator.advance_generation()
        env._command_progress_latest_result = result
    latest = getattr(env, "_command_progress_latest_result", {})
    logs = {
        "Curriculum/command_progress_episodes": latest.get("episodes", 0),
        "Curriculum/command_progress_success_rate": latest.get("success_rate", 0.0),
        "Curriculum/command_progress_mean_ratio": latest.get("mean_progress_ratio", 0.0),
        "Curriculum/command_progress_expanded": float(changed),
        "Curriculum/command_progress_generation": evaluator.generation,
        "Curriculum/command_progress_x_upper": ranges.lin_vel_x[1],
        "Curriculum/command_progress_evaluation_id": evaluator.evaluation_count,
        "Curriculum/command_progress_has_evaluation": float(evaluator.evaluation_count > 0),
        "Curriculum/command_progress_pending_episodes": evaluator.episode_count,
        "Curriculum/command_progress_grouped": float(required_groups is not None),
    }
    for metric in ("active_progress_sum", "inactive_error_sum", "target_distance_sum",
                   "distance_weighted_active_ratio", "distance_weighted_inactive_error_ratio",
                   "distance_weighted_effective_ratio", "min_group_success_rate"):
        logs[f"Curriculum/command_progress_{metric}"] = latest.get(metric, 0.0)
    for group in required_groups or ():
        prefix = f"Curriculum/command_progress_group_{group}"
        logs[f"{prefix}_episodes"] = latest.get("group_episodes", {}).get(group, 0)
        logs[f"{prefix}_success_rate"] = latest.get("group_success_rates", {}).get(group, 0.0)
        logs[f"{prefix}_pending_episodes"] = evaluator.group_episode_counts[group]
    # Emit a fixed key set on every reset: the reference RSL logger uses the
    # first info dictionary's keys. Latest-window metrics repeat deliberately;
    # evaluation_id identifies genuine evaluations, expanded is only a pulse.
    env.extras.setdefault("log", {}).update(logs)
    # Parent reset rebuilds extras; RobotEnv merges this complete snapshot.
    env._command_progress_pending_log = logs
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name: str = "track_ang_vel_z",
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
