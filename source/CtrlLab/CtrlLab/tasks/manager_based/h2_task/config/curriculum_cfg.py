from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from ..mdp import lin_vel_cmd_levels, ang_vel_cmd_levels, terrain_levels_lifetime


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(terrain_levels_lifetime)
    # Legacy comparison: replace func with lin_vel_cmd_levels_legacy and params={};
    # the old reward gate does not accept the new progress thresholds.
    lin_vel_cmd_levels = CurrTerm(
        lin_vel_cmd_levels,
        params={
            "min_episodes": 32,
            # None uses the actual configured maximum episode length in steps.
            "evaluation_interval_steps": None,
            "success_rate": 0.8,
            "min_progress_ratio": 0.8,
            "max_progress_ratio": 1.2,
            "min_episode_fraction": 0.98,
            "min_active_time": 2.0,
            "min_target_distance": 0.5,
            "expansion_step": 0.1,
        },
    )
    # ang_vel_cmd_levels = CurrTerm(ang_vel_cmd_levels)
