import math

from ..mdp import UniformLevelVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        debug_vis=True,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=UniformLevelVelocityCommandCfg.Ranges(
            # Start close to the official standing controller's low-speed regime.
            # The curriculum expands this range toward the limits below.
            lin_vel_x=(-0.1, 0.3), lin_vel_y=(-0.08, 0.08), ang_vel_z=(-0.3, 0.3), heading=(-math.pi, math.pi)
        ),
        limit_ranges=UniformLevelVelocityCommandCfg.Ranges(
            # H2 target range follows a gradual velocity curriculum, capped at the requested range.
            # Unitree's official H2 velocity config also increases command ranges in stages:
            # https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/config/h2/env_cfgs.py
            lin_vel_x=(-1.0, 1.2), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.8, 0.8), heading=(-math.pi, math.pi)
        ),
    )
