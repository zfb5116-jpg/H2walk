"""Policy-only stair curriculum initialized from the accepted flat context policy."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from math import cos
from .stair_ascent_cfg import StairAscentEnvCfg, StairAscentRewardsCfg
from .stair_task_context import append_stair_task_context
from . import stair_ascent_rewards as rewards
from . import stair_clearance_rewards as clearance


@configclass
class StairTransferRewardsCfg(StairAscentRewardsCfg):
    failure = RewTerm(func=rewards.stair_first_qualified_failure, weight=-20.0)
    platform_hold = RewTerm(func=rewards.stair_qualified_platform_hold, weight=1.5)
    world_forward_excess_speed = RewTerm(func=rewards.stair_forward_excess_speed, weight=-33.333333333333336)
    outward_tilt_growth = RewTerm(func=clearance.stair_outward_tilt_growth,
        weight=-20.0/(cos(0.0)-cos(.75)), params={'margin_angle':0.0})
    body_angular_rate = RewTerm(func=clearance.stair_body_angular_rate, weight=-.025)
    swing_clearance_deficit = RewTerm(func=clearance.stair_swing_clearance_deficit, weight=-4.0)


@configclass
class StairTransferEnvCfg(StairAscentEnvCfg):
    stair_task_context_dim: int = 5
    stair_strict_episode_force: bool = True

    def __post_init__(self):
        super().__post_init__()
        append_stair_task_context(self.observations)
        self.rewards = StairTransferRewardsCfg()


@configclass
class StairTransferPlayEnvCfg(StairTransferEnvCfg):
    stair_minimum_height: float = 0.15
    stair_maximum_height: float = 0.15
    stair_height_levels: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.curriculum.terrain_levels = None
