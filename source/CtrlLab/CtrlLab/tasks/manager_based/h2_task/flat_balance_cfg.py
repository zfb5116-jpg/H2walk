"""Independent autonomous direct-flat balance learning configuration."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .direct_initialization import DirectInitializationEnvCfg
from . import flat_balance_env as rewards


@configclass
class FlatBalanceRewardsCfg:
    signed_forward = RewTerm(func=rewards.flat_signed_progress,weight=10.)
    valid_alternating_landing = RewTerm(func=rewards.flat_valid_landing,weight=.2)
    failure = RewTerm(func=rewards.flat_failure,weight=-20.)
    world_xy_velocity_error = RewTerm(func=rewards.flat_velocity_error,weight=-.5)
    world_heading_error = RewTerm(func=rewards.flat_heading_error,weight=-.1)
    tilt_margin = RewTerm(func=rewards.flat_tilt_margin,weight=-1.)
    body_angular_rate = RewTerm(func=rewards.flat_body_angular_rate,weight=-.025)
    supported_slip = RewTerm(func=rewards.flat_support_slip,weight=-1.)
    normalized_drive_request = RewTerm(func=rewards.flat_drive_request,weight=-.02)
    bounded_target_change = RewTerm(func=rewards.flat_target_change,weight=-.02)
    native_overspeed = RewTerm(func=rewards.flat_native_overspeed,weight=-.1)
    actual_soft_excess = RewTerm(func=rewards.flat_soft_excess,weight=-5.)


@configclass
class FlatBalanceEnvCfg(DirectInitializationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards=FlatBalanceRewardsCfg()
        self.episode_length_s=20.
        command=self.commands.base_velocity
        command.heading_command=False
        command.rel_standing_envs=0.
        command.resampling_time_range=(20.,20.)
        for ranges in (command.ranges,command.limit_ranges):
            ranges.lin_vel_x=(.3,.3)
            ranges.lin_vel_y=ranges.ang_vel_z=ranges.heading=(0.,0.)


@configclass
class FlatBalancePlayEnvCfg(FlatBalanceEnvCfg):
    pass
