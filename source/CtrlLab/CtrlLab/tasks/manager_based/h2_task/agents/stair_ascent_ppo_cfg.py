"""The existing H2 architecture and PPO for direct policy stair control."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import H2StairsPPORunnerCfg


@configclass
class H2StairAscentPPORunnerCfg(H2StairsPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h2_stair_ascent_policy"
        self.max_iterations = 8000
        self.save_interval = 500
        # Preserve the RND module/state layout, but don't pay novelty for falls.
        self.algorithm.rnd_cfg.weight = 0.0

