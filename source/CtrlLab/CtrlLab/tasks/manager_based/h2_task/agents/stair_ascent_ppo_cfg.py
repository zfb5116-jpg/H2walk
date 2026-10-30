"""The existing H2 architecture and PPO for direct policy stair control."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import H2StairsPPORunnerCfg


@configclass
class H2StairAscentPPORunnerCfg(H2StairsPPORunnerCfg):
    stair_exploration_budget = {
        "maximum_angle_std": [0.02, 0.015, 0.015, 0.02, 0.01, 0.0075] * 2,
        "initial_fraction": 0.5,
        "minimum_fraction": 0.05,
    }

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "h2_stair_ascent_policy"
        self.max_iterations = 8000
        self.save_interval = 500
        # Preserve the RND module/state layout, but don't pay novelty for falls.
        self.algorithm.rnd_cfg.weight = 0.0
        # Entropy acts on raw coordinates, whose physical scale is now the
        # complete joint range. Avoid spending the gradient budget on noise.
        self.algorithm.entropy_coef = 1.0e-4
