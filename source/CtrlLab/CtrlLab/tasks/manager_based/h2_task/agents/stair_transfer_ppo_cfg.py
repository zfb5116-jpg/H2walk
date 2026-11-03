"""Fresh guarded split PPO; the accepted flat model supplies weights and norms only."""
from isaaclab.utils import configclass
from .flat_balance_ppo_cfg import H2FlatBalancePPORunnerCfg


@configclass
class H2StairTransferPPORunnerCfg(H2FlatBalancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.stair_exploration_budget = dict(self.stair_exploration_budget, initial_fraction=1.0)
        self.experiment_name = 'h2_guarded_stair_transfer'
        self.max_iterations = 1000
        self.save_interval = 500
        self.algorithm.class_name = 'H2GuardedStairPPO'
        self.algorithm.trial_protocol = 'accepted_flat_temporal_credit_stair_v7'
        self.algorithm.lam = .99
