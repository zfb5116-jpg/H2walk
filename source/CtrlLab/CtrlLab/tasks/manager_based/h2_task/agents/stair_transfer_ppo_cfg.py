"""Fresh guarded split PPO; the accepted flat model supplies weights and norms only."""
from isaaclab.utils import configclass
from .flat_balance_ppo_cfg import H2FlatBalancePPORunnerCfg


@configclass
class H2StairTransferPPORunnerCfg(H2FlatBalancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'h2_guarded_stair_transfer'
        self.max_iterations = 5000
        self.save_interval = 500
        self.algorithm.class_name = 'H2GuardedStairPPO'
        self.algorithm.trial_protocol = 'accepted_flat_context_stair_transfer_v1'
