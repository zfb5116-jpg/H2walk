"""Original H2 network with guarded independent actor/critic optimizers."""
from isaaclab.utils import configclass
from .stair_ascent_ppo_cfg import H2StairAscentPPORunnerCfg


@configclass
class H2FlatBalancePPORunnerCfg(H2StairAscentPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='h2_flat_balance_policy'
        self.max_iterations=1000
        self.save_interval=500
        self.algorithm.class_name='H2GuardedPPO'
        self.algorithm.trial_protocol='resource_bounded_reserved_output_proposal_v1'
        self.algorithm.actor_action_budget_penalty_weight=0.
        self.algorithm.actor_action_budget_penalty_epsilon=1.e-6
        self.algorithm.actor_proposal_projection='simultaneous_actual_bank_reserved_linear_v1'
        self.algorithm.actor_proposal_interior_fraction=.5
        self.algorithm.actor_proposal_working_budget_fraction=.5
        self.algorithm.actor_proposal_max_active_rows=None
        self.algorithm.actor_proposal_jacobian_arena_budget_bytes=1073741824
        self.algorithm.actor_proposal_max_solver_sweeps=2048
        self.algorithm.learning_rate=1.e-5
        self.algorithm.actor_learning_rate=1.e-5
        self.algorithm.critic_learning_rate=1.e-3
        self.algorithm.critic_only_iterations=40
        self.algorithm.actor_max_grad_norm=1.
        self.algorithm.critic_max_grad_norm=1.
        self.algorithm.schedule='fixed'
        self.algorithm.entropy_coef=1.e-4
        self.algorithm.num_learning_epochs=5
        self.algorithm.num_mini_batches=4
        self.algorithm.rnd_cfg.weight=0.
        self.algorithm.gamma=.99
        self.algorithm.lam=.95
