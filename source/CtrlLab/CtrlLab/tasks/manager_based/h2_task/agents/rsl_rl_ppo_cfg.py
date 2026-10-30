# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
)

from ..networks import register_rsl_rl_policy

register_rsl_rl_policy()


@configclass
class H2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "h2_flat_manager_based"
    empirical_normalization = True
    clip_actions = 100.0
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
        rnd_cfg=RslRlRndCfg(
            weight=25.0,
            reward_normalization=True,
            state_normalization=True,
        ),
    )


@configclass
class H2TerrainActorCriticCfg(RslRlPpoActorCriticCfg):
    # Proprioception observes the whole body; the stair policy controls 12 leg joints.
    num_joints: int = 29


@configclass
class H2RoughPPORunnerCfg(H2FlatPPORunnerCfg):
    max_iterations = 8000
    experiment_name = "h2_rough_manager_based"
    policy = H2TerrainActorCriticCfg(
        class_name="H2ActorCriticOur",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class H2StairsPPORunnerCfg(H2RoughPPORunnerCfg):
    """The same network and PPO, with gentler initial torque exploration."""

    def __post_init__(self):
        self.policy.init_noise_std = 0.5
