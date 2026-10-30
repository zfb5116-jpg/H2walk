"""H2-compatible TerProg-style actor with terrain perception.

The canvas network was written for a 12-DoF quadruped.  This implementation
keeps its structure while deriving all dimensions from the H2 observation
layout: proprio history, velocity/contact/predictor encoders, terrain encoder,
fusion bottleneck, and a low-level action head.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


CMD_DIM = 3
HISTORY_LENGTH = 5
HEIGHT_SCAN_DIM = 33 * 17


def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[name.lower()]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list[int], activation: str = "elu"):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for width in hidden:
            layers.extend((nn.Linear(last, width), _activation(activation)))
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class H2TerrainEncoderActor(nn.Module):
    """TerProg fusion path with dimensions derived from the H2 action layout."""

    def __init__(
        self,
        obs_dim: int,
        actions: int,
        hidden: list[int],
        activation: str,
        *,
        history_length: int = HISTORY_LENGTH,
        num_joints: int | None = None,
        height_scan_dim: int = HEIGHT_SCAN_DIM,
        detach_estimates: bool = False,
    ):
        super().__init__()
        self.num_joints = actions if num_joints is None else num_joints
        if self.num_joints < actions:
            raise ValueError("Observed joint count cannot be smaller than controlled action count")
        self.history_length = history_length
        self.height_scan_dim = height_scan_dim
        self.detach_estimates = detach_estimates
        self.term_widths = (3, 3, self.num_joints, self.num_joints, self.num_joints)
        self.proprio_hist_dim = self.history_length * sum(self.term_widths)
        self.current_obs_dim = CMD_DIM + sum(self.term_widths)
        expected_obs_dim = CMD_DIM + self.proprio_hist_dim + self.height_scan_dim
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"H2 terrain actor observation layout mismatch: expected {expected_obs_dim} "
                f"(3 commands + {self.history_length}x proprio + {self.height_scan_dim} height samples), "
                f"got {obs_dim}. Check the configured joint order, history length, and height scanner."
            )
        self.obs_dim = obs_dim
        self.proprio_start = CMD_DIM
        self.proprio_end = CMD_DIM + self.proprio_hist_dim
        self.scan_start = self.proprio_end
        self.scan_end = self.scan_start + self.height_scan_dim
        self.velocity_encoder = MLP(self.proprio_hist_dim, 32, [128, 64], activation)
        self.contact_encoder = MLP(self.proprio_hist_dim, 32, [256, 128, 64], activation)
        self.predict_encoder = MLP(self.proprio_hist_dim, 32, [256, 128, 64], activation)
        self.terrain_encoder = MLP(self.height_scan_dim, 32, [256, 128, 64], activation)
        fusion_in = self.current_obs_dim + 32 + 32 + 32 + 32
        self.fusion = MLP(fusion_in, 64, [256, 128], activation)
        self.actor = MLP(self.current_obs_dim + 32 + 64 + 32, actions, hidden, activation)

    def _current_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # ObservationManager appends each history with the newest frame last.
        p = obs[:, self.proprio_start : self.proprio_end]
        i = 0
        latest = []
        for width in self.term_widths:
            latest.append(
                p[:, i + (self.history_length - 1) * width : i + self.history_length * width]
            )
            i += self.history_length * width
        return torch.cat((obs[:, :CMD_DIM], *latest), dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs[:, self.proprio_start : self.proprio_end]
        terrain = obs[:, self.scan_start : self.scan_end]
        current = self._current_obs(obs)
        velocity = self.velocity_encoder(proprio)
        contact = self.contact_encoder(proprio)
        prediction = self.predict_encoder(proprio)
        if self.detach_estimates:
            velocity = velocity.detach()
            contact = contact.detach()
            prediction = prediction.detach()
        terrain_latent = self.terrain_encoder(terrain)
        fusion = self.fusion(torch.cat((current, velocity, contact, prediction, terrain_latent), dim=-1))
        return self.actor(torch.cat((current, velocity, fusion, terrain_latent), dim=-1))


class H2ActorCriticOur(nn.Module):
    """RSL-RL-compatible actor-critic used by the rough H2 task."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_joints: int | None = None,
        history_length: int = HISTORY_LENGTH,
        height_scan_dim: int = HEIGHT_SCAN_DIM,
        detach_estimates: bool = False,
        **_: object,
    ):
        super().__init__()
        self.actor = H2TerrainEncoderActor(
            num_actor_obs,
            num_actions,
            list(actor_hidden_dims),
            activation,
            history_length=history_length,
            num_joints=num_joints,
            height_scan_dim=height_scan_dim,
            detach_estimates=detach_estimates,
        )
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        del dones

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        del kwargs
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor):
        return self.actor(observations)

    def get_actions_log_prob(self, actions: torch.Tensor):
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        del kwargs
        return self.critic(critic_observations)


def register_rsl_rl_policy() -> None:
    """Make the custom class discoverable by Isaac Lab's RSL-RL resolver."""
    try:
        import rsl_rl.modules as modules
    except ImportError as exc:
        raise ImportError("Install Isaac Lab/RSL-RL before importing H2ActorCriticOur") from exc
    modules.H2ActorCriticOur = H2ActorCriticOur
    # RSL-RL 2.3 resolves the configured name in this module's globals.
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.H2ActorCriticOur = H2ActorCriticOur
