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
NUM_JOINTS = 28
TERM_DIM = 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS
PROPRIO_HIST_DIM = HISTORY_LENGTH * TERM_DIM
HEIGHT_SCAN_DIM = 33 * 17
CUR_OBS_DIM = CMD_DIM + 3 + 3 + NUM_JOINTS * 3


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
    """TerProg fusion path adapted to H2's 28-joint proprioception."""

    def __init__(self, obs_dim: int, actions: int, hidden: list[int], activation: str):
        super().__init__()
        self.obs_dim = obs_dim
        self.proprio_start = CMD_DIM
        self.proprio_end = CMD_DIM + PROPRIO_HIST_DIM
        self.scan_start = self.proprio_end
        self.scan_end = self.scan_start + HEIGHT_SCAN_DIM
        self.velocity_encoder = MLP(PROPRIO_HIST_DIM, 32, [128, 64], activation)
        self.contact_encoder = MLP(PROPRIO_HIST_DIM, 32, [256, 128, 64], activation)
        self.predict_encoder = MLP(PROPRIO_HIST_DIM, 32, [256, 128, 64], activation)
        self.terrain_encoder = MLP(HEIGHT_SCAN_DIM, 32, [256, 128, 64], activation)
        fusion_in = CUR_OBS_DIM + 32 + 32 + 32 + 32
        self.fusion = MLP(fusion_in, 64, [256, 128], activation)
        self.actor = MLP(CUR_OBS_DIM + 32 + 64 + 32, actions, hidden, activation)

    def _current_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # ObservationManager appends each history with the newest frame last.
        p = obs[:, self.proprio_start : self.proprio_end]
        i = 0
        latest = []
        for width in (3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS):
            latest.append(p[:, i + (HISTORY_LENGTH - 1) * width : i + HISTORY_LENGTH * width])
            i += HISTORY_LENGTH * width
        return torch.cat((obs[:, :CMD_DIM], *latest), dim=-1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs[:, self.proprio_start : self.proprio_end]
        terrain = obs[:, self.scan_start : self.scan_end]
        current = self._current_obs(obs)
        velocity = self.velocity_encoder(proprio).detach()
        contact = self.contact_encoder(proprio).detach()
        prediction = self.predict_encoder(proprio).detach()
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
        **_: object,
    ):
        super().__init__()
        if num_actor_obs < CMD_DIM + PROPRIO_HIST_DIM + HEIGHT_SCAN_DIM:
            raise ValueError(
                f"H2 terrain actor needs at least {CMD_DIM + PROPRIO_HIST_DIM + HEIGHT_SCAN_DIM} "
                f"policy observations, got {num_actor_obs}"
            )
        self.actor = H2TerrainEncoderActor(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
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
