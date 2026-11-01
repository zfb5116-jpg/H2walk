"""Frozen legacy empirical prefixes and a separate fixed task-context block.

The five new channels have no fitted empirical moments or sample count. This
module imports no task, environment, current-ROOT metadata checker or simulator.
"""
from __future__ import annotations

import torch
from torch import nn
from rsl_rl.modules.normalizer import EmpiricalNormalization


NORMALIZATION_KIND = "frozen_empirical_prefix_fixed_task_context_v1"
CONTEXT_ORDER = (
    "root_link_heading_vx_m_s",
    "root_link_heading_vy_m_s",
    "world_plusX_heading_x",
    "world_plusX_heading_y",
    "force_credit_eligible",
)


class FrozenPrefixContextNorm(nn.Module):
    """Preserve one real saved empirical prefix, then scale a raw SI suffix."""

    def __init__(self, prefix_width, context_dim=5, eps=.01, until=1e8):
        super().__init__()
        if type(prefix_width) is not int or prefix_width not in (1029, 866):
            raise ValueError("Context normalization requires an original H2 actor/critic prefix")
        if type(context_dim) is not int or context_dim != 5 or eps != .01 or until != 1e8:
            raise ValueError("Undeclared frozen-prefix/fixed-context normalization ABI")
        self.prefix_width = prefix_width
        self.context_dim = context_dim
        self.input_width = prefix_width + context_dim
        self.prefix = EmpiricalNormalization(prefix_width, eps=eps, until=until)
        self.register_buffer("fixed_context_scale", torch.tensor([1/.3, 1/.3, 1., 1., 1.], dtype=torch.float32))
        self.eval()

    @property
    def prefix_count(self):
        """This count belongs only to the original empirical prefix."""
        return self.prefix.count

    def train(self, mode=True):
        # RSL toggles all normalizers into train mode at every learn entry.
        # A frozen block must remain frozen even under that ordinary call.
        del mode
        return super().train(False)

    def update(self, value):
        del value
        raise RuntimeError("The inherited prefix and fixed context cannot fit new empirical statistics")

    def load_prefix_state_dict(self, state):
        if set(state) != {"_mean", "_var", "_std", "count"}:
            raise ValueError("Expected the saved plain empirical prefix state only")
        if state["count"].shape != torch.Size([]) or state["count"].dtype != torch.long or int(state["count"]) != 24576000:
            raise ValueError("The reviewed prefix empirical sample count changed")
        for key in ("_mean", "_var", "_std"):
            if (tuple(state[key].shape) != (1, self.prefix_width) or state[key].dtype != torch.float32
                    or not bool(torch.isfinite(state[key]).all())):
                raise ValueError("Invalid saved empirical prefix: " + key)
        if not bool((state["_std"] >= 0).all() & (state["_var"] >= 0).all()):
            raise ValueError("Invalid saved empirical prefix variance")
        self.prefix.load_state_dict(state, strict=True)
        self.eval()

    def forward(self, value):
        if value.ndim != 2 or value.shape[-1] != self.input_width:
            raise ValueError("Raw input does not match the frozen-prefix/context width")
        if self.training or self.prefix.training:
            raise RuntimeError("Frozen-prefix context normalization entered training mode")
        return torch.cat((self.prefix(value[:, :self.prefix_width]),
                          value[:, self.prefix_width:] * self.fixed_context_scale), dim=-1)

    def inverse(self, value):
        if value.ndim != 2 or value.shape[-1] != self.input_width:
            raise ValueError("Normalized input does not match the declared width")
        return torch.cat((self.prefix.inverse(value[:, :self.prefix_width]),
                          value[:, self.prefix_width:] / self.fixed_context_scale), dim=-1)


def make_task_context_normalizers(device):
    return {name: FrozenPrefixContextNorm(width).to(device).eval()
            for name, width in (("actor", 1029), ("critic", 866))}


def normalization_metadata(normalizers):
    """Report actual block state; this is not a full-width fitted-count claim."""
    if set(normalizers) != {"actor", "critic"}:
        raise ValueError("Expected exactly the actor and critic normalization blocks")
    actor, critic = normalizers["actor"], normalizers["critic"]
    if not all(isinstance(item, FrozenPrefixContextNorm) for item in (actor, critic)):
        raise ValueError("The explicit-context profile requires the real block normalizers")
    if (actor.prefix_width, critic.prefix_width) != (1029, 866):
        raise ValueError("Actor and critic empirical prefixes were exchanged")
    scales = [item.fixed_context_scale.detach().cpu().tolist() for item in (actor, critic)]
    if scales[0] != scales[1]:
        raise ValueError("Actor and critic fixed context scales differ")
    result = dict(kind=NORMALIZATION_KIND, raw_context_width=5, context_order=list(CONTEXT_ORDER),
                  fixed_context_scale=scales[0], context_statistics_fitted=False,
                  context_empirical_count=None,
                  prefix_empirical_counts={name: int(norm.prefix_count) for name, norm in normalizers.items()},
                  frozen=all(not norm.training and not norm.prefix.training for norm in normalizers.values()))
    for name, norm in normalizers.items():
        result[name] = dict(prefix_width=norm.prefix_width, input_width=norm.input_width,
                            prefix_epsilon=norm.prefix.eps, prefix_until=norm.prefix.until)
    return result
