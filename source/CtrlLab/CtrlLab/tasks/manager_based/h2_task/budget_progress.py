"""Finite route-progress budget and displacement costs, without Isaac imports.

The route is fixed by the first valid pose and body-frame command. New spatial
high water is consumed immediately, including unpaid excess: rushing ahead
cannot create a bank that pays out while standing or revisiting old positions.
These tensor properties do not establish that a robot can walk.
"""
from __future__ import annotations

import math

import torch


BUDGET_PROGRESS_PARAMETERS = {
    "semantic_version": "budget_v3",
    "min_command_speed": .1,
    "max_step_speed": 5.,
    "overspeed_ratio": 1.2,
    "progress_budget": 20.,
    "retreat_weight": 20.,
    "overspeed_weight": .25,
    "lateral_weight": .25,
    "command_change_tolerance": 1e-6,
    "position_continuity_tolerance": 1e-5,
    "quaternion_min_norm": 1e-6,
    "transition_distance_tolerance": 1e-6,
}


class BudgetProgressState:
    """One fixed command route and one finite budget per natural episode.

    Reward outputs are rates for a manager that multiplies by control dt.
    ``advance_rate`` integrates to at most one and its external weight is the
    point budget. ``retreat_rate`` measures actual backward displacement,
    including motion below the start; it is not a high-water difference.
    Position-derived rewards never use the reported root velocity. Its
    consistency error is diagnostic only, and unknown diagnostics are NaN.
    """

    def __init__(self, num_envs: int, device: str | torch.device,
                 goal_distance: float, parameters: dict | None = None):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if (isinstance(goal_distance, bool) or not isinstance(goal_distance, (int, float))
                or not math.isfinite(goal_distance) or goal_distance <= 0.):
            raise ValueError("goal_distance must be finite and positive")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary")
        self.parameters = {**BUDGET_PROGRESS_PARAMETERS, **(parameters or {})}
        for name in BUDGET_PROGRESS_PARAMETERS:
            if name == "semantic_version":
                continue
            value = self.parameters[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0.):
                raise ValueError(f"{name} must be finite and positive")
        if self.parameters["overspeed_ratio"] < 1.:
            raise ValueError("overspeed_ratio must be at least one")
        self.num_envs, self.device = num_envs, torch.device(device)
        self.goal_distance = float(goal_distance)
        for name in ("advance_rate", "retreat_rate", "overspeed_cost", "lateral_cost",
                     "raw_route_progress", "highwater", "paid_distance", "displacement_velocity_error", "_speed"):
            setattr(self, name, torch.zeros(num_envs, device=device))
        for name in ("route_valid", "command_valid", "velocity_diagnostic_valid", "_initialized",
                     "_invalid", "_has_previous"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool, device=device))
        self._origin = torch.zeros(num_envs, 2, device=device)
        self._direction = torch.zeros_like(self._origin)
        self._expected_position = torch.zeros_like(self._origin)
        self._reference_command = torch.zeros(num_envs, 3, device=device)
        self._pending = False
        self._dt = 0.
        for name in ("forward_velocity", "lateral_velocity", "lateral_distance"):
            setattr(self, name, torch.full((num_envs,), torch.nan, device=device))

    @property
    def direction(self) -> torch.Tensor:
        """Fixed world route direction; NaN until an anchor is established."""
        return torch.where(self._initialized[:, None], self._direction, torch.nan)

    @property
    def command_speed(self) -> torch.Tensor:
        """Original route command speed; changing commands never reanchor it."""
        return torch.where(self._initialized, self._speed, torch.nan)

    def _tensor(self, value: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or value.device != self.device):
            raise ValueError(f"{name} must be a floating tensor on {self.device} with shape {shape}")

    @torch.no_grad()
    def begin_transition(self, root_xy, root_quat_w, command, dt) -> None:
        """Capture the executed command and pre-physics position once."""
        if self._pending:
            raise RuntimeError("Previous budget progress transition is pending")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0.:
            raise ValueError("dt must be finite and positive")
        self._tensor(root_xy, "root_xy", (self.num_envs, 2))
        self._tensor(root_quat_w, "root_quat_w", (self.num_envs, 4))
        self._tensor(command, "command", (self.num_envs, 3))
        p = self.parameters
        finite_command = torch.isfinite(command).all(dim=-1)
        safe_command = torch.where(finite_command[:, None], command, 0.)
        speed = safe_command[:, :2].norm(dim=-1)
        unchanged = (safe_command - self._reference_command).abs().amax(dim=-1) <= p["command_change_tolerance"]
        self.command_valid.copy_(finite_command & torch.isfinite(speed)
                                 & (speed >= p["min_command_speed"]) & (~self._initialized | unchanged))
        finite_quat = torch.isfinite(root_quat_w).all(dim=-1)
        safe_quat = torch.where(finite_quat[:, None], root_quat_w, 0.)
        norm = safe_quat.norm(dim=-1)
        quat_valid = finite_quat & torch.isfinite(norm) & (norm > p["quaternion_min_norm"])
        quat = torch.where(quat_valid[:, None], safe_quat / norm.clamp_min(p["quaternion_min_norm"])[:, None], 0.)
        w, x, y, z = quat.unbind(-1)
        yaw = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
        world_command = torch.stack((yaw.cos()*safe_command[:, 0] - yaw.sin()*safe_command[:, 1],
                                     yaw.sin()*safe_command[:, 0] + yaw.cos()*safe_command[:, 1]), dim=-1)
        direction = world_command / speed.clamp_min(p["min_command_speed"])[:, None]
        continuous = (~self._has_previous | (torch.isfinite(self._expected_position).all(dim=-1)
                      & ((root_xy - self._expected_position).norm(dim=-1) <= p["position_continuity_tolerance"])))
        valid_input = torch.isfinite(root_xy).all(dim=-1) & quat_valid & self.command_valid & continuous
        fresh = ~self._initialized & ~self._invalid & valid_input
        self._origin = torch.where(fresh[:, None], root_xy, self._origin)
        self._direction = torch.where(fresh[:, None], direction, self._direction)
        self._reference_command = torch.where(fresh[:, None], safe_command, self._reference_command)
        self._speed = torch.where(fresh, speed, self._speed)
        self._initialized |= fresh
        self._invalid |= ~valid_input
        self.route_valid.copy_(self._initialized & ~self._invalid)
        self._position = root_xy.clone()
        self._dt = float(dt)
        self._pending = True
        self._clear_outputs()

    @torch.no_grad()
    def finish_transition(self, root_xy, root_velocity_w) -> None:
        """Consume finite high water even when this transition earns nothing."""
        if not self._pending:
            raise RuntimeError("Budget progress requires begin_transition before finish_transition")
        self._tensor(root_xy, "root_xy", (self.num_envs, 2))
        if (not isinstance(root_velocity_w, torch.Tensor) or root_velocity_w.ndim != 2
                or root_velocity_w.shape[0] != self.num_envs or root_velocity_w.shape[1] < 2
                or not root_velocity_w.is_floating_point() or root_velocity_w.device != self.device):
            raise ValueError("root_velocity_w must be a floating [env, >=2] tensor on the state device")
        p = self.parameters
        delta = root_xy - self._position
        finite_delta = torch.isfinite(delta).all(dim=-1)
        velocity = delta / self._dt
        forward = (velocity*self._direction).sum(dim=-1)
        side = velocity[:, 0]*(-self._direction[:, 1]) + velocity[:, 1]*self._direction[:, 0]
        speed = self._speed.clamp_min(p["min_command_speed"])
        overspeed = (forward/speed - p["overspeed_ratio"]).clamp_min(0.).square()
        lateral = (side/speed).square()
        finite_cost = torch.isfinite(overspeed) & torch.isfinite(lateral)
        plausible = finite_delta & (delta.norm(dim=-1) <= p["max_step_speed"]*self._dt + p["transition_distance_tolerance"])
        self._invalid |= ~plausible | ~finite_cost
        self.route_valid.copy_(self._initialized & ~self._invalid)

        signed_position = ((root_xy - self._origin)*self._direction).sum(dim=-1)
        observed = self._initialized & torch.isfinite(root_xy).all(dim=-1) & torch.isfinite(signed_position)
        self.raw_route_progress.copy_(torch.where(observed, signed_position, torch.nan))
        lateral_position = ((root_xy - self._origin) * torch.stack(
            (-self._direction[:, 1], self._direction[:, 0]), dim=-1)).sum(dim=-1)
        self.lateral_distance.copy_(torch.where(observed & torch.isfinite(lateral_position),
                                                lateral_position, torch.nan))
        self.forward_velocity.copy_(torch.where(self._initialized & finite_delta & torch.isfinite(forward),
                                                forward, torch.nan))
        self.lateral_velocity.copy_(torch.where(self._initialized & finite_delta & torch.isfinite(side),
                                                side, torch.nan))
        candidate = signed_position.clamp(0., self.goal_distance)
        next_highwater = torch.where(observed, torch.maximum(self.highwater, candidate), self.highwater)
        newly_observed = (next_highwater - self.highwater).clamp_min(0.)
        credited = torch.minimum(newly_observed, self._speed*self._dt)
        credited = torch.where(self.route_valid, credited, 0.)
        self.highwater.copy_(next_highwater)
        self.paid_distance += credited
        self.advance_rate.copy_(credited / (self.goal_distance*self._dt))
        self.retreat_rate.copy_(torch.where(finite_delta & torch.isfinite(forward) & self._initialized,
                                            (-forward).clamp_min(0.)/self.goal_distance, 0.))
        self.overspeed_cost.copy_(torch.where(finite_cost & finite_delta & self._initialized, overspeed, 0.))
        self.lateral_cost.copy_(torch.where(finite_cost & finite_delta & self._initialized, lateral, 0.))
        error = (root_velocity_w[:, :2] - velocity).norm(dim=-1)
        diagnostic_valid = (torch.isfinite(root_velocity_w).all(dim=-1) & finite_delta
                            & torch.isfinite(error))
        self.velocity_diagnostic_valid.copy_(diagnostic_valid)
        self.displacement_velocity_error.copy_(torch.where(diagnostic_valid, error, torch.nan))
        self._expected_position.copy_(root_xy)
        self._has_previous.fill_(True)
        self._pending = False

    def _clear_outputs(self) -> None:
        for name in ("advance_rate", "retreat_rate", "overspeed_cost", "lateral_cost"):
            getattr(self, name).zero_()
        self.velocity_diagnostic_valid.zero_()
        self.displacement_velocity_error.fill_(torch.nan)
        for name in ("forward_velocity", "lateral_velocity", "lateral_distance"):
            getattr(self, name).fill_(torch.nan)

    @torch.no_grad()
    def reset(self, ids) -> None:
        """Reset only natural new-episode rows; do not use this as fault recovery."""
        if self._pending:
            raise RuntimeError("Cannot reset budget progress while a transition is pending")
        selected = torch.as_tensor(ids, device=self.device)
        if selected.ndim != 1 or (selected.numel() and selected.dtype not in (torch.int32, torch.int64)):
            raise ValueError("reset ids must be a one-dimensional integer sequence")
        selected = selected.to(torch.long)
        if bool(((selected < 0) | (selected >= self.num_envs)).any()):
            raise ValueError("reset ids are out of range")
        for name in ("advance_rate", "retreat_rate", "overspeed_cost", "lateral_cost", "raw_route_progress",
                     "highwater", "paid_distance", "_speed", "route_valid", "command_valid",
                     "velocity_diagnostic_valid", "_initialized", "_invalid", "_has_previous", "_origin",
                     "_direction", "_expected_position", "_reference_command"):
            getattr(self, name)[selected] = 0
        self.displacement_velocity_error[selected] = torch.nan
        for name in ("forward_velocity", "lateral_velocity", "lateral_distance"):
            getattr(self, name)[selected] = torch.nan

    @torch.no_grad()
    def abort_step(self) -> None:
        """Invalidate interrupted episodes without replenishing their budgets."""
        self._pending = False
        self._invalid.fill_(True)
        self.route_valid.zero_()
        self._clear_outputs()
