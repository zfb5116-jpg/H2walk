"""Per-episode commanded distance and signed physical progress, without Isaac imports.

The environment captures a transition before physics, settles terminal rows
before reset/curriculum, and settles remaining rows after the parent step.
This state is diagnostic curriculum bookkeeping, not a policy observation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class CommandProgressTracker:
    """Accumulate movement in the direction of the command actually executed.

    Pure yaw defines the horizontal command frame. Backtracking is signed, so
    oscillating around a fixed starting point cannot accumulate positive credit.
    An implausible single-step displacement or a pose discontinuity between
    steps invalidates the entire episode. This is a data guard, not a fall or
    contact detector, and cannot distinguish every possible external teleport.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        min_command_speed: float = 0.1,
        max_step_speed: float = 5.0,
    ):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(min_command_speed) or min_command_speed <= 0:
            raise ValueError("min_command_speed must be finite and positive")
        if not math.isfinite(max_step_speed) or max_step_speed <= min_command_speed:
            raise ValueError("max_step_speed must be finite and exceed min_command_speed")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.min_command_speed = min_command_speed
        self.max_step_speed = max_step_speed
        for name in ("elapsed_time", "active_time", "target_distance", "signed_progress",
                     "forward_active_time", "forward_target_distance", "forward_signed_progress",
                     "inactive_forward_error"):
            setattr(self, name, torch.zeros(num_envs, device=self.device))
        self.device = self.elapsed_time.device
        for name in ("had_translation", "had_forward_command", "reset_ready", "reported", "pending", "_has_last_position"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.bool, device=self.device))
        self.valid = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.episode_generation = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._start_position = torch.zeros(num_envs, 2, device=self.device)
        self._last_position = torch.zeros_like(self._start_position)
        self._direction = torch.zeros_like(self._start_position)
        self._forward_direction = torch.zeros_like(self._start_position)
        self._world_command = torch.zeros_like(self._start_position)
        self._speed = torch.zeros(num_envs, device=self.device)
        self._forward_speed = torch.zeros_like(self._speed)
        self._moving = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._forward_moving = torch.zeros_like(self._moving)
        self._input_valid = torch.ones_like(self.valid)
        self._dt = 0.0

    def _check_shape(self, value: torch.Tensor, shape: tuple[int, ...], name: str):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be a tensor with shape {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("env_ids must be a one-dimensional integer sequence")
        ids = ids.to(torch.long)
        if torch.any((ids < 0) | (ids >= self.num_envs)):
            raise ValueError("env_ids out of range")
        return ids.unique()

    @torch.no_grad()
    def begin_step(
        self,
        position_xy: torch.Tensor,
        quaternion_w: torch.Tensor,
        command: torch.Tensor,
        dt: float,
    ):
        """Snapshot pre-physics pose and command, before command resampling."""
        self._check_shape(position_xy, (self.num_envs, 2), "position_xy")
        self._check_shape(quaternion_w, (self.num_envs, 4), "quaternion_w")
        self._check_shape(command, (self.num_envs, 3), "command")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if self.pending.any():
            raise RuntimeError("Previous command-progress transition is still pending")
        pose_finite = torch.isfinite(position_xy).all(dim=-1)
        unchanged = torch.isclose(position_xy, self._last_position, atol=1e-5, rtol=0).all(dim=-1)
        self.valid &= (~self._has_last_position) | unchanged
        quat_finite = torch.isfinite(quaternion_w).all(dim=-1)
        safe_quat = torch.where(quat_finite[:, None], quaternion_w, 0.0)
        quat_norm = safe_quat.norm(dim=-1)
        safe_quat = safe_quat / quat_norm.clamp_min(1e-12)[:, None]
        w, x, y, z = safe_quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        command_finite = torch.isfinite(command).all(dim=-1)
        safe_command = torch.where(command_finite[:, None], command[:, :2], 0.0)
        raw_speed = safe_command.norm(dim=-1)
        speed_finite = torch.isfinite(raw_speed)
        speed = torch.where(speed_finite, raw_speed, 0.0)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        world_command = torch.stack(
            (cos_yaw * safe_command[:, 0] - sin_yaw * safe_command[:, 1],
             sin_yaw * safe_command[:, 0] + cos_yaw * safe_command[:, 1]), dim=-1,
        )
        self._direction.copy_(torch.where(
            speed_finite[:, None], world_command / speed.clamp_min(1e-12)[:, None], 0.0,
        ))
        self._speed.copy_(speed)
        self._moving.copy_(command_finite & (raw_speed >= self.min_command_speed))
        # X range expansion requires actual X tracking. Lateral displacement
        # cannot stand in for forward/backward motion on diagonal commands.
        self._forward_direction.copy_(torch.stack((cos_yaw, sin_yaw), dim=-1)
                                      * safe_command[:, :1].sign())
        self._world_command.copy_(torch.where(speed_finite[:, None], world_command, 0.0))
        self._forward_speed.copy_(safe_command[:, 0].abs())
        self._forward_moving.copy_(command_finite & (self._forward_speed >= self.min_command_speed))
        self._input_valid.copy_(pose_finite & quat_finite & torch.isfinite(quat_norm)
                                & (quat_norm > 1e-6) & command_finite & speed_finite)
        self._start_position.copy_(position_xy)
        self._dt = dt
        self.pending.fill_(True)
        self.reset_ready.fill_(False)

    @torch.no_grad()
    def finish_step(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        position_xy: torch.Tensor,
        *,
        for_reset: bool = False,
    ) -> torch.Tensor:
        """Consume each pending row once, before a terminal pose is reset."""
        self._check_shape(position_xy, (self.num_envs, 2), "position_xy")
        ids = self._ids(env_ids)
        ids = ids[self.pending[ids]]
        if not ids.numel():
            return ids
        delta = position_xy[ids] - self._start_position[ids]
        finite_position = torch.isfinite(position_xy[ids]).all(dim=-1)
        good = self._input_valid[ids] & finite_position & (
            delta.norm(dim=-1) <= self.max_step_speed * self._dt + 1e-6
        )
        self.valid[ids] &= good
        moving = self._moving[ids]
        self.elapsed_time[ids] += self._dt
        self.active_time[ids] += moving * self._dt
        self.target_distance[ids] += self._speed[ids] * self._dt * moving
        projection = (delta * self._direction[ids]).sum(dim=-1)
        self.signed_progress[ids] += torch.where(good & moving, projection, 0.0)
        self.had_translation[ids] |= moving
        forward = self._forward_moving[ids]
        self.forward_active_time[ids] += forward * self._dt
        self.forward_target_distance[ids] += self._forward_speed[ids] * self._dt * forward
        forward_projection = (delta * self._forward_direction[ids]).sum(dim=-1)
        self.forward_signed_progress[ids] += torch.where(good & forward, forward_projection, 0.0)
        self.had_forward_command[ids] |= forward
        # Inactive X commands cannot earn progress. Use the full XY residual:
        # turning cannot hide a stop-command retreat as unmonitored side drift.
        # Correctly commanded lateral/small motion still has zero residual.
        inactive_error = (delta - self._world_command[ids] * self._dt).norm(dim=-1)
        self.inactive_forward_error[ids] += torch.where(good & ~forward, inactive_error, 0.0)
        self._last_position[ids] = position_xy[ids]
        self._has_last_position[ids] = finite_position
        self.pending[ids] = False
        if for_reset:
            self.reset_ready[ids] = True
        return ids

    @torch.no_grad()
    def reset(self, env_ids: Sequence[int] | torch.Tensor, generation: int = 0):
        """Clear selected episodes after curriculum consumes terminal evidence."""
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        ids = self._ids(env_ids)
        for name in ("elapsed_time", "active_time", "target_distance", "signed_progress",
                     "forward_active_time", "forward_target_distance", "forward_signed_progress",
                     "inactive_forward_error"):
            getattr(self, name)[ids] = 0.0
        for name in ("had_translation", "had_forward_command", "reset_ready", "reported", "pending", "_has_last_position"):
            getattr(self, name)[ids] = False
        self.valid[ids] = True
        self.episode_generation[ids] = generation

    @torch.no_grad()
    def abort_step(self):
        """An interrupted parent step cannot provide trustworthy episode evidence."""
        self.valid[self.pending] = False
        self.pending.fill_(False)
        self.reset_ready.fill_(False)
