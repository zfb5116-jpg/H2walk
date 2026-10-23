"""Completed-episode command curriculum, independent of simulator imports.

Progress is accumulated by CommandProgressTracker in the actual step/reset
path. This module only evaluates completed episodes and proposes X ranges;
it does not advance a simulator, load a policy or update an optimizer.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def expanded_x_range(
    current: Sequence[float], limits: Sequence[float], increment: float
) -> tuple[float, float]:
    """Propose a bounded X expansion without changing other command axes."""
    if len(current) != 2 or len(limits) != 2:
        raise ValueError("Command ranges must contain two endpoints")
    low, high = map(float, current)
    limit_low, limit_high = map(float, limits)
    if not all(math.isfinite(value) for value in (low, high, limit_low, limit_high, increment)):
        raise ValueError("Command endpoints and increment must be finite")
    if not (limit_low <= low <= high <= limit_high) or increment <= 0.0:
        raise ValueError("Require ordered current range inside limits and increment > 0")
    return max(low - increment, limit_low), min(high + increment, limit_high)


class ProgressCurriculum:
    """Aggregate episode success across asynchronous reset batches.

    Failed/short natural episodes with X-command exposure stay in the
    denominator. Valid standing/sideways-only and external/initial resets are
    not samples. Invalid natural episodes count as failures conservatively,
    even when a malformed command prevented recording X exposure.
    Each environment is consumed once per reset via ``tracker.reported``.
    """

    def __init__(
        self,
        *,
        interval_steps: int,
        min_episodes: int = 32,
        success_rate: float = 0.8,
        min_progress_ratio: float = 0.8,
        max_progress_ratio: float = 1.2,
        min_episode_fraction: float = 0.98,
        min_active_time: float = 2.0,
        min_target_distance: float = 0.5,
        generation: int = 0,
    ):
        for name, value in (("interval_steps", interval_steps), ("min_episodes", min_episodes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        scalars = (success_rate, min_progress_ratio, max_progress_ratio,
                   min_episode_fraction, min_active_time, min_target_distance)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Curriculum thresholds must be finite")
        if not 0.0 < success_rate <= 1.0:
            raise ValueError("success_rate must satisfy 0 < value <= 1")
        if not 0.0 < min_progress_ratio <= max_progress_ratio:
            raise ValueError("Require 0 < min_progress_ratio <= max_progress_ratio")
        if not 0.0 < min_episode_fraction <= 1.0:
            raise ValueError("min_episode_fraction must satisfy 0 < value <= 1")
        if min_active_time <= 0.0 or min_target_distance <= 0.0:
            raise ValueError("Active time and target distance thresholds must be positive")
        self.interval_steps = interval_steps
        self.min_episodes = min_episodes
        self.success_rate = success_rate
        self.min_progress_ratio = min_progress_ratio
        self.max_progress_ratio = max_progress_ratio
        self.min_episode_fraction = min_episode_fraction
        self.min_active_time = min_active_time
        self.min_target_distance = min_target_distance
        self.generation = generation
        self.last_evaluation_step = 0
        self._last_observation_step = 0
        self.episode_count = 0
        self.success_count = 0
        self._progress_ratio_sum = 0.0
        self._progress_ratio_count = 0

    def observe(
        self,
        tracker,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        max_episode_length_s: float,
        current_step: int,
    ) -> dict[str, float | int | bool] | None:
        """Consume newly ready terminal episodes and evaluate at most once.

        Reset state must still describe the terminating episode. An unchanged
        tracker/reset batch can be passed repeatedly without counting it twice.
        """
        if not math.isfinite(max_episode_length_s) or max_episode_length_s <= 0.0:
            raise ValueError("max_episode_length_s must be finite and positive")
        if (isinstance(current_step, bool) or not isinstance(current_step, int)
                or current_step < self._last_observation_step):
            raise ValueError("current_step must be a nondecreasing nonnegative integer")
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=tracker.elapsed_time.device)
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
        self._last_observation_step = current_step
        ids = ids.unique()
        ready = tracker.reset_ready[ids] & ~tracker.reported[ids]
        ready_ids = ids[ready]
        tracker.reported[ready_ids] = True
        natural_done = terminated[ready_ids] | time_outs[ready_ids]
        candidate = (natural_done & (tracker.had_forward_command[ready_ids] | ~tracker.valid[ready_ids])
                     & (tracker.episode_generation[ready_ids] == self.generation))
        sample_ids = ready_ids[candidate]
        if sample_ids.numel():
            elapsed = tracker.elapsed_time[sample_ids]
            active = tracker.forward_active_time[sample_ids]
            target = tracker.forward_target_distance[sample_ids]
            progress = tracker.forward_signed_progress[sample_ids]
            inactive_error = tracker.inactive_forward_error[sample_ids]
            finite = (torch.isfinite(elapsed) & torch.isfinite(active)
                      & torch.isfinite(target) & torch.isfinite(progress)
                      & torch.isfinite(inactive_error))
            valid = tracker.valid[sample_ids] & finite & (inactive_error >= 0.0)
            raw_ratio = progress / target.clamp_min(1e-12)
            # Displacement during stop/tiny-X commands must not hide the
            # return leg of an apparent successful forward excursion. Correct
            # tiny-command tracking has zero error and is not penalized.
            ratio = (progress - inactive_error) / target.clamp_min(1e-12)
            success = (
                valid & time_outs[sample_ids] & ~terminated[sample_ids]
                & (elapsed >= self.min_episode_fraction * max_episode_length_s)
                & (active >= self.min_active_time)
                & (target >= self.min_target_distance)
                & (raw_ratio >= self.min_progress_ratio) & (raw_ratio <= self.max_progress_ratio)
                & (ratio >= self.min_progress_ratio)
            )
            self.episode_count += sample_ids.numel()
            self.success_count += int(success.sum().item())
            metric_valid = valid & (target > 0.0) & torch.isfinite(ratio)
            self._progress_ratio_sum += float(ratio[metric_valid].double().sum().item())
            self._progress_ratio_count += int(metric_valid.sum().item())
        if (current_step - self.last_evaluation_step < self.interval_steps
                or self.episode_count < self.min_episodes):
            return None
        fraction = self.success_count / self.episode_count
        result = {
            "episodes": self.episode_count,
            "successes": self.success_count,
            "success_rate": fraction,
            "mean_progress_ratio": self._progress_ratio_sum / max(self._progress_ratio_count, 1),
            "progress_ratio_samples": self._progress_ratio_count,
            "passed": fraction >= self.success_rate,
        }
        self.last_evaluation_step = current_step
        self.episode_count = self.success_count = 0
        self._progress_ratio_sum = 0.0
        self._progress_ratio_count = 0
        return result

    def advance_generation(self) -> int:
        """Retire evidence from episodes started before an actual expansion."""
        self.generation += 1
        return self.generation
