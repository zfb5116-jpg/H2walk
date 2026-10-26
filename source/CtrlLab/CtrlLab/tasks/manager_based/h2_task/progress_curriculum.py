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
        required_group_ids: Sequence[int] | None = None,
        min_group_episodes: int | None = None,
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
        if min_group_episodes is None:
            min_group_episodes = min_episodes
        if isinstance(min_group_episodes, bool) or not isinstance(min_group_episodes, int) or min_group_episodes <= 0:
            raise ValueError("min_group_episodes must be a positive integer")
        if required_group_ids is not None:
            try:
                required_group_ids = tuple(required_group_ids)
            except TypeError as exc:
                raise ValueError("required_group_ids must be a sequence of integers") from exc
            if (not required_group_ids
                    or any(isinstance(group, bool) or not isinstance(group, int) or group < 0
                           for group in required_group_ids)
                    or len(set(required_group_ids)) != len(required_group_ids)):
                raise ValueError("required_group_ids must be nonempty, unique nonnegative integers")
            required_group_ids = tuple(sorted(required_group_ids))
        self.interval_steps = interval_steps
        self.min_episodes = min_episodes
        self.success_rate = success_rate
        self.min_progress_ratio = min_progress_ratio
        self.max_progress_ratio = max_progress_ratio
        self.min_episode_fraction = min_episode_fraction
        self.min_active_time = min_active_time
        self.min_target_distance = min_target_distance
        self.generation = generation
        self.required_group_ids = required_group_ids
        self.min_group_episodes = min_group_episodes
        self.group_episode_counts = {group: 0 for group in required_group_ids or ()}
        self.group_success_counts = {group: 0 for group in required_group_ids or ()}
        self._group_ids_snapshot: torch.Tensor | None = None
        self.evaluation_count = 0
        self.last_evaluation_step = 0
        self._last_observation_step = 0
        self.episode_count = 0
        self.success_count = 0
        self._progress_ratio_sum = 0.0
        self._progress_ratio_count = 0
        self._active_progress_sum = 0.0
        self._inactive_error_sum = 0.0
        self._target_distance_sum = 0.0

    def observe(
        self,
        tracker,
        env_ids: Sequence[int] | torch.Tensor,
        *,
        terminated: torch.Tensor,
        time_outs: torch.Tensor,
        max_episode_length_s: float,
        current_step: int,
        group_ids: torch.Tensor | None = None,
    ) -> dict[str, object] | None:
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
        group_indices = None
        groups_valid = None
        if self.required_group_ids is not None:
            if (not isinstance(group_ids, torch.Tensor) or group_ids.dtype != torch.long
                    or group_ids.shape != tracker.elapsed_time.shape
                    or group_ids.device != tracker.elapsed_time.device):
                raise ValueError("group_ids must be a full-environment one-dimensional long tensor on the tracker device")
            required = torch.tensor(self.required_group_ids, device=group_ids.device, dtype=torch.long)
            group_indices = torch.searchsorted(required, group_ids).clamp(max=len(self.required_group_ids) - 1)
            groups_valid = (required[group_indices] == group_ids).all()
            if self._group_ids_snapshot is not None:
                if (group_ids.shape != self._group_ids_snapshot.shape
                        or group_ids.device != self._group_ids_snapshot.device):
                    raise ValueError("group_ids must retain fixed terrain assignments")
                # An identity check alone misses in-place swaps between legal
                # columns. Fold content stability into the existing compact
                # validity aggregate before consuming any reset evidence.
                groups_valid &= (group_ids == self._group_ids_snapshot).all()
        elif group_ids is not None:
            raise ValueError("group_ids requires configured required_group_ids")
        ids = ids.unique()
        ready = tracker.reset_ready[ids] & ~tracker.reported[ids]
        ready_ids = ids[ready]
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
            metric_valid = valid & (target > 0.0) & torch.isfinite(ratio)
            summaries = [success.sum().double(), ratio[metric_valid].double().sum(),
                         metric_valid.sum().double(), progress[metric_valid].double().sum(),
                         inactive_error[metric_valid].double().sum(), target[metric_valid].double().sum()]
            if group_indices is not None:
                sample_groups = group_indices[sample_ids]
                group_count = len(self.required_group_ids)
                counts = torch.bincount(sample_groups, minlength=group_count)
                successes = torch.bincount(sample_groups, weights=success.double(), minlength=group_count)
                summaries.extend(counts.unbind())
                summaries.extend(successes.unbind())
                summaries.append(groups_valid.double())
            # Transfer a compact aggregate once, rather than synchronizing the
            # GPU separately for every terrain column or each metric.
            values = torch.stack(summaries).cpu().tolist()
            if groups_valid is not None and not values[-1]:
                raise ValueError("group_ids must retain fixed, declared terrain assignments")
            if group_ids is not None and self._group_ids_snapshot is None:
                self._group_ids_snapshot = group_ids.clone()
            self.episode_count += sample_ids.numel()
            self.success_count += int(values[0])
            self._progress_ratio_sum += values[1]
            self._progress_ratio_count += int(values[2])
            self._active_progress_sum += values[3]
            self._inactive_error_sum += values[4]
            self._target_distance_sum += values[5]
            if group_indices is not None:
                for offset, group in enumerate(self.required_group_ids):
                    self.group_episode_counts[group] += int(values[6 + offset])
                    self.group_success_counts[group] += int(values[6 + len(self.required_group_ids) + offset])
        elif groups_valid is not None:
            if not bool(groups_valid.item()):
                raise ValueError("group_ids must retain fixed, declared terrain assignments")
            if self._group_ids_snapshot is None:
                self._group_ids_snapshot = group_ids.clone()
        self._last_observation_step = current_step
        tracker.reported[ready_ids] = True
        if (current_step - self.last_evaluation_step < self.interval_steps
                or self.episode_count < self.min_episodes
                or any(count < self.min_group_episodes for count in self.group_episode_counts.values())):
            return None
        fraction = self.success_count / self.episode_count
        group_rates = {group: self.group_success_counts[group] / count
                       for group, count in self.group_episode_counts.items()}
        distance = self._target_distance_sum
        self.evaluation_count += 1
        result = {
            "episodes": self.episode_count,
            "successes": self.success_count,
            "success_rate": fraction,
            "mean_progress_ratio": self._progress_ratio_sum / max(self._progress_ratio_count, 1),
            "progress_ratio_samples": self._progress_ratio_count,
            "passed": fraction >= self.success_rate and all(rate >= self.success_rate for rate in group_rates.values()),
            "evaluation_id": self.evaluation_count,
            "active_progress_sum": self._active_progress_sum,
            "inactive_error_sum": self._inactive_error_sum,
            "target_distance_sum": distance,
            "distance_weighted_active_ratio": self._active_progress_sum / distance if distance > 0.0 else 0.0,
            "distance_weighted_inactive_error_ratio": self._inactive_error_sum / distance if distance > 0.0 else 0.0,
            "distance_weighted_effective_ratio": (self._active_progress_sum - self._inactive_error_sum) / distance if distance > 0.0 else 0.0,
            "group_episodes": dict(self.group_episode_counts),
            "group_success_rates": group_rates,
            "min_group_success_rate": min(group_rates.values(), default=0.0),
        }
        self.last_evaluation_step = current_step
        self.episode_count = self.success_count = 0
        self._progress_ratio_sum = 0.0
        self._progress_ratio_count = 0
        self._active_progress_sum = self._inactive_error_sum = self._target_distance_sum = 0.0
        for group in self.group_episode_counts:
            self.group_episode_counts[group] = self.group_success_counts[group] = 0
        return result

    def advance_generation(self) -> int:
        """Retire evidence from episodes started before an actual expansion."""
        self.generation += 1
        return self.generation
