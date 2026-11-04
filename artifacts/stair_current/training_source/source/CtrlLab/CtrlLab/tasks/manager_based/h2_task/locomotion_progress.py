"""Complete-episode physical progress used by the H2 velocity curriculum."""

from collections import deque

import numpy as np
import torch


class FootfallTracker:
    """Valid alternating landings, excluding initial contact, hops and chatter."""

    def __init__(self, num_envs, device):
        self.air_time = torch.zeros((num_envs, 2), device=device)
        self.landings = torch.zeros_like(self.air_time)
        self.seen_contact = torch.zeros_like(self.air_time, dtype=torch.bool)
        self.previous_contact = torch.zeros_like(self.seen_contact)
        self.last_landing = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids):
        for value in (self.air_time, self.landings, self.seen_contact, self.previous_contact):
            value[env_ids] = 0
        self.last_landing[env_ids] = -1

    def update(self, contact, dt, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        current = contact if env_ids is None else contact[ids]
        previous = self.previous_contact[ids]
        landed = current & ~previous
        other_support = current.flip(-1) & previous.flip(-1)
        valid = (landed & self.seen_contact[ids] & other_support
                 & (self.air_time[ids] >= 0.08 - 1e-6) & (self.air_time[ids] <= 0.6 + 1e-6))
        foot = valid.long().argmax(-1)
        alternate = valid.sum(-1).eq(1) & (foot != self.last_landing[ids])
        counts = self.landings[ids]
        counts[:, 0] += alternate & foot.eq(0)
        counts[:, 1] += alternate & foot.eq(1)
        self.landings[ids] = counts
        self.last_landing[ids] = torch.where(alternate, foot, self.last_landing[ids])
        self.air_time[ids] = torch.where(current, 0.0, self.air_time[ids] + dt)
        self.seen_contact[ids] |= current
        self.previous_contact[ids] = current


def curriculum_ready(metrics, *, min_episodes=1024, min_moving_episodes=256,
                     min_survival_rate=0.9, success_rate=0.8):
    return (metrics["episode_count"] >= min_episodes
            and metrics["moving_episode_count"] >= min_moving_episodes
            and metrics["survival_fraction"] >= min_survival_rate
            and metrics["moving_walking_success_fraction"] >= success_rate)


class LocomotionProgress:
    """Finalize terminal transitions before scene reset; reject partial/old-generation episodes."""

    def __init__(self, num_envs, device, episode_length_s=20.0, window_size=5000):
        self.num_envs = num_envs
        self.device = device
        self.episode_length_s = episode_length_s
        self.generation = 0
        self.recent = deque(maxlen=window_size)
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.valid_episode = torch.zeros_like(self.pending)
        self.episode_generation = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.duration = torch.zeros(num_envs, device=device)
        self.active_time = torch.zeros_like(self.duration)
        self.target_distance = torch.zeros_like(self.duration)
        self.projected_distance = torch.zeros_like(self.duration)
        self.footfalls = FootfallTracker(num_envs, device)
        self.air_time = self.footfalls.air_time
        self.landings = self.footfalls.landings
        self.pre_xy = torch.zeros((num_envs, 2), device=device)
        self.desired_xy = torch.zeros_like(self.pre_xy)
        self.dt = 0.0

    def begin_step(self, position_xy, quaternion, command, dt):
        self.pre_xy.copy_(position_xy)
        q = quaternion
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.desired_xy[:, 0] = c * command[:, 0] - s * command[:, 1]
        self.desired_xy[:, 1] = s * command[:, 0] + c * command[:, 1]
        self.dt = dt
        self.pending[:] = True

    def finish_step(self, position_xy, foot_contact, env_ids=None):
        ids = self.pending.nonzero(as_tuple=False).flatten() if env_ids is None else env_ids[self.pending[env_ids]]
        if not ids.numel():
            return
        delta = position_xy[ids] - self.pre_xy[ids]
        if not torch.isfinite(delta).all():
            raise RuntimeError("Non-finite episode displacement")
        speed = self.desired_xy[ids].norm(dim=-1)
        active = speed >= 0.1
        direction = self.desired_xy[ids] / speed[:, None].clamp_min(1e-6)
        self.duration[ids] += self.dt
        self.active_time[ids] += active * self.dt
        self.target_distance[ids] += torch.where(active, speed * self.dt, 0.0)
        self.projected_distance[ids] += torch.where(active, (delta * direction).sum(-1), 0.0)
        self.footfalls.update(foot_contact, self.dt, ids)
        self.pending[ids] = False

    def record_episodes(self, env_ids, terminated):
        ids = env_ids[self.valid_episode[env_ids]
                      & (self.episode_generation[env_ids] == self.generation)
                      & (self.duration[env_ids] > 0)]
        if not ids.numel():
            return
        survived = (self.duration[ids] >= 0.98 * self.episode_length_s) & ~terminated[ids]
        bilateral = (self.landings[ids] >= 2).all(-1)
        rows = torch.stack((self.duration[ids], survived.float(), self.active_time[ids],
                            self.target_distance[ids], self.projected_distance[ids],
                            bilateral.float()), dim=1).cpu().numpy()
        self.recent.extend(rows)

    def reset(self, env_ids):
        # A reset before the first transition does not make a random-length bootstrap episode valid.
        self.valid_episode[env_ids] |= self.duration[env_ids] > 0
        self.episode_generation[env_ids] = self.generation
        for value in (self.duration, self.active_time, self.target_distance,
                      self.projected_distance, self.air_time, self.landings):
            value[env_ids] = 0.0
        self.footfalls.reset(env_ids)
        self.pending[env_ids] = False

    def advance_generation(self):
        self.generation += 1
        self.recent.clear()

    def summary(self, *, min_active_time=2.0, min_target_distance=0.5,
                min_progress_ratio=0.8, max_progress_ratio=1.2):
        if not self.recent:
            return {"episode_count": 0, "moving_episode_count": 0, "survival_fraction": 0.0,
                    "mean_progress_ratio": 0.0, "moving_walking_success_fraction": 0.0}
        rows = np.asarray(self.recent)
        moving = (rows[:, 2] >= min_active_time) & (rows[:, 3] >= min_target_distance)
        ratio = rows[:, 4] / np.maximum(rows[:, 3], 1e-6)
        success = ((rows[:, 1] > 0) & (rows[:, 5] > 0)
                   & (ratio >= min_progress_ratio) & (ratio <= max_progress_ratio))
        return {"episode_count": len(rows), "moving_episode_count": int(moving.sum()),
                "survival_fraction": float(rows[:, 1].mean()),
                "mean_progress_ratio": float(np.clip(ratio[moving], -2, 2).mean()) if moving.any() else 0.0,
                "moving_walking_success_fraction": float(success[moving].mean()) if moving.any() else 0.0}
