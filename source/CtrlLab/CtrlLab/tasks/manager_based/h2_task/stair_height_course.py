"""Real terrain rows and reset-only intermediate-skill promotion; no controller."""
from collections import deque
from copy import deepcopy
import torch
from isaaclab.terrains import TerrainImporter
from .stair_ascent_terrain import ascent_stair_generator


class FixedTargetTerrainImporter(TerrainImporter):
    def _compute_env_origins_curriculum(self, num_envs, origins):
        # Preserve the inherited initial randint call. Rebind actual rows here,
        # during scene construction, before assets, resets or observations.
        result = super()._compute_env_origins_curriculum(num_envs, origins)
        if num_envs != 1024 or origins.shape[:2] != (3, 4) or self.cfg.max_init_terrain_level != 0:
            raise RuntimeError("Unexpected fixed-target terrain construction")
        self.terrain_levels[:256] = 2
        self.terrain_levels[256:] = 0
        result.copy_(origins[self.terrain_levels, self.terrain_types])
        self.fixed_target_assignment_before_observations = True
        self.fixed_target_initial_row_counts = torch.bincount(self.terrain_levels, minlength=3).cpu().tolist()
        return result


def configure_height_course(cfg, binding):
    """Only a fresh training config instance receives this profile."""
    from stair_curriculum_binding import course_config_errors
    if course_config_errors(binding):
        raise ValueError("Undeclared height course")
    cfg.stair_height_course = deepcopy(binding)
    cfg.stair_minimum_height, cfg.stair_maximum_height, cfg.stair_height_levels = 0.05, 0.15, 3
    cfg.scene.terrain = deepcopy(cfg.scene.terrain)
    cfg.scene.terrain.class_type = FixedTargetTerrainImporter
    cfg.scene.terrain.terrain_generator = ascent_stair_generator(levels=3, minimum=0.05, maximum=0.15)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None


class StairHeightCourse:
    def __init__(self, env):
        self.env = env
        self.contract = deepcopy(env.cfg.stair_height_course)
        self.terrain = env.scene.terrain
        if (type(self.terrain) is not FixedTargetTerrainImporter
                or self.terrain.fixed_target_assignment_before_observations is not True
                or self.terrain.fixed_target_initial_row_counts != [768, 0, 256]):
            raise RuntimeError("Actual early terrain assignment is absent")
        self.seen_certificate = torch.zeros(1024, dtype=torch.bool, device=env.device)
        self.history = torch.zeros(1024, 5, dtype=torch.bool, device=env.device)
        self.history_count = torch.zeros(1024, dtype=torch.long, device=env.device)
        self.history_next = torch.zeros_like(self.history_count)
        self.promotion_count = torch.zeros_like(self.history_count)
        self.last_recorded_end_counter = torch.full_like(self.history_count, -1)
        self.controls = 0
        self.target_completed = self.curriculum_completed = 0
        self.target_recent = deque(maxlen=5000)
        self.height_recent = [deque(maxlen=5000) for _ in range(3)]
        self.height_completed = [0, 0, 0]
        self.last_observed_counter = -1
        self._assert_rows()

    def _assert_rows(self):
        row = self.terrain.terrain_levels
        if (row.shape != (1024,) or not bool((row[:256] == 2).all())
                or not bool(((row >= 0) & (row <= 2)).all())
                or not torch.equal(self.terrain.env_origins,
                    self.terrain.terrain_origins[row, self.terrain.terrain_types])):
            raise RuntimeError("Actual height-course rows/origins changed")
        expected_heights = torch.tensor([.05, .10, .15], device=self.env.device)[row]
        if not torch.equal(self.env._stair_heights(), expected_heights):
            raise RuntimeError("Actual height lookup disagrees with the three real rows")

    def observe(self):
        p = self.env.stair_progress
        counter = int(self.env.common_step_counter)
        if counter == self.last_observed_counter:
            return
        if counter != self.controls + 1:
            raise RuntimeError("Height course missed an actual control")
        self._assert_rows()
        self.last_observed_counter = counter
        self.controls += 1
        alive = ~self.env.termination_manager.terminated & ~self.env.termination_manager.time_outs
        certificate = (alive & ~p.partial & ~p.ever_invalid & p.force_credit_eligible
            & p.episode_positive_control_seen.all(-1) & (p.duration >= 20.0)
            & p.verified[:, :2].all(dim=(1, 2)) & p.transferred[:, :2].all(-1)
            & (p.root_rise > 0)
            & (torch.acos((-self.env.asset.data.projected_gravity_b[:, 2]).clamp(-1, 1)) < 0.5))
        self.seen_certificate |= certificate

    def finish_episodes(self, ids):
        """Called after true terminal recording, before the inherited reset."""
        p = self.env.stair_progress
        ids = ids[p.last_episode["valid"][ids].bool()]
        if not ids.numel():
            return
        if (bool((p.end_counter[ids] != self.controls).any())
                or bool((self.last_recorded_end_counter[ids] == p.end_counter[ids]).any())):
            raise RuntimeError("Course terminal records are missing or duplicated")
        self.last_recorded_end_counter[ids] = p.end_counter[ids]
        row = self.terrain.terrain_levels[ids].clone()
        if not torch.equal(p.terrain_level[ids], row):
            raise RuntimeError("Episode height changed before terminal accounting")
        rows = torch.stack([p.last_episode[k][ids].float() for k in p.SUMMARY_FIELDS], -1).cpu().tolist()
        env_ids, levels = ids.cpu().tolist(), row.cpu().tolist()
        for env_id, level, record in zip(env_ids, levels, rows):
            self.height_recent[level].append(record)
            self.height_completed[level] += 1
            if env_id < 256:
                if level != 2:
                    raise RuntimeError("A fixed target episode was not15cm")
                self.target_recent.append(record)
                self.target_completed += 1
            else:
                self.curriculum_completed += 1
        curriculum_ids = ids[ids >= 256]
        if not curriculum_ids.numel():
            return
        certified = (self.seen_certificate[curriculum_ids] & ~p.ever_invalid[curriculum_ids]
                     & p.force_credit_eligible[curriculum_ids]
                     & p.episode_positive_control_seen[curriculum_ids].all(-1))
        self.history[curriculum_ids, self.history_next[curriculum_ids]] = certified
        self.history_next[curriculum_ids] = (self.history_next[curriculum_ids] + 1) % 5
        self.history_count[curriculum_ids] += 1
        promote = ((self.history_count[curriculum_ids] >= 5)
                   & (self.history[curriculum_ids].sum(-1) >= 4)
                   & (self.terrain.terrain_levels[curriculum_ids] < 2))
        changed = curriculum_ids[promote]
        if changed.numel():
            # No inherited overflow/random-demotion routine: this is one
            # deterministic row advance at a real reset, never an active move.
            self.terrain.terrain_levels[changed] += 1
            self.terrain.env_origins[changed] = self.terrain.terrain_origins[
                self.terrain.terrain_levels[changed], self.terrain.terrain_types[changed]]
            self.promotion_count[changed] += 1
            self.history[changed] = False
            self.history_count[changed] = self.history_next[changed] = 0
        self._assert_rows()

    def reset(self, ids):
        self.seen_certificate[ids] = False

    def snapshot(self):
        self._assert_rows()
        return dict(contract=deepcopy(self.contract), controls=self.controls,
            cohort_assignment_before_observations=True, fixed_target_all_row2=True,
            actual_row_counts=torch.bincount(self.terrain.terrain_levels, minlength=3).cpu().tolist(),
            target_completed_episodes=self.target_completed,
            curriculum_completed_episodes=self.curriculum_completed,
            promotions=int(self.promotion_count.sum()),
            initial_row_counts=list(self.terrain.fixed_target_initial_row_counts))

    def target_summary(self):
        self._assert_rows()
        actual = self.env._stair_heights()[:256]
        return dict(cohort_ids_inclusive=[0, 255], authored_riser_height_m=0.15,
            actual_riser_height_m=float(actual[0]),
            actual_target_riser_min_max_m=[float(actual.min()), float(actual.max())],
            motion=self.env.stair_progress.summary(records=self.target_recent, completed_count=self.target_completed))

    def per_height_summary(self):
        return [dict(row=i, height_m=self.contract["heights_m"][i],
            motion=self.env.stair_progress.summary(records=self.height_recent[i], completed_count=self.height_completed[i]))
                for i in range(3)]
