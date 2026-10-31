from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .clearance_reward import FootState, swing_score
from .curriculum import StairCurriculum
from .stair_rewards import (SequentialTreadReward, StairGeometry, sole_contact_ratios,
                            stair_boundary_count, stair_riser_count)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "assets" / "h2_lowerbody.xml"

# Keep these names aligned with CtrlLab/tasks/manager_based/h2_task/assets/robot.py.
LOWER_BODY_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
)
UPPER_BODY_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

STANDING_POSE = {
    "left_hip_pitch_joint": -0.25,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.5,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": -0.25,
    "right_hip_pitch_joint": -0.25,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.5,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": -0.25,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


class H2StairsEnv(gym.Env[np.ndarray, np.ndarray]):
    """MuJoCo H2 locomotion on a deterministic 20 cm staircase.

    The terrain observation is privileged ground truth: it is calculated from
    the analytical stair profile, with no camera, lidar, noise, or estimator.
    Curriculum starts on flat ground and advances only after successful runs.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        render_mode: str | None = None,
        terrain_mode: str = "20cm",
        max_episode_steps: int = 1000,
        low_step_height: float = 0.05,
        start_stage: str = "flat",
        curriculum_window: int = 20,
        curriculum_success_rate: float = 0.8,
        clearance_margin: float = 0.05,
        swing_reward_weight: float = 2.5,
        stair_boundary_weight: float = -0.05,
        stair_riser_weight: float = -0.5,
        stair_tread_weight: float = 0.5,
        stair_lane_half_width: float = 0.60,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if terrain_mode not in ("flat", "low", "20cm", "curriculum"):
            raise ValueError("terrain_mode must be flat, low, 20cm, or curriculum")
        if not 0.0 < low_step_height <= 0.20 or max_episode_steps < 1:
            raise ValueError("low_step_height must be in (0, 0.20] and episode length positive")
        if not 0.0 < clearance_margin <= 0.15 or swing_reward_weight < 0.0:
            raise ValueError("clearance_margin must be in (0, 0.15] and reward weight nonnegative")
        if (not np.isfinite([stair_boundary_weight, stair_riser_weight, stair_tread_weight]).all()
                or stair_boundary_weight > 0.0 or stair_riser_weight > 0.0 or stair_tread_weight < 0.0):
            raise ValueError("stair penalty weights must be finite and <=0; tread weight must be finite and >=0")
        if not 0.0 < stair_lane_half_width <= 1.6:
            raise ValueError("stair_lane_half_width must be in (0, 1.6]")
        self.render_mode = render_mode
        self.terrain_mode = terrain_mode
        self.max_episode_steps = int(max_episode_steps)
        self.low_step_height = float(low_step_height)
        self.clearance_margin = float(clearance_margin)
        self.swing_reward_weight = float(swing_reward_weight)
        self.stair_boundary_weight = float(stair_boundary_weight)
        self.stair_riser_weight = float(stair_riser_weight)
        self.stair_tread_weight = float(stair_tread_weight)
        self.curriculum = StairCurriculum(start_stage, curriculum_window, curriculum_success_rate)
        selected = model_path or os.environ.get("CTRLLAB_H2_MJCF", str(DEFAULT_MODEL))
        self.model_path = Path(selected).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.model_path}. "
                "Set CTRLLAB_H2_MJCF to the official Unitree H2 MJCF."
            )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._joint_ids = self._resolve_joint_ids(LOWER_BODY_JOINTS, required=True)
        self._upper_joint_ids = self._resolve_joint_ids(UPPER_BODY_JOINTS, required=False)
        self._left_foot = self._find_body("left_ankle_pitch_link", "left_foot")
        self._right_foot = self._find_body("right_ankle_pitch_link", "right_foot")
        self._torso = self._find_body("pelvis", "torso", "base")
        self._stair_geom_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"stair_{i:02d}")
                for i in range(1, 7)
            ],
            dtype=np.int32,
        )
        if np.any(self._stair_geom_ids < 0):
            raise ValueError("The MJCF scene must contain stair_01 through stair_06; "
                             "a robot-only MJCF has no physical stairs for terrain observations.")
        landing = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "stair_landing")
        self._terrain_geom_ids = list(self._stair_geom_ids) + ([landing] if landing >= 0 else [])
        self._landing_end = 4.3 if landing >= 0 else 3.0
        self._stair_geometry = StairGeometry(landing_end=self._landing_end,
                                             lane_half_width=stair_lane_half_width)
        self._tread_reward = SequentialTreadReward(self._stair_geometry)
        self._terrain_collision = [(int(self.model.geom_contype[g]), int(self.model.geom_conaffinity[g]))
                                   for g in self._terrain_geom_ids]
        self._foot_geoms = [self._collision_geoms(b) for b in (self._left_foot, self._right_foot)]
        self._geom_to_foot = {g: i for i, geoms in enumerate(self._foot_geoms) for g in geoms}
        corners = np.asarray(list(product((-1.0, 1.0), repeat=3)))
        # MuJoCo's local geom bounds include the collision shape offset. Their
        # transformed corners conservatively account for toe/heel pitch and roll.
        self._foot_corners = {g: self.model.geom_aabb[g, :3] + corners * self.model.geom_aabb[g, 3:]
                              for geoms in self._foot_geoms for g in geoms}
        # Fixed sample density makes raw point-count penalties reproducible.
        volume_grid = np.asarray(list(product(np.linspace(-1, 1, 5), np.linspace(-1, 1, 3),
                                              np.linspace(-1, 1, 3))))
        sole_grid = np.asarray([(x, y, -1.0) for x in np.linspace(-1, 1, 7)
                                for y in np.linspace(-1, 1, 5)])
        self._foot_volume_local = {g: self.model.geom_aabb[g, :3] + volume_grid * self.model.geom_aabb[g, 3:]
                                   for geoms in self._foot_geoms for g in geoms}
        self._foot_sole_local = {g: self.model.geom_aabb[g, :3] + sole_grid * self.model.geom_aabb[g, 3:]
                                 for geoms in self._foot_geoms for g in geoms}
        self._qpos_adrs = np.asarray([self.model.jnt_qposadr[i] for i in self._joint_ids], dtype=np.int32)
        self._qvel_adrs = np.asarray([self.model.jnt_dofadr[i] for i in self._joint_ids], dtype=np.int32)
        self._upper_qpos_adrs = np.asarray(
            [self.model.jnt_qposadr[i] for i in self._upper_joint_ids], dtype=np.int32
        )
        self._upper_qvel_adrs = np.asarray(
            [self.model.jnt_dofadr[i] for i in self._upper_joint_ids], dtype=np.int32
        )
        self._upper_home = np.asarray(
            [STANDING_POSE.get(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(i)), 0.0) for i in self._upper_joint_ids], dtype=np.float64
        )
        self._lower_home = np.asarray(
            [STANDING_POSE[name] for name in LOWER_BODY_JOINTS], dtype=np.float64
        )
        self.action_scale = 0.45
        self._kp = np.asarray([180.0, 100.0, 100.0, 180.0, 50.0, 60.0] * 2, dtype=np.float64)
        self._kd = np.asarray([8.0, 5.0, 5.0, 8.0, 2.0, 2.0] * 2, dtype=np.float64)
        self._torque_limit = np.asarray([360.0, 100.0, 100.0, 360.0, 66.0, 66.0] * 2, dtype=np.float64)
        self.command = np.array([0.65, 0.0], dtype=np.float32)
        self._step_count = 0
        self._last_x = 0.0
        self._renderer = None
        self._support_heights = np.zeros(2)
        self._best_landed_levels = np.zeros(2, dtype=int)
        self._previous_foot_x = np.zeros(2)
        self._goal_hold = 0.0
        self._stall_time = 0.0
        self._episode_done = False

        # proprioception (qpos/qvel), command, and privileged terrain profile.
        self.terrain_x = np.linspace(-0.45, 1.85, 24, dtype=np.float32)
        obs_size = 3 + 3 + 12 + 12 + 2 + len(self.terrain_x)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (len(LOWER_BODY_JOINTS),), dtype=np.float32)

    def _resolve_joint_ids(self, names: tuple[str, ...], required: bool) -> np.ndarray:
        ids = []
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                ids.append(jid)
            elif required:
                raise ValueError(f"Model is missing required H2 joint: {name}")
        return np.asarray(ids, dtype=np.int32)

    def _find_body(self, *names: str) -> int:
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid
        raise ValueError(f"Model does not contain any of the required bodies: {names}")

    def _collision_geoms(self, foot_body: int) -> list[int]:
        geoms = []
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            ancestor = int(body_id)
            while ancestor and ancestor != foot_body:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == foot_body and (self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id]):
                geoms.append(geom_id)
        if not geoms:
            raise ValueError("Each foot must have collision geometry for sole clearance rewards")
        return geoms

    def _terrain_step_height(self) -> float:
        if self.terrain_mode == "20cm":
            return 0.20
        if self.terrain_mode == "flat":
            return 0.0
        if self.terrain_mode == "low":
            return self.low_step_height
        return self.curriculum.height

    def get_curriculum_state(self) -> dict:
        return self.curriculum.state()

    def set_curriculum_state(self, state: dict) -> None:
        self.curriculum.restore(state)

    def terrain_height(self, x: np.ndarray | float) -> np.ndarray | float:
        """Analytical height profile used both for the XML stairs and truth input."""
        height = self._terrain_step_height()
        x_arr = np.asarray(x)
        # The first step starts at x=0.9; each tread is 0.35 m long.
        n_steps = np.floor(np.maximum(x_arr - 0.9, 0.0) / 0.35) + 1.0
        out = np.where((x_arr < 0.9) | (x_arr >= self._landing_end),
                       0.0, np.minimum(n_steps, 6.0) * height)
        return float(out) if np.ndim(x) == 0 else out

    def _set_stair_height(self) -> None:
        """Keep the bundled XML geometry consistent with curriculum input."""
        height = self._terrain_step_height()
        for index, geom_id in enumerate(self._terrain_geom_ids, start=1):
            half_height = max(0.5 * min(index, 6) * height, 0.001)
            self.model.geom_pos[geom_id, 2] = half_height
            self.model.geom_size[geom_id, 2] = half_height
            self.model.geom_aabb[geom_id, 3:] = self.model.geom_size[geom_id]
            self.model.geom_rbound[geom_id] = np.linalg.norm(self.model.geom_size[geom_id])
            collision = self._terrain_collision[index - 1] if height > 0.0 else (0, 0)
            self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = collision
            self.model.geom_rgba[geom_id, 3] = float(height > 0.0)

    def _lock_upper_body(self) -> None:
        if self._upper_qpos_adrs.size:
            self.data.qpos[self._upper_qpos_adrs] = self._upper_home
            self.data.qvel[self._upper_qvel_adrs] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _obs(self) -> np.ndarray:
        torso_pos = self.data.xpos[self._torso]
        torso_mat = self.data.xmat[self._torso].reshape(3, 3)
        # MuJoCo body frame z axis in world coordinates; projected gravity in body frame.
        gravity_b = torso_mat.T @ np.array([0.0, 0.0, -1.0])
        terrain = self.terrain_height(torso_pos[0] + self.terrain_x) - torso_pos[2]
        obs = np.concatenate(
            [
                torso_pos[1:2],
                torso_pos[2:3],
                torso_pos[:1],
                gravity_b,
                self.data.qpos[self._qpos_adrs] - self._lower_home,
                self.data.qvel[self._qvel_adrs],
                self.command,
                terrain.astype(np.float64, copy=False),
            ]
        )
        return np.asarray(obs, dtype=np.float32)

    def _foot_states(self, dt: float) -> tuple[list[FootState], bool]:
        supported = [False, False]
        stumble = False
        force = np.zeros(6)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            for foot_geom, other_geom in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                foot_id = self._geom_to_foot.get(int(foot_geom))
                # Ignore self-collisions: only static terrain can support a foot.
                if foot_id is None or self.model.body_weldid[self.model.geom_bodyid[other_geom]] != 0:
                    continue
                mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                if force[0] > 5.0:
                    if abs(contact.frame[2]) > 0.5:
                        supported[foot_id] = True
                    else:
                        stumble = True
        states = []
        for i, geoms in enumerate(self._foot_geoms):
            points = np.concatenate([self._foot_corners[g] @ self.data.geom_xmat[g].reshape(3, 3).T
                                     + self.data.geom_xpos[g] for g in geoms])
            rear, toe, sole = float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 2].min())
            center = 0.5 * (rear + toe)
            states.append(FootState(rear, toe, sole, (center - self._previous_foot_x[i]) / dt,
                                    supported[i], float(self._support_heights[i])))
            self._previous_foot_x[i] = center
        return states, stumble

    def _landed_level(self, foot: FootState) -> int:
        height = self._terrain_step_height()
        if not foot.support or height == 0.0:
            return 0
        center = 0.5 * (foot.rear_x + foot.toe_x)
        level = int(round(self.terrain_height(center) / height))
        if not 1 <= level <= 6 or abs(foot.sole_z - level * height) > 0.035:
            return 0
        start = 0.9 + (level - 1) * 0.35
        end = self._landing_end if level == 6 else start + 0.35
        return level if foot.rear_x >= start + 0.005 and foot.toe_x <= end - 0.005 else 0

    def _world_foot_points(self, local_points: dict[int, np.ndarray]) -> list[np.ndarray]:
        return [np.concatenate([local_points[g] @ self.data.geom_xmat[g].reshape(3, 3).T
                                + self.data.geom_xpos[g] for g in geoms])
                for geoms in self._foot_geoms]

    def _stair_reward_terms(self, feet: list[FootState], dt: float) -> tuple[dict, dict]:
        """Three WM-LOCO-inspired terms; no contribution on a flat stage."""
        height = self._terrain_step_height()
        if height > 0.0:
            volume_points = self._world_foot_points(self._foot_volume_local)
            boundary = stair_boundary_count(volume_points, height, self._stair_geometry)
            riser = stair_riser_count(volume_points, height, self._stair_geometry)
            ratios = sole_contact_ratios(self._world_foot_points(self._foot_sole_local),
                                         [f.support for f in feet], height, self._stair_geometry)
        else:
            boundary, riser = 0, 0
            ratios = np.zeros((2, self._stair_geometry.step_count))
        sequential = self._tread_reward.step(ratios, float(self.command[0]), dt, active=height > 0.0)
        terms = {"stair_boundary": self.stair_boundary_weight * boundary,
                 "stair_riser": self.stair_riser_weight * riser,
                 "stair_tread": self.stair_tread_weight * sequential}
        diagnostics = {"stair_boundary_points": boundary, "stair_riser_points": riser,
                       "stair_sole_contact_ratios": ratios.tolist(),
                       "stair_completed_treads": self._tread_reward.current_index,
                       "stair_window_remaining": self._tread_reward.window_remaining}
        return terms, diagnostics

    def _is_fallen(self) -> bool:
        torso_pos = self.data.xpos[self._torso]
        torso_mat = self.data.xmat[self._torso].reshape(3, 3)
        local_ground = float(self.terrain_height(torso_pos[0]))
        return bool(torso_pos[2] - local_ground < 0.60 or torso_mat[2, 2] < 0.45
                    or abs(torso_pos[1]) > 1.6)

    def _action_target(self, action: np.ndarray) -> np.ndarray:
        scales = np.full(12, self.action_scale)
        stair_fraction = self._terrain_step_height() / 0.20
        # Keep the flat gait's mapping exactly. Gradually make room for hip
        # flexion and knee lift on stairs: the old +/-0.45 rad mapping could
        # not reach the 25 cm sole target at the nominal pelvis height.
        for hip, knee in ((0, 3), (6, 9)):
            if action[hip] < 0.0:
                scales[hip] += 0.35 * stair_fraction
            if action[knee] > 0.0:
                scales[knee] += 0.65 * stair_fraction
        target = self._lower_home + scales * action
        limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        bounds = self.model.jnt_range[self._joint_ids]
        target[limited] = np.clip(target[limited], bounds[limited, 0], bounds[limited, 1])
        return target

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.terrain_mode == "curriculum":
            self.curriculum.advance()
        mujoco.mj_resetData(self.model, self.data)
        self._set_stair_height()
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self._qpos_adrs] = self._lower_home
        self._lock_upper_body()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_x = float(self.data.xpos[self._torso, 0])
        self._support_heights[:] = 0.0
        self._best_landed_levels[:] = 0
        self._tread_reward.reset()
        self._foot_states(1.0)  # Initialize foot positions before differencing.
        self._goal_hold = 0.0
        self._stall_time = 0.0
        self._episode_done = False
        self.command = np.array([self.np_random.uniform(0.45, 0.75), 0.0], dtype=np.float32)
        return self._obs(), {"terrain_input": "ground_truth", "locked_upper_body": True}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        target = self._action_target(action)
        x_before = float(self.data.xpos[self._torso, 0])
        for _ in range(5):
            # Apply joint-space PD directly in generalized coordinates. This
            # works with the checked-in model and with official H2 MJCFs whose
            # actuator types/ranges differ.
            self.data.ctrl[:] = 0.0
            q = self.data.qpos[self._qpos_adrs]
            qd = self.data.qvel[self._qvel_adrs]
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self._qvel_adrs] = np.clip(
                self._kp * (target - q) - self._kd * qd,
                -self._torque_limit,
                self._torque_limit,
            )
            self._lock_upper_body()
            mujoco.mj_step(self.model, self.data)
            self._lock_upper_body()
        self._step_count += 1
        mujoco.mj_forward(self.model, self.data)
        x_after = float(self.data.xpos[self._torso, 0])
        dt = self.model.opt.timestep * 5.0
        progress = (x_after - x_before) / dt
        velocity_error = (progress - float(self.command[0])) ** 2
        upright = float(np.clip(self.data.xmat[self._torso, 8], -1.0, 1.0))
        feet, stumble = self._foot_states(dt)
        height = self._terrain_step_height()
        scores = [swing_score(foot, other_foot_support=feet[1 - i].support, step_height=height,
                              command_speed=float(self.command[0]), margin=self.clearance_margin)
                  for i, foot in enumerate(feet)]
        swing_count = max(sum(not f.support for f in feet), 1)
        for i, foot in enumerate(feet):
            level = self._landed_level(foot)
            if level > self._best_landed_levels[i]:
                self._best_landed_levels[i] = level
            if foot.support:
                # Only a sole near a horizontal surface can update the support
                # reference; pressing a toe against a vertical riser cannot.
                ground = float(self.terrain_height(0.5 * (foot.rear_x + foot.toe_x)))
                if abs(foot.sole_z - ground) < 0.035:
                    self._support_heights[i] = ground
        self._stall_time = self._stall_time + dt if progress < 0.03 else 0.0
        reward_terms = {
            "forward": 1.8 * min(progress, float(self.command[0])),
            "velocity_tracking": -0.35 * velocity_error,
            "upright": 0.8 * upright,
            "swing_clearance": (self.swing_reward_weight if height > 0.0 else 0.5)
                               * sum(s.clearance for s in scores) / swing_count,
            "low_clearance": -0.8 * sum(s.low_clearance for s in scores) / swing_count,
            "stumble": -0.25 * float(stumble),
            "stall": -0.75 * float(self._stall_time >= 2.0),
            "action": -0.002 * float(np.square(action).sum()),
            "fall": 0.0,
            "success": 0.0,
        }
        stair_terms, stair_diagnostics = self._stair_reward_terms(feet, dt)
        reward_terms.update(stair_terms)
        reward = sum(reward_terms.values())
        terminated = self._is_fallen()
        feet_on_top = height == 0.0 or (np.all(self._best_landed_levels == 6)
                                       and all(f.sole_z >= 6.0 * height - 0.035 for f in feet))
        stable_goal = x_after > 3.0 and feet_on_top and any(f.support for f in feet) and not terminated
        self._goal_hold = self._goal_hold + dt if stable_goal else 0.0
        success = self._goal_hold >= 0.15
        truncated = self._step_count >= self.max_episode_steps
        if terminated:
            reward -= 8.0
            reward_terms["fall"] = -8.0
        if success:
            reward += 10.0
            reward_terms["success"] = 10.0
            terminated = True
        if (terminated or truncated) and not self._episode_done:
            if self.terrain_mode == "curriculum":
                self.curriculum.record(success)
            self._episode_done = True
        info = {
            "x_position": x_after,
            "forward_speed": progress,
            "terrain_step_height": self._terrain_step_height(),
            "success": success,
            "locked_upper_body": True,
            "terrain_input": "ground_truth",
            "reward_terms": reward_terms,
            "foot_target_z": [s.target_z for s in scores],
            "foot_sole_z": [f.sole_z for f in feet],
            "foot_support": [f.support for f in feet],
            "max_landed_level": int(self._best_landed_levels.max()),
            "curriculum_level": self.curriculum.level,
            **stair_diagnostics,
        }
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            # Keep the environment headless-safe; use scripts/viewer.py for an interactive viewer.
            return None
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register("CtrlLab-H2-Stairs-MuJoCo-v0", entry_point="mujoco_stairs:H2StairsEnv")
