from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from dataclasses import replace
import hashlib
from .rewards import REWARD_VERSION, REWARD_WEIGHTS, compute_standard_rewards, weighted_rewards

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
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
CONTROLLED_JOINTS = LOWER_BODY_JOINTS + WAIST_JOINTS
CONTRACT_VERSION = "h2_legs_waist_15_v2"
ARM_JOINTS = (
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
    """MuJoCo H2 locomotion with active leg and waist control.

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
        stair_direction: str = "mixed",
        stair_lane_half_width: float = 0.60,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if terrain_mode not in ("flat", "low", "20cm", "curriculum"):
            raise ValueError("terrain_mode must be flat, low, 20cm, or curriculum")
        if not 0.0 < low_step_height <= 0.20 or max_episode_steps < 1:
            raise ValueError("low_step_height must be in (0, 0.20] and episode length positive")
        if stair_direction not in ("up", "down", "mixed"):
            raise ValueError("stair_direction must be up, down, or mixed")
        self.stair_direction = stair_direction
        self.episode_direction = "up"
        if not 0.0 < stair_lane_half_width <= 1.6:
            raise ValueError("stair_lane_half_width must be in (0, 1.6]")
        self.render_mode = render_mode
        self.terrain_mode = terrain_mode
        self.max_episode_steps = int(max_episode_steps)
        self.low_step_height = float(low_step_height)
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
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or self.model.jnt_qposadr[0] != 0:
            raise ValueError("The model must start with the robot free-root joint")
        self._joint_ids = self._resolve_joint_ids(CONTROLLED_JOINTS, required=True)
        self._arm_joint_ids = self._resolve_joint_ids(ARM_JOINTS, required=False)
        self._left_foot = self._find_body("left_ankle_pitch_link", "left_foot")
        self._right_foot = self._find_body("right_ankle_pitch_link", "right_foot")
        self._torso = self._find_body("pelvis", "base")
        self._upper_body = self._find_body("torso", "upper_mass")
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
        self._start_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "stair_start")
        if stair_direction != "up" and self._start_geom < 0:
            raise ValueError("Descending scenes require a stair_start platform at x=-4..0.9")
        self._landing_end = 4.3 if landing >= 0 else 3.0
        self._stair_geometry = StairGeometry(landing_end=self._landing_end,
                                             lane_half_width=stair_lane_half_width)
        self._tread_reward = SequentialTreadReward(self._stair_geometry)
        self._foot_geoms = [self._collision_geoms(b) for b in (self._left_foot, self._right_foot)]
        self._geom_to_foot = {g: i for i, geoms in enumerate(self._foot_geoms) for g in geoms}
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
        self._arm_qpos_adrs = np.asarray(
            [self.model.jnt_qposadr[i] for i in self._arm_joint_ids], dtype=np.int32
        )
        self._arm_qvel_adrs = np.asarray(
            [self.model.jnt_dofadr[i] for i in self._arm_joint_ids], dtype=np.int32
        )
        self._arm_home = np.asarray(
            [STANDING_POSE.get(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(i)), 0.0) for i in self._arm_joint_ids], dtype=np.float64
        )
        self._controlled_home = np.asarray(
            [STANDING_POSE[name] for name in CONTROLLED_JOINTS], dtype=np.float64
        )
        self.action_scales = np.array([.8, .35, .4, 1.1, .3, .5] * 2 + [.5, .35, .6])
        self._kp = np.asarray([180.0, 100.0, 100.0, 180.0, 50.0, 60.0] * 2 + [120., 150., 150.], dtype=np.float64)
        self._kd = np.asarray([8.0, 5.0, 5.0, 8.0, 2.0, 2.0] * 2 + [8., 10., 10.], dtype=np.float64)
        self._torque_limit = np.asarray([360.0, 100.0, 100.0, 360.0, 66.0, 66.0] * 2 + [100., 100., 100.], dtype=np.float64)
        self.command = np.array([.3, 0., 0.], dtype=np.float32)
        self._step_count = 0
        self._renderer = self._viewer = None
        self._goal_hold = 0.0
        self._stall_time = 0.0
        self._episode_done = False

        # Versioned proprioception, previous actions, commands and GT terrain.
        self.terrain_x = np.linspace(-0.45, 1.85, 24, dtype=np.float32)
        self._velocity_limits = np.array([12.] * 12 + [8.] * 3)
        self._previous_action = np.zeros(15)
        self._previous_velocity = np.zeros(15)
        self._air_time = np.zeros(2)
        self._previous_support = np.zeros(2, dtype=bool)
        self._target_heading = 0.
        self.dt = self.model.opt.timestep * 5
        obs_size = 86
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (len(CONTROLLED_JOINTS),), dtype=np.float32)

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
            raise ValueError("Each foot must have collision geometry for foot contact rewards")
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
        region = np.minimum(np.floor(np.maximum(x_arr - .9, 0.) / .35).astype(int), 5)
        levels = region + 1 if self.episode_direction == "up" else 5 - region
        before = 0. if self.episode_direction == "up" else 6 * height
        out = np.where(x_arr < .9, before, np.where(x_arr < self._landing_end, levels * height, 0.))
        return float(out) if out.ndim == 0 else out

    def _set_box_height(self, geom, height):
        half = max(height / 2., .001)
        self.model.geom_pos[geom, 2] = half if height > 0 else -.001
        self.model.geom_size[geom, 2] = half
        self.model.geom_aabb[geom, 3:] = self.model.geom_size[geom]
        self.model.geom_rbound[geom] = np.linalg.norm(self.model.geom_size[geom])
        self.model.geom_contype[geom] = self.model.geom_conaffinity[geom] = int(height > 0)
        self.model.geom_rgba[geom, 3] = float(height > 0)

    def _set_stair_height(self):
        h = self._terrain_step_height()
        for i, geom in enumerate(self._terrain_geom_ids):
            self._set_box_height(geom, self._stair_geometry.tread_height(min(i, 5), h))
        if self._start_geom >= 0:
            self._set_box_height(self._start_geom, 6 * h if self.episode_direction == "down" else 0.)

    def _body_velocity(self, body):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return velocity

    def _heading_error(self):
        rot = self.data.xmat[self._torso].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        return float(np.arctan2(np.sin(self._target_heading - yaw), np.cos(self._target_heading - yaw)))

    def _obs(self):
        rotation = self.data.xmat[self._torso].reshape(3, 3)
        upper = self.data.xmat[self._upper_body].reshape(3, 3)
        velocity = self._body_velocity(self._torso)
        pos = self.data.xpos[self._torso]
        error = self._heading_error()
        terrain = self.terrain_height(pos[0] + self.terrain_x * rotation[0, 0]) - pos[2]
        return np.concatenate((rotation.T @ velocity[3:], rotation.T @ velocity[:3],
                               -rotation[2], -upper[2], self.data.qpos[self._qpos_adrs] - self._controlled_home,
                               self.data.qvel[self._qvel_adrs], self._previous_action, self.command,
                               [np.sin(error), np.cos(error)], terrain)).astype(np.float32)

    def _foot_support(self):
        """Only upward static-terrain contacts above 5 N support a foot."""
        supported = np.zeros(2, dtype=bool)
        force = np.zeros(6)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            for foot_geom, other_geom in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                foot_id = self._geom_to_foot.get(int(foot_geom))
                if foot_id is None or self.model.body_weldid[self.model.geom_bodyid[other_geom]] != 0:
                    continue
                mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                if force[0] > 5. and abs(contact.frame[2]) > .5:
                    supported[foot_id] = True
        return supported

    def _world_foot_points(self, local_points: dict[int, np.ndarray]) -> list[np.ndarray]:
        return [np.concatenate([local_points[g] @ self.data.geom_xmat[g].reshape(3, 3).T
                                + self.data.geom_xpos[g] for g in geoms])
                for geoms in self._foot_geoms]

    def _stair_reward_terms(self, support, dt: float) -> tuple[dict, dict]:
        """Three WM-LOCO-inspired terms; no contribution on a flat stage."""
        height = self._terrain_step_height()
        if height > 0.0:
            volume_points = self._world_foot_points(self._foot_volume_local)
            boundary = stair_boundary_count(volume_points, height, self._stair_geometry)
            riser = stair_riser_count(volume_points, height, self._stair_geometry)
            ratios = sole_contact_ratios(self._world_foot_points(self._foot_sole_local),
                                         support, height, self._stair_geometry)
        else:
            boundary, riser = 0, 0
            ratios = np.zeros((2, self._stair_geometry.step_count))
        sequential = self._tread_reward.step(ratios, float(self.command[0]), dt, active=height > 0.0)
        terms = {"stair_boundary": float(boundary),
                 "stair_riser": float(riser),
                 "stair_tread": sequential}
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

    def _action_target(self, action):
        target = self._controlled_home + self.action_scales * action
        limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        bounds = self.model.jnt_range[self._joint_ids]
        target[limited] = np.clip(target[limited], bounds[limited, 0], bounds[limited, 1])
        return target

    def get_training_contract(self):
        return {"contract_version": CONTRACT_VERSION, "reward_version": REWARD_VERSION,
                "controlled_joints": list(CONTROLLED_JOINTS), "observations": 86, "actions": 15,
                "reward_weights": dict(REWARD_WEIGHTS), "control_dt": self.dt,
                "stair_direction": self.stair_direction,
                "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                "action_scales": self.action_scales.tolist(),
                "kp": self._kp.tolist(), "kd": self._kd.tolist(),
                "torque_limits": self._torque_limit.tolist(),
                "velocity_limits": self._velocity_limits.tolist(),
                "stair_lane_half_width": self._stair_geometry.lane_half_width}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.terrain_mode == "curriculum":
            self.curriculum.advance()
        self.episode_direction = str(self.np_random.choice(("up", "down"))) if self.stair_direction == "mixed" else self.stair_direction
        self._stair_geometry = replace(self._stair_geometry, direction=self.episode_direction)
        self._tread_reward = SequentialTreadReward(self._stair_geometry)
        self._set_stair_height()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_adrs] = self._controlled_home
        self.data.qpos[self._arm_qpos_adrs] = self._arm_home
        self.data.qpos[2] += self.terrain_height(0.)
        mujoco.mj_forward(self.model, self.data)
        self._previous_action.fill(0.)
        self._previous_velocity.fill(0.)
        self._air_time.fill(0.)
        self._previous_support[:] = self._foot_support()
        self._stall_time = self._goal_hold = self._target_heading = 0.
        self._step_count = 0
        self._episode_done = False
        self.command[:] = [self.np_random.uniform(.15, .45), 0., 0.]
        if options and "command" in options:
            command = np.asarray(options["command"], dtype=float)
            if command.shape != (3,) or not np.isfinite(command).all() or command[0] < 0:
                raise ValueError("command must be finite [vx>=0, vy, wz]")
            self.command[:] = command
        return self._obs(), {"training_contract": self.get_training_contract(), "stair_direction": self.episode_direction}

    def _undesired_contacts(self):
        bodies = set()
        force = np.zeros(6)
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            mujoco.mj_contactForce(self.model, self.data, ci, force)
            if force[0] <= 5.:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            if b1 and b2:
                bodies.update((b1, b2))
            else:
                geom = g1 if b1 else g2
                if geom not in self._geom_to_foot:
                    bodies.add(int(self.model.geom_bodyid[geom]))
        return len(bodies)

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("reset() is required after an episode terminates")
        action = np.asarray(action, dtype=float)
        if action.shape != (15,) or not np.isfinite(action).all():
            raise ValueError("Expected 15 finite leg+waist actions; legacy 12-action policies are incompatible")
        action = np.clip(action, -1., 1.)
        previous_action = self._previous_action.copy()
        target = self._action_target(action)
        before = float(self.data.xpos[self._torso, 0])
        for _ in range(5):
            self.data.ctrl[:] = 0.
            self.data.qfrc_applied[:] = 0.
            torque = np.clip(self._kp * (target - self.data.qpos[self._qpos_adrs])
                             - self._kd * self.data.qvel[self._qvel_adrs], -self._torque_limit, self._torque_limit)
            self.data.qfrc_applied[self._qvel_adrs] = torque
            # Arms use finite PD; waist is excluded and never reset in a step.
            self.data.qfrc_applied[self._arm_qvel_adrs] = np.clip(40 * (self._arm_home - self.data.qpos[self._arm_qpos_adrs])
                                                                   - 4 * self.data.qvel[self._arm_qvel_adrs], -30., 30.)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        pos = self.data.xpos[self._torso]
        progress = float((pos[0] - before) / self.dt)
        self._target_heading += float(self.command[2]) * self.dt
        support = self._foot_support()
        self._air_time += ~support * self.dt
        landed_air_time = np.where(support & ~self._previous_support, self._air_time, 0.)
        self._air_time[support] = 0.
        self._previous_support[:] = support
        self._stall_time = self._stall_time + self.dt if self.command[0] > .05 and progress < .1 * self.command[0] else 0.
        rotation = self.data.xmat[self._torso].reshape(3, 3)
        upper = self.data.xmat[self._upper_body].reshape(3, 3)
        velocity = self._body_velocity(self._torso)
        foot_bodies = [self._left_foot, self._right_foot]
        foot_velocity = np.array([self._body_velocity(b)[3:] for b in foot_bodies])
        sole_points = self._world_foot_points(self._foot_sole_local)
        plane_error = np.array([np.square(p[:, 2] - self.terrain_height(p[:, 0])).mean() for p in sole_points])
        dq = self.data.qvel[self._qvel_adrs].copy()
        acceleration = (dq - self._previous_velocity) / self.dt
        self._previous_velocity[:] = dq
        fallen = self._is_fallen()
        limited = self.model.jnt_limited[self._joint_ids].astype(bool)
        bounds = self.model.jnt_range[self._joint_ids]
        state = dict(command=self.command, linear_velocity=rotation.T @ velocity[3:],
                     angular_velocity=rotation.T @ velocity[:3], heading_error=self._heading_error(),
                     joint_deviation=self.data.qpos[self._qpos_adrs] - self._controlled_home,
                     joint_position=self.data.qpos[self._qpos_adrs], joint_bounds=bounds,
                     joint_limited=limited, joint_velocity=dq, joint_acceleration=acceleration,
                     velocity_limits=self._velocity_limits, torque=torque, torque_limits=self._torque_limit,
                     action=action, previous_action=previous_action, support=support,
                     landed_air_time=landed_air_time, forward_speed=progress, stall_time=self._stall_time,
                     foot_position=self.data.xpos[foot_bodies], foot_velocity=foot_velocity,
                     foot_up=np.array([self.data.xmat[b].reshape(3, 3)[:, 2] for b in foot_bodies]),
                     sole_plane_error=plane_error, hip_anchor_indices=[1, 2, 7, 8],
                     torso_gravity=-upper[2], pelvis_gravity=-rotation[2], fallen=fallen,
                     undesired_contacts=self._undesired_contacts())
        raw = compute_standard_rewards(state)
        stair_raw, diagnostics = self._stair_reward_terms(support, self.dt)
        raw.update(stair_raw)
        terms = weighted_rewards(raw, self.dt)
        self._previous_action[:] = action
        sequence_done = self._terrain_step_height() == 0 or self._tread_reward.current_index == 6
        stable_goal = (pos[0] > 3.0 and abs(pos[1]) < self._stair_geometry.lane_half_width
                       and sequence_done and support.any() and rotation[2, 2] > .8 and not fallen)
        self._goal_hold = self._goal_hold + self.dt if stable_goal else 0.
        success = self._goal_hold >= .2
        terminated = bool(fallen or success)
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        if terminated or truncated:
            if self.terrain_mode == "curriculum":
                self.curriculum.record(success)
            self._episode_done = True
        info = {"x_position": float(pos[0]), "y_position": float(pos[1]), "forward_speed": progress,
                "terrain_step_height": self._terrain_step_height(), "stair_direction": self.episode_direction,
                "success": bool(success), "fallen": fallen, "reward_terms": terms, "reward_raw": raw,
                "reward_version": REWARD_VERSION, "waist_controlled": True,
                "waist_position": self.data.qpos[self._qpos_adrs[-3:]].tolist(),
                "foot_support": support.tolist(), "curriculum_level": self.curriculum.level, **diagnostics}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(sum(terms.values())), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.data.xpos[self._torso]
            camera.distance, camera.azimuth, camera.elevation = 4., 135., -15.
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render()

    def close(self):
        for obj in (self._renderer, self._viewer):
            if obj is not None:
                obj.close()
        self._renderer = self._viewer = None


gym.register("CtrlLab-H2-Stairs-MuJoCo-v1", entry_point="mujoco_stairs:H2StairsEnv")
