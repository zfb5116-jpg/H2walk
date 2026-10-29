from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


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
    Set ``terrain_mode='curriculum'`` to ramp step height from 5 to 20 cm.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        render_mode: str | None = None,
        terrain_mode: str = "20cm",
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if terrain_mode not in ("20cm", "curriculum"):
            raise ValueError("terrain_mode must be '20cm' or 'curriculum'")
        self.render_mode = render_mode
        self.terrain_mode = terrain_mode
        self.max_episode_steps = int(max_episode_steps)
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
        self._episode_count = 0
        self._curriculum_height = 0.20 if terrain_mode == "20cm" else 0.05
        self._last_x = 0.0
        self._renderer = None

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

    def _terrain_step_height(self) -> float:
        if self.terrain_mode == "20cm":
            return 0.20
        # Seven episode bands, ending at the requested 20 cm target.
        return self._curriculum_height

    def terrain_height(self, x: np.ndarray | float) -> np.ndarray | float:
        """Analytical height profile used both for the XML stairs and truth input."""
        height = self._terrain_step_height()
        x_arr = np.asarray(x)
        # The first step starts at x=0.9; each tread is 0.35 m long.
        n_steps = np.floor(np.maximum(x_arr - 0.9, 0.0) / 0.35) + 1.0
        out = np.where(x_arr < 0.9, 0.0, np.minimum(n_steps * height, 2.0))
        return float(out) if np.ndim(x) == 0 else out

    def _set_stair_height(self) -> None:
        """Keep the bundled XML geometry consistent with curriculum input."""
        height = self._terrain_step_height()
        for index, geom_id in enumerate(self._stair_geom_ids, start=1):
            if geom_id < 0:
                continue
            half_height = 0.5 * index * height
            self.model.geom_pos[geom_id, 2] = half_height
            self.model.geom_size[geom_id, 2] = half_height

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

    def _foot_z(self) -> tuple[float, float]:
        return float(self.data.xpos[self._left_foot, 2]), float(self.data.xpos[self._right_foot, 2])

    def _is_fallen(self) -> bool:
        torso_pos = self.data.xpos[self._torso]
        torso_mat = self.data.xmat[self._torso].reshape(3, 3)
        return bool(torso_pos[2] < 0.60 or torso_mat[2, 2] < 0.45)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_count += 1
        if self.terrain_mode == "curriculum":
            self._curriculum_height = float(min(0.05 + 0.025 * (self._episode_count // 8), 0.20))
        mujoco.mj_resetData(self.model, self.data)
        self._set_stair_height()
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self._qpos_adrs] = self._lower_home
        self._lock_upper_body()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._last_x = float(self.data.xpos[self._torso, 0])
        self.command = np.array([self.np_random.uniform(0.45, 0.75), 0.0], dtype=np.float32)
        return self._obs(), {"terrain_input": "ground_truth", "locked_upper_body": True}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        target = self._lower_home + self.action_scale * action
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
        x_after = float(self.data.xpos[self._torso, 0])
        dt = self.model.opt.timestep * 5.0
        progress = (x_after - x_before) / dt
        velocity_error = (progress - float(self.command[0])) ** 2
        upright = float(np.clip(self.data.xmat[self._torso, 8], -1.0, 1.0))
        foot_z = self._foot_z()
        terrain_under = float(self.terrain_height(x_after))
        foot_clearance = max(0.0, max(foot_z) - terrain_under)
        reward = 1.8 * progress - 0.35 * velocity_error + 0.8 * upright
        reward += 0.25 * min(foot_clearance / 0.20, 1.0) - 0.002 * float(np.square(action).sum())
        terminated = self._is_fallen()
        success = x_after > 0.9 + 6.0 * 0.35
        truncated = self._step_count >= self.max_episode_steps
        if terminated:
            reward -= 8.0
        if success:
            reward += 10.0
            terminated = True
        info = {
            "x_position": x_after,
            "forward_speed": progress,
            "terrain_step_height": self._terrain_step_height(),
            "success": success,
            "locked_upper_body": True,
            "terrain_input": "ground_truth",
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
