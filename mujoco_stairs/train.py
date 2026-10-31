from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

import mujoco_stairs  # noqa: F401 - registers the environment


def save_training_state(model: PPO, path: Path, args: argparse.Namespace) -> None:
    model.save(str(path))
    state = {"terrain_mode": args.terrain_mode, "num_timesteps": model.num_timesteps,
             "arguments": vars(args),
             "environments": model.get_env().env_method("get_curriculum_state")}
    path.with_suffix(".curriculum.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


class TrainingDiagnostics(BaseCallback):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.term_sums: dict[str, float] = {}
        self.samples = 0
        self.height_sum = 0.0
        self.last_save = 0

    def _on_training_start(self) -> None:
        self.last_save = self.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for name, value in info.get("reward_terms", {}).items():
                self.term_sums[name] = self.term_sums.get(name, 0.0) + value
            self.height_sum += info["terrain_step_height"]
            self.samples += 1
        if self.args.save_every > 0 and self.num_timesteps - self.last_save >= self.args.save_every:
            save_training_state(self.model, Path(self.args.run_dir) / f"ckpt_{self.num_timesteps}", self.args)
            self.last_save = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        for name, total in self.term_sums.items():
            self.logger.record(f"reward/{name}", total / max(self.samples, 1))
        self.logger.record("terrain/mean_step_height", self.height_sum / max(self.samples, 1))
        heights = self.training_env.env_method("_terrain_step_height")
        self.logger.record("terrain/min_step_height", float(np.min(heights)))
        self.logger.record("terrain/max_step_height", float(np.max(heights)))
        self.term_sums.clear()
        self.samples = 0
        self.height_sum = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H2: flat -> low stairs -> 20 cm stairs")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Additional steps to train, including after --init-from")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--terrain-mode", choices=("flat", "low", "20cm", "curriculum"), default="curriculum")
    parser.add_argument("--start-stage", choices=("flat", "low", "20cm"), default="flat")
    parser.add_argument("--low-step-height", type=float, default=0.05)
    parser.add_argument("--curriculum-window", type=int, default=20)
    parser.add_argument("--curriculum-success-rate", type=float, default=0.8)
    parser.add_argument("--clearance-margin", type=float, default=0.05)
    parser.add_argument("--swing-reward-weight", type=float, default=2.5)
    parser.add_argument("--stair-boundary-weight", type=float, default=-0.05)
    parser.add_argument("--stair-riser-weight", type=float, default=-0.5)
    parser.add_argument("--stair-tread-weight", type=float, default=0.5)
    parser.add_argument("--stair-lane-half-width", type=float, default=0.60)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--resume-curriculum", action="store_true",
                        help="Restore curriculum sidecar as well as policy/optimizer state")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-dir", default="logs/mujoco_h2_stairs")
    args = parser.parse_args()
    if min(args.n_envs, args.total_timesteps, args.batch_size) < 1 or args.n_steps < 2:
        parser.error("env count, training steps and batch size must be positive; n-steps must be >=2")
    if args.resume_curriculum and (not args.init_from or args.terrain_mode != "curriculum"):
        parser.error("--resume-curriculum requires --init-from and --terrain-mode curriculum")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    env_kwargs = {name: getattr(args, name) for name in (
        "terrain_mode", "low_step_height", "start_stage", "curriculum_window",
        "curriculum_success_rate", "clearance_margin", "swing_reward_weight",
        "stair_boundary_weight", "stair_riser_weight", "stair_tread_weight", "stair_lane_half_width")}
    if args.model_path:
        env_kwargs["model_path"] = args.model_path
    env = make_vec_env("CtrlLab-H2-Stairs-MuJoCo-v0", n_envs=args.n_envs,
                       seed=args.seed, env_kwargs=env_kwargs,
                       monitor_kwargs={"info_keywords": ("success", "terrain_step_height", "max_landed_level")})
    try:
        if args.resume_curriculum:
            path = Path(args.init_from).with_suffix(".curriculum.json")
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["terrain_mode"] != "curriculum" or len(state["environments"]) != args.n_envs:
                parser.error("saved curriculum must be from curriculum mode with the same n-envs")
            for index, saved in enumerate(state["environments"]):
                env.env_method("set_curriculum_state", saved, indices=index)
        lr = args.learning_rate if args.learning_rate is not None else (1e-4 if args.init_from else 3e-4)
        batch_size = min(args.batch_size, args.n_envs * args.n_steps)
        if args.init_from:
            model = PPO.load(args.init_from, env=env, device=args.device,
                             learning_rate=lr, n_steps=args.n_steps, batch_size=batch_size,
                             tensorboard_log=args.run_dir, seed=args.seed)
        else:
            model = PPO("MlpPolicy", env, n_steps=args.n_steps, batch_size=batch_size,
                        learning_rate=lr, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                        ent_coef=0.005,
                        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
                        tensorboard_log=args.run_dir, verbose=1, device=args.device, seed=args.seed)
        model.learn(total_timesteps=args.total_timesteps, callback=TrainingDiagnostics(args),
                    reset_num_timesteps=not bool(args.init_from))
        save_training_state(model, Path(args.run_dir) / "h2_stairs_lowerbody_ppo", args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
