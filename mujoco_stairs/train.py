from __future__ import annotations

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import mujoco_stairs  # noqa: F401 - registers the environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lower-body-only H2 on 20 cm stairs in MuJoCo")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--terrain-mode", choices=("20cm", "curriculum"), default="curriculum")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-dir", default="logs/mujoco_h2_stairs")
    args = parser.parse_args()
    env_kwargs = {"terrain_mode": args.terrain_mode}
    if args.model_path:
        env_kwargs["model_path"] = args.model_path
    env = make_vec_env("CtrlLab-H2-Stairs-MuJoCo-v0", n_envs=args.n_envs, env_kwargs=env_kwargs)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=2048,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        tensorboard_log=args.run_dir,
        verbose=1,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
    model.save(f"{args.run_dir}/h2_stairs_lowerbody_ppo")
    env.close()


if __name__ == "__main__":
    main()
