from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import mujoco_stairs  # noqa: F401 - registers the environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a lower-body H2 stair policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    kwargs = {"terrain_mode": "20cm", "render_mode": "human"}
    if args.model_path:
        kwargs["model_path"] = args.model_path
    env = gym.make("CtrlLab-H2-Stairs-MuJoCo-v0", **kwargs)
    policy = PPO.load(args.checkpoint)
    successes = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        success = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = bool(info.get("success", False))
        successes.append(success)
    env.close()
    print(f"20cm stair success: {np.mean(successes):.1%} ({sum(successes)}/{len(successes)})")


if __name__ == "__main__":
    main()
