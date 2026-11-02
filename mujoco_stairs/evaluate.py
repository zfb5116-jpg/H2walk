from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import mujoco_stairs  # noqa: F401 - registers the environment
from mujoco_stairs.checkpoint import validate_checkpoint_contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a WM-LOCO leg+waist H2 policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--terrain-mode", choices=("flat", "low", "20cm"), default="20cm")
    parser.add_argument("--low-step-height", type=float, default=0.05)
    parser.add_argument("--stair-direction", choices=("up", "down", "mixed"), default="mixed")
    parser.add_argument("--stair-lane-half-width", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write per-episode diagnostics as JSON")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    kwargs = {"terrain_mode": args.terrain_mode, "low_step_height": args.low_step_height}
    kwargs.update(stair_direction=args.stair_direction, stair_lane_half_width=args.stair_lane_half_width)
    if args.model_path:
        kwargs["model_path"] = args.model_path
    env = gym.make("CtrlLab-H2-Stairs-MuJoCo-v1", **kwargs)
    try:
        validate_checkpoint_contract(args.checkpoint, env, allow_direction_change=True)
        policy = PPO.load(args.checkpoint, device="cpu")
    except Exception:
        env.close()
        raise
    if policy.observation_space != env.observation_space or policy.action_space != env.action_space:
        env.close()
        raise ValueError("Expected a new 86-observation/15-action policy; old leg-only checkpoints are incompatible")
    successes = []
    episodes = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        success = False
        max_x = -np.inf
        steps = 0
        totals = {}
        max_treads = 0
        peak_landing_speed = peak_tilt = peak_slide = 0.
        stable_steps = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = bool(info.get("success", False))
            steps += 1
            max_x = max(max_x, info["x_position"])
            peak_landing_speed = max(peak_landing_speed, max(info["landing_downward_speed"]))
            peak_tilt = max(peak_tilt, info["base_tilt"])
            peak_slide = max(peak_slide, info["supported_foot_slide_speed"])
            stable_steps += int(any(info["stable_foot_support"]))
            max_treads = max(max_treads, info["stair_completed_treads"])
            for name, value in info["reward_terms"].items():
                totals[name] = totals.get(name, 0.0) + value
        successes.append(success)
        episodes.append({"success": success, "max_x": float(max_x), "steps": steps,
                         "stair_direction": info["stair_direction"],
                         "sequential_treads": max_treads, "reward_totals": totals,
                         "fallen": info["fallen"], "peak_landing_speed": peak_landing_speed,
                         "peak_base_tilt_rad": peak_tilt, "peak_support_slide_speed": peak_slide,
                         "stable_support_fraction": stable_steps / steps})
    contract = env.unwrapped.get_training_contract()
    env.close()
    print(f"{args.terrain_mode} success: {np.mean(successes):.1%} ({sum(successes)}/{len(successes)})")
    print(f"mean max x: {np.mean([e['max_x'] for e in episodes]):.3f} m; "
          f"mean completed sequential treads: {np.mean([e['sequential_treads'] for e in episodes]):.2f}")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"terrain_mode": args.terrain_mode, "training_contract": contract, "episodes": episodes}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
