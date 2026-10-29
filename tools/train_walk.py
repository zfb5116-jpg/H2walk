"""Explicit scanned flat-to-stairs entry for RSL-RL 2.3.3.

Bootstrap is a fresh fixed-scale/RND-disabled experiment. Normalized or
unmarked older weights cannot resume here. --max_iterations is additional
work, never evidence of completed work. Import and --help do not launch Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
TASKS = tuple(
    "Unitree-H2-Velocity-Walk-" + name
    for name in ("Bootstrap", "Shallow", "Stairs08", "Stairs12", "Stairs16", "Stairs20")
)


def build_parser(
    add_launcher_args: Callable[[argparse.ArgumentParser], None] | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Isaac AppLauncher options (including --headless and --device) are accepted at runtime.",
    )
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Additional iterations for this run.",
    )
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", default="")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit same-protocol checkpoint only.",
    )
    parser.add_argument("--expected_completed_iterations", type=int, default=None)
    if add_launcher_args is not None:
        add_launcher_args(parser)
    return parser


def parse_args(
    argv: list[str] | None = None,
    add_launcher_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = build_parser(add_launcher_args)
    args = parser.parse_args(argv)
    for name in (
        "num_envs",
        "max_iterations",
        "save_interval",
        "expected_completed_iterations",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(name + " must be positive")
    if args.expected_completed_iterations is not None and args.checkpoint is None:
        parser.error("expected_completed_iterations requires --checkpoint")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_name):
        parser.error("run_name must be a single 1-80 character ASCII name")
    return args


def load_config(entry: str) -> Any:
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def registry_configs(
    gym: Any, task: str, loader: Callable[[str], Any] = load_config
) -> tuple[Any, Any]:
    spec = gym.spec(task)
    return loader(spec.kwargs["env_cfg_entry_point"]), loader(
        spec.kwargs["rsl_rl_cfg_entry_point"]
    )


def resolved_configuration_signature(
    env_cfg: Any, agent_cfg: Any, fingerprint: Callable[..., str]
) -> str:
    """Pin input/action transforms including wrapper clipping, before environment creation."""
    return fingerprint(
        env_cfg.observations,
        env_cfg.actions,
        agent_cfg.policy,
        {
            "physics_dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "default_joint_pos": env_cfg.scene.robot.init_state.joint_pos,
            "wrapper_clip_actions": agent_cfg.clip_actions,
        },
    )


def source_provenance(root: Path = ROOT) -> dict[str, Any]:
    """Record actual local Python content even when Git is absent or checkout dirty."""
    files = sorted(
        (root / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task").rglob("*.py")
    )
    files += [root / "tools/train_walk.py", root / "tools/walk_protocol.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    commit = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        candidate = result.stdout.strip()
        if re.fullmatch("[0-9a-f]{40}", candidate):
            commit = candidate
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    actual = list(sys.argv[1:] if argv is None else argv)
    if "--help" in actual or "-h" in actual:
        parse_args(["--help"])
        return
    # Importing this module or asking for help must not create a simulation app.
    from isaaclab.app import AppLauncher

    args = parse_args(actual, AppLauncher.add_app_launcher_args)
    app = AppLauncher(args).app
    try:
        _run(args)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> None:
    # Everything below is an explicitly user-invoked training path. CPU tests
    # never call it, construct its objects or execute learn/optimizer updates.
    import gymnasium as gym
    import torch
    from importlib.metadata import PackageNotFoundError, version
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    dependencies: dict[str, str | None] = {
        "rsl_rl": version("rsl-rl-lib"),
        "pytorch": str(torch.__version__),
    }
    for name in ("isaaclab", "isaaclab_rl"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    if dependencies["rsl_rl"] != "2.3.3":
        raise RuntimeError(
            "train_walk.py requires the reviewed rsl-rl-lib==2.3.3 interface"
        )
    sys.path.insert(0, str(ROOT / "tools"))
    from walk_protocol import (
        configuration_fingerprint,
        guarded_runner_class,
        make_protocol,
        restore_checkpoint,
        validate_checkpoint,
        validate_runner_config,
    )

    sys.path.insert(0, str(ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based"))
    import h2_task

    env_cfg, agent_cfg = registry_configs(gym, args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = agent_cfg.device = args.device
    for name in ("max_iterations", "save_interval"):
        if getattr(args, name) is not None:
            setattr(agent_cfg, name, getattr(args, name))
    agent_cfg.logger = "tensorboard"
    agent_cfg.resume = args.checkpoint is not None
    validate_runner_config(agent_cfg.to_dict())
    signature = resolved_configuration_signature(
        env_cfg, agent_cfg, configuration_fingerprint
    )
    protocol = make_protocol(
        env_cfg.walking_stage,
        env_cfg.walking_cross_time_mode,
        env_cfg.walking_max_time_difference,
        source_provenance(),
        configuration_sha256=signature,
    )
    # Read once, with weights-only restrictions, and reject bad input before
    # gym.make or runner/model/optimizer construction. No unsafe load fallback.
    saved = None
    completed = 0
    checkpoint = (
        args.checkpoint.resolve(strict=True) if args.checkpoint is not None else None
    )
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        completed = validate_checkpoint(
            saved,
            protocol,
            expected_completed_iterations=args.expected_completed_iterations,
        )
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        run_name += "_" + args.run_name
    log_dir = ROOT / "logs/rsl_rl/h2_walk_staged_fixed_scale" / run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.io_descriptors_output_dir = str(log_dir)
    dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
    metadata = {
        "task": args.task,
        "task_source": h2_task.__file__,
        "protocol": protocol,
        "dependency_versions": dependencies,
        "asset": env_cfg.scene.robot.spawn.usd_path,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "device": agent_cfg.device,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "starting_completed_iterations": completed,
        "additional_iterations_requested": agent_cfg.max_iterations,
        "target_completed_iterations": completed + agent_cfg.max_iterations,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval,
        "status": "initializing",
    }

    def write_metadata() -> None:
        (log_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    write_metadata()
    print("[H2-Walk] " + json.dumps(metadata), flush=True)
    env = None
    try:
        env = gym.make(args.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        obs, extras = env.get_observations()
        critic = extras["observations"]["critic"]
        if (
            obs.ndim != 2
            or obs.shape[1] != 1029
            or critic.ndim != 2
            or critic.shape[1] != 866
            or env.num_actions != 29
            or obs.shape[0] != args.num_envs
            or critic.shape[0] != args.num_envs
        ):
            raise ValueError(
                "Runtime dimensions disagree with the fixed walking protocol"
            )
        if not torch.isfinite(obs).all() or not torch.isfinite(critic).all():
            raise ValueError("Non-finite initial actor/critic observations")
        metadata["runtime_dimensions"] = {
            "actor": list(obs.shape),
            "critic": list(critic.shape),
            "actions": env.num_actions,
        }
        runner_type = guarded_runner_class(OnPolicyRunner, protocol)
        runner = runner_type(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        if saved is not None:
            restore_checkpoint(
                runner,
                saved,
                protocol,
                expected_completed_iterations=args.expected_completed_iterations,
            )
            del saved
        metadata["status"] = "training"
        write_metadata()
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=False,
        )
        metadata["status"] = "completed"
        metadata["completed_iterations"] = runner.current_learning_iteration + 1
        write_metadata()
    except BaseException:
        metadata["status"] = "interrupted_or_failed"
        write_metadata()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
