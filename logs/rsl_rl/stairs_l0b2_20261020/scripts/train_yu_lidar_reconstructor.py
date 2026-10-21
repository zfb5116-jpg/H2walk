#!/usr/bin/env python3
"""Phase-1: supervise LidarTerrainReconstructor (range image → H^b).

Default = flat + L0 stones + walking policy. Actor still eats GT H^b.
Reconstructor proprio = current obs + policy hat_v. Yaw is not locked (train or preview).
Delayed lidar frames are ego-warped into the current lidar pose (YU_LIDAR_EGO_WARP).
Flat+L0..L4 run:
  bash scripts/go2_terrain/train_yu_lidar_reconstructor_flat_l0_l4.sh

Usage:
  bash scripts/go2_terrain/train_yu_lidar_reconstructor.sh --steps 20000 --num_envs 64
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_PAPER_LOG = _REPO / "logs/rsl_rl/go2_yu_paper_stones_manager_based"
_DEFAULT_PAPER_CKPT = _DEFAULT_PAPER_LOG / "2026-08-28_00-17-27" / "model_39999.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Unitree-Go2-YuPaperStones-Play")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--mix_every", type=int, default=64)
parser.add_argument("--pin", action="store_true")
parser.add_argument("--terrain_level", type=int, default=3)
parser.add_argument("--terrain_type", type=int, default=1)
parser.add_argument("--stones_only", action="store_true")
parser.add_argument(
    "--stairs",
    action="store_true",
    help="RoughYu stair map: flat + up/down stairs. Continuous height loss, no void split.",
)
parser.add_argument("--row_min", type=int, default=0)
parser.add_argument("--row_max", type=int, default=0)
parser.add_argument("--no_refine", action="store_true")
parser.add_argument("--policy_ckpt", type=Path, default=None)
parser.add_argument("--init_ckpt", type=Path, default=None, help="Warm-start reconstructor weights.")
parser.add_argument(
    "--freeze_backbone",
    action="store_true",
    help="Train only the edge-clean layer. Use with --init_ckpt on a finished stair run.",
)
parser.add_argument("--void_weight", type=float, default=None, help="Override model void_weight (default 3.0).")
parser.add_argument(
    "--gru_delay_s",
    type=float,
    default=None,
    help="GRU token lag in seconds (default=PAPER_LIDAR_DELAY_S). 0 disables Δ-align.",
)
parser.add_argument(
    "--no_gru_delay_align",
    action="store_true",
    help="Disable GRU [t−Δ,t] token sequence (old per-step single token).",
)
parser.add_argument("--random_actions", action="store_true")
parser.add_argument("--lin_vel_x", type=float, default=0.6)
parser.add_argument(
    "--unlock_yaw",
    action="store_true",
    help="Sample ωz∈[-1.2,1.2] while walking (paper zeros yaw if vx>0). Preview does not lock yaw.",
)
parser.add_argument(
    "--hb_splat",
    action="store_true",
    help="Unproject lidar into the yaw-aligned H^b grid before the decoder.",
)
parser.add_argument(
    "--height_splat",
    action="store_true",
    help="Stairs: max-z splat prior, then reconstruct the full H^b including cells with no return.",
)
parser.add_argument(
    "--flat_copies",
    type=int,
    default=1,
    help="Repeat flat cell this many times in the MIX (higher → less stone-gap hallucination).",
)
parser.add_argument("--out-dir", type=Path, default=Path("logs/rsl_rl/yu_lidar_reconstructor"))
parser.add_argument("--fig-dir", type=Path, default=Path("docs/wtr_assets/yu_lidar_recon_spatial"))
parser.add_argument("--fig-every", type=int, default=1000, help="Write preview PNG every N steps (0=only at end).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import CtrlLab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from CtrlLab.tasks.manager_based.go2_task.robot_env_cfg import PAPER_LIDAR_DELAY_S
from rsl_rl.modules.encoder_actor import (
    PAPER_LIDAR_AZ_STRIDE,
    PAPER_LIDAR_CHANNELS,
    PAPER_LIDAR_FRAMES,
    PAPER_LIDAR_N_AZ,
    PAPER_LIDAR_OBS_N_AZ,
    PAPER_LIDAR_RANGE_DIM,
    RECON_PROPRIO_DIM,
    recon_proprio,
    split_policy_lidar,
)
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.networks import LidarTerrainReconstructor
from rsl_rl.networks.terrain_reconstructor import HB_DIM, HB_NX, HB_NY


def _resolve_walk_ckpt(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--policy_ckpt not found: {p}")
        return p
    try:
        resolver = _REPO / "scripts/go2_terrain/resolve_latest_ckpt.py"
        spec = importlib.util.spec_from_file_location("resolve_latest_ckpt", resolver)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if _DEFAULT_PAPER_LOG.is_dir():
            return mod.latest_checkpoint(_DEFAULT_PAPER_LOG)
    except Exception:
        pass
    if _DEFAULT_PAPER_CKPT.is_file():
        return _DEFAULT_PAPER_CKPT.resolve()
    raise FileNotFoundError(
        "No locomotion checkpoint found. Pass --policy_ckpt PATH or train YuPaperStones first."
    )


def _load_walk_policy(ckpt_path: Path, device: torch.device) -> dict:
    from rsl_rl.modules.actor_critic_our import ActorCriticOur

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pmean = ck["obs_norm_state_dict"]["_mean"].detach().float().reshape(-1)
    n_obs = int(pmean.numel())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        policy = ActorCriticOur(
            num_actor_obs=n_obs,
            num_critic_obs=int(ck["privileged_obs_norm_state_dict"]["_mean"].numel()),
            num_actions=12,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
            noise_std_type="scalar",
            add_encoder=True,
            add_vae=True,
            num_latent=32,
            height_scan_latent_dim=32,
            height_scan_encoder_hidden_dims=[256, 128, 64],
            fusion_latent_dim=64,
            feet_height_latent_dim=16,
            use_feet_height_encoder=False,
        )
    policy.load_state_dict(ck["model_state_dict"], strict=False)
    policy.eval().to(device)
    normalizer = EmpiricalNormalization(shape=[n_obs], until=1.0e8).to(device)
    normalizer.load_state_dict(ck["obs_norm_state_dict"])
    pt = int(ck.get("obs_norm_passthrough_last", 0))
    if pt > 0:
        normalizer.set_passthrough_last(pt)
    normalizer.eval()
    return {
        "policy": policy,
        "normalizer": normalizer,
        "n_obs": n_obs,
        "path": ckpt_path,
        "iter": int(ck.get("iter", -1)),
    }


@torch.no_grad()
def _walk_actions(walker: dict, policy_obs: torch.Tensor) -> torch.Tensor:
    n_obs = int(walker["n_obs"])
    if policy_obs.shape[-1] < n_obs:
        raise RuntimeError(f"policy obs dim {policy_obs.shape[-1]} < walker n_obs {n_obs}")
    obs_n = walker["normalizer"](policy_obs[..., :n_obs])
    return walker["policy"].act_inference(obs_n)


@torch.no_grad()
def _hat_v(walker: dict | None, policy_obs: torch.Tensor) -> torch.Tensor:
    """Policy velocity_encoder hat_v (normalized obs, same as Actor)."""
    n = int(policy_obs.shape[0])
    if walker is None:
        return policy_obs.new_zeros(n, 3)
    n_obs = int(walker["n_obs"])
    obs_n = walker["normalizer"](policy_obs[..., :n_obs])
    return walker["policy"].actor.encode_base_vel(obs_n)


def _recon_proprio(model, policy_obs: torch.Tensor, walker: dict | None) -> torch.Tensor:
    in_dim = int(getattr(model, "proprio_dim", model.proprio_encoder[0].in_features))
    return recon_proprio(policy_obs, _hat_v(walker, policy_obs), in_dim)


def _pad_proprio_linear(model: LidarTerrainReconstructor, sd: dict) -> dict:
    """Copy old 45-d proprio Linear into 48-d; extra hat_v columns start at 0."""
    key = "proprio_encoder.0.weight"
    if key not in sd:
        return sd
    old = sd[key]
    new = model.state_dict()[key]
    if tuple(old.shape) == tuple(new.shape):
        return sd
    if old.ndim == 2 and new.ndim == 2 and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
        padded = new.detach().clone().zero_()
        padded[:, : old.shape[1]] = old.to(device=padded.device, dtype=padded.dtype)
        sd = dict(sd)
        sd[key] = padded
        print(
            f"[lidar-recon] padded {key} {tuple(old.shape)} → {tuple(new.shape)} (hat_v cols=0)",
            flush=True,
        )
    return sd


def _terrain_meta(env) -> tuple[object, int, int]:
    terrain = env.unwrapped.scene.terrain
    if getattr(terrain, "terrain_levels", None) is None:
        return None, 0, 0
    return terrain, int(terrain.max_terrain_level) - 1, int(terrain.terrain_origins.shape[1])


def _sync_env_origins(env) -> None:
    terrain = env.unwrapped.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        return
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.unwrapped.scene.env_origins[:] = terrain.env_origins


def _pin_terrain(env, row: int, col: int) -> None:
    terrain, max_row, num_cols = _terrain_meta(env)
    if terrain is None:
        return
    terrain.terrain_levels[:] = int(np.clip(row, 0, max_row))
    terrain.terrain_types[:] = int(np.clip(col, 0, num_cols - 1))
    _sync_env_origins(env)


def _lidar_root_pose(env):
    raw = env.unwrapped
    sensor = raw.scene.sensors["lidar"]
    robot = raw.scene["robot"]
    # Ray start in the base frame. Range is measured from here, not from the base origin.
    origin_b = sensor.ray_starts[:, 0, :]
    return sensor.data.pos_w, sensor.data.quat_w, robot.data.root_pos_w, robot.data.root_quat_w, origin_b


def _recon_forward(model, env, scan, mask, proprio, hidden):
    """Range-CNN path, stone gap splat, or stair max-z splat."""
    if not getattr(model, "use_hb_splat", False) and not getattr(model, "use_height_splat", False):
        return model(scan, mask, proprio, hidden)
    frame = int(model.n_ch * model.n_az)
    rng = scan[:, -frame:]
    m = mask[:, -frame:]
    pos, quat, root_p, root_q, origin_b = _lidar_root_pose(env)
    if getattr(model, "use_height_splat", False):
        return model.forward_height_splat(
            rng, m, pos, quat, root_p, root_q, proprio, hidden, origin_b
        )
    return model.forward_yaw_splat(rng, m, pos, quat, root_p, root_q, proprio, hidden)


def _lock_root_yaw_zero(env) -> None:
    """Keep XY, force identity yaw so body-frame H^b shows an axis-aligned lattice."""
    robot = env.unwrapped.scene["robot"]
    n = int(robot.data.root_pos_w.shape[0])
    pos = robot.data.root_pos_w.clone()
    quat = torch.zeros(n, 4, device=pos.device, dtype=pos.dtype)
    quat[:, 0] = 1.0
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    vel = torch.zeros(n, 6, device=pos.device, dtype=pos.dtype)
    robot.write_root_velocity_to_sim(vel)


def _nudge_stair_spawn(env, dx: float, env_ids: torch.Tensor | None = None) -> None:
    """Move stair-column envs to the platform edge. Flat envs stay put.

    Training used to reset on the 3 m platform, so the unobserved half of H^b
    was flat and the completion learned a constant. The preview stands on the
    stairs, where that constant is wrong.
    """
    raw = env.unwrapped
    terrain = raw.scene.terrain
    types = getattr(terrain, "terrain_types", None)
    if types is None:
        return
    robot = raw.scene["robot"]
    n = int(robot.data.root_pos_w.shape[0])
    stair = types.view(-1) > 0
    if env_ids is not None:
        keep = torch.zeros(n, dtype=torch.bool, device=stair.device)
        keep[env_ids.long().view(-1)] = True
        stair = stair & keep
    if not bool(stair.any()):
        return
    origin = raw.scene.env_origins
    pos = robot.data.root_pos_w.clone()
    quat = robot.data.root_quat_w.clone()
    pos[stair, 0] = origin[stair, 0] + float(dx)
    pos[stair, 1] = origin[stair, 1]
    q0 = torch.zeros_like(quat)
    q0[:, 0] = 1.0
    quat = torch.where(stair.unsqueeze(-1), q0, quat)
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    vel = robot.data.root_vel_w.clone()
    vel[stair] = 0
    robot.write_root_velocity_to_sim(vel)


def _place_root_on_tile(env, dx: float, dy: float = 0.0) -> None:
    """Put every env at tile center + (dx, dy), yaw 0, so H^b faces the stair edge."""
    raw = env.unwrapped
    robot = raw.scene["robot"]
    origin = raw.scene.env_origins
    pos = robot.data.root_pos_w.clone()
    pos[:, 0] = origin[:, 0] + float(dx)
    pos[:, 1] = origin[:, 1] + float(dy)
    quat = torch.zeros(pos.shape[0], 4, device=pos.device, dtype=pos.dtype)
    quat[:, 0] = 1.0
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    vel = torch.zeros(pos.shape[0], 6, device=pos.device, dtype=pos.dtype)
    robot.write_root_velocity_to_sim(vel)


def _mix_terrain(
    env,
    *,
    stones_only: bool,
    generator: torch.Generator,
    row_min: int,
    row_max: int,
    flat_copies: int = 1,
    stairs: bool = False,
) -> None:
    """Mix training cells.

    Stones: flat (optionally repeated) + stones rows ``[row_min, row_max]``.
    Stairs: equal weight of flat, up-stairs, and down-stairs. Stair drops are
    height, not voids — do not pair this with the stone void loss.
    """
    terrain, max_row, num_cols = _terrain_meta(env)
    if terrain is None:
        return
    n = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    r0 = int(np.clip(row_min, 0, max_row))
    r1 = max_row if row_max < 0 else int(np.clip(row_max, r0, max_row))
    cells: list[tuple[int, int]] = []
    if stairs:
        copies = max(1, int(flat_copies))
        for _ in range(copies):
            cells.append((0, 0))
        up_c = min(1, max(0, num_cols - 1))
        down_c = min(2, max(0, num_cols - 1))
        for r in range(r0, r1 + 1):
            cells.append((r, up_c))
            if down_c != up_c:
                cells.append((r, down_c))
    else:
        stone_c = min(1, num_cols - 1) if num_cols > 0 else 0
        if (not stones_only) and num_cols > 1:
            for _ in range(max(1, int(flat_copies))):
                cells.append((0, 0))  # flat column
        for r in range(r0, r1 + 1):
            cells.append((r, stone_c))
    if not cells:
        cells.append((r0, 0))
    pick = torch.randint(0, len(cells), (n,), device=device, generator=generator)
    levels = torch.zeros(n, dtype=torch.long, device=device)
    types = torch.zeros(n, dtype=torch.long, device=device)
    for i, (rr, cc) in enumerate(cells):
        m = pick == i
        if m.any():
            levels[m] = int(rr)
            types[m] = int(cc)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    _sync_env_origins(env)


_HB_EXTENT = (-0.5, 1.1, -0.4, 0.4)
# 3 m platform, edge at 1.5 m. H^b covers body x in [-0.5, 1.1].
# 1.7 m puts the back of the window on the platform and the front across several treads.
_STAIR_VIEW_X = 1.7


def _save_compare_rows(path: Path, rows: list[dict], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[lidar-recon] skip figure ({exc})", flush=True)
        return
    fig, axes = plt.subplots(len(rows), 6, figsize=(18.0, max(3.2, 2.9 * len(rows))), squeeze=False)
    rmax = max(float(np.max(r["scan"])) for r in rows)
    if rmax <= 0.0:
        rmax = 1.0

    def _scan_view(img: np.ndarray, msk: np.ndarray) -> np.ndarray:
        hit = msk > 0.5
        if not np.any(hit):
            return img
        ch = np.where(hit.any(axis=1))[0]
        i0 = max(0, int(ch[0]) - 1)
        i1 = min(img.shape[0], int(ch[-1]) + 2)
        return img[i0:i1]

    for ri, row in enumerate(rows):
        resid = np.abs(row["refined"] - row["gt"])
        label = row.get("label", "")
        prefix = f"{label}  " if label else ""
        scan_v = _scan_view(row["scan"], row["mask"])
        mask_v = _scan_view(row["mask"], row["mask"])
        hb_maps = np.stack([row["gt"], row["rough"], row["refined"]])
        vmin, vmax = float(hb_maps.min()), float(hb_maps.max())
        # A flat platform spans ~0. Stretching it paints sensor noise as fake steps.
        if vmax <= vmin + 0.02:
            mid = 0.5 * (vmin + vmax)
            vmin, vmax = mid - 0.05, mid + 0.05
        span = float(row["gt"].max() - row["gt"].min())
        panels = [
            (scan_v, f"{prefix}range", "turbo", (0.0, rmax), "auto"),
            (mask_v, f"{prefix}valid", "gray", (0.0, 1.0), "auto"),
            (row["gt"], f"{prefix}GT H^b  Δz={span:.2f}", "terrain", (vmin, vmax), "equal"),
            (row["rough"], f"{prefix}H_rough  L1={row['l1_rough']:.3f}", "terrain", (vmin, vmax), "equal"),
            (row["refined"], f"{prefix}H_refined  L1={row['l1_ref']:.3f}", "terrain", (vmin, vmax), "equal"),
            (resid, f"{prefix}|ref−GT|", "magma", (0.0, max(float(resid.max()), 1e-3)), "equal"),
        ]
        for ax, (img, title_i, cmap, scale, aspect) in zip(axes[ri], panels):
            is_hb = aspect == "equal"
            im = ax.imshow(
                img,
                cmap=cmap,
                vmin=scale[0],
                vmax=scale[1],
                origin="lower" if is_hb else "upper",
                aspect=aspect,
                interpolation="nearest",
                extent=_HB_EXTENT if is_hb else None,
            )
            ax.set_title(title_i, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[lidar-recon] wrote {path}", flush=True)


@torch.no_grad()
def _snapshot(
    model,
    env,
    row: int,
    col: int,
    label: str,
    *,
    walker,
    rng,
    warm_steps: int = 80,
    lock_yaw: bool = True,
    stair_view: bool = False,
) -> dict:
    raw = env.unwrapped
    _pin_terrain(env, row, col)
    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    n_env = int(raw.num_envs)
    act_dim = int(env.action_space.shape[-1])
    # Stair columns: stand near the platform edge facing +x. Per-step yaw lock
    # also zeros velocity, which left the robot on the flat center and H^b blank.
    on_stairs = bool(stair_view) and int(col) != 0
    if on_stairs:
        _place_root_on_tile(env, _STAIR_VIEW_X, 0.0)
    model.reset_temporal_state()
    delay_k = int(getattr(model, "gru_delay_steps", 0))
    warm_steps = max(int(warm_steps), delay_k + 5)
    hidden = None
    for _ in range(int(warm_steps)):
        if lock_yaw and not on_stairs:
            _lock_root_yaw_zero(env)
        if walker is not None:
            actions = _walk_actions(walker, policy_obs)
        else:
            actions = 0.15 * torch.randn(n_env, act_dim, device=raw.device, generator=rng)
        obs, _, _, _, _ = env.step(actions)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        hb, scan, mask = split_policy_lidar(policy_obs)
        assert scan is not None and mask is not None
        proprio = _recon_proprio(model, policy_obs, walker)
        h_rough, h_refined, hidden = _recon_forward(model, env, scan, mask, proprio, hidden)
        hidden = hidden.detach()
    if lock_yaw and not on_stairs:
        _lock_root_yaw_zero(env)
    if walker is not None:
        actions = _walk_actions(walker, policy_obs)
    else:
        actions = 0.15 * torch.randn(n_env, act_dim, device=raw.device, generator=rng)
    obs, _, _, _, _ = env.step(actions)
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    hb, scan, mask = split_policy_lidar(policy_obs)
    assert scan is not None and mask is not None
    proprio = _recon_proprio(model, policy_obs, walker)
    h_rough, h_refined, _ = _recon_forward(model, env, scan, mask, proprio, hidden)
    gt = LidarTerrainReconstructor.reshape_hb(hb[0:1]).cpu().numpy()[0]
    hit = getattr(model, "_last_hit", None)
    if hit is not None:
        w = hit[0:1]
        den = w.sum().clamp_min(1.0)
        l1_hit = float(((h_refined[0:1] - hb[0:1]).abs() * w).sum() / den)
    else:
        l1_hit = float("nan")
    return {
        "label": label,
        "gt": gt,
        "gt_span": float(gt.max() - gt.min()),
        "l1_hit": l1_hit,
        "scan": model.reshape_range(scan[0:1]).cpu().numpy()[0],
        "mask": model.reshape_range(mask[0:1]).cpu().numpy()[0],
        "rough": LidarTerrainReconstructor.reshape_hb(h_rough[0:1]).cpu().numpy()[0],
        "refined": LidarTerrainReconstructor.reshape_hb(h_refined[0:1]).cpu().numpy()[0],
        "l1_rough": float(torch.mean(torch.abs(h_rough - hb))),
        "l1_ref": float(torch.mean(torch.abs(h_refined - hb))),
    }


def _preview_cells(args, num_cols: int, max_row: int) -> list[tuple[int, int, str]]:
    """One panel per trained terrain: flat (optional) + stones L{row_min}..L{row_max}."""
    stone_c = min(1, num_cols - 1) if num_cols > 0 else 0
    if args.pin:
        return [(int(args.terrain_level), int(args.terrain_type), "pin")]
    cells: list[tuple[int, int, str]] = []
    if bool(getattr(args, "stairs", False)):
        cells.append((0, 0, "flat"))
        show_rows = (int(args.row_min), 4, max_row if int(args.row_max) < 0 else int(args.row_max))
        seen: set[int] = set()
        up_c = min(1, max(0, num_cols - 1))
        down_c = min(2, max(0, num_cols - 1))
        for r in show_rows:
            rr = int(np.clip(r, 0, max_row))
            if rr in seen:
                continue
            seen.add(rr)
            cells.append((rr, up_c, f"up L{rr}"))
            if down_c != up_c:
                cells.append((rr, down_c, f"down L{rr}"))
        return cells
    if num_cols > 1 and not args.stones_only:
        cells.append((0, 0, "flat"))
    r0 = max(0, int(args.row_min))
    r1 = max_row if int(args.row_max) < 0 else int(np.clip(int(args.row_max), r0, max_row))
    for r in range(r0, r1 + 1):
        cells.append((r, stone_c, f"L{r} stones"))
    return cells


@torch.no_grad()
def _rows_from_batch(
    model,
    hb: torch.Tensor,
    scan: torch.Tensor,
    mask: torch.Tensor,
    h_rough: torch.Tensor,
    h_refined: torch.Tensor,
    types: torch.Tensor,
    cells: list[tuple[int, int, str]],
) -> list[dict]:
    """One row per preview cell from the current training batch (no env reset)."""
    rows: list[dict] = []
    for _r, c, name in cells:
        hit = (types == int(c)).nonzero(as_tuple=False).view(-1)
        i = int(hit[0].item()) if hit.numel() > 0 else 0
        sl = slice(i, i + 1)
        rows.append(
            {
                "label": name,
                "gt": LidarTerrainReconstructor.reshape_hb(hb[sl]).cpu().numpy()[0],
                "scan": model.reshape_range(scan[sl]).cpu().numpy()[0],
                "mask": model.reshape_range(mask[sl]).cpu().numpy()[0],
                "rough": LidarTerrainReconstructor.reshape_hb(h_rough[sl]).cpu().numpy()[0],
                "refined": LidarTerrainReconstructor.reshape_hb(h_refined[sl]).cpu().numpy()[0],
                "l1_rough": float(torch.mean(torch.abs(h_rough[sl] - hb[sl]))),
                "l1_ref": float(torch.mean(torch.abs(h_refined[sl] - hb[sl]))),
            }
        )
    return rows


def _write_step_preview(
    *,
    model,
    hb: torch.Tensor,
    scan: torch.Tensor,
    mask: torch.Tensor,
    h_rough: torch.Tensor,
    h_refined: torch.Tensor,
    types: torch.Tensor,
    cells: list[tuple[int, int, str]],
    fig_dir: Path,
    step: int,
    stats: dict[str, float],
) -> None:
    rows = _rows_from_batch(model, hb, scan, mask, h_rough, h_refined, types, cells)
    if not rows:
        return
    l1_f = float(np.mean([s["l1_ref"] for s in rows]))
    l1_r = float(np.mean([s["l1_rough"] for s in rows]))
    _save_compare_rows(
        fig_dir / f"lidar_recon_step_{step:05d}.png",
        rows,
        title=(
            f"step {step}  L1_rough={l1_r:.3f} L1_ref={l1_f:.3f}  "
            f"surf={stats.get('l1_surf', float('nan')):.3f} void={stats.get('l1_void', float('nan')):.3f}"
        ),
    )


def _edge_penalty(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-env mean |pred step − GT step|."""
    pred_m = pred.view(-1, 1, HB_NY, HB_NX)
    gt_m = gt.view(-1, 1, HB_NY, HB_NX)
    dx_p = pred_m[..., :, 1:] - pred_m[..., :, :-1]
    dx_g = gt_m[..., :, 1:] - gt_m[..., :, :-1]
    dy_p = pred_m[..., 1:, :] - pred_m[..., :-1, :]
    dy_g = gt_m[..., 1:, :] - gt_m[..., :-1, :]
    return (dx_p - dx_g).abs().flatten(1).mean(dim=-1) + (dy_p - dy_g).abs().flatten(1).mean(dim=-1)


def _continuous_height_loss(
    h_rough: torch.Tensor,
    h_refined: torch.Tensor,
    h_gt: torch.Tensor,
    hit: torch.Tensor | None = None,
    match_edges: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE(rough) + L1(refined). Stair drops stay height; no void class.

    When ``hit`` is set, both terms average only over cells with a lidar return.
    ``match_edges`` adds step matching.

    Each env's height loss is weighted by the square root of its GT height span
    (floor 5 cm). The step-matching term is not span-weighted, so a short riser
    is not discounted. Logged L1 stays an unweighted absolute error.
    Mixed-pixel bridges are removed in the range image before the splat.
    """
    if hit is None:
        span = (h_gt.amax(dim=-1) - h_gt.amin(dim=-1)).detach().clamp(min=0.05)
        # Square root, so a tall stair still counts more, but a short L0 step is
        # not an order of magnitude cheaper than L9.
        w = span.sqrt() / span.sqrt().mean().clamp_min(1e-6)
        mse_i = (h_rough - h_gt).square().mean(dim=-1)
        l1_i = (h_refined - h_gt).abs().mean(dim=-1)
        total_i = mse_i + l1_i
        edge_v = h_refined.new_zeros(())
        if match_edges:
            edge = _edge_penalty(h_refined, h_gt)
            # Edge match is not span-weighted: a 2 cm riser is only one cell pair.
            total = (w * total_i).mean() + 0.5 * edge.mean()
            edge_v = edge.detach().mean()
        else:
            total = (w * total_i).mean()
        with torch.no_grad():
            l1_rough = (h_rough - h_gt).abs().mean()
            l1_ref = (h_refined - h_gt).abs().mean()
        return total, {
            "loss": float(total.detach()),
            "l1_rough": float(l1_rough),
            "l1_plain": float(l1_ref),
            "l1_surf": float(l1_ref),
            "l1_void": 0.0,
            "hit_frac": 1.0,
            "edge": float(edge_v),
            "pred_mean": float(h_refined.detach().mean()),
            "span_max": float(span.max()),
        }
    w = hit.detach()
    if w.shape != h_gt.shape:
        w = w.reshape_as(h_gt)
    den = w.sum().clamp_min(1.0)
    mse = ((h_rough - h_gt).square() * w).sum() / den
    l1 = ((h_refined - h_gt).abs() * w).sum() / den
    total = mse + l1
    with torch.no_grad():
        l1_rough = ((h_rough - h_gt).abs() * w).sum() / den
        l1_ref = ((h_refined - h_gt).abs() * w).sum() / den
    return total, {
        "loss": float(total.detach()),
        "l1_rough": float(l1_rough),
        "l1_plain": float(l1_ref),
        "l1_surf": float(l1_ref),
        "l1_void": 0.0,
        "hit_frac": float(w.mean()),
        "pred_mean": float((h_refined.detach() * w).sum() / den),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if bool(args_cli.stairs):
        from CtrlLab.tasks.manager_based.go2_task.robot_env_cfg import _configure_yu_lidar

        env_cfg.scene.depth_camera = None
        env_cfg.observations.policy.depth_camera = None
        # Match RoughYu GT height scan: low treads are surfaces, not gaps.
        _configure_yu_lidar(env_cfg, void_hit_z_max=-2.0)
        print("[lidar-recon] stairs: continuous height loss, void_hit_z_max=-2", flush=True)
        if bool(args_cli.height_splat):
            print(
                "[lidar-recon] height splat: median of recent upper-tread splats, nearest-hit hole fill",
                flush=True,
            )
            print(
                "[lidar-recon] cell band=2cm under max; span weight is sqrt; edge term unweighted",
                flush=True,
            )
            print(
                "[lidar-recon] hits stay on the splat; only empty cells get ±8 cm; refined 3x3 median",
                flush=True,
            )
    walker: dict | None = None
    if not args_cli.random_actions:
        v = float(args_cli.lin_vel_x)
        wz = (-1.2, 1.2) if bool(args_cli.unlock_yaw) else (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = wz
        if hasattr(env_cfg.commands.base_velocity, "yaw_while_moving"):
            env_cfg.commands.base_velocity.yaw_while_moving = bool(args_cli.unlock_yaw)
        if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
            env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (v, v)
            env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
            env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = wz

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    device = raw.device
    terrain, max_row, num_cols = _terrain_meta(env)
    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    if not args_cli.random_actions:
        ckpt = _resolve_walk_ckpt(args_cli.policy_ckpt)
        walker = _load_walk_policy(ckpt, device)
        print(f"[lidar-recon] walk policy={walker['path']} iter={walker['iter']}", flush=True)
    else:
        print("[lidar-recon] drive=RANDOM jitter", flush=True)

    if args_cli.pin:
        _pin_terrain(env, args_cli.terrain_level, args_cli.terrain_type)
        print("[lidar-recon] mode=PINNED", flush=True)
    else:
        _mix_terrain(
            env,
            stones_only=args_cli.stones_only,
            generator=rng,
            row_min=args_cli.row_min,
            row_max=args_cli.row_max,
            flat_copies=int(args_cli.flat_copies),
            stairs=bool(args_cli.stairs),
        )
        print(
            f"[lidar-recon] mode={'STAIRS' if args_cli.stairs else 'MIX'} "
            f"rows={args_cli.row_min}..{args_cli.row_max if args_cli.row_max >= 0 else max_row}  "
            f"flat_copies={int(args_cli.flat_copies)}",
            flush=True,
        )

    obs, _ = env.reset()
    if bool(args_cli.stairs) and bool(args_cli.height_splat):
        _nudge_stair_spawn(env, _STAIR_VIEW_X)
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    hb0, scan0, mask0 = split_policy_lidar(policy_obs)
    if scan0 is None or mask0 is None:
        raise RuntimeError(f"no lidar tail on policy obs (dim={policy_obs.shape[-1]})")
    if hb0.shape[-1] != HB_DIM:
        raise RuntimeError(f"expected H^b dim {HB_DIM}, got {hb0.shape[-1]}")
    if int(scan0.shape[-1]) != PAPER_LIDAR_RANGE_DIM:
        raise RuntimeError(
            f"lidar range dim {scan0.shape[-1]} != {PAPER_LIDAR_RANGE_DIM} "
            f"(hist {PAPER_LIDAR_FRAMES - 1}×{PAPER_LIDAR_CHANNELS}×{PAPER_LIDAR_N_AZ} "
            f"+ now {PAPER_LIDAR_CHANNELS}×{PAPER_LIDAR_N_AZ})"
        )

    step_dt = float(getattr(raw, "step_dt", 0.02))
    if bool(args_cli.no_gru_delay_align):
        gru_delay_steps = 0
        gru_delay_s = 0.0
    else:
        gru_delay_s = float(PAPER_LIDAR_DELAY_S if args_cli.gru_delay_s is None else args_cli.gru_delay_s)
        gru_delay_steps = 0 if gru_delay_s <= 0.0 else max(1, int(round(gru_delay_s / max(step_dt, 1e-6))))

    model = LidarTerrainReconstructor(
        n_ch=PAPER_LIDAR_CHANNELS,
        n_az=PAPER_LIDAR_N_AZ,
        n_frames=PAPER_LIDAR_FRAMES,
        proprio_dim=RECON_PROPRIO_DIM,
        use_refine=not bool(args_cli.no_refine),
        void_weight=float(args_cli.void_weight) if args_cli.void_weight is not None else 3.0,
        gru_delay_steps=gru_delay_steps,
        use_hb_splat=bool(args_cli.hb_splat),
        use_height_splat=bool(args_cli.height_splat),
    ).to(device)
    print(
        f"[lidar-recon] proprio_dim={RECON_PROPRIO_DIM} (cur obs + policy hat_v)  "
        f"preview {'yaw-open' if args_cli.unlock_yaw else 'yaw-locked'}  "
        f"cmd ωz={'±1.2' if args_cli.unlock_yaw else '0'}  "
        f"hb_splat={bool(args_cli.hb_splat)}  height_splat={bool(args_cli.height_splat)}  "
        f"GRU Δ-align steps={gru_delay_steps} ({gru_delay_s:g}s @ dt={step_dt:g})",
        flush=True,
    )
    if args_cli.init_ckpt is not None:
        init_path = args_cli.init_ckpt.expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f"--init_ckpt not found: {init_path}")
        blob = torch.load(init_path, map_location=device, weights_only=False)
        sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
        sd = _pad_proprio_linear(model, sd)
        incompat = model.load_state_dict(sd, strict=False)
        print(f"[lidar-recon] warm-start from {init_path}", flush=True)
        if hasattr(incompat, "missing_keys") and (incompat.missing_keys or incompat.unexpected_keys):
            print(
                f"[lidar-recon] init missing={len(incompat.missing_keys)} "
                f"unexpected={len(incompat.unexpected_keys)}",
                flush=True,
            )
    if bool(args_cli.freeze_backbone):
        if getattr(model, "edge_clean", None) is None:
            raise RuntimeError("--freeze_backbone needs --height_splat so edge_clean exists")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("edge_clean.")
        n_train = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"[lidar-recon] backbone frozen, training edge_clean ({n_train} params)", flush=True)
    opt = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=float(args_cli.lr))
    hidden = None
    n_env = int(raw.num_envs)
    act_dim = int(env.action_space.shape[-1])
    log_every = max(50, int(args_cli.steps) // 20)
    fig_every = int(args_cli.fig_every)
    fig_dir = args_cli.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells = _preview_cells(args_cli, num_cols, max_row)
    print(f"[lidar-recon] preview cells={[n for _, _, n in cells]}", flush=True)

    for t in range(int(args_cli.steps)):
        if (not args_cli.pin) and args_cli.mix_every > 0 and t > 0 and t % int(args_cli.mix_every) == 0:
            _mix_terrain(
                env,
                stones_only=args_cli.stones_only,
                generator=rng,
                row_min=args_cli.row_min,
                row_max=args_cli.row_max,
                flat_copies=int(args_cli.flat_copies),
                stairs=bool(args_cli.stairs),
            )
            obs, _ = env.reset()
            if bool(args_cli.stairs) and bool(args_cli.height_splat):
                _nudge_stair_spawn(env, _STAIR_VIEW_X)
            hidden = None
            model.reset_temporal_state()
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs

        if walker is not None:
            actions = _walk_actions(walker, policy_obs)
        else:
            actions = 0.15 * torch.randn(n_env, act_dim, device=device, generator=rng)
        obs, _, _, _, _ = env.step(actions)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        hb, scan, mask = split_policy_lidar(policy_obs)
        assert scan is not None and mask is not None
        proprio = _recon_proprio(model, policy_obs, walker)

        if hasattr(raw, "episode_length_buf"):
            done = raw.episode_length_buf <= 1
            if done.any():
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if bool(args_cli.stairs) and bool(args_cli.height_splat):
                    _nudge_stair_spawn(env, _STAIR_VIEW_X, done_ids)
                model.reset_temporal_state(done_ids)
                if hidden is not None:
                    hidden = hidden.clone()
                    hidden[:, done] = 0.0

        h_rough, h_refined, hidden = _recon_forward(model, env, scan, mask, proprio, hidden)
        if bool(args_cli.stairs):
            loss, stats = _continuous_height_loss(
                h_rough,
                h_refined,
                hb.detach(),
                match_edges=bool(args_cli.height_splat),
            )
        else:
            loss, stats = model.loss(h_rough, h_refined, hb.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hidden = hidden.detach()

        last = t == int(args_cli.steps) - 1
        if t % log_every == 0 or last:
            print(
                f"[lidar-recon] step={t:05d}  loss={stats['loss']:.4f}  "
                f"l1_rough={stats['l1_rough']:.4f}  l1_ref={stats['l1_plain']:.4f}  "
                f"l1_surf={stats.get('l1_surf', float('nan')):.4f}  "
                f"l1_void={stats.get('l1_void', float('nan')):.4f}  "
                f"hit={stats.get('hit_frac', float('nan')):.2f}  "
                f"edge={stats.get('edge', float('nan')):.4f}  "
                f"pred_mean={stats.get('pred_mean', float('nan')):.3f}",
                flush=True,
            )
        if fig_every > 0 and (t % fig_every == 0 or last):
            # Full per-terrain snapshots so every trained cell appears (not just batch types).
            rows_viz = []
            for r, c, name in cells:
                snap = _snapshot(
                    model,
                    env,
                    r,
                    c,
                    name,
                    walker=walker,
                    rng=rng,
                    lock_yaw=not bool(args_cli.unlock_yaw),
                    stair_view=bool(args_cli.stairs),
                )
                rows_viz.append(snap)
                print(
                    f"[lidar-recon] fig@{t} {name}: L1_rough={snap['l1_rough']:.4f} "
                    f"L1_ref={snap['l1_ref']:.4f}  L1_hit={snap['l1_hit']:.4f}  gt_span={snap['gt_span']:.3f}",
                    flush=True,
                )
            l1_f = float(np.mean([s["l1_ref"] for s in rows_viz]))
            l1_r = float(np.mean([s["l1_rough"] for s in rows_viz]))
            _save_compare_rows(
                fig_dir / f"lidar_recon_step_{t:05d}.png",
                rows_viz,
                title=(
                    f"step {t}  L1_rough={l1_r:.3f} L1_ref={l1_f:.3f}  "
                    f"surf={stats.get('l1_surf', float('nan')):.3f} "
                    f"void={stats.get('l1_void', float('nan')):.3f}"
                ),
            )
            # Snapshots pin cells — restore MIX and clear GRU for continued training.
            if not args_cli.pin:
                _mix_terrain(
                    env,
                    stones_only=args_cli.stones_only,
                    generator=rng,
                    row_min=args_cli.row_min,
                    row_max=args_cli.row_max,
                    flat_copies=int(args_cli.flat_copies),
                    stairs=bool(args_cli.stairs),
                )
                obs, _ = env.reset()
                if bool(args_cli.stairs) and bool(args_cli.height_splat):
                    _nudge_stair_spawn(env, _STAIR_VIEW_X)
                hidden = None
                model.reset_temporal_state()
                policy_obs = obs["policy"] if isinstance(obs, dict) else obs

    out_dir = args_cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "lidar_reconstructor_phase1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": int(args_cli.steps),
            "task": args_cli.task,
            "n_ch": PAPER_LIDAR_CHANNELS,
            "n_az": PAPER_LIDAR_N_AZ,
            "n_az_full": PAPER_LIDAR_N_AZ,
            "az_stride": PAPER_LIDAR_AZ_STRIDE,
            "n_frames": PAPER_LIDAR_FRAMES,
            "proprio_dim": int(getattr(model, "proprio_dim", RECON_PROPRIO_DIM)),
            "gru_delay_steps": int(getattr(model, "gru_delay_steps", 0)),
            "gru_delay_s": float(gru_delay_s),
            "use_hb_splat": bool(getattr(model, "use_hb_splat", False)),
            "use_height_splat": bool(getattr(model, "use_height_splat", False)),
            "surface_loss": bool(args_cli.stairs),
            "delay_s": float(PAPER_LIDAR_DELAY_S),
        },
        ckpt_path,
    )
    print(f"[lidar-recon] saved {ckpt_path}", flush=True)

    rows_viz = []
    for r, c, name in cells:
        snap = _snapshot(
            model,
            env,
            r,
            c,
            name,
            walker=walker,
            rng=rng,
            lock_yaw=not bool(args_cli.unlock_yaw),
            stair_view=bool(args_cli.stairs),
        )
        rows_viz.append(snap)
        print(
            f"[lidar-recon] preview {name}: L1_rough={snap['l1_rough']:.4f} "
            f"L1_ref={snap['l1_ref']:.4f}  L1_hit={snap['l1_hit']:.4f}  gt_span={snap['gt_span']:.3f}",
            flush=True,
        )

    l1_f = float(np.mean([s["l1_ref"] for s in rows_viz]))
    l1_r = float(np.mean([s["l1_rough"] for s in rows_viz]))
    _save_compare_rows(
        args_cli.fig_dir / "lidar_recon_gt_vs_pred.png",
        rows_viz,
        title=f"Lidar [t-{PAPER_LIDAR_DELAY_S:g}s,t]→H^b  mean L1_rough={l1_r:.3f} L1_ref={l1_f:.3f}",
    )
    l1_zero = float(np.mean([np.abs(s["gt"]).mean() for s in rows_viz]))
    print(f"[lidar-recon] L1_rough={l1_r:.4f} L1_ref={l1_f:.4f}  L1(0,GT)={l1_zero:.4f}", flush=True)
    if l1_f < l1_zero:
        print("[PASS] refined closer to GT than zero map", flush=True)
    else:
        print("[WARN] not better than zero yet — train longer", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
