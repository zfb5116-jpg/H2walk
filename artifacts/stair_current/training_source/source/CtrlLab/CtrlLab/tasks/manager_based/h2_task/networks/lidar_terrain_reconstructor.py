# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lidar terrain reconstructor: spherical range stack → paper H^b.

Input is subsampled frames ``[scan(t−delay), …, scan(t)]`` (channels × azimuth),
not a 5 cm crop. A spatial decoder maps the scan onto 17×33.

GRU temporal alignment (default): each control step encodes a token
``[pooled_cnn ∥ z_proprio]``, rings it for ``gru_delay_steps`` (≈ lidar Δ),
then runs the GRU on the length-2 sequence ``[token(t−Δ), token(t)]`` so the
recurrent history beat matches the fixed lidar delay—not every 0.02 s alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .terrain_reconstructor import HB_NX, HB_NY, _RefineUNet


def spherical_ray_dirs(
    n_ch: int,
    n_az: int,
    vfov: tuple[float, float] = (-45.0, 14.0),
    hfov: tuple[float, float] = (-180.0, 180.0),
) -> torch.Tensor:
    """Sensor-frame unit directions ``(3, n_ch, n_az)`` matching Isaac Lab lidar grid."""
    wrap = abs(abs(hfov[0] - hfov[1]) - 360.0) < 1e-6
    n_lin = int(n_az) + 1 if wrap else int(n_az)
    va = torch.deg2rad(torch.linspace(vfov[0], vfov[1], int(n_ch)))
    ha = torch.deg2rad(torch.linspace(hfov[0], hfov[1], n_lin))
    if wrap:
        ha = ha[:-1]
    if int(ha.numel()) != int(n_az):
        ha = torch.deg2rad(torch.linspace(hfov[0], hfov[1], int(n_az), endpoint=not wrap))
    va, ha = torch.meshgrid(va, ha, indexing="ij")
    dx = torch.cos(va) * torch.cos(ha)
    dy = torch.cos(va) * torch.sin(ha)
    dz = torch.sin(va)
    return torch.stack([dx, dy, dz], dim=0)


def _circ_pad_az(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Circular pad on azimuth (last dim) for a 360° scan."""
    if pad <= 0:
        return x
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


def _isolated_between_jumps(
    r: torch.Tensor,
    valid: torch.Tensor,
    jump: float,
    max_run: int,
) -> torch.Tensor:
    """Drop valid runs of length ≤ ``max_run`` that sit between two range jumps.

    A mixed pixel is a one- or two-beam bridge between two surfaces (Tuley 2005;
    Ye & Borenstein). Long surfaces, including a flat ground whose channel-to-channel
    range change stays under ``jump``, are kept.
    """
    n, length = r.shape
    if length < 3:
        return torch.zeros_like(valid)
    diff = (r[:, 1:] - r[:, :-1]).abs()
    jump_after = valid[:, 1:] & valid[:, :-1] & (diff > jump)
    start = torch.ones(n, length, dtype=torch.bool, device=r.device)
    start[:, 1:] = (~valid[:, 1:]) | (~valid[:, :-1]) | jump_after
    seg = start.long().cumsum(dim=1) - 1
    nseg = int(seg[:, -1].max().item()) + 1
    counts = r.new_zeros(n, nseg)
    counts.scatter_add_(1, seg, valid.to(dtype=r.dtype))
    length_at = counts.gather(1, seg)
    left = r.new_zeros(n, nseg)
    left_seed = (start & torch.cat([jump_after.new_zeros(n, 1), jump_after], dim=1)).to(dtype=r.dtype)
    left.scatter_reduce_(1, seg, left_seed, reduce="amax", include_self=True)
    right = r.new_zeros(n, nseg)
    end = torch.cat([start[:, 1:], torch.ones(n, 1, dtype=torch.bool, device=r.device)], dim=1)
    right_seed = (end & torch.cat([jump_after, jump_after.new_zeros(n, 1)], dim=1)).to(dtype=r.dtype)
    right.scatter_reduce_(1, seg, right_seed, reduce="amax", include_self=True)
    short = valid & (length_at <= float(max_run)) & (left.gather(1, seg) > 0.5) & (right.gather(1, seg) > 0.5)
    return short


def _suppress_mixed_pixels(
    rng: torch.Tensor,
    mask: torch.Tensor,
    n_ch: int,
    n_az: int,
    *,
    jump_ch: float = 0.22,
    jump_az: float = 0.15,
    max_run: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero returns that bridge a range discontinuity, before they are splatted."""
    frame = int(n_ch) * int(n_az)
    if rng.ndim != 2 or int(rng.shape[-1]) != frame:
        return rng, mask
    img = rng.reshape(-1, int(n_ch), int(n_az))
    valid = mask.reshape(-1, int(n_ch), int(n_az)) > 0.5
    # Elevation: one column is a channel line. Azimuth: one row is an azimuth line.
    ch_r = img.permute(0, 2, 1).reshape(-1, int(n_ch))
    ch_v = valid.permute(0, 2, 1).reshape(-1, int(n_ch))
    drop_ch = _isolated_between_jumps(ch_r, ch_v, jump_ch, max_run)
    drop_ch = drop_ch.reshape(img.shape[0], int(n_az), int(n_ch)).permute(0, 2, 1)
    az_r = img.reshape(-1, int(n_az))
    az_v = valid.reshape(-1, int(n_az))
    drop_az = _isolated_between_jumps(az_r, az_v, jump_az, max_run)
    drop_az = drop_az.reshape(img.shape)
    keep = valid & ~drop_ch & ~drop_az
    out_mask = keep.reshape(rng.shape).to(dtype=mask.dtype)
    out_rng = torch.where(keep.reshape(rng.shape), rng, torch.zeros_like(rng))
    return out_rng, out_mask


def _median_filter(h: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Edge-preserving denoise. A one-cell spike disappears; a tread wider than ``k`` stays."""
    p = int(k) // 2
    x = F.pad(h, (p, p, p, p), mode="reflect")
    patches = x.unfold(2, int(k), 1).unfold(3, int(k), 1)
    return patches.reshape(*patches.shape[:4], int(k) * int(k)).median(dim=-1).values


def _nearest_fill(h: torch.Tensor, hit: torch.Tensor, iters: int = 16) -> torch.Tensor:
    """Spread measured heights into empty cells, one 5 cm ring at a time."""
    filled = h
    known = hit > 0.5
    for _ in range(int(iters)):
        weight = known.float()
        num = F.avg_pool2d(F.pad(filled * weight, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        den = F.avg_pool2d(F.pad(weight, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        take = (~known) & (den > 1.0e-6)
        filled = torch.where(take, num / den.clamp_min(1.0e-6), filled)
        known = known | take
    return filled


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply wxyz quaternion(s) to vectors. Broadcasts on the leading dims."""
    xyz = quat[..., 1:]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    conj = quat.clone()
    conj[..., 1:] = -conj[..., 1:]
    return _quat_apply(conj, vec)


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only wxyz quaternion (rotation about world Z)."""
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = torch.zeros_like(quat)
    out[..., 0] = half.cos()
    out[..., 3] = half.sin()
    return out


class _RangeEncoder(nn.Module):
    """CNN on ``(B, C, n_ch, n_az)`` → spatial feature map (no global pool)."""

    def __init__(self, in_ch: int = 5, out_ch: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=(1, 2), padding=(1, 0)),
            nn.ELU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _circ_pad_az(x, 1)
        return self.net(self.stem(x))


class _EdgeClean(nn.Module):
    """Small map head. Zero-init so it starts as the identity, then trims speckle and steps."""

    def __init__(self, span: float = 0.15):
        super().__init__()
        self.span = float(span)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode="reflect"),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h + self.span * torch.tanh(self.net(h))).clamp(-2.0, 2.0)


class _RoughHead(nn.Module):
    """Fuse spatial scan features + GRU style → ``(B, 1, 17, 33)``."""

    def __init__(self, feat_ch: int, style_ch: int, hb_ny: int, hb_nx: int):
        super().__init__()
        self.hb_ny = int(hb_ny)
        self.hb_nx = int(hb_nx)
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch + style_ch, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        feat = F.interpolate(feat, size=(self.hb_ny, self.hb_nx), mode="bilinear", align_corners=False)
        style = style.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hb_ny, self.hb_nx)
        return self.net(torch.cat([feat, style], dim=1))


class LidarTerrainReconstructor(nn.Module):
    """CNN(range, mask, ray_dir) → spatial 17×33 + GRU → Rough → U-Net Refine."""

    def __init__(
        self,
        in_channels: int = 2,
        n_ch: int = 16,
        n_az: int = 224,
        n_frames: int = 2,
        max_range: float = 2.0,
        vfov: tuple[float, float] = (-45.0, 14.0),
        hfov: tuple[float, float] = (-180.0, 180.0),
        map_feat_dim: int = 128,
        style_ch: int = 32,
        proprio_dim: int = 48,
        z_prop_dim: int = 64,
        gru_hidden_dim: int = 256,
        hb_ny: int = HB_NY,
        hb_nx: int = HB_NX,
        use_refine: bool = True,
        void_weight: float = 3.0,
        miss_fill: float = 1.0,
        # Match GRU history to lidar delay: 0 disables (old per-step single token).
        gru_delay_steps: int = 75,
        # Unproject range into the yaw-aligned H^b grid before the decoder so a
        # rotated lattice stays rotated (range-CNN upsample cannot).
        use_hb_splat: bool = False,
        # Stairs: max-z splat only. No min/gap blend, so a tread drop stays a step.
        use_height_splat: bool = False,
        hb_res: float = 0.05,
        hb_offset_xy: tuple[float, float] = (0.3, 0.0),
        hb_size: tuple[float, float] = (1.6, 0.8),
        **_ignored,
    ):
        super().__init__()
        self.n_ch = int(n_ch)
        self.n_az = int(n_az)
        self.n_frames = int(n_frames)
        self.max_range = float(max_range)
        self.hb_ny = int(hb_ny)
        self.hb_nx = int(hb_nx)
        self.hb_dim = self.hb_ny * self.hb_nx
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.use_refine = bool(use_refine)
        self.void_weight = float(void_weight)
        self.miss_fill = float(miss_fill)
        self.style_ch = int(style_ch)
        self.proprio_dim = int(proprio_dim)
        self.map_feat_dim = int(map_feat_dim)
        self.z_prop_dim = int(z_prop_dim)
        self.token_dim = int(map_feat_dim) + int(z_prop_dim)
        self.gru_delay_steps = max(0, int(gru_delay_steps))
        # (range+mask) × frames + ray xyz
        self.in_channels = 2 * self.n_frames + 3

        self.register_buffer(
            "ray_dirs",
            spherical_ray_dirs(self.n_ch, self.n_az, vfov, hfov),
            persistent=True,
        )
        self.map_cnn = _RangeEncoder(in_ch=self.in_channels, out_ch=int(map_feat_dim))
        self.proprio_encoder = nn.Sequential(
            nn.Linear(int(proprio_dim), 128),
            nn.ELU(inplace=True),
            nn.Linear(128, z_prop_dim),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=self.token_dim,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
        )
        self.gru_to_style = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, self.style_ch),
            nn.ELU(inplace=True),
        )
        self.rough_head = _RoughHead(int(map_feat_dim), self.style_ch, self.hb_ny, self.hb_nx)
        self.refine = _RefineUNet(base_ch=32) if self.use_refine else None
        self.use_hb_splat = bool(use_hb_splat)
        self.use_height_splat = bool(use_height_splat)
        if self.use_hb_splat and self.use_height_splat:
            raise ValueError("use_hb_splat and use_height_splat are mutually exclusive")
        self.hb_res = float(hb_res)
        self.hb_x0 = -0.5 * float(hb_size[0]) + float(hb_offset_xy[0])
        self.hb_y0 = -0.5 * float(hb_size[1]) + float(hb_offset_xy[1])
        if self.use_hb_splat:
            # Per frame: h_max, h_min, hit, gap(=max-min) × {now, lag}
            self.hb_enc = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ELU(inplace=True),
            )
            self.hb_delta = nn.Conv2d(32 + self.style_ch, 1, kernel_size=3, padding=1)
            nn.init.zeros_(self.hb_delta.weight)
            nn.init.zeros_(self.hb_delta.bias)
            if self.refine is not None:
                nn.init.zeros_(self.refine.dec[-1].weight)
                nn.init.zeros_(self.refine.dec[-1].bias)
            self.hb_token_dim = 32 + int(z_prop_dim)
            self.hb_gru = nn.GRU(self.hb_token_dim, self.gru_hidden_dim, batch_first=True)
            self.hb_gru_to_style = nn.Sequential(
                nn.Linear(self.gru_hidden_dim, self.style_ch),
                nn.ELU(inplace=True),
            )
        if self.use_height_splat:
            # Per frame: max height, hit. Reflect pad so the map border is not a zero-pad frame.
            self.height_enc = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=3, padding=1, padding_mode="reflect"),
                nn.ELU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="reflect"),
                nn.ELU(inplace=True),
            )
            ys = torch.linspace(-1.0, 1.0, self.hb_ny)
            xs = torch.linspace(-1.0, 1.0, self.hb_nx)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self.register_buffer("hb_xy", torch.stack([gx, gy], dim=0).unsqueeze(0), persistent=False)
            self.height_delta = nn.Conv2d(
                32 + self.style_ch, 1, kernel_size=3, padding=1, padding_mode="reflect"
            )
            nn.init.zeros_(self.height_delta.weight)
            nn.init.zeros_(self.height_delta.bias)
            if self.refine is not None:
                nn.init.zeros_(self.refine.dec[-1].weight)
                nn.init.zeros_(self.refine.dec[-1].bias)
            self.height_token_dim = 32 + int(z_prop_dim)
            self.height_gru = nn.GRU(self.height_token_dim, self.gru_hidden_dim, batch_first=True)
            self.height_gru_to_style = nn.Sequential(
                nn.Linear(self.gru_hidden_dim, self.style_ch),
                nn.ELU(inplace=True),
            )
            self.edge_clean = _EdgeClean()
        else:
            self.edge_clean = None
        # Ring of encoded tokens for Δ-aligned GRU (not in state_dict).
        self._tok_buf: torch.Tensor | None = None
        self._tok_ptr: torch.Tensor | None = None
        self._tok_fill: torch.Tensor | None = None
        self._pose_rng: torch.Tensor | None = None
        self._pose_mask: torch.Tensor | None = None
        self._pose_pos: torch.Tensor | None = None
        self._pose_quat: torch.Tensor | None = None
        self._pose_ptr: torch.Tensor | None = None
        self._pose_fill: torch.Tensor | None = None

    def _as_range_image(self, rng: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        frame = self.n_ch * self.n_az
        n_fr = self.n_frames
        if rng.ndim == 2:
            if rng.shape[-1] == frame * n_fr:
                rng = rng.view(-1, n_fr, self.n_ch, self.n_az)
            else:
                rng = rng.view(-1, 1, self.n_ch, self.n_az).expand(-1, n_fr, -1, -1).clone()
        elif rng.ndim == 3:
            rng = rng.unsqueeze(1).expand(-1, n_fr, -1, -1).clone()
        elif rng.ndim == 4 and rng.shape[1] == n_fr:
            pass
        else:
            raise ValueError(f"unexpected range shape {tuple(rng.shape)}")
        if mask.ndim == 2:
            if mask.shape[-1] == frame * n_fr:
                mask = mask.view(-1, n_fr, self.n_ch, self.n_az)
            else:
                mask = mask.view(-1, 1, self.n_ch, self.n_az).expand(-1, n_fr, -1, -1).clone()
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1).expand(-1, n_fr, -1, -1).clone()
        elif mask.ndim == 4 and mask.shape[1] == n_fr:
            pass
        else:
            raise ValueError(f"unexpected mask shape {tuple(mask.shape)}")
        rng = (rng / max(self.max_range, 1.0e-6)).clamp(0.0, 1.0)
        miss = mask < 0.5
        rng = torch.where(miss, torch.full_like(rng, self.miss_fill), rng)
        dirs = self.ray_dirs.to(device=rng.device, dtype=rng.dtype).unsqueeze(0).expand(rng.shape[0], -1, -1, -1)
        return torch.cat([rng, mask, dirs], dim=1)

    def reset_temporal_state(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear Δ token ring and yaw-splat pose ring (episode / terrain reset)."""
        self._reset_ring(self._tok_buf, self._tok_ptr, self._tok_fill, env_ids)
        self._reset_ring(self._pose_rng, self._pose_ptr, self._pose_fill, env_ids)
        if env_ids is None:
            if self._pose_mask is not None:
                self._pose_mask.zero_()
            if self._pose_pos is not None:
                self._pose_pos.zero_()
            if self._pose_quat is not None:
                self._pose_quat.zero_()
                self._pose_quat[..., 0] = 1.0
            return
        ids = env_ids.long().view(-1)
        if ids.numel() == 0:
            return
        if self._pose_mask is not None:
            self._pose_mask[ids] = 0
        if self._pose_pos is not None:
            self._pose_pos[ids] = 0
        if self._pose_quat is not None:
            self._pose_quat[ids] = 0
            self._pose_quat[ids, :, 0] = 1.0

    @staticmethod
    def _reset_ring(buf, ptr, fill, env_ids: torch.Tensor | None) -> None:
        if buf is None or ptr is None or fill is None:
            return
        if env_ids is None:
            buf.zero_()
            ptr.zero_()
            fill.zero_()
            return
        ids = env_ids.long().view(-1)
        if ids.numel() == 0:
            return
        buf[ids] = 0
        ptr[ids] = 0
        fill[ids] = 0

    def _ensure_token_ring(self, n: int, device: torch.device, dtype: torch.dtype) -> None:
        L = max(1, self.gru_delay_steps)
        need = (
            self._tok_buf is None
            or self._tok_buf.shape[0] != n
            or self._tok_buf.shape[1] != L
            or self._tok_buf.device != device
            or self._tok_buf.dtype != dtype
        )
        if need:
            self._tok_buf = torch.zeros(n, L, self.token_dim, device=device, dtype=dtype)
            self._tok_ptr = torch.zeros(n, dtype=torch.long, device=device)
            self._tok_fill = torch.zeros(n, dtype=torch.long, device=device)

    def _pop_lag_and_push(self, token_now: torch.Tensor) -> torch.Tensor:
        """Return token from Δ steps ago (or ``token_now`` if ring cold), then push now."""
        n = int(token_now.shape[0])
        self._ensure_token_ring(n, token_now.device, token_now.dtype)
        assert self._tok_buf is not None and self._tok_ptr is not None and self._tok_fill is not None
        L = int(self._tok_buf.shape[1])
        k = int(self.gru_delay_steps)
        rows = torch.arange(n, device=token_now.device)
        if k <= 0:
            lag = token_now
        else:
            warm = self._tok_fill >= k
            idx = (self._tok_ptr - k) % L
            lag_buf = self._tok_buf[rows, idx]
            lag = torch.where(warm.unsqueeze(-1), lag_buf, token_now)
        self._tok_buf[rows, self._tok_ptr] = token_now.detach()
        self._tok_ptr = (self._tok_ptr + 1) % L
        self._tok_fill = torch.clamp(self._tok_fill + 1, max=L)
        return lag

    def encode_token(
        self, rng: torch.Tensor, mask: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CNN spatial map + GRU token ``[pooled ∥ z_prop]``."""
        feat = self.map_cnn(self._as_range_image(rng, mask))
        pooled = feat.mean(dim=(2, 3))
        z_prop = self.proprio_encoder(proprio)
        token = torch.cat([pooled, z_prop], dim=-1)
        return feat, token

    def forward(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, token_now = self.encode_token(rng, mask, proprio)
        if self.gru_delay_steps > 0:
            token_lag = self._pop_lag_and_push(token_now)
            gru_in = torch.stack([token_lag, token_now], dim=1)  # (B, 2, D) ≡ [t−Δ, t]
        else:
            gru_in = token_now.unsqueeze(1)
        if hidden is None:
            out, hidden = self.gru(gru_in)
        else:
            out, hidden = self.gru(gru_in, hidden)
        style = self.gru_to_style(out[:, -1])
        h_rough = self.rough_head(feat, style).clamp(-1.0, 1.0)
        if self.refine is None:
            return h_rough.flatten(1), h_rough.flatten(1), hidden
        refined_map = self.refine(h_rough).clamp(-1.0, 1.0)
        return h_rough.flatten(1), refined_map.flatten(1), hidden

    def splat_range_to_hb(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Max/min-z splat into the yaw-aligned ``H^b`` grid.

        Thin stone gaps (~1 HF cell) disappear under max-only splat when any
        stone-top hit lands in the cell. Returning min-z and ``gap=max-min``
        keeps a void cue for the decoder.

        Returns ``h_max, h_min, hit, gap`` as ``(B, 1, ny, nx)``. Empty cells
        use the per-env mean of ``h_max`` hits (FOV edge); ``gap`` is 0 there.
        """
        b, n = rng.shape
        dirs = self.ray_dirs.reshape(3, -1).transpose(0, 1).to(device=rng.device, dtype=rng.dtype)
        if int(dirs.shape[0]) != n:
            raise RuntimeError(f"ray dirs {dirs.shape[0]} != range {n}")
        hit_s = dirs.unsqueeze(0) * rng.unsqueeze(-1)
        q = lidar_quat.unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        hit_w = lidar_pos.unsqueeze(1) + _quat_apply(q, hit_s.reshape(b * n, 3)).view(b, n, 3)
        yaw_q = _yaw_quat(root_quat).unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        delta = (hit_w - root_pos.unsqueeze(1)).reshape(b * n, 3)
        hit_b = _quat_apply_inverse(yaw_q, delta).view(b, n, 3)
        ix = torch.round((hit_b[..., 0] - self.hb_x0) / self.hb_res).long()
        iy = torch.round((hit_b[..., 1] - self.hb_y0) / self.hb_res).long()
        valid = (mask > 0.5) & (ix >= 0) & (ix < self.hb_nx) & (iy >= 0) & (iy < self.hb_ny)
        flat = torch.where(valid, iy * self.hb_nx + ix, torch.zeros_like(ix)).clamp(0, self.hb_ny * self.hb_nx - 1)
        z_hi = torch.where(valid, hit_b[..., 2], torch.full_like(hit_b[..., 2], -1.0e4))
        z_lo = torch.where(valid, hit_b[..., 2], torch.full_like(hit_b[..., 2], 1.0e4))
        acc_max = rng.new_full((b, self.hb_ny * self.hb_nx), -1.0e4)
        acc_min = rng.new_full((b, self.hb_ny * self.hb_nx), 1.0e4)
        acc_max.scatter_reduce_(1, flat, z_hi, reduce="amax", include_self=True)
        acc_min.scatter_reduce_(1, flat, z_lo, reduce="amin", include_self=True)
        hit = acc_max > -1.0e3
        mean_h = (acc_max.clamp(min=-1.0e3) * hit.float()).sum(dim=-1) / hit.float().sum(dim=-1).clamp_min(1.0)
        fill = mean_h.unsqueeze(-1).expand_as(acc_max)
        h_max = torch.where(hit, acc_max, fill)
        h_min = torch.where(hit, acc_min, fill)
        gap = torch.where(hit, (acc_max - acc_min).clamp(min=0.0), torch.zeros_like(acc_max))
        shape = (b, 1, self.hb_ny, self.hb_nx)
        return (
            h_max.view(*shape),
            h_min.view(*shape),
            hit.float().view(*shape),
            gap.view(*shape),
        )

    def splat_max_height(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        origin_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Max-z splat into yaw-aligned ``H^b``. Empty cells stay 0; no min/gap fill.

        ``origin_b`` is the sensor-frame ray start (lidar offset on the base).
        Range is measured from that point, not from the base origin.

        Returns ``h_max, hit`` as ``(B, 1, ny, nx)``. A cell that sees two treads
        keeps the higher one, so a step stays a step.
        """
        b, n = rng.shape
        dirs = self.ray_dirs.reshape(3, -1).transpose(0, 1).to(device=rng.device, dtype=rng.dtype)
        if int(dirs.shape[0]) != n:
            raise RuntimeError(f"ray dirs {dirs.shape[0]} != range {n}")
        hit_s = dirs.unsqueeze(0) * rng.unsqueeze(-1)
        q = lidar_quat.unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        if origin_b is None:
            start_w = lidar_pos
        else:
            start_w = lidar_pos + _quat_apply(lidar_quat, origin_b)
        hit_w = start_w.unsqueeze(1) + _quat_apply(q, hit_s.reshape(b * n, 3)).view(b, n, 3)
        yaw_q = _yaw_quat(root_quat).unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        delta = (hit_w - root_pos.unsqueeze(1)).reshape(b * n, 3)
        hit_b = _quat_apply_inverse(yaw_q, delta).view(b, n, 3)
        ix = torch.round((hit_b[..., 0] - self.hb_x0) / self.hb_res).long()
        iy = torch.round((hit_b[..., 1] - self.hb_y0) / self.hb_res).long()
        valid = (mask > 0.5) & (ix >= 0) & (ix < self.hb_nx) & (iy >= 0) & (iy < self.hb_ny)
        flat = torch.where(valid, iy * self.hb_nx + ix, torch.zeros_like(ix)).clamp(0, self.hb_ny * self.hb_nx - 1)
        z_hi = torch.where(valid, hit_b[..., 2], torch.full_like(hit_b[..., 2], -1.0e4))
        acc_max = rng.new_full((b, self.hb_ny * self.hb_nx), -1.0e4)
        acc_max.scatter_reduce_(1, flat, z_hi, reduce="amax", include_self=True)
        hit = acc_max > -1.0e3
        h_max = torch.where(hit, acc_max, torch.zeros_like(acc_max))
        shape = (b, 1, self.hb_ny, self.hb_nx)
        return h_max.view(*shape), hit.float().view(*shape)

    def splat_upper_height(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        origin_b: torch.Tensor | None = None,
        band: float = 0.02,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upper-tread height. Rays more than ``band`` below the cell max are dropped.

        ``band`` is 2 cm, about one sigma of the range noise. A shorter stair
        than that stays out of the average. Mixed-pixel bridges are removed in
        the range image first.
        """
        rng, mask = _suppress_mixed_pixels(rng, mask, self.n_ch, self.n_az)
        b, n = rng.shape
        dirs = self.ray_dirs.reshape(3, -1).transpose(0, 1).to(device=rng.device, dtype=rng.dtype)
        if int(dirs.shape[0]) != n:
            raise RuntimeError(f"ray dirs {dirs.shape[0]} != range {n}")
        hit_s = dirs.unsqueeze(0) * rng.unsqueeze(-1)
        q = lidar_quat.unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        if origin_b is None:
            start_w = lidar_pos
        else:
            start_w = lidar_pos + _quat_apply(lidar_quat, origin_b)
        hit_w = start_w.unsqueeze(1) + _quat_apply(q, hit_s.reshape(b * n, 3)).view(b, n, 3)
        yaw_q = _yaw_quat(root_quat).unsqueeze(1).expand(-1, n, -1).reshape(b * n, 4)
        delta = (hit_w - root_pos.unsqueeze(1)).reshape(b * n, 3)
        hit_b = _quat_apply_inverse(yaw_q, delta).view(b, n, 3)
        ix = torch.round((hit_b[..., 0] - self.hb_x0) / self.hb_res).long()
        iy = torch.round((hit_b[..., 1] - self.hb_y0) / self.hb_res).long()
        valid = (mask > 0.5) & (ix >= 0) & (ix < self.hb_nx) & (iy >= 0) & (iy < self.hb_ny)
        flat = torch.where(valid, iy * self.hb_nx + ix, torch.zeros_like(ix)).clamp(0, self.hb_ny * self.hb_nx - 1)
        z = hit_b[..., 2]
        z_hi = torch.where(valid, z, torch.full_like(z, -1.0e4))
        acc_max = rng.new_full((b, self.hb_ny * self.hb_nx), -1.0e4)
        acc_max.scatter_reduce_(1, flat, z_hi, reduce="amax", include_self=True)
        z_top = acc_max.gather(1, flat)
        close = valid & ((z_top - z) <= float(band)) & ((z_top - z) >= 0.0)
        acc_sum = rng.new_zeros((b, self.hb_ny * self.hb_nx))
        acc_cnt = rng.new_zeros((b, self.hb_ny * self.hb_nx))
        acc_sum.scatter_add_(1, flat, torch.where(close, z, torch.zeros_like(z)))
        acc_cnt.scatter_add_(1, flat, close.float())
        hit = acc_cnt > 0.0
        h = torch.where(hit, acc_sum / acc_cnt.clamp_min(1.0), torch.zeros_like(acc_sum))
        shape = (b, 1, self.hb_ny, self.hb_nx)
        return h.view(*shape), hit.float().view(*shape)

    def _fuse_recent_heights(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        origin_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Median of upper-tread splats over the pose ring, in the current yaw frame.

        A spike that appears in one frame drops out. A tread seen from several
        poses stays. Frames the ring does not have yet are ignored.
        """
        current = self.splat_upper_height(rng, mask, lidar_pos, lidar_quat, root_pos, root_quat, origin_b)
        ring = self._pose_rng
        if ring is None or self.gru_delay_steps <= 0:
            return current
        b = int(rng.shape[0])
        k = int(ring.shape[1])
        ages = [a for a in (0, 10, 20, 30, 40, 50, 60, 70) if a < k]
        if not ages:
            return current
        assert self._pose_ptr is not None and self._pose_fill is not None
        assert self._pose_mask is not None and self._pose_pos is not None and self._pose_quat is not None
        age_t = torch.tensor(ages, device=rng.device, dtype=torch.long)
        idx = (self._pose_ptr.unsqueeze(1) - 1 - age_t.unsqueeze(0)) % k
        alive = self._pose_fill.unsqueeze(1) > age_t.unsqueeze(0)
        rows = torch.arange(b, device=rng.device)
        heights = []
        hits = []
        for t in range(int(age_t.shape[0])):
            h_t, hit_t = self.splat_upper_height(
                ring[rows, idx[:, t]],
                self._pose_mask[rows, idx[:, t]],
                self._pose_pos[rows, idx[:, t]],
                self._pose_quat[rows, idx[:, t]],
                root_pos,
                root_quat,
                origin_b,
            )
            keep = alive[:, t].view(b, 1, 1, 1)
            heights.append(h_t)
            hits.append(hit_t * keep.float())
        stacked_h = torch.cat(heights, dim=1)
        stacked_hit = torch.cat(hits, dim=1) > 0.5
        order = torch.where(stacked_hit, stacked_h, torch.full_like(stacked_h, 1.0e4))
        order, _ = order.sort(dim=1)
        count = stacked_hit.sum(dim=1)
        pick = ((count - 1).clamp(min=0) // 2).unsqueeze(1)
        med = order.gather(1, pick)
        seen = (count > 0).unsqueeze(1)
        med = torch.where(seen, med, torch.zeros_like(med))
        return med, seen.float()

    def _push_pose_frame(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the frame from Δ steps ago (or the current frame if cold), then push."""
        b, n = rng.shape
        k = max(1, int(self.gru_delay_steps))
        need = (
            self._pose_rng is None
            or self._pose_rng.shape[0] != b
            or self._pose_rng.shape[1] != k
            or self._pose_rng.shape[2] != n
            or self._pose_rng.device != rng.device
        )
        if need:
            self._pose_rng = rng.new_zeros(b, k, n)
            self._pose_mask = mask.new_zeros(b, k, n)
            self._pose_pos = lidar_pos.new_zeros(b, k, 3)
            self._pose_quat = lidar_quat.new_zeros(b, k, 4)
            self._pose_quat[..., 0] = 1.0
            self._pose_ptr = torch.zeros(b, dtype=torch.long, device=rng.device)
            self._pose_fill = torch.zeros(b, dtype=torch.long, device=rng.device)
        assert self._pose_rng is not None and self._pose_ptr is not None and self._pose_fill is not None
        rows = torch.arange(b, device=rng.device)
        warm = self._pose_fill >= k
        idx = (self._pose_ptr - k) % k
        lag_r = torch.where(warm.unsqueeze(-1), self._pose_rng[rows, idx], rng)
        lag_m = torch.where(warm.unsqueeze(-1), self._pose_mask[rows, idx], mask)
        lag_p = torch.where(warm.unsqueeze(-1), self._pose_pos[rows, idx], lidar_pos)
        lag_q = torch.where(warm.unsqueeze(-1), self._pose_quat[rows, idx], lidar_quat)
        self._pose_rng[rows, self._pose_ptr] = rng.detach()
        self._pose_mask[rows, self._pose_ptr] = mask.detach()
        self._pose_pos[rows, self._pose_ptr] = lidar_pos.detach()
        self._pose_quat[rows, self._pose_ptr] = lidar_quat.detach()
        self._pose_ptr = (self._pose_ptr + 1) % k
        self._pose_fill = torch.clamp(self._pose_fill + 1, max=k)
        return lag_r, lag_m, lag_p, lag_q

    def forward_yaw_splat(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splat current + Δ-delayed hits into yaw-H^b, then residual U-Net."""
        if not self.use_hb_splat:
            raise RuntimeError("forward_yaw_splat requires use_hb_splat=True")
        if self.gru_delay_steps > 0:
            lag_r, lag_m, lag_p, lag_q = self._push_pose_frame(rng, mask, lidar_pos, lidar_quat)
        else:
            lag_r, lag_m, lag_p, lag_q = rng, mask, lidar_pos, lidar_quat
        mx_n, mn_n, hit_n, gap_n = self.splat_range_to_hb(rng, mask, lidar_pos, lidar_quat, root_pos, root_quat)
        mx_l, mn_l, hit_l, gap_l = self.splat_range_to_hb(lag_r, lag_m, lag_p, lag_q, root_pos, root_quat)
        # Soft blend: gap=0 → max (flat); gap≥0.25m → min (true pit/seam).
        alpha = (gap_n / 0.25).clamp(0.0, 1.0) * hit_n
        h_base = (1.0 - alpha) * mx_n + alpha * mn_n
        feat = self.hb_enc(torch.cat([mx_n, mn_n, hit_n, gap_n, mx_l, mn_l, hit_l, gap_l], dim=1))
        pooled = feat.mean(dim=(2, 3))
        z_prop = self.proprio_encoder(proprio)
        token_now = torch.cat([pooled, z_prop], dim=-1)
        # Reuse the token ring, but the stored width is hb_token_dim.
        saved = self.token_dim
        self.token_dim = self.hb_token_dim
        if self.gru_delay_steps > 0:
            token_lag = self._pop_lag_and_push(token_now)
            gru_in = torch.stack([token_lag, token_now], dim=1)
        else:
            gru_in = token_now.unsqueeze(1)
        self.token_dim = saved
        if hidden is None:
            out, hidden = self.hb_gru(gru_in)
        else:
            out, hidden = self.hb_gru(gru_in, hidden)
        style = self.hb_gru_to_style(out[:, -1])
        style_map = style.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hb_ny, self.hb_nx)
        delta = self.hb_delta(torch.cat([feat, style_map], dim=1))
        h_rough = (h_base + delta).clamp(-1.0, 1.0)
        if self.refine is None:
            flat = h_rough.flatten(1)
            return flat, flat, hidden
        # Bound residual so clamp(-1,1) cannot saturate and kill refine grads
        # (scratch + high void_weight previously collapsed to pred≡−1).
        refined = (h_rough + 0.5 * torch.tanh(self.refine(h_rough))).clamp(-1.0, 1.0)
        return h_rough.flatten(1), refined.flatten(1), hidden

    def forward_height_splat(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        lidar_pos: torch.Tensor,
        lidar_quat: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
        origin_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Median of recent upper-tread splats. Empty cells start from the nearest hit.

        ``_last_hit`` is diagnostic.
        """
        if not self.use_height_splat:
            raise RuntimeError("forward_height_splat requires use_height_splat=True")
        if self.gru_delay_steps > 0:
            lag_r, lag_m, lag_p, lag_q = self._push_pose_frame(rng, mask, lidar_pos, lidar_quat)
        else:
            lag_r, lag_m, lag_p, lag_q = rng, mask, lidar_pos, lidar_quat
        h_n, hit_n = self._fuse_recent_heights(
            rng, mask, lidar_pos, lidar_quat, root_pos, root_quat, origin_b
        )
        h_l, hit_l = self.splat_upper_height(
            lag_r, lag_m, lag_p, lag_q, root_pos, root_quat, origin_b
        )
        # Holes continue the last measured tread. A global mean paints one disk.
        near = _nearest_fill(h_n, hit_n).detach()
        h_n_in = torch.where(hit_n > 0.5, h_n, near)
        h_l_in = torch.where(hit_l > 0.5, h_l, near)
        # Cell coordinates, so an empty region is not a constant field. Without
        # them a conv of (hit=0, h=mean) can only paint one height in the hole.
        xy = self.hb_xy.to(dtype=h_n.dtype).expand(h_n.shape[0], -1, -1, -1)
        feat = self.height_enc(torch.cat([h_n_in, hit_n, h_l_in, hit_l, xy], dim=1))
        pooled = feat.mean(dim=(2, 3))
        z_prop = self.proprio_encoder(proprio)
        token_now = torch.cat([pooled, z_prop], dim=-1)
        saved = self.token_dim
        self.token_dim = self.height_token_dim
        try:
            if self.gru_delay_steps > 0:
                token_lag = self._pop_lag_and_push(token_now)
                gru_in = torch.stack([token_lag, token_now], dim=1)
            else:
                gru_in = token_now.unsqueeze(1)
            if hidden is None:
                out, hidden = self.height_gru(gru_in)
            else:
                out, hidden = self.height_gru(gru_in, hidden)
        finally:
            self.token_dim = saved
        style = self.height_gru_to_style(out[:, -1])
        style_map = style.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hb_ny, self.hb_nx)
        delta = self.height_delta(torch.cat([feat, style_map], dim=1))
        # Measured cells stay on the splat. The learned heads were painting a
        # second edge on those treads. Only empty cells may move, by ±8 cm.
        seen = hit_n > 0.5
        guess = (near + 0.08 * torch.tanh(delta)).clamp(-2.0, 2.0)
        h_rough = torch.where(seen, h_n, guess)
        self._last_hit = hit_n.detach().flatten(1)
        if self.refine is None:
            refined = h_rough
        else:
            filled = (h_rough + 0.08 * torch.tanh(self.refine(h_rough))).clamp(-2.0, 2.0)
            refined = torch.where(seen, h_rough, filled)
        if self.edge_clean is not None:
            cleaned = self.edge_clean(refined.view(-1, 1, self.hb_ny, self.hb_nx)).view_as(refined)
            refined = torch.where(seen, refined, cleaned)
        # Hits are the raw splat, so speckle is still there. A 3×3 median removes
        # that grain and a one-cell false ridge, and does not blend across a riser.
        refined = _median_filter(refined)
        return h_rough.flatten(1), refined.flatten(1), hidden

    def loss(
        self,
        h_rough: torch.Tensor,
        h_refined: torch.Tensor,
        h_gt: torch.Tensor,
        *,
        rough_weight: float = 1.0,
        refine_weight: float = 1.0,
        void_thr: float = -0.9,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Surface / void terms are normalized separately so a high void fraction
        # cannot collapse the optimum to "predict all −1".
        void = (h_gt <= float(void_thr)).float()
        surf = 1.0 - void
        surf_den = surf.sum().clamp_min(1.0)
        void_den = void.sum().clamp_min(1.0)
        mse_surf = ((h_rough - h_gt).pow(2) * surf).sum() / surf_den
        mse_void = ((h_rough - h_gt).pow(2) * void).sum() / void_den
        mse = mse_surf + self.void_weight * mse_void
        l1_surf = ((h_refined - h_gt).abs() * surf).sum() / surf_den
        l1_void = ((h_refined - h_gt).abs() * void).sum() / void_den
        l1 = l1_surf + self.void_weight * l1_void
        total = float(rough_weight) * mse + float(refine_weight) * l1
        with torch.no_grad():
            l1_rough = (h_rough - h_gt).abs().mean()
            l1_ref = (h_refined - h_gt).abs().mean()
            resid = (h_refined - h_rough).abs().mean()
            pred_mean = float(h_refined.mean())
        return total, {
            "loss": float(total.detach()),
            "mse_rough": float(mse.detach()),
            "l1_refine": float(l1.detach()),
            "l1_plain": float(l1_ref),
            "l1_rough": float(l1_rough),
            "l1_void": float(l1_void.detach()),
            "l1_surf": float(l1_surf.detach()),
            "resid_mean": float(resid),
            "pred_mean": pred_mean,
            "use_refine": 1.0 if self.refine is not None else 0.0,
        }

    def predict(
        self,
        rng: torch.Tensor,
        mask: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_refined, hidden = self.forward(rng, mask, proprio, hidden)
        return h_refined, hidden

    @staticmethod
    def reshape_hb(flat: torch.Tensor, ny: int = HB_NY, nx: int = HB_NX) -> torch.Tensor:
        return flat.view(-1, ny, nx)

    def reshape_range(self, flat: torch.Tensor) -> torch.Tensor:
        """Current-frame range/mask image ``(B, n_ch, n_az)``.

        History frames may be azimuth-strided, so a stacked vector is wider
        than one frame but not a uniform ``n_frames × n_ch × n_az`` image.
        The current frame is always the tail.
        """
        frame = self.n_ch * self.n_az
        if flat.shape[-1] > frame:
            flat = flat[..., -frame:]
        return flat.view(-1, self.n_ch, self.n_az)
