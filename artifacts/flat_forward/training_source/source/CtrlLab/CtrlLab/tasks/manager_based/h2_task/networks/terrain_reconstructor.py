# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""START / Yu TR-Net terrain reconstructor (paper Fig.3).

Pipeline:
  depth [d(t−2s), d(t)] → CNN → (+ z_prop) → GRU
    → Rough MLP  → Ĥ_rough   (MSE vs H^b)
    → U-Net Refine → Ĥ_refined (L1 vs H^b)

Depth layout matches ``depth_image_delayed_pair``: channel0=lag, channel1=now.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEPTH_H = 60
DEPTH_W = 60
DEPTH_FRAMES = 2  # [lag, now]
HB_NY = 17
HB_NX = 33
HB_DIM = HB_NY * HB_NX  # 561


def split_delayed_depth_pair(
    depth: torch.Tensor,
    *,
    frames: int = DEPTH_FRAMES,
    height: int = DEPTH_H,
    width: int = DEPTH_W,
) -> torch.Tensor:
    """Return depth as ``(B, 2, H, W)`` with ``[lag, now]``."""
    if depth.ndim == 2:
        b = depth.shape[0]
        depth = depth.view(b, frames, height, width)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1).expand(-1, frames, -1, -1).clone()
    elif depth.ndim == 4 and depth.shape[1] == frames:
        pass
    else:
        raise ValueError(f"unexpected depth shape {tuple(depth.shape)}")
    return depth


class _DepthCNN(nn.Module):
    """Paper-style CNN encoder on stacked depth frames → feature vector."""

    def __init__(self, in_frames: int = DEPTH_FRAMES, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=5, stride=2, padding=2),  # 30×30
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 15×15
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8×8
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 4×4
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ELU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        depth = split_delayed_depth_pair(depth)
        x = self.net(depth).flatten(1)
        return self.fc(x)


class _RoughMLP(nn.Module):
    """Paper Rough decoder: GRU state → flat heightmap (561)."""

    def __init__(self, in_dim: int, hb_dim: int = HB_DIM, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [512, 512]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, hb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class _RefineUNet(nn.Module):
    """Paper Refine decoder: U-Net on ``(B, 1, 17, 33)`` rough map → refined map."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ELU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1),
            nn.ELU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, rough: torch.Tensor) -> torch.Tensor:
        # rough: (B, 1, ny, nx)
        e1 = self.enc1(rough)
        e2 = self.down(e1)
        b = self.bottleneck(e2)
        u = self.up(b)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([u, e1], dim=1))


class TerrainReconstructor(nn.Module):
    """Paper TR-Net: CNN → GRU → Rough MLP → U-Net Refine."""

    def __init__(
        self,
        depth_frames: int = DEPTH_FRAMES,
        depth_feat_dim: int = 128,
        proprio_dim: int = 45,
        z_prop_dim: int = 64,
        gru_hidden_dim: int = 256,
        hb_ny: int = HB_NY,
        hb_nx: int = HB_NX,
        use_refine: bool = True,
        rough_hidden: list[int] | None = None,
    ):
        super().__init__()
        if int(depth_frames) != 2:
            raise ValueError("expects 2 depth frames [lag, now]")
        self.depth_frames = 2
        self.hb_ny = int(hb_ny)
        self.hb_nx = int(hb_nx)
        self.hb_dim = self.hb_ny * self.hb_nx
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.use_refine = bool(use_refine)

        self.depth_cnn = _DepthCNN(in_frames=self.depth_frames, out_dim=depth_feat_dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(int(proprio_dim), 128),
            nn.ELU(inplace=True),
            nn.Linear(128, z_prop_dim),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=depth_feat_dim + z_prop_dim,
            hidden_size=self.gru_hidden_dim,
            batch_first=True,
        )
        self.rough_decoder = _RoughMLP(
            in_dim=self.gru_hidden_dim,
            hb_dim=self.hb_dim,
            hidden=rough_hidden,
        )
        self.refine = _RefineUNet(base_ch=32) if self.use_refine else None

    def encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.proprio_encoder(proprio)

    def forward(
        self,
        depth: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h_rough, h_refined : (B, 561)
        hidden : GRU state
        """
        depth_feat = self.depth_cnn(depth)
        z_prop = self.encode_proprio(proprio)
        gru_in = torch.cat([depth_feat, z_prop], dim=-1).unsqueeze(1)
        if hidden is None:
            out, hidden = self.gru(gru_in)
        else:
            out, hidden = self.gru(gru_in, hidden)
        h = out.squeeze(1)

        h_rough = self.rough_decoder(h).clamp(-1.0, 1.0)
        if self.refine is None:
            return h_rough, h_rough, hidden

        rough_map = h_rough.view(-1, 1, self.hb_ny, self.hb_nx)
        # Paper: U-Net refinement decoder on the rough map (full refined output).
        refined_map = self.refine(rough_map).clamp(-1.0, 1.0)
        h_refined = refined_map.flatten(1)
        return h_rough, h_refined, hidden

    def loss(
        self,
        h_rough: torch.Tensor,
        h_refined: torch.Tensor,
        h_gt: torch.Tensor,
        *,
        rough_weight: float = 1.0,
        refine_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Paper hybrid loss: MSE(rough) + L1(refined)."""
        mse = F.mse_loss(h_rough, h_gt)
        l1 = F.l1_loss(h_refined, h_gt)
        total = float(rough_weight) * mse + float(refine_weight) * l1
        with torch.no_grad():
            l1_rough = (h_rough - h_gt).abs().mean()
            l1_ref = (h_refined - h_gt).abs().mean()
            resid = (h_refined - h_rough).abs().mean()
        return total, {
            "loss": float(total.detach()),
            "mse_rough": float(mse.detach()),
            "l1_refine": float(l1.detach()),
            "l1_plain": float(l1_ref),
            "l1_rough": float(l1_rough),
            "resid_mean": float(resid),
            "use_refine": 1.0 if self.refine is not None else 0.0,
        }

    def predict(
        self,
        depth: torch.Tensor,
        proprio: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deploy with refined map (paper)."""
        _, h_refined, hidden = self.forward(depth, proprio, hidden)
        return h_refined, hidden

    @staticmethod
    def reshape_hb(flat: torch.Tensor, ny: int = HB_NY, nx: int = HB_NX) -> torch.Tensor:
        return flat.view(-1, ny, nx)
