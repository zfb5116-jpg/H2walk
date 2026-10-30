"""Batched H2 USD leg geometry; joint coordinates are never approximated as planar."""

import json
from pathlib import Path

import torch


class H2LegKinematics:
    def __init__(self, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        data = json.loads((Path(__file__).parent / "assets/h2_leg_chain.json").read_text())
        legs = [data["legs"][s] for s in ("left", "right")]
        self.joint_names = [name for leg in legs for name in leg["joint_names"]]
        tensor = lambda value: torch.tensor(value, dtype=dtype, device=device)
        self.a = tensor([[j["A"] for j in leg["joints"]] for leg in legs])
        self.b_inv = tensor([[j["B_inv"] for j in leg["joints"]] for leg in legs])
        basis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
        self.axes = tensor([[basis[j["axis"]] for j in leg["joints"]] for leg in legs])
        self.sole_offset = tensor([leg["sole_center_local_offset_m"] for leg in legs])
        self.home = tensor([leg["home_leveled_q_rad"] for leg in legs])
        limits = tensor([[j["soft_position_limits_rad"] for j in leg["joints"]] for leg in legs])
        self.lower, self.upper = limits[..., 0], limits[..., 1]
        self.mass = tensor([[j["child_mass_kg"] for j in leg["joints"]] for leg in legs])
        self.com_local = tensor([[j["child_com_local_m"] for j in leg["joints"]] for leg in legs])
        self.eye3 = torch.eye(3, device=device, dtype=dtype)
        self.eye6 = torch.eye(6, device=device, dtype=dtype)
        u = self.axes
        self.cross_axes = torch.zeros((2, 6, 3, 3), device=device, dtype=dtype)
        self.cross_axes[..., 0, 1], self.cross_axes[..., 0, 2] = -u[..., 2], u[..., 1]
        self.cross_axes[..., 1, 0], self.cross_axes[..., 1, 2] = u[..., 2], -u[..., 0]
        self.cross_axes[..., 2, 0], self.cross_axes[..., 2, 1] = -u[..., 1], u[..., 0]

    def forward(self, q, *, jacobian=False, frames=False):
        q = q.reshape(-1, 2, 6)
        n = q.shape[0]
        transform = torch.eye(4, device=q.device, dtype=q.dtype).expand(n, 2, 4, 4)
        origins, axes, children = [], [], []
        for i in range(6):
            frame = transform @ self.a[None, :, i]
            origins.append(frame[..., :3, 3])
            axes.append((frame[..., :3, :3] @ self.axes[None, :, i, :, None]).squeeze(-1))
            angle = q[..., i, None, None]
            unit = self.axes[None, :, i]
            rotation = (angle.cos() * self.eye3 + (1 - angle.cos()) * unit[..., :, None] * unit[..., None, :]
                        + angle.sin() * self.cross_axes[None, :, i])
            joint = torch.zeros_like(frame)
            joint[..., :3, :3], joint[..., 3, 3] = rotation, 1.0
            transform = frame @ joint @ self.b_inv[None, :, i]
            children.append(transform)
        rotation = transform[..., :3, :3]
        sole = transform[..., :3, 3] + (rotation @ self.sole_offset[None, ..., None]).squeeze(-1)
        result = [sole, rotation]
        if jacobian or frames:
            origins, axes = torch.stack(origins, 2), torch.stack(axes, 2)
            linear = torch.cross(axes, sole[..., None, :] - origins, dim=-1)
            result.append(torch.cat((linear.transpose(-1, -2), axes.transpose(-1, -2)), dim=-2))
        if frames:
            result.extend((origins, axes, torch.stack(children, 2)))
        return tuple(result)

    def solve(self, position, rotation, initial, *, iterations=5, damping=0.008):
        q = initial.reshape(-1, 2, 6).clone()
        for _ in range(iterations):
            current, current_rotation, jac = self.forward(q, jacobian=True)
            delta_rotation = rotation @ current_rotation.transpose(-1, -2)
            vee = torch.stack((delta_rotation[..., 2, 1] - delta_rotation[..., 1, 2],
                               delta_rotation[..., 0, 2] - delta_rotation[..., 2, 0],
                               delta_rotation[..., 1, 0] - delta_rotation[..., 0, 1]), -1) * 0.5
            sine = vee.norm(dim=-1, keepdim=True)
            cosine = ((delta_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).unsqueeze(-1)
            orientation_error = vee * torch.atan2(sine, cosine) / sine.clamp_min(1e-9)
            error = torch.cat((position - current, 0.25 * orientation_error), -1)
            jac = jac.clone()
            jac[..., 3:, :] *= 0.25
            increment = (jac.transpose(-1, -2) @ torch.linalg.solve(
                jac @ jac.transpose(-1, -2) + damping**2 * self.eye6, error[..., None])).squeeze(-1)
            q = torch.maximum(torch.minimum(q + increment.clamp(-0.12, 0.12), self.upper), self.lower)
        final, final_rotation = self.forward(q)
        return q, (final - position).norm(dim=-1)

    def support_torque(self, q, force, contact_point, gravity_direction=None):
        """Gravity plus planned ground-reaction inverse statics, for each leg."""
        _, _, _, origins, axes, children = self.forward(q, frames=True)
        centers = (children[..., :3, :3] @ self.com_local[None, ..., None]).squeeze(-1) + children[..., :3, 3]
        if gravity_direction is None:
            gravity = torch.zeros_like(centers)
            gravity[..., 2] = -9.81 * self.mass
        else:
            gravity = 9.81 * gravity_direction[:, None, None, :] * self.mass[None, ..., None]
        torque = torch.zeros_like(q)
        for i in range(6):
            own = torch.cross(centers[..., i:, :] - origins[..., i, None, :], gravity[..., i:, :], dim=-1).sum(-2)
            reaction = torch.cross(contact_point - origins[..., i, :], force, dim=-1)
            torque[..., i] = -((own + reaction) * axes[..., i, :]).sum(-1)
        return torque
