"""H2 sole-surface height reference on the collision mesh, without Isaac imports.

The 57 planar hull vertices are in ankle_pitch_link local metres, extracted
from configuration/H2_dae_base.usd at commit 0b012ad284c6e924a77d4d6892b212fa0ace7ff8
(blob 04ce88228fb6e11318ab78aa86e144c62b5a2d88). H2_dae_physics.usd
(blob 994802cc301d139ed3a2b0951c52ede5ebd12b2e) binds the foot collision
meshes with identity mesh-to-link transforms. The right surface is the left
surface mirrored across local Y. Order is left, right, matching H2 feet config.

These vertices describe only the lowest planar sole surface in the source mesh.
They are not the complete collision hull, PhysX's cooked hull, contact points,
or a measurement of actual support coverage. When tilted, bevel-side vertices
outside this surface can lie slightly lower than this reference.
"""

from __future__ import annotations

import torch


_H2_LEFT_SOLE_SURFACE_B = (
    (-0.05740509554743767, 0.0639863908290863, -0.05429999902844429),
    (-0.060996752232313156, 0.06344228237867355, -0.05429999902844429),
    (-0.062104593962430954, 0.06315868347883224, -0.05429999902844429),
    (-0.06513179838657379, 0.061915576457977295, -0.05429999902844429),
    (-0.06540420651435852, 0.0617472305893898, -0.05429999902844429),
    (-0.0663413554430008, 0.06099813058972359, -0.05429999902844429),
    (-0.06709180772304535, 0.06006176024675369, -0.05429999902844429),
    (-0.06726360321044922, 0.059790484607219696, -0.05429999902844429),
    (-0.06768493354320526, 0.05901521444320679, -0.05429999902844429),
    (-0.06863021850585938, 0.05649099126458168, -0.05429999902844429),
    (-0.06945429742336273, 0.05246516689658165, -0.05429999902844429),
    (-0.06990192830562592, 0.04736962169408798, -0.05429999902844429),
    (-0.07000000774860382, 0.04259999841451645, -0.05429999902844429),
    (-0.07000000774860382, 0.032600000500679016, -0.05429999902844429),
    (-0.06970280408859253, 0.02489575371146202, -0.05429999902844429),
    (-0.06949540972709656, 0.023036915808916092, -0.05429999902844429),
    (-0.06854389607906342, 0.0184171199798584, -0.05429999902844429),
    (-0.06766007840633392, 0.01613430678844452, -0.05429999902844429),
    (-0.06705890595912933, 0.01508894469588995, -0.05429999902844429),
    (-0.066864013671875, 0.014813588000833988, -0.05429999902844429),
    (-0.06629231572151184, 0.01415297668427229, -0.05429999902844429),
    (-0.06430710852146149, 0.012851488776504993, -0.05429999902844429),
    (-0.06084860488772392, 0.011724456213414669, -0.05429999902844429),
    (-0.057354386895895004, 0.011209326796233654, -0.05429999902844429),
    (-0.05502733215689659, 0.01110092829912901, -0.05429999902844429),
    (0.09467753767967224, 0.004889592994004488, -0.05429999902844429),
    (0.10461880266666412, 0.004922960419207811, -0.05429999902844429),
    (0.11087755113840103, 0.005317386705428362, -0.05429999902844429),
    (0.11861834675073624, 0.006254349369555712, -0.05429999902844429),
    (0.12681181728839874, 0.007912960834801197, -0.05429999902844429),
    (0.12965404987335205, 0.008693368174135685, -0.05429999902844429),
    (0.13461636006832123, 0.01039541233330965, -0.05429999902844429),
    (0.13812316954135895, 0.011944458819925785, -0.05429999902844429),
    (0.1431206911802292, 0.014951261691749096, -0.05429999902844429),
    (0.14514930546283722, 0.01663696952164173, -0.05429999902844429),
    (0.14668187499046326, 0.018252087756991386, -0.05429999902844429),
    (0.14780522882938385, 0.01977250911295414, -0.05429999902844429),
    (0.14917480945587158, 0.0225372351706028, -0.05429999902844429),
    (0.14987650513648987, 0.0255312267690897, -0.05429999902844429),
    (0.15000000596046448, 0.02759999968111515, -0.05429999902844429),
    (0.15000000596046448, 0.047600001096725464, -0.05429999902844429),
    (0.14969009160995483, 0.050812095403671265, -0.05429999902844429),
    (0.14948853850364685, 0.051661841571331024, -0.05429999902844429),
    (0.1480235904455185, 0.055078234523534775, -0.05429999902844429),
    (0.14782357215881348, 0.055399008095264435, -0.05429999902844429),
    (0.145180806517601, 0.0585334412753582, -0.05429999902844429),
    (0.14496350288391113, 0.05873505771160126, -0.05429999902844429),
    (0.14196372032165527, 0.06106007844209671, -0.05429999902844429),
    (0.14001041650772095, 0.0622577928006649, -0.05429999902844429),
    (0.13842149078845978, 0.06310655921697617, -0.05429999902844429),
    (0.13313332200050354, 0.06536534428596497, -0.05429999902844429),
    (0.1269296407699585, 0.06725707650184631, -0.05429999902844429),
    (0.12236975878477097, 0.068283811211586, -0.05429999902844429),
    (0.11898121237754822, 0.06888804584741592, -0.05429999902844429),
    (0.11019960045814514, 0.06994011998176575, -0.05429999902844429),
    (0.10283351689577103, 0.07033537328243256, -0.05429999902844429),
    (0.09467753767967224, 0.07031040638685226, -0.05429999902844429),

)
H2_SOLE_SURFACE_POINTS_B = (
    _H2_LEFT_SOLE_SURFACE_B,
    tuple((x, -y, z) for x, y, z in _H2_LEFT_SOLE_SURFACE_B),
)


def lowest_sole_height_world(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    sole_points_b=None,
) -> torch.Tensor:
    """Minimum world Z of each foot's transformed sole-surface vertices.

    Positions have shape [env, foot, 3], quaternions [env, foot, 4] in wxyz
    order, points [foot, point, 3]. Points default to the H2 left/right pair.
    Each foot's full world orientation is used; finite nonzero quaternions are
    normalized. Invalid positions or quaternions produce NaN for that foot so
    the reward caller can exclude it without disabling the other foot.
    """
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError("body_pos_w must have shape [env, foot, 3]")
    if body_quat_w.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError("body_quat_w must have shape [env, foot, 4]")
    if not body_pos_w.is_floating_point() or not body_quat_w.is_floating_point():
        raise ValueError("body positions and quaternions must be floating point")
    if body_quat_w.device != body_pos_w.device or body_quat_w.dtype != body_pos_w.dtype:
        raise ValueError("body positions and quaternions must share dtype and device")
    points = torch.as_tensor(
        H2_SOLE_SURFACE_POINTS_B if sole_points_b is None else sole_points_b,
        device=body_pos_w.device, dtype=body_pos_w.dtype,
    )
    if (points.ndim != 3 or points.shape[0] != body_pos_w.shape[1]
            or points.shape[1] == 0 or points.shape[2] != 3):
        raise ValueError("sole_points_b must have shape [foot, point >= 1, 3]")
    if sole_points_b is not None and not torch.isfinite(points).all():
        raise ValueError("sole_points_b must contain only finite points")

    finite_quat = torch.isfinite(body_quat_w).all(dim=-1)
    safe_quat = torch.where(finite_quat.unsqueeze(-1), body_quat_w, 0.0)
    norm = torch.linalg.vector_norm(safe_quat, dim=-1)
    valid_quat = finite_quat & torch.isfinite(norm) & (norm > 0.0)
    quat = safe_quat / torch.where(valid_quat, norm, 1.0).unsqueeze(-1)
    w, x, y, z = quat.unbind(-1)
    # Only the rotation matrix's Z row is needed, avoiding [env, foot, point, xyz].
    rotation_z = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                              1 - 2 * (x.square() + y.square())), dim=-1)
    sole_z = torch.einsum("efc,fpc->efp", rotation_z, points).amin(dim=-1) + body_pos_w[..., 2]
    valid_pose = valid_quat & torch.isfinite(body_pos_w).all(dim=-1)
    return torch.where(valid_pose, sole_z, torch.nan)
