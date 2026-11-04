"""Pure batched helpers for a measured-pose contact handover.

These functions own no clocks, pressure anchors, contact predicates, buffers or
simulator state. A C2 Cartesian restore weight is not a global joint-C2 claim.
"""

import torch


def blend_contact_quaternion(measured, nominal, restore_weight):
    """Shortest-hemisphere normalized blend of complete wxyz rotations.

    Supports (...,4) quaternions and (...) weights, including N-by2 feet. The
    known local IsaacLab quat_slerp does not support batches and mutates q2 in
    its antipodal branch, so it is deliberately not called here.
    """
    if measured.shape != nominal.shape or measured.shape[-1] != 4:
        raise ValueError('Measured and nominal rotations must share(...,4)shape')
    if restore_weight.shape != measured.shape[:-1]:
        raise ValueError('Restore weight must match quaternion batchshape')
    first = measured/measured.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    last = nominal/nominal.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    last = torch.where((first*last).sum(-1,keepdim=True)<0,-last,last)
    weight = restore_weight.clamp(0,1)[...,None]
    result = (1-weight)*first+weight*last
    return result/result.norm(dim=-1,keepdim=True).clamp_min(1e-9)


def rebase_landed_joint_reference(reference_position, reference_velocity,
        measured_position, measured_velocity, fresh_contact, maximum_rate):
    """Rebase only a newly contacting foot and retain its measured velocity.

    Foot mask is(...,2); joint tensors are(...,2,6). Position rebasing changes
    the published joint target intentionally. Velocity matching avoids dividing
    preceding swing tracking error by one20ms control interval. All other
    joints retain the ordinary IK/rate-limited command bit-for-bit.
    """
    if reference_position.shape != reference_velocity.shape or measured_position.shape != reference_position.shape:
        raise ValueError('Joint state/reference shapes mustmatch')
    if measured_velocity.shape != reference_position.shape or fresh_contact.shape != reference_position.shape[:-1]:
        raise ValueError('Measured velocity/footmask shape mismatch')
    if maximum_rate <= 0:
        raise ValueError('Maximum reference joint rate must bepositive')
    return (torch.where(fresh_contact[...,None],measured_position,reference_position),
            torch.where(fresh_contact[...,None],measured_velocity.clamp(-maximum_rate,maximum_rate),reference_velocity))
