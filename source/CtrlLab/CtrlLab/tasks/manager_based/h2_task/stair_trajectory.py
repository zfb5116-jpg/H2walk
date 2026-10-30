"""One pure Cartesian stair trajectory shared by command and preview IK."""

import torch


def smooth(value):
    u=value.clamp(0,1)
    return u**3*(10-15*u+6*u*u)


def swing_pitch(gait,ids,offsets=(0.,),*,cross_advancing=None):
    """A small toe-down angle relaxes rear-lift ankle flexion; level to land."""
    offset=gait.q.new_tensor(offsets)
    cf=gait.cross_advancing if cross_advancing is None else cross_advancing
    time=(gait.time[ids,None]+offset).clamp_min(0)
    cross=(gait.cross_elapsed[ids,None]+cf[ids,None]*offset).clamp_min(0)
    rising=(gait.swing_target[ids,2]-gait.swing_start[ids,2]>.04)&~gait.preparing[ids]
    amplitude=gait.options.get("swing_toe_down_rad",0.)-gait.options.get("swing_toe_up_rad",0.)
    return amplitude*rising[:,None]*smooth(time/.12)*(1-smooth(cross/(gait.options["cross_duration"]*.75)))


def swing_position_local(gait,ids,offsets=(0.,),*,cross_advancing=None,descent_advancing=None):
    """N x T x 3 planned points; no contact/map/clock mutation or prediction.

    An upper-step flight first approaches its bounded pre-riser point at a
    modest clearance. Actual full-sole clearance releases crossing; the later
    higher peak is reached while the foot is already moving forward. This
    avoids demanding an unreachable rear-foot high lift under a forward body.
    Ground preparation and level flights retain their previous lift profile.
    """
    offset=gait.q.new_tensor(offsets)
    if offset.ndim!=1:raise ValueError("Trajectory offsets must be one-dimensional")
    cf=gait.cross_advancing if cross_advancing is None else cross_advancing
    df=gait.descent_advancing if descent_advancing is None else descent_advancing
    if cf.shape!=gait.time.shape or df.shape!=gait.time.shape:
        raise ValueError("Trajectory clock flags must be full environment tensors")
    time=(gait.time[ids,None]+offset).clamp_min(0)
    cross_time=(gait.cross_elapsed[ids,None]+cf[ids,None]*offset).clamp_min(0)
    descent_time=(gait.descent_elapsed[ids,None]+df[ids,None]*offset).clamp_min(0)
    start,end=gait.swing_start[ids],gait.swing_target[ids]
    rising=(end[:,2]-start[:,2]>.04)&~gait.preparing[ids]
    lift=gait.lift_target[ids]
    lift_blend=smooth(time/(gait.options["flight_duration"]*.5))
    before=torch.lerp(start[:,None],lift[:,None],lift_blend[...,None])
    cross=(cross_time/gait.options["cross_duration"]).clamp(0,1)
    cross=torch.where(gait.clearance_passed[ids,None],cross,0)
    moving=torch.lerp(lift[:,None],end[:,None],smooth(cross)[...,None])
    clearance=torch.where(gait.preparing[ids],gait.options["preparation_clearance"],gait.options["clearance"])
    peak=torch.maximum(start[:,2],end[:,2])+clearance
    raised=torch.lerp(lift[:,2,None],peak[:,None],smooth(cross/.65))
    original_lift=torch.lerp(start[:,2,None],peak[:,None],lift_blend)
    moving[...,2]=torch.where(rising[:,None],raised,original_lift)
    moving=torch.where((rising[:,None]&(cross_time<=0))[...,None],before,moving)
    drop=torch.lerp(peak[:,None],end[:,2,None],smooth(descent_time/gait.options["descent_duration"]))
    moving[...,2]=torch.where((gait.descend_time[ids]>=0)[:,None],drop,moving[...,2])
    return moving
