"""Two-dimensional load allocation on a split-position double support."""

import torch


def allocate_two_feet(point, soles, bounds, preferred=None, minimum_share=.05):
    """Closest L-infinity feasible common CoP and its vertical load fractions.

    The support set is the Minkowski sum of the segment between sole centres
    and the caller's conservative pressure rectangle. A foot's load cannot be
    inferred from world Y alone when the feet also differ substantially in X.
    Every solution remains a convex combination of the two real supports.
    """
    error = point-soles[:,1]
    delta = soles[:,0]-soles[:,1]
    tiny = delta.abs()<1e-8
    denominator = torch.where(tiny,torch.ones_like(delta),delta)
    candidates = [torch.full_like(error[:,:1],minimum_share),
                  torch.full_like(error[:,:1],1-minimum_share),error/denominator]
    for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1)):
        d = sx*delta[:,0]-sy*delta[:,1]
        e = sx*error[:,0]-sy*error[:,1]-bounds[0]+bounds[1]
        fraction = e/torch.where(d.abs()>1e-8,d,torch.ones_like(d))
        candidates.append(torch.where(d.abs()>1e-8,fraction,.5)[:,None])
    candidates = torch.cat(candidates,-1).clamp(minimum_share,1-minimum_share)
    residual = (error[:,None]-candidates[:,:,None]*delta[:,None]).abs()-bounds
    violation,which = residual.clamp_min(0).amax(-1).min(-1)
    best = candidates.gather(1,which[:,None]).squeeze(-1)
    # At the optimum, intersect both coordinate intervals and retain the
    # preceding load where possible, avoiding arbitrary load flips in DS.
    radius = bounds+violation[:,None]+2e-7
    a,b = (error-radius)/denominator,(error+radius)/denominator
    lower = torch.where(tiny,torch.full_like(a,-float('inf')),torch.minimum(a,b)).amax(-1)
    upper = torch.where(tiny,torch.full_like(a,float('inf')),torch.maximum(a,b)).amin(-1)
    lower,upper = lower.clamp(minimum_share,1-minimum_share),upper.clamp(minimum_share,1-minimum_share)
    preferred = best if preferred is None else preferred
    left = torch.where(lower<=upper,torch.maximum(torch.minimum(preferred,upper),lower),best)
    center = soles[:,1]+left[:,None]*delta
    projected = torch.maximum(torch.minimum(point,center+bounds),center-bounds)
    return torch.stack((left,1-left),-1),projected,violation


def settle_curve_violation(start, speed, acceleration, end, duration, omega,
                           soles, bounds, *, settle_profile, vertical_limit=None):
    """Exactly the original33 support knots, flattened by env then fraction.

    This helper does not choose durations or add the signed span gate. The
    caller retains those original decisions. In particular vertical_limit=None
    preserves the original initial four attempts, which exclude vertical excess.
    Only the subsequent13 duration trials pass the existing vertical limit.
    """
    n = start.shape[0]
    # Original _shift uses the current default dtype for its linspace too.
    fractions = torch.linspace(0, 1, 33, device=start.device)
    def repeated(value):
        return value[:, None].expand(n, 33, *value.shape[1:]).reshape(n*33, *value.shape[1:])
    knot_time = (duration[:, None]*fractions).reshape(-1)
    position, _, acc = settle_profile(knot_time, repeated(duration), repeated(start),
        repeated(speed), repeated(acceleration), repeated(end))
    point = position[:, :2]-acc[:, :2]/repeated(omega).square()
    violation = allocate_two_feet(point, repeated(soles), bounds)[2]
    if vertical_limit is not None:
        violation = torch.maximum(violation, ((9.81+acc[:, 2])-vertical_limit).clamp_min(0))
    return torch.maximum(torch.zeros_like(duration), violation.reshape(n, 33).amax(-1))
