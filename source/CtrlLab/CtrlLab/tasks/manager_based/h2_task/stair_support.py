"""Closest contact-pressure points satisfying actual loaded joint limits."""

import torch


def project_wrench_points(kin, q, root_inverse, forces_root, points, root, soles,
                         foot_rotation, effort_limits, corners, *, support_pose=None):
    """Keep a valid request, otherwise project it onto the exact2D intersection.

    The two foot loads and actual joint axes define separate affine torque
    constraints. Actual sole geometry supplies the pressure polygon. No neutral
    ankle bias or hypothetical full-body load replaces a valid braking point.
    Points, soles and root use the fixed heading frame; forces use root frame.
    """
    relative = (root_inverse[:,None]@(points-root[:,None])[...,None]).squeeze(-1)
    torque = (kin.support_torque(q,forces_root,relative,-root_inverse[...,2]) if support_pose is None
              else cached_support_torque(support_pose,forces_root,relative))
    local = (foot_rotation.transpose(-1,-2)@(points-soles)[...,None]).squeeze(-1)[...,:2]
    half = corners[...,:2].abs().amax(-2)-.008
    geometric_good = (local.abs()<=half+1e-6).all(-1)
    torque_good = (torque.abs()<=effort_limits+1e-3).all(-1)
    needs = ~(geometric_good&torque_good)
    result = points.clone()
    feasible = ~needs
    if not needs.any():
        return result,torque,feasible
    rows,feet = needs.nonzero(as_tuple=True)
    for offset in range(0,rows.numel(),64):
        ids,side = rows[offset:offset+64],feet[offset:offset+64]
        if support_pose is None:
            _,_,_,_,axes,_ = kin.forward(q[ids],frames=True)
        else:
            axes = support_pose["axes"][ids]
        basis = root_inverse[ids].transpose(-1,-2)[:,:2]
        moment = torch.cross(basis,forces_root[ids,side,None],dim=-1)
        derivative = -(axes[torch.arange(ids.numel(),device=q.device),side,:,None]*moment[:,None]).sum(-1)
        geometry = foot_rotation[ids,side].transpose(-1,-2)[:,:2,:2]
        a = torch.cat((derivative,-derivative,geometry,-geometry),1)
        b = torch.cat((effort_limits[ids,side]-torque[ids,side],
                       effort_limits[ids,side]+torque[ids,side],
                       half[side]-local[ids,side],half[side]+local[ids,side]),-1)
        norm = a.norm(dim=-1)
        active = norm>1e-6
        impossible_constant = ((~active)&(b<-.001)).any(-1)
        a = a/norm.clamp_min(1e-6)[...,None]
        b = torch.where(active,b/norm.clamp_min(1e-6),torch.full_like(b,100.))
        i,j = torch.triu_indices(a.shape[1],a.shape[1],1,device=q.device)
        first,second = a[:,i],a[:,j]
        determinant = first[...,0]*second[...,1]-first[...,1]*second[...,0]
        nonparallel = determinant.abs()>1e-6
        denominator = torch.where(nonparallel,determinant,torch.ones_like(determinant))
        intersection = torch.stack((b[:,i]*second[...,1]-b[:,j]*first[...,1],
                                     first[...,0]*b[:,j]-second[...,0]*b[:,i]),-1)/denominator[...,None]
        candidate = torch.cat((torch.zeros_like(a[:,:1]),a*b[...,None],intersection),1)
        candidate_valid = torch.cat((torch.ones(ids.numel(),1,dtype=torch.bool,device=q.device),
                                     active,nonparallel),-1)
        violation = (torch.einsum('nci,nki->nkc',a,candidate)-b[:,None]).amax(-1)
        valid = candidate_valid&(violation<=2e-6)&~impossible_constant[:,None]
        distance = candidate.square().sum(-1)
        score = torch.where(valid,distance,float('inf'))
        best,choice = score.min(-1)
        # An empty intersection remains explicitly infeasible. Retain the
        # least-violating finite point for a bounded physical failure record.
        fallback = torch.where(candidate_valid,violation+1e-4*distance,float('inf')).argmin(-1)
        choice = torch.where(torch.isfinite(best),choice,fallback)
        delta = candidate[torch.arange(ids.numel(),device=q.device),choice]
        result[ids,side,:2] += delta
        feasible[ids,side] = torch.isfinite(best)
    relative = (root_inverse[:,None]@(result-root[:,None])[...,None]).squeeze(-1)
    torque = (kin.support_torque(q,forces_root,relative,-root_inverse[...,2]) if support_pose is None
              else cached_support_torque(support_pose,forces_root,relative))
    feasible &= (torque.abs()<=effort_limits+.02).all(-1)
    return result,torque,feasible


def closest_2d_intersection(a,b):
    """Minimum norm delta in {a delta <= b}; empty sets return finite zero/False.

    No physical inequality is relaxed. The caller supplies inward rounding
    protection for newly projected boundary points and rechecks full physics.
    """
    n,m,_=a.shape
    eps=torch.finfo(a.dtype).eps
    norm=a.norm(dim=-1)
    active=norm>eps*32
    impossible=((~active)&(b<0)).any(-1)
    unit=a/norm.clamp_min(eps*32)[...,None]
    bound=torch.where(active,b/norm.clamp_min(eps*32),torch.ones_like(b))
    i,j=torch.triu_indices(m,m,1,device=a.device)
    first,second=unit[:,i],unit[:,j]
    det=first[...,0]*second[...,1]-first[...,1]*second[...,0]
    nonparallel=det.abs()>eps*64
    den=torch.where(nonparallel,det,torch.ones_like(det))
    crossing=torch.stack((bound[:,i]*second[...,1]-bound[:,j]*first[...,1],
                          first[...,0]*bound[:,j]-second[...,0]*bound[:,i]),-1)/den[...,None]
    candidate=torch.cat((torch.zeros_like(unit[:,:1]),unit*bound[...,None],crossing),1)
    usable=torch.cat((torch.ones(n,1,dtype=torch.bool,device=a.device),active,nonparallel),-1)
    residual=torch.einsum('nmd,nkd->nkm',unit,candidate)-bound[:,None]
    # This tolerance only absorbs evaluation of an already inward-guarded
    # normalized intersection. Returned native caps are checked unrelaxed.
    valid=usable&(residual.amax(-1)<=eps*64)&~impossible[:,None]
    best,which=torch.where(valid,candidate.square().sum(-1),float('inf')).min(-1)
    found=torch.isfinite(best)
    delta=candidate[torch.arange(n,device=a.device),which]
    return torch.where(found[:,None],delta,0),found


def ss_forces_and_torque(kin,q,root_inverse,points,root,com,mass,vertical,omega2,support):
    n=q.shape[0];row=torch.arange(n,device=q.device)
    cop=points[row,support,:2]
    grf=torch.cat((omega2[:,None]*(com[:,:2]-cop),vertical[:,None]),-1)*mass[:,None]
    forces=torch.zeros_like(points)
    forces[row,support]=(root_inverse@grf[...,None]).squeeze(-1)
    relative=(root_inverse[:,None]@(points-root[:,None])[...,None]).squeeze(-1)
    torque=kin.support_torque(q,forces,relative,-root_inverse[...,2])
    return forces,torque


@torch.no_grad()
def ss_affine_native_project(kin,q,root_inverse,points,root,com,mass,vertical,omega2,
                             support,soles,foot_rotation,effort_limits,corners):
    """Project only loaded sole pressure; original native limits/8mm are fixed.

    The unloaded leg retains its own gravity-compensation cap check. Its
    arbitrary no-contact pressure marker is not a support constraint.
    Valid physical requests are returned bit-for-bit unchanged. An empty
    coupled intersection is explicit False, never an enlarged cap or a
    silently clipped torque. Pure single-support only; no DS fallback.
    """
    n=q.shape[0];row=torch.arange(n,device=q.device)
    support=support.long()
    if q.shape!=(n,2,6) or points.shape!=(n,2,3) or support.shape!=(n,):
        raise ValueError('Expected N by2 by6 q, N by2 by3 points and N support IDs')
    if not bool(((support==0)|(support==1)).all()):raise ValueError('Invalid support foot')
    if not bool((mass>0).all()&(vertical>0).all()&(omega2>0).all()):
        raise ValueError('Positive mass/vertical/omega2 are required')
    force,tau0=ss_forces_and_torque(kin,q,root_inverse,points,root,com,mass,vertical,omega2,support)
    foot=foot_rotation[row,support]
    half=corners[support,:,:2].abs().amax(-2)-.008
    local=(foot.transpose(-1,-2)@(points[row,support]-soles[row,support])[...,None]).squeeze(-1)[:,:2]
    valid_request=(local.abs()<=half).all(-1)&(tau0.abs()<=effort_limits).all((1,2))
    _,_,_,origins,axes,_=kin.forward(q,frames=True)
    point_root=(root_inverse@(points[row,support]-root)[...,None]).squeeze(-1)
    arm=point_root[:,None]-origins[row,support]
    basis=root_inverse[:,:,:2].transpose(-1,-2)
    dforce=-mass[:,None,None]*omega2[:,None,None]*basis
    moment=(torch.cross(basis[:,None],force[row,support,None,None],dim=-1)
            +torch.cross(arm[:,:,None],dforce[:,None],dim=-1))
    derivative=-(axes[row,support,:,None]*moment).sum(-1)
    geometry=foot.transpose(-1,-2)[:,:2,:2]
    ts=tau0[row,support];limit=effort_limits[row,support]
    a=torch.cat((derivative,-derivative,geometry,-geometry),1)
    b=torch.cat((limit-ts,limit+ts,half-local,half+local),-1)
    eps=torch.finfo(q.dtype).eps
    # Tightening, not enlargement, protects a newly computed boundary point
    # against floating-point cross products and casting in this same dtype.
    torque_guard=32*eps*(1+limit+ts.abs())
    point_scale=1+points[row,support].abs().amax(-1)+soles[row,support].abs().amax(-1)
    geometry_guard=32*eps*point_scale[:,None].expand(-1,2)
    protected=b-torch.cat((torque_guard,torque_guard,geometry_guard,geometry_guard),-1)
    delta,nonempty=closest_2d_intersection(a,protected)
    result=points.clone()
    result[row,support,:2]+=delta
    result=torch.where(valid_request[:,None,None],points,result)
    force,tau=ss_forces_and_torque(kin,q,root_inverse,result,root,com,mass,vertical,omega2,support)
    final_local=(foot.transpose(-1,-2)@(result[row,support]-soles[row,support])[...,None]).squeeze(-1)[:,:2]
    other=1-support
    unloaded_ok=(tau0[row,other].abs()<=effort_limits[row,other]).all(-1)
    feasible=(valid_request|nonempty)&unloaded_ok
    feasible&=(final_local.abs()<=half).all(-1)&(tau.abs()<=effort_limits).all((1,2))
    # Never hand an empty/inexact candidate to a caller as a valid pressure.
    result=torch.where(feasible[:,None,None],result,points)
    _,final_tau=ss_forces_and_torque(kin,q,root_inverse,result,root,com,mass,vertical,omega2,support)
    return result,final_tau,feasible,dict(valid_request=valid_request,intersection_nonempty=nonempty,
        unloaded_gravity_within_cap=unloaded_ok,total_derivative=derivative,tau_at_request=tau0,
        original_native_limits=effort_limits,maximum_final_cap_excess=(final_tau.abs()-effort_limits).amax((1,2)),
        support_geometry_margin=(half-final_local.abs()).amin(-1),
        numerical_inward_torque_guard=torque_guard,numerical_inward_geometry_guard=geometry_guard)


def prepare_support_pose(kin, q, gravity_direction=None):
    """One actual pose for this compose only; never reuse after q/root changes.

    Cache full gravity moment vectors before their joint-axis projection. This
    retains support_torque's exact (own+reaction)*axis arithmetic, rather than
    separately projecting gravity/contact and changing the reduction order.
    """
    _, _, _, origins, axes, children = kin.forward(q, frames=True)
    centers = (children[..., :3, :3] @ kin.com_local[None, ..., None]).squeeze(-1) + children[..., :3, 3]
    if gravity_direction is None:
        gravity = torch.zeros_like(centers)
        gravity[..., 2] = -9.81 * kin.mass
    else:
        gravity = 9.81 * gravity_direction[:, None, None, :] * kin.mass[None, ..., None]
    own = []
    for i in range(6):
        own.append(torch.cross(centers[..., i:, :] - origins[..., i, None, :],
                               gravity[..., i:, :], dim=-1).sum(-2))
    return dict(q=q, origins=origins, axes=axes, gravity_moment=torch.stack(own, -2))


def cached_support_torque(pose, force, contact_point):
    """Same inverse statics for changed force/point at the cached actual pose."""
    torque = torch.zeros_like(pose["q"])
    for i in range(6):
        reaction = torch.cross(contact_point - pose["origins"][..., i, :], force, dim=-1)
        torque[..., i] = -((pose["gravity_moment"][..., i, :] + reaction) * pose["axes"][..., i, :]).sum(-1)
    return torque
