"""Whole-leg geometric and loaded-drive screen for a direct observed entry.

This replaces a Cartesian stride heuristic, not any physical joint limit.
XY is planned in total-COM space and Z is planned in pelvis-height space.
Future total COMZ follows the leg mass moment; it is not fixed while a leg lifts.
The measured-root swing IK, contact, pressure and independent landing gates
remain active at runtime.
"""

import torch
from .stair_support import project_wrench_points
from .stair_lift import pre_riser_lift
from .stair_trajectory import smooth
from .stair_load import allocate_two_feet
from .stair_impedance import stair_gain_factor
from .stair_swing_ik import _separation_margin


def _rotation(q):
    w,x,y,z = (q/q.norm(dim=-1,keepdim=True).clamp_min(1e-9)).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


@torch.no_grad()
def direct_entry(gait, ids, soles, pair_world, *, chunk_size=8):
    """Screen both swings, split-height DS, and the subsequent body rise.

    Nine bounded yaw candidates per foot are checked over the complete route.
    Both legs share each future pelvis pose. The chosen combined solution is
    also checked with each pose's real mass, native effort limits and projected
    sole pressure points; gravity preload must stay inside soft joint limits.
    No current-floor cells are invented: the retained feet are real contacts,
    and pair_world must already have passed the live observed-plane selector.
    """
    n = ids.numel()
    joint_reserve=gait.options.get("joint_kinematic_reserve",.008)
    accepted = torch.zeros(n,dtype=torch.bool,device=gait.device)
    root_height = gait.time.new_full((n,),gait.options["pelvis_height"])
    selected_yaw = gait.time.new_zeros((n,2))
    support_backshift = gait.time.new_full((n,),.035)
    com_error_out = gait.time.new_full((n,),float('inf'))
    root_height_error_out = gait.time.new_full((n,),float('inf'))
    worst_error = gait.time.new_full((n,),float('inf'))
    worst_joint = gait.time.new_full((n,),-float('inf'))
    heading_all = _rotation(gait.flat.heading[ids])
    feet_local = (heading_all[:,None].transpose(-1,-2)@soles[ids,:,:,None]).squeeze(-1)
    next_side = feet_local[:,:,0].argmin(-1)
    legal_order = next_side != gait.flat.stance[ids]
    matrix = gait.robot.root_physx_view.get_generalized_mass_matrices()
    size = gait.robot.num_joints+6
    if matrix.shape != (gait.n,size,size):
        raise ValueError("Direct entry requires native floating-base inertia")
    yaw_grid = torch.linspace(-.4,.4,9,device=gait.device)
    # The lower values are reachability backtracking, never a new stair size.
    height_grid = (gait.options["pelvis_height"],.925,.90,.875,.85)
    def leg_moment(q):
        children=gait.kin.forward(q,frames=True)[-1]
        centres=children[...,:3,3]+(children[...,:3,:3]@gait.kin.com_local[None,...,None]).squeeze(-1)
        return (centres*gait.kin.mass[None,...,None]).sum((1,2))
    for offset in range(0,n,chunk_size):
        at = torch.arange(offset,min(offset+chunk_size,n),device=gait.device)
        sub = ids[at]
        count = sub.numel()
        row = torch.arange(count,device=gait.device)
        lead = next_side[at]
        trail = 1-lead
        measured_feet = soles[sub].clone()
        feet = measured_feet.clone()
        plane = gait.flat.origin[sub,2]
        feet[:,:,2] = plane[:,None]-gait.cfg["ground_contact_depth"]
        target = pair_world[at].clone()
        target[:,:,2] -= gait.cfg["ground_contact_depth"]
        heading = _rotation(gait.flat.heading[sub])
        inverse = heading.transpose(-1,-2)
        actual_rotation = _rotation(gait.robot.data.body_quat_w[sub][:,gait.flat.feet].reshape(-1,4)).reshape(count,2,3,3)
        old_rotation = inverse[:,None]@actual_rotation
        # _entry_plan runs inside flat.compute(), before the stair owner's
        # actual_com is refreshed. Use the just-measured articulation COM.
        com = (gait.robot.data.body_com_pos_w[sub]*gait.masses[sub,:,None]).sum(1)/gait.total_mass[sub,None]
        com_offset = com-gait.robot.data.root_pos_w[sub]
        measured = gait.robot.data.joint_pos[sub][:,gait.joint_ids].reshape(count,2,6)
        actual_root_rotation=_rotation(gait.robot.data.root_quat_w[sub])
        actual_com_body=(actual_root_rotation.transpose(-1,-2)@com_offset[...,None]).squeeze(-1)
        upper_moment=gait.total_mass[sub,None]*actual_com_body-leg_moment(measured)
        medial=gait.options["support_medial_offset"]
        support_xy = feet[row,trail,:2].clone()
        support_xy[:,1] += torch.where(trail==0,-medial,medial)
        high_xy = target[row,lead,:2].clone()
        high_xy[:,1] += torch.where(lead==0,-medial,medial)
        paths,centres,turned,loads,tilts = [],[],[],[],[]
        tilt=torch.zeros(count,2,device=gait.device)
        def add(foot,xy,turn,load):
            paths.append(foot.clone());centres.append(xy.clone())
            turned.append(turn.clone());loads.append(load.clone())
            tilts.append(tilt.clone())
        turn = torch.zeros(count,2,dtype=torch.bool,device=gait.device)
        half = gait.time.new_full((count,2),.5)
        for u in (0.,.5,1.):
            add(feet,torch.lerp(com[:,:2],support_xy,u),turn,half)
        working = feet.clone()
        for side,body_xy in ((lead,support_xy),(trail,high_xy)):
            other = 1-side
            start = working[row,side].clone()
            start[:,2] = plane
            end = target[row,side]
            peak = torch.maximum(start[:,2],end[:,2])+gait.options["clearance"]
            lifted,_ = pre_riser_lift(gait.map,sub,start,end,gait.foot_corners[side],
                                      max_clearance=.03,margin=.02)
            high = end.clone();high[:,2] = peak
            turn[row,side] = True
            single = torch.zeros_like(half);single[row,other] = 1
            middle=.5*(lifted+high)
            middle[:,2]=torch.lerp(lifted[:,2],peak,smooth(torch.full_like(peak,.5/.65)))
            amplitude=gait.options["swing_toe_down_rad"]
            point_pitch=(0.,amplitude,amplitude*(17/81),0.,0.,0.)
            for point,pitch_value in zip((start,lifted,middle,high,.5*(high+end),end),point_pitch):
                tilt[row,side]=pitch_value
                pose = working.clone();pose[row,side] = point
                add(pose,body_xy,turn,single)
            tilt[row,side]=0
            working[row,side] = end
            if bool((side==lead).all()):
                for u in (0.,.25,.5,.75,1.):
                    load = torch.zeros_like(half);load[row,trail] = 1-u;load[row,lead] = u
                    add(working,torch.lerp(support_xy,high_xy,u),turn,load)
        # Both feet are now on the first tread; restore leg extension there.
        for u in (0.,.5,1.):
            add(working,high_xy,turn,half)
        feet_path = torch.stack(paths,1)
        nominal_xy = torch.stack(centres,1)
        rotate_now = torch.stack(turned,1)
        load = torch.stack(loads,1)
        pitch_path=torch.stack(tilts,1)
        poses = feet_path.shape[1]
        rise = target[:,:,2].mean(-1)-feet[:,:,2].mean(-1)
        body_rise = gait.time.new_zeros(count,poses)
        body_rise[:,-3:] = rise[:,None]*gait.time.new_tensor([0.,.5,1.])[None]
        k = gait.robot.data.joint_stiffness[sub][:,gait.joint_ids].reshape(count,2,6)
        native_joints = torch.as_tensor(gait.joint_ids,device=gait.device)+6
        inertia = matrix[sub[:,None],native_joints[None],native_joints[None]].reshape(count,2,6)
        factor=stair_gain_factor(k.reshape(count,12),inertia.reshape(count,12),gait.options["stair_drive_frequency_rad_s"])
        k=k*factor.reshape(count,2,6)
        limits = gait.robot.data.joint_effort_limits[sub][:,gait.joint_ids].reshape(count,2,6)

        def footprint_evidence(world_position,world_rotation):
            # Candidate axes are [environment,yaw,pose,leg]. Certify each
            # leg at its own landing; retained measured floor contacts do not
            # invent LiDAR evidence in the sensor's near-field blind area.
            angles=world_position.shape[1]
            corners=world_position[...,None,:]+(world_rotation[...,None,:,:]
                @gait.foot_corners[None,None,None,...,None]).squeeze(-1)
            landing=torch.where(lead[:,None]==torch.arange(2,device=gait.device)[None],8,19)
            landing_corners=corners.permute(0,1,3,2,4,5)[
                row[:,None,None],torch.arange(angles,device=gait.device)[None,:,None],
                torch.arange(2,device=gait.device)[None,None],landing[:,None]]
            covered=gait.map.footprint_supported(
                sub[:,None,None].expand(-1,angles,2).reshape(-1),
                landing_corners.reshape(-1,4,3),
                (target[:,:,2]+gait.cfg["ground_contact_depth"])[:,None].expand(-1,angles,-1).reshape(-1),
                margin=.008).reshape(count,angles,2)
            return corners,covered

        def collision_pairs(corners,world_position,world_rotation):
            # Reuse the runtime rectangle SAT rule, including the actual
            # foot orientation. Test all nine-by-nine yaw combinations, not
            # two independent choices that could point into one another.
            angles=corners.shape[1]
            shape=(count,angles,angles,poses)
            left=corners[:,:,:,0][:,:,None].expand(shape+(4,3))
            right=corners[:,:,:,1][:,None].expand(shape+(4,3))
            left_r=world_rotation[:,:,:,0][:,:,None].expand(shape+(3,3))
            right_r=world_rotation[:,:,:,1][:,None].expand(shape+(3,3))
            gap=_separation_margin(left,right,left_r,right_r)
            same_level=(world_position[:,:,:,0,2][:,:,None]
                        -world_position[:,:,:,1,2][:,None]).abs()<.04
            return ((~same_level)|(gap>.008)).all(-1)

        def capture_evidence(world_rotation,desired,anchors,weights):
            # Match the runtime static limit of its horizontal foot-frame
            # capture gate. COM velocity is zero only for these support-pose
            # samples; this is not a prediction of dynamic capture success.
            delta=desired[:,:,:2][:,:,None]-anchors[...,:2]
            foot_delta=(world_rotation[...,:2,:2].transpose(-1,-2)
                        @delta[:,None,...,None]).squeeze(-1)
            single=(weights>0).sum(-1)==1
            support=(weights>0)&single[...,None]
            fits=(foot_delta[...,0].abs()<.085)&(foot_delta[...,1].abs()<.035)
            return ((~support[:,None])|fits).all(2),foot_delta

        def select_loaded_yaws(q_candidates,orientations,pelvis,position,*,coupled_check=False):
            angles=q_candidates.shape[1]
            achieved,rotation=gait.kin.forward(q_candidates.reshape(-1,2,6))
            achieved=achieved.reshape(count,angles,poses,2,3)
            rotation=rotation.reshape(count,angles,poses,2,3,3)
            error=(achieved-position[:,None]).norm(dim=-1)
            rotation_error=(rotation-orientations).abs().amax((-1,-2))
            joint_margin=torch.minimum(q_candidates-gait.kin.lower,
                                       gait.kin.upper-q_candidates).amin(-1)
            expand=lambda value:value[:,None].expand(-1,angles,*value.shape[1:])
            points,tau,wrench_ok=project_wrench_points(
                gait.kin,q_candidates.reshape(-1,2,6),
                inverse[:,None,None].expand(-1,angles,poses,-1,-1).reshape(-1,3,3),
                expand(force_root).reshape(-1,2,3),expand(pressure).reshape(-1,2,3),
                expand(pelvis).reshape(-1,3),expand(feet_path).reshape(-1,2,3),
                (heading[:,None,None,None]@rotation).reshape(-1,2,3,3),
                limits[:,None,None].expand(-1,angles,poses,-1,-1).reshape(-1,2,6),gait.foot_corners)
            drive=q_candidates+tau.reshape(count,angles,poses,2,6)/k[:,None,None]
            drive_margin=torch.minimum(drive-gait.kin.lower,gait.kin.upper-drive).amin(-1)
            wrench_ok=wrench_ok.reshape(count,angles,poses,2)
            world_position=pelvis[:,None,:,None]+(heading[:,None,None,None]@achieved[...,None]).squeeze(-1)
            world_rotation=heading[:,None,None,None]@rotation
            corners,covered=footprint_evidence(world_position,world_rotation)
            collision_free=collision_pairs(corners,world_position,world_rotation)
            capture_ok,_=capture_evidence(world_rotation,desired_com,feet_path,load)
            valid=((error<.001)&(rotation_error<.005)&(joint_margin>joint_reserve)
                   &(drive_margin>.004)&wrench_ok).all(2)&covered&capture_ok
            quality=torch.minimum(joint_margin.amin(2),drive_margin.amin(2))
            pairs=valid[:,:,None,0]&valid[:,None,:,1]&collision_free
            score=torch.minimum(quality[:,:,None,0],quality[:,None,:,1])
            score-=.01*(yaw_grid.abs()[None,:,None]+yaw_grid.abs()[None,None,:])
            best,pair=torch.where(pairs,score,-float('inf')).flatten(1).max(-1)
            # An infeasible coarse route can still seed the shared-COM
            # correction. Choose its least violating bounded pair instead
            # of silently falling back to the first (-0.4 rad) yaw.
            penalty=(50*error.amax(2)+rotation_error.amax(2)
                     +10*(joint_reserve-joint_margin.amin(2)).clamp_min(0)
                     +10*(.004-drive_margin.amin(2)).clamp_min(0)
                     +(~covered).float()+(~capture_ok).float()+(~wrench_ok.all(2)).float())
            pair_penalty=penalty[:,:,None,0]+penalty[:,None,:,1]+(~collision_free).float()
            pair=torch.where(torch.isfinite(best),pair,pair_penalty.flatten(1).argmin(-1))
            choice=torch.stack((pair//angles,pair%angles),-1)
            selected=q_candidates.permute(0,3,1,2,4)[
                row[:,None],torch.arange(2,device=gait.device)[None],choice].transpose(1,2)
            selected_rotation=orientations.permute(0,3,1,2,4,5)[
                row[:,None],torch.arange(2,device=gait.device)[None],choice].transpose(1,2)
            if coupled_check:
                # Retain the original one-dimensional argsort per environment,
                # including its tie ordering, but run shared-COM/IK for the
                # next <=16 candidates of all unfinished environments together.
                ordered_rows=[]
                for env_row in range(count):
                    ordered=torch.where(pairs[env_row],score[env_row],-float('inf')).flatten().argsort(descending=True)
                    ordered_rows.append(ordered[pairs[env_row].flatten()[ordered]])
                maximum=max((ordered.numel() for ordered in ordered_rows),default=0)
                if maximum:
                    lengths=torch.tensor([ordered.numel() for ordered in ordered_rows],
                                         dtype=torch.long,device=gait.device)
                    ordered_table=torch.zeros((count,maximum),dtype=torch.long,device=gait.device)
                    for env_row,ordered in enumerate(ordered_rows):
                        ordered_table[env_row,:ordered.numel()]=ordered
                    found_env=torch.zeros(count,dtype=torch.bool,device=gait.device)
                    slots=torch.arange(16,device=gait.device)
                    for begin in range(0,maximum,16):
                        active_env=((~found_env)&(lengths>begin)).nonzero().flatten()
                        if not active_env.numel():break
                        ranks=(begin+slots)[None].expand(active_env.numel(),-1)
                        use=ranks<lengths[active_env,None]
                        pair_env=active_env[:,None].expand(-1,16)[use]
                        trial=ordered_table[pair_env,ranks[use]]
                        number=trial.numel()
                        left,right=trial//angles,trial%angles
                        pair_checks.index_add_(0,pair_env,torch.ones_like(pair_env))
                        pair_q=torch.stack((q_candidates[pair_env,left,:,0],
                                            q_candidates[pair_env,right,:,1]),2)
                        pair_rotation=torch.stack((orientations[pair_env,left,:,0],
                                                   orientations[pair_env,right,:,1]),2)
                        pair_heading=heading[pair_env]
                        pair_inverse=inverse[pair_env]
                        pair_feet=feet_path[pair_env]
                        for _ in range(4):
                            future=(upper_moment[pair_env,None]+leg_moment(pair_q.reshape(-1,2,6))
                                    .reshape(number,poses,3))/gait.total_mass[sub[pair_env],None,None]
                            pair_offset=(pair_heading[:,None]@future[...,None]).squeeze(-1)
                            pair_pelvis=torch.cat((desired_com[pair_env,:,:2]-pair_offset[...,:2],
                                planned_root_z[pair_env,:,None]),dim=-1)
                            pair_position=(pair_inverse[:,None,None]@(pair_feet-pair_pelvis[:,:,None])[...,None]).squeeze(-1)
                            pair_q,_=gait.kin.solve(pair_position.reshape(-1,2,3),pair_rotation.reshape(-1,2,3,3),
                                                   pair_q.reshape(-1,2,6),iterations=12)
                            pair_q=pair_q.reshape(number,poses,2,6)
                        final_position,final_rotation=gait.kin.forward(pair_q.reshape(-1,2,6))
                        final_position=final_position.reshape(number,poses,2,3)
                        final_rotation=final_rotation.reshape(number,poses,2,3,3)
                        pe=(final_position-pair_position).norm(dim=-1).amax((1,2))
                        re=(final_rotation-pair_rotation).abs().amax((1,2,3,4))
                        jm=torch.minimum(pair_q-gait.kin.lower,gait.kin.upper-pair_q).amin((1,2,3))
                        future=(upper_moment[pair_env,None]+leg_moment(pair_q.reshape(-1,2,6))
                                .reshape(number,poses,3))/gait.total_mass[sub[pair_env],None,None]
                        pair_com=pair_pelvis+(pair_heading[:,None]@future[...,None]).squeeze(-1)
                        ce_xy=(pair_com[...,:2]-desired_com[pair_env,:,:2]).norm(dim=-1).amax(-1)
                        ze=(pair_pelvis[...,2]-planned_root_z[pair_env]).abs().amax(-1)
                        world_r=pair_heading[:,None,None]@final_rotation
                        world_p=pair_pelvis[:,:,None]+(pair_heading[:,None,None]@final_position[...,None]).squeeze(-1)
                        pair_corners=world_p[...,None,:]+(world_r[...,None,:,:]
                            @gait.foot_corners[None,None,...,None]).squeeze(-1)
                        landing=torch.where(lead[pair_env,None]==torch.arange(2,device=gait.device)[None],8,19)
                        landing_corners=pair_corners[torch.arange(number,device=gait.device)[:,None],
                                                     landing,torch.arange(2,device=gait.device)[None]]
                        covered=gait.map.footprint_supported(sub[pair_env,None].expand(-1,2).reshape(-1),
                            landing_corners.reshape(-1,4,3),
                            (target[pair_env,:,2]+gait.cfg["ground_contact_depth"]).reshape(-1),
                            margin=.008).reshape(number,2).all(-1)
                        gap=_separation_margin(pair_corners[:,:,0],pair_corners[:,:,1],world_r[:,:,0],world_r[:,:,1])
                        same_level=(world_p[:,:,0,2]-world_p[:,:,1,2]).abs()<.04
                        apart=((~same_level)|(gap>.008)).all(-1)
                        capture_ok,_=capture_evidence(world_r[:,None],desired_com[pair_env],
                            pair_feet,load[pair_env])
                        _,tau,feasible=project_wrench_points(gait.kin,pair_q.reshape(-1,2,6),
                            pair_inverse[:,None].expand(-1,poses,-1,-1).reshape(-1,3,3),
                            force_root[pair_env].reshape(-1,2,3),
                            pressure[pair_env].reshape(-1,2,3),pair_pelvis.reshape(-1,3),
                            pair_feet.reshape(-1,2,3),world_r.reshape(-1,2,3,3),
                            limits[pair_env,None].expand(-1,poses,-1,-1).reshape(-1,2,6),gait.foot_corners)
                        request=pair_q+tau.reshape(number,poses,2,6)/k[pair_env,None]
                        dm=torch.minimum(request-gait.kin.lower,gait.kin.upper-request).amin((1,2,3))
                        valid_pair=((pe<.001)&(re<.005)&(jm>joint_reserve)&(dm>.004)&(ce_xy<.001)&(ze<1e-6)
                                    &feasible.reshape(number,poses,2).all((1,2))&covered&apart&capture_ok[:,0].all(-1))
                        # Flattening is environment-major and score-order-major.
                        # The first passing row for each owner is therefore the
                        # exact first-valid candidate of the serial algorithm.
                        first=torch.full((count,),number,dtype=torch.long,device=gait.device)
                        candidate=torch.arange(number,device=gait.device)
                        first.scatter_reduce_(0,pair_env,torch.where(valid_pair,candidate,number),
                                              reduce="amin",include_self=True)
                        accepted_env=(first<number).nonzero().flatten()
                        found=first[accepted_env]
                        choice[accepted_env]=torch.stack((left[found],right[found]),-1)
                        selected[accepted_env]=pair_q[found]
                        selected_rotation[accepted_env]=pair_rotation[found]
                        found_env[accepted_env]=True
            return choice,selected,selected_rotation

        for backshift,height in ((x,z) for x in (.035,.055) for z in height_grid):
            remaining = ~accepted[at]
            if not remaining.any():break
            body_xy=nominal_xy.clone()
            pair_checks=torch.zeros(count,dtype=torch.long,device=gait.device)
            weight=gait.time.new_ones(poses);weight[:3]=gait.time.new_tensor([0.,.5,1.])
            body_xy[:,:,0]-=backshift*weight[None]
            planned_root_z=plane[:,None]+height+body_rise
            # desired_com is queried only in XY. Its initial Z is a seed,
            # never a controlled total-COMZ target or an acceptance constraint.
            desired_com=torch.cat((body_xy,(planned_root_z+com_offset[:,2,None])[:,:,None]),-1)
            pelvis = torch.cat((body_xy-com_offset[:,None,:2],planned_root_z[:,:,None]),-1)
            position = (inverse[:,None,None]@(feet_path-pelvis[:,:,None])[...,None]).squeeze(-1)
            orientation = old_rotation[:,None,None].expand(-1,9,poses,-1,-1,-1).clone()
            c,s = yaw_grid.cos(),yaw_grid.sin()
            rz = gait.time.new_zeros(9,3,3)
            rz[:,0,0],rz[:,0,1],rz[:,1,0],rz[:,1,1],rz[:,2,2] = c,-s,s,c,1
            candidate = gait.flat.home_foot_rotation[None]@rz[:,None]
            orientation = torch.where(rotate_now[:,None,:,:,None,None],
                                      candidate[None,:,None],orientation)
            ry=gait.time.new_zeros(count,poses,2,3,3)
            ry[...,0,0]=ry[...,2,2]=pitch_path.cos()
            ry[...,0,2],ry[...,2,0],ry[...,1,1]=pitch_path.sin(),-pitch_path.sin(),1
            orientation=orientation@ry[:,None]
            actual_load=load.clone()
            bounds=gait.time.new_tensor([.06,.025])
            allocated,_,_=allocate_two_feet(body_xy.reshape(-1,2),feet_path[:,:,:,:2].reshape(-1,2,2),bounds)
            multi=(load>0).sum(-1)==2
            actual_load=torch.where(multi[...,None],allocated.reshape(count,poses,2),actual_load)
            centre=(feet_path[:,:,:,:2]*actual_load[...,None]).sum(2)
            pressure=feet_path.clone();pressure[:,:,:,:2]+=(body_xy-centre)[:,:,None]
            force_world = torch.zeros_like(feet_path)
            force_world[...,2] = gait.total_mass[sub,None,None]*9.81*actual_load
            force_root = (inverse[:,None,None]@force_world[...,None]).squeeze(-1)
            desired = position[:,None].expand(-1,9,-1,-1,-1)
            seed = measured[:,None,None].expand(-1,9,poses,-1,-1)
            q,_ = gait.kin.solve(desired.reshape(-1,2,3),orientation.reshape(-1,2,3,3),
                                seed.reshape(-1,2,6),iterations=24)
            q = q.reshape(count,9,poses,2,6)
            choice,chosen,chosen_rotation=select_loaded_yaws(q,orientation,pelvis,position,coupled_check=True)
            # Recompute one shared pelvis from the complete chosen two-leg
            # mass moment. Individual yaw choices must not imply two pelvises
            # or reuse a frozen COM offset during a rear-leg lift.
            for selection_round in range(3):
                if selection_round:
                    desired=position[:,None].expand(-1,9,-1,-1,-1)
                    seed=chosen[:,None].expand(-1,9,-1,-1,-1)
                    q,_=gait.kin.solve(desired.reshape(-1,2,3),orientation.reshape(-1,2,3,3),
                                       seed.reshape(-1,2,6),iterations=24)
                    q=q.reshape(count,9,poses,2,6)
                    choice,chosen,chosen_rotation=select_loaded_yaws(q,orientation,pelvis,position,coupled_check=True)
                for _ in range(4):
                    future_body=(upper_moment[:,None]+leg_moment(chosen.reshape(-1,2,6)).reshape(count,poses,3))/gait.total_mass[sub,None,None]
                    future_offset=(heading[:,None]@future_body[...,None]).squeeze(-1)
                    pelvis=torch.cat((desired_com[...,:2]-future_offset[...,:2],planned_root_z[:,:,None]),-1)
                    position=(inverse[:,None,None]@(feet_path-pelvis[:,:,None])[...,None]).squeeze(-1)
                    chosen,_=gait.kin.solve(position.reshape(-1,2,3),chosen_rotation.reshape(-1,2,3,3),
                        chosen.reshape(-1,2,6),iterations=12)
                    chosen=chosen.reshape(count,poses,2,6)
            actual_position,actual_rotation=gait.kin.forward(chosen.reshape(-1,2,6))
            chosen_position_error=(actual_position.reshape(count,poses,2,3)-position).norm(dim=-1)
            chosen_rotation_error=(actual_rotation.reshape(count,poses,2,3,3)-chosen_rotation).abs().amax((-1,-2))
            joint_margin=torch.minimum(chosen-gait.kin.lower,gait.kin.upper-chosen).amin((1,2,3))
            final_body=(upper_moment[:,None]+leg_moment(chosen.reshape(-1,2,6)).reshape(count,poses,3))/gait.total_mass[sub,None,None]
            future_com=pelvis+(heading[:,None]@final_body[...,None]).squeeze(-1)
            com_error=(future_com[...,:2]-desired_com[...,:2]).norm(dim=-1).amax(-1)
            root_height_error=(pelvis[...,2]-planned_root_z).abs().amax(-1)
            # Re-evaluate after the shared COM iteration; coarse yaw geometry
            # is a seed, not a substitute for the final coupled pose check.
            both=remaining&legal_order[at]&(chosen_position_error.amax((1,2))<.001)
            both&=(chosen_rotation_error.amax((1,2))<.005)&(joint_margin>joint_reserve)&(com_error<.001)&(root_height_error<1e-6)
            actual_position=actual_position.reshape(count,poses,2,3)
            actual_rotation=actual_rotation.reshape(count,poses,2,3,3)
            world_position=pelvis[:,:,None]+(heading[:,None,None]@actual_position[...,None]).squeeze(-1)
            world_rotation=heading[:,None,None]@actual_rotation
            final_corners,final_covered=footprint_evidence(world_position[:,None],world_rotation[:,None])
            final_collision_free=collision_pairs(final_corners,world_position[:,None],world_rotation[:,None])[:,0,0]
            final_capture_fit,final_capture_delta=capture_evidence(world_rotation[:,None],desired_com,feet_path,load)
            final_capture_ok=final_capture_fit[:,0].all(-1)
            both&=final_covered[:,0].all(-1)&final_collision_free&final_capture_ok
            points,torque,feasible = project_wrench_points(gait.kin,chosen.reshape(-1,2,6),
                inverse[:,None].expand(-1,poses,-1,-1).reshape(-1,3,3),
                force_root.reshape(-1,2,3),pressure.reshape(-1,2,3),pelvis.reshape(-1,3),
                feet_path.reshape(-1,2,3),(heading[:,None,None]@chosen_rotation).reshape(-1,2,3,3),
                limits[:,None].expand(-1,poses,-1,-1).reshape(-1,2,6),gait.foot_corners)
            torque = torque.reshape(count,poses,2,6)
            drive = chosen+torque/k[:,None]
            drive_margin = torch.minimum(drive-gait.kin.lower,gait.kin.upper-drive).amin((-1,-2,-3))
            both &= feasible.reshape(count,poses,2).all((1,2))&(drive_margin>.004)
            accepted[at] |= both
            root_height[at] = torch.where(both,height,root_height[at])
            selected_yaw[at] = torch.where(both[:,None],yaw_grid[choice],selected_yaw[at])
            support_backshift[at]=torch.where(both,backshift,support_backshift[at])
            com_error_out[at]=torch.where(remaining,com_error,com_error_out[at])
            root_height_error_out[at]=torch.where(remaining,root_height_error,root_height_error_out[at])
            chosen_error = chosen_position_error.amax((1,2))
            chosen_margin = joint_margin
            worst_error[at] = torch.where(remaining,chosen_error,worst_error[at])
            worst_joint[at] = torch.where(remaining,chosen_margin,worst_joint[at])
    return dict(valid=accepted,root_height=root_height,yaw=selected_yaw,support_backshift=support_backshift,
                COM_error=com_error_out,COM_XY_error=com_error_out,root_height_error=root_height_error_out,
                leading=next_side,IK_error=worst_error,joint_margin=worst_joint)
