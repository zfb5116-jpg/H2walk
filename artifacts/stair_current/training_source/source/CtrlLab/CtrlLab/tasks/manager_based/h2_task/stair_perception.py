"""Persistent stair-surface evidence from finite real LiDAR mesh returns.

No reconstruction-filled cells, synthetic floor hits, high-altitude height
scanner or configured stair geometry enter this controller map. A separately
marked, non-propagating single-cell plane estimate may bridge bounded dropout.
"""

import torch
import torch.nn.functional as functional


class LidarTreadMap:
    def __init__(self, env, *, resolution=0.025):
        self.env, self.device, self.n = env, env.device, env.num_envs
        self.sensor = env.scene["lidar"]
        self.resolution = resolution
        self.x0, self.y0 = -1.0, -0.75
        self.nx, self.ny = 240, 60
        shape = (self.n, self.ny * self.nx)
        self.minimum = torch.full(shape, float("inf"), device=self.device)
        self.maximum = torch.full(shape, -float("inf"), device=self.device)
        self.first_seen = torch.full(shape, 1_000_000, device=self.device, dtype=torch.int32)
        self.last_seen = torch.full(shape, -1, device=self.device, dtype=torch.int32)
        self.clock = 0
        self.last_step = -1
        yy, xx = torch.meshgrid(torch.arange(self.ny, device=self.device),
                                torch.arange(self.nx, device=self.device), indexing="ij")
        self.x = self.x0 + (xx.flatten() + .5) * resolution
        self.plane_x = self.x
        self.y = self.y0 + (yy.flatten() + .5) * resolution
        self.cached_step = -1
        self.safe = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.surface = torch.zeros(shape, device=self.device)
        self.real_hit_count = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.plane_half_extent = torch.tensor([5.5 * resolution, 3.5 * resolution], device=self.device)
        self.modelled = torch.zeros(shape,dtype=torch.bool,device=self.device)
        self.model_height = torch.zeros(shape,device=self.device)
        self.tactile_known = torch.zeros(shape,dtype=torch.bool,device=self.device)
        self.tactile_invalid = torch.zeros_like(self.tactile_known)
        self.tactile_height = torch.zeros(shape,device=self.device)
        self.fusion_conflict = torch.zeros_like(self.tactile_known)
        self.tactile = None

    def connect_feet(self,robot,feet,mass,rotation_from_quat):
        from .tactile_support import TactileSupport
        self.tactile = TactileSupport(self.env,robot,feet,mass,rotation_from_quat)

    def _tactile_conflict(self):
        disagreement = (self.tactile_known & (self.last_seen >= 0)
                        & (((self.maximum-self.tactile_height) > .008)
                           | ((self.tactile_height-self.minimum) > .008)))
        return self.fusion_conflict | self.tactile_invalid | disagreement

    def _tactile_valid(self):
        return self.tactile_known & ~self._tactile_conflict()

    def _fused_cells(self, env_ids=None, indices=None):
        """One validity/provenance rule for selection and footprint checks."""
        select = lambda value: value if env_ids is None else value[env_ids[:, None], indices]
        low, high = select(self.minimum), select(self.maximum)
        last, first = select(self.last_seen), select(self.first_seen)
        touch_known, tz = select(self.tactile_known), select(self.tactile_height)
        conflict = select(self.fusion_conflict) | select(self.tactile_invalid)
        conflict |= (touch_known & (last >= 0) & (((high-tz) > .008) | ((tz-low) > .008)))
        touched = touch_known & ~conflict
        model = select(self.modelled) & (last < 0) & ~touch_known & ~conflict
        raw = (last-first >= 2) & ((high-low) < .022) & ~conflict
        # Compatible raw intervals remain evidence; contact does not erase their spread.
        touch_low = torch.where(last >= 0, torch.minimum(low, tz), tz)
        touch_high = torch.where(last >= 0, torch.maximum(high, tz), tz)
        low, high = torch.where(touched, touch_low, low), torch.where(touched, touch_high, high)
        estimate = select(self.model_height)
        low, high = torch.where(model, estimate, low), torch.where(model, estimate, high)
        valid = (raw | touched | model) & ~conflict
        return valid, low, high, model

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.minimum[ids] = float("inf")
        self.maximum[ids] = -float("inf")
        self.first_seen[ids] = 1_000_000
        self.last_seen[ids] = -1
        self.real_hit_count[ids] = 0
        self.modelled[ids] = False
        self.tactile_known[ids],self.tactile_invalid[ids] = False,False
        self.fusion_conflict[ids] = False
        if self.tactile is not None:
            self.tactile.reset(ids)
        self.cached_step = -1

    def update(self):
        step = int(self.env.common_step_counter)
        if step == self.last_step:
            return
        self.last_step = step
        self.clock += 1
        hits = self.sensor.data.ray_hits_w
        origins = self.env.scene.env_origins
        local = hits - origins[:, None]
        ranges = (hits - self.sensor.data.pos_w[:, None]).norm(dim=-1)
        finite = torch.isfinite(local).all(-1) & (ranges >= .1) & (ranges <= 2.0)
        # The map covers the forward approach/stair corridor in the fixed yaw
        # frame. This task has zero commanded yaw and does not claim turning.
        ix = torch.floor((local[..., 0] - self.x0) / self.resolution).long()
        iy = torch.floor((local[..., 1] - self.y0) / self.resolution).long()
        good = finite & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        good &= (local[..., 2] >= -.10) & (local[..., 2] < 2.0)
        index = (iy.clamp(0, self.ny - 1) * self.nx + ix.clamp(0, self.nx - 1))
        self.minimum.scatter_reduce_(1, index, torch.where(good, local[..., 2], float("inf")),
                                     reduce="amin", include_self=True)
        self.maximum.scatter_reduce_(1, index, torch.where(good, local[..., 2], -float("inf")),
                                     reduce="amax", include_self=True)
        self.first_seen.scatter_reduce_(1, index, torch.where(good, self.clock, 1_000_000).to(torch.int32),
                                        reduce="amin", include_self=True)
        self.last_seen.scatter_reduce_(1, index, torch.where(good, self.clock, -1).to(torch.int32),
                                       reduce="amax", include_self=True)
        self.real_hit_count += good.sum(-1)
        if self.tactile is not None:
            self.tactile.update(self)
        conflict = self._tactile_conflict()
        if bool((conflict & ~self.fusion_conflict).any()):
            self.cached_step = -1
        self.fusion_conflict |= conflict

    def _single_cell_models(self):
        # Use raw evidence only: neither another estimate nor a filled decoder
        # cell can support a new model. A run of missing cells stays unknown.
        conflict = self._tactile_conflict()
        for offset in range(0,self.n,64):
            sub = slice(offset,min(self.n,offset+64))
            low,high = self.minimum[sub],self.maximum[sub]
            raw = ((self.last_seen[sub]-self.first_seen[sub]>=2)&((high-low)<.008)
                   & ~conflict[sub])
            height = torch.where(raw,(low+high)*.5,0).view(-1,self.ny,self.nx)
            good = raw.view(-1,self.ny,self.nx)
            hp = functional.pad(height,(1,1,1,1),value=0)
            gp = functional.pad(good,(1,1,1,1),value=False)
            get = lambda a,dy,dx:a[:,1+dy:1+dy+self.ny,1+dx:1+dx+self.nx]
            north,south = get(hp,1,0),get(hp,-1,0)
            ns = get(gp,1,0)&get(gp,-1,0)
            east = torch.stack([get(hp,dy,1) for dy in (1,0,-1)],-1)
            west = torch.stack([get(hp,dy,-1) for dy in (1,0,-1)],-1)
            eok = ns & torch.stack([get(gp,dy,1) for dy in (1,0,-1)],-1).all(-1)
            wok = ns & torch.stack([get(gp,dy,-1) for dy in (1,0,-1)],-1).all(-1)
            side = torch.where(eok[...,None],east,west)
            z0,dy = (north+south)*.5,(north-south)*.5
            side_mean = side.mean(-1)
            residual = (side-torch.stack((side_mean+dy,side_mean,side_mean-dy),-1)).abs().amax(-1)
            support = torch.cat((north[...,None],south[...,None],side),-1)
            spread = support.amax(-1)-support.amin(-1)
            missing = (self.last_seen[sub]<0).view(-1,self.ny,self.nx)
            neighbor_conflict = torch.zeros_like(missing)
            for yy,xx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                neighbor_conflict |= get(gp,yy,xx)&((get(hp,yy,xx)-z0).abs()>=.008)
            allowed = (missing & (eok|wok) & (spread<.008) & (residual<.002) & ~neighbor_conflict
                       & ~self.tactile_known[sub].view_as(missing) & ~conflict[sub].view_as(missing))
            self.modelled[sub] = allowed.flatten(1)
            self.model_height[sub] = torch.where(allowed,z0,0).flatten(1)

    def surfaces(self):
        if self.clock - self.cached_step < 5:
            return self.safe, self.surface
        self._single_cell_models()
        flat, minimum, maximum, used_model = self._fused_cells()
        shape = (self.n, 1, self.ny, self.nx)
        # Eleven by seven observed25mm cells represent27.5x17.5cm. This
        # contains the USD24x8.56cm foot including home yaw, while a13-column
        # window rejects the35cm tread's clean cells beside mixed risers.
        # Coverage is the measured cells' area (12.5mm discretization), never
        # nearest-filled holes or an unobserved floor. Outside stays unknown.
        padding = (5, 5, 3, 3)
        unknown = functional.max_pool2d(functional.pad((~flat).float().view(shape), padding, value=1), (7, 11), 1)
        high = functional.max_pool2d(functional.pad(maximum.view(shape), padding, value=-float("inf")), (7, 11), 1)
        low = -functional.max_pool2d(functional.pad(-minimum.view(shape), padding, value=-float("inf")), (7, 11), 1)
        count = functional.avg_pool2d(used_model.float().view(shape),(7,11),1,(3,5))*77
        self.safe = ((unknown == 0) & ((high - low) < .022) & (count<=1.0001)).flatten(1)
        self.surface = ((high + low) * .5).flatten(1)
        self.cached_step = self.clock
        return self.safe, self.surface

    def choose(self, ids, *, minimum_x, maximum_x, foot_y, low_z, high_z, desired_x=None,
               interior=False):
        safe, height = self.surfaces()
        mask = (safe[ids] & (self.plane_x[None] >= minimum_x[:, None])
                & (self.plane_x[None] <= maximum_x[:, None])
                & ((self.y[None] - foot_y[:, None]).abs() <= .04)
                & (height[ids] >= low_z[:, None]) & (height[ids] <= high_z[:, None]))
        if desired_x is None:
            # Nearest higher observed tread; never skip a visible lower level.
            score = self.plane_x[None] + 4 * height[ids]
        else:
            score = (self.plane_x[None] - desired_x[:, None]).abs()
        score = score + 2 * (self.y[None] - foot_y[:, None]).abs()
        score = torch.where(mask, score, float("inf"))
        best, index = score.min(-1)
        valid = torch.isfinite(best)
        if interior:
            # Centre the contiguous observed run on this same tread/scan row.
            # Neither a gap, a higher level nor another row can bridge the run.
            shift = torch.arange(0,7,device=self.device)
            reaches = []
            for direction in (-1,1):
                candidate = index[:,None]+direction*shift[None]
                inside = (candidate>=0)&(candidate<self.nx*self.ny)
                inside &= torch.div(candidate,self.nx,rounding_mode="floor")==torch.div(index,self.nx,rounding_mode="floor")[:,None]
                bounded = candidate.clamp(0,self.nx*self.ny-1)
                same = inside&mask.gather(1,bounded)
                same &= (height[ids[:,None],bounded]-height[ids,index,None]).abs()<.018
                run = same.long().cumprod(-1)
                reaches.append((run*shift).amax(-1))
            index += torch.div(reaches[1]-reaches[0],2,rounding_mode="floor")
        point = torch.stack((self.plane_x[index], self.y[index], height[ids, index]), -1)
        point = point + self.env.scene.env_origins[ids]
        return point, valid

    def contains(self, ids, world_point):
        local = world_point - self.env.scene.env_origins[ids]
        ix = torch.floor((local[:, 0] - self.x0) / self.resolution).long()
        iy = torch.floor((local[:, 1] - self.y0) / self.resolution).long()
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        index = iy.clamp(0, self.ny - 1) * self.nx + ix.clamp(0, self.nx - 1)
        safe, height = self.surfaces()
        return (inside & safe[ids, index] & ((local[:, 2] - height[ids, index]).abs() <= .03)
                & (self.extent_at(ids,world_point).amin(-1)>0))

    def extent_at(self, ids, world_point):
        """Grow only across additional real, temporally observed plane cells."""
        local = world_point - self.env.scene.env_origins[ids]
        self.surfaces()
        ix = torch.floor((local[:,0]-self.x0)/self.resolution).long()
        iy = torch.floor((local[:,1]-self.y0)/self.resolution).long()
        half_y = torch.zeros(ids.numel(),device=self.device)
        for rows in (7,9,11,13):
            dy,dx = torch.meshgrid(torch.arange(-(rows//2),rows//2+1,device=self.device),
                                   torch.arange(-5,6,device=self.device),indexing="ij")
            x,y = ix[:,None]+dx.flatten(),iy[:,None]+dy.flatten()
            inside = (x>=0)&(x<self.nx)&(y>=0)&(y<self.ny)
            index = y.clamp(0,self.ny-1)*self.nx+x.clamp(0,self.nx-1)
            seen, low, high, model = self._fused_cells(ids, index)
            valid = (inside&seen&((high-low)<.022)).all(-1)
            valid &= model.sum(-1)<=1
            valid &= high.amax(-1)-low.amin(-1)<.022
            valid &= ((high.amax(-1)+low.amin(-1))*.5-local[:,2]).abs()<.03
            half_y = torch.where(valid,rows*self.resolution*.5,half_y)
        return torch.stack((torch.where(half_y>0,5.5*self.resolution,0),half_y),-1)

    def footprint_supported(self,ids,corners,target_surface_z,margin=.008):
        from .stair_footprint import footprint_supported
        self.surfaces()
        return footprint_supported(self,ids,corners,target_surface_z,margin)

    def summary(self):
        return {"source": "finite_real_lidar_and_measured_contact_support_with_bounded_single_cell_model", "synthetic_hits_used": False,
                "privileged_height_scanner_used": False, "resolution_m": self.resolution,
                "safe_patch_observed_cells": [11, 7], "height_cell_half_width_uncertainty_m": self.resolution * .5,
                "maximum_modelled_cells_per_patch":1,"modelled_cells":int(self.modelled.sum()),
                "tactile_contact_polygon_cells":int(self.tactile_known.sum()),
                "tactile_usable_cells":int(self._tactile_valid().sum()),
                "fusion_conflict_cells":int(self._tactile_conflict().sum()),
                "tactile_diagnostics":self.tactile.summary() if self.tactile is not None else None,
                "model_requires_raw_repeated_opposite_neighbors_and_noncollinear_triplet":True,
                "real_hit_count_min": int(self.real_hit_count.min()),
                "safe_tread_cells_min": int(self.safe.sum(-1).min())}
