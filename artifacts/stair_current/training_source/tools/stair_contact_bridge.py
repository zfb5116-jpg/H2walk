"""Bounded, training-only future-contact labels and local-support credit.

No policy observation or joint command is written here. Sensor quantities are
copies of the original post-control path; a label never becomes a stair event.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stair_contact_bridge_binding import bridge_contract, bridge_config_errors


class BridgeRejected(RuntimeError):
    """One fail-closed auxiliary admission, with no automatic retry."""


def require(ok, message):
    if not ok:
        raise BridgeRejected(message)


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for child in v: visit(child)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class ContactPredictor(nn.Module):
    """Explicit initialization consumes only the dedicated CPU generator."""
    def __init__(self, generator, *, widths=(883,256,128,10)):
        super().__init__()
        require(tuple(widths) in ((883,128,64,10),(883,256,128,10)), "Undeclared bridge model shape")
        for i, (a, b) in enumerate(zip(widths[:-1],widths[1:])):
            w = torch.empty((b, a), dtype=torch.float32, device="cpu")
            w.uniform_(-math.sqrt(6/(a+b)), math.sqrt(6/(a+b)), generator=generator)
            self.register_parameter(f"w{i}", nn.Parameter(w))
            self.register_parameter(f"b{i}", nn.Parameter(torch.zeros(b,dtype=torch.float32,device="cpu")))

    def forward(self, x):
        h1 = F.elu(F.linear(x, self.w0, self.b0))
        h2 = F.elu(F.linear(h1, self.w1, self.b1))
        return F.linear(h2, self.w2, self.b2)


RECORD = np.dtype([("x", "<f4", (883,)), ("prediction", "<f4", (10,)),
    ("root", "<f4", (7,)), ("target", "<f4", (2, 5)),
    ("acc", "<f4", (2, 8)), ("header", "<u4", (24,)),
    ("flags", "u1", (2, 8)), ("padding", "u1", (24,))])
assert RECORD.itemsize == 3840
# header: control,generation,env,row,close_control,admitted,unused,queued;
# then two feet x (phase,contact_control,region,loaded_run,unloaded_run,
#                 achieved_support,physical_same_foot_level_event,floor_air_ok).
# flags/foot: live,seen_unloaded,location_mask,contact_mask,joint_mask,unused...


def actual_geometry(soles, rotations, origins, heights, corners, *, approach, depth, width,
                    floor_start, platform_end):
    """FP64 field on the six real faces; no selected target or model in P."""
    local = soles.double()-origins.double()[:, None]
    xyz = torch.einsum("nfij,fkj->nfki", rotations.double(), corners.double())+local[:, :, None]
    lo, hi = xyz.amin(-2), xyz.amax(-2)
    rear, front, bottom = lo[..., 0], hi[..., 0], lo[..., 2]
    edges = approach+torch.arange(6, dtype=torch.float64, device=soles.device)*depth
    span = front-rear
    u = ((front[..., None]-(edges-.60))/.35).clamp(0, 1)
    c = ((front[..., None]-(edges-.25))/(.25+span[..., None]+.008)).clamp(0, 1)
    crossed = rear[..., None] >= edges+.008
    lower = torch.arange(6, dtype=torch.float64, device=soles.device)*heights.double()[:, None]
    b = ((bottom[..., None]-lower[:, None]) /
         (heights.double()[:, None, None]+.02*(~crossed))).clamp(0, 1)
    lateral = (lo[..., 1] >= -width/2+.02) & (hi[..., 1] <= width/2-.02)
    potential = lateral*(.25*u+.25*u*b+.50*c*b).sum(-1)/6
    level = (torch.floor((local[..., 0]-approach)/depth).long()+1).clamp(0, 6)
    start = torch.where(level == 0, floor_start, approach+(level-1)*depth)
    end = torch.where(level == 6, platform_end, approach+level*depth)
    footprint = lateral & (rear >= start+.008) & (front <= end-.008)
    plane = level*heights.double()[:, None]
    first_ok = footprint & ((bottom-plane).abs() <= .02)
    full_plane = footprint & ((xyz[..., 2]-plane[..., None]).abs() <= .02).all(-1)
    require(bool(torch.isfinite(potential).all()) and bool(((potential >= 0)&(potential <= 1)).all()),
            "Nonfinite or unbounded actual bridge geometry")
    return dict(potential=potential, level=level, first_ok=first_ok,
                full_plane=full_plane, local=local, corners=xyz)


def conservative_event(previous, following, weight, eligible, quality, dtype):
    delta = .99*following-previous
    paid = delta.clamp_max(0)+delta.clamp_min(0)*weight[:, None]*eligible*quality
    # Round publication down, including negative debt. No epsilon top-up.
    value = paid.sum(-1)
    rounded = value.to(dtype)
    down = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf))
    rounded = torch.where(rounded.double() > value, down, rounded)
    require(bool(torch.isfinite(rounded).all()), "Nonfinite bridge reward publication")
    return delta, rounded


def masked_loss(prediction, target, mask):
    p = prediction.reshape(-1, 2, 5)
    loc, a, b = 2*torch.tanh(p[..., :3]), p[..., 3], p[..., 4]
    log_joint = F.logsigmoid(a)+F.logsigmoid(b)
    log_not_joint = torch.logsumexp(torch.stack((torch.zeros_like(a), a, b)), dim=0)-F.softplus(a)-F.softplus(b)
    terms = [F.smooth_l1_loss((loc-target[..., :3])/.10, torch.zeros_like(loc), reduction="none").mean(-1),
             F.binary_cross_entropy_with_logits(a, target[..., 3], reduction="none"),
             -target[..., 4]*log_joint-(1-target[..., 4])*log_not_joint]
    result = prediction.sum()*0
    for term, valid in zip(terms, mask.unbind(-1)):
        if bool(valid.any()): result = result+term[valid].mean()
    return result


def binary_metrics(probability, target):
    p, y = np.asarray(probability, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(p.size > 0 and np.isfinite(p).all(), "Empty/nonfinite calibration labels")
    brier = float(np.mean((p-y)**2))
    bins = np.minimum((p*10).astype(np.int64), 9)
    ece = sum(float(np.mean(bins == j))*abs(float(p[bins == j].mean())-float(y[bins == j].mean()))
              for j in range(10) if (bins == j).any())
    pos, neg = int(y.sum()), int((1-y).sum())
    require(pos > 0 and neg > 0, "Calibration needs both observed classes")
    order = np.argsort(p, kind="stable"); ranks = np.empty(p.size, dtype=np.float64)
    sorted_p = p[order]; start = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1])+1]
    end = np.r_[start[1:], p.size]
    for lo, hi in zip(start, end): ranks[order[lo:hi]] = (lo+1+hi)/2
    return dict(Brier=brier, ECE10=ece, AUROC=float((ranks[y == 1].sum()-pos*(pos+1)/2)/(pos*neg)),
                positive=pos, negative=neg, count=int(y.size))


class ContactBridge:
    """One live collector, two small models, and one proposal seal."""
    def __init__(self, raw, algorithm, binding, directory):
        require(not bridge_config_errors(binding) and raw.num_envs == 1024
                and raw.step_dt == .02 and algorithm.gamma == .99,
                "Bridge requires its exact original1024/.02/gamma source")
        self.raw, self.alg = raw, algorithm
        self.contract = copy.deepcopy(binding)
        self.directory = Path(directory); self.directory.mkdir(exist_ok=False)
        self.log = (self.directory/"auxiliary_transactions.jsonl").open("x")
        self.rng = torch.Generator(device="cpu").manual_seed(20261104)
        self.learner = ContactPredictor(self.rng,widths=binding["layers"]).to(algorithm.device)
        self.service = copy.deepcopy(self.learner).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=3e-4, betas=(.9,.999), eps=1e-8,
            weight_decay=0., amsgrad=False, foreach=False, fused=False)
        self.parameter_count = sum(p.numel() for p in self.learner.parameters())
        require(self.parameter_count == binding["parameters"] == 260490,
                "Bridge actual independent parameter count differs")
        actor_ids = {id(p) for p in self.alg.policy.parameters()}
        require(not actor_ids.intersection(id(p) for p in self.learner.parameters()), "Bridge shares policy parameters")
        self.pending = np.zeros((64, 1024), dtype=RECORD)
        self.closed = np.zeros(49152, dtype=RECORD)
        self.closed_cursor = self.closed_count = 0
        self.controls = self.data_epoch = self.seal_id = self.anchors = 0
        self.serving_forwards = self.learner_forwards = self.calibration_forwards = 0
        self.backwards = self.Adam_attempts = self.Adam_commits = 0
        self.aux_rolled_back_commits = self.aux_failed_Adam_attempts = self.skipped_data_rollouts = 0
        self.serving_version = self.last_reward_model_version = 0
        self.armed = self.closed_down = False
        self.ledger = {k: 0 for k in ("joint_positive", "joint_negative", "contact_positive", "contact_negative",
            "location_known", "fully_censored", "initial_partial", "no_contact", "timeout", "budget_tail",
            "external_cut", "physical_or_force_failure", "first_contact_bad", "no_local_recovery", "FIFO_evicted")}
        self._received_epoch = self._endpoint_epoch = 0
        self._critic = None; self._critic_epoch = 0
        self._contact = np.zeros((1024,2), dtype=bool)
        self._seen_contact = np.zeros_like(self._contact)
        self._air = np.zeros((1024,2), dtype=np.float32)
        self._events = np.zeros((1024,2), dtype=np.int64)
        self._true_end = np.zeros(1024,dtype=bool)
        self._reset_admitted = np.zeros(1024,dtype=bool)
        self.previous = torch.zeros((1024,2), dtype=torch.float64, device=raw.device)
        self.initialized = torch.zeros(1024, dtype=torch.bool, device=raw.device)
        self.latched = torch.zeros_like(self.initialized)
        self.quality = torch.ones_like(self.previous)
        self._baseline = None; self._seal_digest = None; self._lease_epoch = None
        self.calibration = None
        self.last_loss = None
        self.last_added_min = self.last_added_max = 0.0
        self.last_delta_min = self.last_delta_max = 0.0
        self._gpu_scratch = 0
        geometry = raw.scene.terrain.cfg.terrain_generator.sub_terrains["ascending_stairway"]
        length = float(raw.scene.terrain.cfg.terrain_generator.size[0])
        self.geometry = dict(approach=raw.stair_progress.approach, depth=raw.stair_progress.depth,
            width=raw.stair_progress.width, floor_start=-float(geometry.spawn_origin_x),
            platform_end=length-float(geometry.spawn_origin_x)-float(geometry.top_end_margin))
        self.generation = raw.initialization_reset_generation.detach().cpu().numpy().astype(np.uint32).copy()
        self._journal("initialized", self.snapshot())

    def _journal(self, kind, record):
        self.log.write(json.dumps(dict(kind=kind, controls=self.controls, data_epoch=self.data_epoch,
            record=record), allow_nan=False)+"\n"); self.log.flush()

    def _geometry(self, soles, rotations):
        p = self.raw.stair_progress
        return actual_geometry(soles, rotations, p.origins, p.heights, p.corners, **self.geometry)

    def cache_critic(self, critic):
        require(self._critic is None and tuple(critic.shape) == (1024,871)
                and bool(torch.isfinite(critic).all()), "Missing/mismatched actual normalized critic input")
        self._critic = critic.detach()
        self._critic_epoch = self.controls+1

    def begin(self, soles, rotations):
        require(not self.closed_down and int(self.raw.common_step_counter) == self.controls
                and self._critic_epoch == self.controls+1 and self._endpoint_epoch == self.controls,
                "Bridge precontrol epoch/critic source mismatch")
        require(not self._reset_admitted.any()
                and bool(torch.isfinite(self.raw.asset.data.root_pos_w).all())
                and bool(torch.isfinite(self.raw.asset.data.root_quat_w).all()),
                "Bridge unclosed reset or nonfinite actual anchor pose")
        self.controls += 1; self.data_epoch += 1; self.anchors += 1024
        require(np.array_equal(self.generation, self.raw.initialization_reset_generation.detach().cpu().numpy()),
                "Unaccounted actual reset generation")
        self._events.fill(0)
        self.last_reward_model_version = self.serving_version
        p = self.raw.stair_progress
        if self.armed:
            init = ~self.initialized & ~p.partial & ~self.latched
            value = self._geometry(soles, rotations)["potential"]
            self.previous[init] = value[init]; self.initialized[init] = True
        slot = (self.controls-1)%64
        require(not self.pending["flags"][slot,:,:,0].any(), "Pending bridge record collision at64 controls")
        self.pending[slot] = np.zeros(1024, dtype=RECORD)
        r = self.pending[slot]
        r["header"][:,0] = self.controls; r["header"][:,1] = self.generation
        r["header"][:,2] = np.arange(1024); r["header"][:,3] = p.terrain_level.cpu().numpy()
        admitted = (~p.partial).cpu().numpy()
        r["header"][:,5] = admitted
        r["flags"][:,:,0] = admitted[:,None]
        r["flags"][:,:,1] = (~self._contact) & self._seen_contact
        r["root"][:,:3] = self.raw.asset.data.root_pos_w.detach().cpu().numpy()
        r["root"][:,3:] = self.raw.asset.data.root_quat_w.detach().cpu().numpy()
        self.ledger["initial_partial"] += int((~admitted).sum())*2
        self.ledger["fully_censored"] += int((~admitted).sum())*2

    def received_action(self):
        """Called after the original first native write, before its physics step."""
        if self._received_epoch == self.controls:
            return
        require(self._received_epoch == self.controls-1 and self._critic_epoch == self.controls,
                "Bridge first native action is not from its pending transition")
        actual = self.raw.action_manager.get_term("JointPositionAction").raw_actions
        require(tuple(actual.shape) == (1024,12) and bool(torch.isfinite(actual).all()),
                "Bridge actual action source differs")
        x = torch.cat((self._critic, torch.tanh(actual)), dim=-1)
        slot = (self.controls-1)%64
        self.pending["x"][slot] = x.detach().cpu().numpy()
        if self.armed:
            self.serving_forwards += 1
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False): out = self.service(x)
            require(bool(torch.isfinite(out).all()), "Nonfinite bridge serving output")
            shaped = out.reshape(1024,2,5)
            self.quality.copy_(.25+.75*torch.sigmoid(shaped[...,3]).double()*torch.sigmoid(shaped[...,4]).double())
            self.pending["prediction"][slot] = out.detach().cpu().numpy()
            self._gpu_scratch = x.numel()*4+out.numel()*4+1024*sum(self.contract["layers"][1:-1])*8
        self._critic = None; self._received_epoch = self.controls

    def note_physical_foot(self, ids, foot, level):
        require(self._received_epoch == self.controls, "Physical foot event without actual bridge command")
        self._events[ids.detach().cpu().numpy(), foot] = level.detach().cpu().numpy()

    def endpoint(self, soles, rotations, root_tilt, root_ang_vel, terminated, timed_out,
                 valid_physical, positive_weight):
        require(self._endpoint_epoch == self.controls-1 and self._received_epoch == self.controls,
                "Missing bridge endpoint/first native write")
        p = self.raw.stair_progress
        require(bool(torch.isfinite(root_ang_vel).all()) and tuple(root_ang_vel.shape) == (1024,3),
                "Missing actual root angular velocity")
        g = self._geometry(soles, rotations)
        failure = terminated | p.ever_invalid | ~p.force_credit_eligible
        endpoint = failure | timed_out
        self._true_end[:] = (terminated|timed_out).detach().cpu().numpy()
        final = self.controls == 48000
        if final: endpoint = torch.ones_like(endpoint)
        if self.armed:
            value = torch.where((endpoint|self.latched|~self.initialized)[:,None], 0., g["potential"])
            fz = p.ground_force_end[...,2]
            Q = (valid_physical[:,None] & ~endpoint[:,None] & ~self.latched[:,None]
                 & self.initialized[:,None] & p.current_force_valid.all(-1)[:,None]
                 & (root_tilt < .5)[:,None] & (fz < p.minimum_load[:,None])
                 & (fz.flip(-1) >= .7*p.weight[:,None]))
            require(bool(torch.isfinite(positive_weight).all())
                    and bool(((positive_weight >= 0)&(positive_weight <= 1)).all()), "Invalid original positive weight")
            delta, added = conservative_event(self.previous, value, positive_weight.double(), Q,
                                            self.quality, p.events["potential"].dtype)
            self.last_added_min, self.last_added_max = float(added.min()), float(added.max())
            self.last_delta_min, self.last_delta_max = float(delta.min()), float(delta.max())
            # Round the merged event conservatively too: no positive cast gain.
            merged = p.events["potential"].double()+added.double()
            cast = merged.to(p.events["potential"].dtype)
            cast = torch.where(cast.double() > merged,
                torch.nextafter(cast, torch.full_like(cast,-torch.inf)), cast)
            p.events["potential"].copy_(cast)
            self.previous.copy_(value); self.latched |= endpoint
        self._labels(g, soles, root_tilt, root_ang_vel, failure, timed_out, final)
        self._endpoint_epoch = self.controls; self.data_epoch += 1

    def _labels(self, g, soles, tilt, ang, failure, timeout, final):
        p, r = self.raw.stair_progress, self.pending
        fz = p.ground_force_end[...,2].detach().cpu().numpy()
        force_valid = p.current_force_valid.detach().cpu().numpy()
        force_eligible = p.force_credit_eligible.detach().cpu().numpy()
        contact = np.maximum(p.ground_force_end.norm(dim=-1).cpu().numpy(), p.raw_force_end.norm(dim=-1).cpu().numpy()) > 5
        first_ok = g["first_ok"].cpu().numpy(); full_plane = g["full_plane"].cpu().numpy()
        level = g["level"].cpu().numpy(); sole = soles.detach().cpu().numpy()
        tilt = tilt.detach().cpu().numpy(); angular = ang[:,:2].norm(dim=-1).detach().cpu().numpy()
        fail = failure.detach().cpu().numpy(); timeout = timeout.detach().cpu().numpy()
        prior_air = self._air.copy(); seen_prior = self._seen_contact.copy()
        self._air[:] = np.where(contact, 0., self._air+.02)
        self._seen_contact |= contact; self._contact[:] = contact
        flags, target, acc = r["flags"], r["target"], r["acc"]
        ph = r["header"][...,8:].reshape(64,1024,2,8)
        live = flags[...,0].astype(bool)
        search = live & (ph[...,0] == 0)
        first = search & flags[...,1].astype(bool) & contact[None]
        a,e,f = np.nonzero(first)
        if len(a):
            delta = sole[e,f]-r["root"][a,e,:3]
            q = r["root"][a,e,3:]
            t = 2*np.cross(-q[:,1:], delta)
            local = delta+q[:,:1]*t+np.cross(-q[:,1:],t)
            target[a,e,f,:3] = local; flags[a,e,f,2] = 1
            ok = first_ok[e,f] & force_valid[e,f] & force_eligible[e] & ~fail[e]
            target[a,e,f,3] = ok; flags[a,e,f,3] = 1
            ph[a,e,f,0] = 1; ph[a,e,f,1] = self.controls; ph[a,e,f,2] = level[e,f]
            ph[a,e,f,7] = seen_prior[e,f] & (prior_air[e,f] >= .08-1e-6) & (prior_air[e,f] <= .6+1e-6)
            acc[a,e,f,1] = 1.
            bad = np.zeros_like(first); bad[a,e,f] = ~ok
            self._close_feet(bad, known=0, cause="first_contact_bad")
        flags[...,1] |= (search & ~contact[None]).astype(np.uint8)
        live = flags[...,0].astype(bool)
        failed = live & fail[None,:,None]
        unknown_contact = failed & ~flags[...,3].astype(bool)
        target[...,3][unknown_contact] = 0; flags[...,3][unknown_contact] = 1
        self._close_feet(failed, known=0, cause="physical_or_force_failure")
        live = flags[...,0].astype(bool)
        age = self.controls-ph[...,1].astype(np.int64)
        recovering = live & (ph[...,0] == 1) & (age >= 1) & (age <= 32)
        matched = level[None] == ph[...,2]
        weight = p.weight.detach().cpu().numpy()
        load = (recovering & matched & full_plane[None] & force_valid[None]
                & (fz[None] >= .7*weight[None,:,None]))
        off = fz[:,::-1] < p.minimum_load.detach().cpu().numpy()[:,None]
        ph[...,3] = np.where(load, ph[...,3]+1, 0)
        ph[...,4] = np.where(load & off[None], ph[...,4]+1, 0)
        acc[...,3] = np.where(load, np.maximum(acc[...,3], ph[...,4]), 0)
        ph[...,5] |= (recovering & (ph[...,3] >= 4) & (acc[...,3] >= 2)).astype(np.uint32)
        ph[...,6] |= (recovering & (self._events[None] == ph[...,2]) & (ph[...,2] > 0)).astype(np.uint32)
        acc[...,2] = np.where(recovering & (tilt[None,:,None] >= .5), 1., acc[...,2])
        tail = recovering & (age >= 29)
        acc[...,0] = np.where(recovering & (age == 29), tilt[None,:,None], acc[...,0])
        acc[...,1] = np.where(tail,
            acc[...,1]*((tilt[None,:,None] <= .15)&(angular[None,:,None] <= .75)), acc[...,1])
        done = recovering & (age == 32)
        good = (done & (acc[...,2] == 0) & (ph[...,5] != 0) & (acc[...,1] != 0)
                & (tilt[None,:,None]-acc[...,0] <= .01)
                & np.where(ph[...,2] == 0, ph[...,7] != 0, ph[...,6] != 0))
        self._close_feet(good, known=1, cause=None)
        self._close_feet(done & ~good, known=0, cause="no_local_recovery")
        live = flags[...,0].astype(bool)
        no_contact = live & (ph[...,0] == 0) & (self.controls-r["header"][...,0,None].astype(np.int64) >= 31)
        self._close_feet(no_contact, known=None, cause="no_contact")
        self._close_feet(flags[...,0].astype(bool) & timeout[None,:,None], known=None, cause="timeout")
        if final:
            self._close_feet(flags[...,0].astype(bool), known=None, cause="budget_tail")
        self._enqueue_finished()

    def _close_feet(self, mask, known, cause):
        mask &= self.pending["flags"][...,0].astype(bool)
        n = int(mask.sum())
        if not n: return
        flags, target = self.pending["flags"], self.pending["target"]
        if known is not None:
            target[...,4][mask] = known; flags[...,4][mask] = 1
        if cause: self.ledger[cause] += n
        for col, name in ((3,"contact"), (4,"joint")):
            observed = mask & flags[...,col].astype(bool)
            self.ledger[name+"_positive"] += int((observed & (target[...,col] == 1)).sum())
            self.ledger[name+"_negative"] += int((observed & (target[...,col] == 0)).sum())
        self.ledger["location_known"] += int((mask & flags[...,2].astype(bool)).sum())
        self.ledger["fully_censored"] += int((mask & ~flags[...,2:5].any(-1)).sum())
        flags[...,0][mask] = 0

    def _enqueue_finished(self):
        r = self.pending
        done = ((r["header"][...,0] > 0) & (r["header"][...,7] == 0)
                & ~r["flags"][...,0].any(-1))
        a,e = np.nonzero(done)
        order = np.lexsort((e,r["header"][a,e,0])); a,e = a[order],e[order]
        for start in range(0,len(a),512):
            aa, ee = a[start:start+512],e[start:start+512]
            records = r[aa,ee]  # bounded <=512 records, never a full-ring clone
            r["header"][aa,ee,7] = 1
            records["header"][:,4] = self.controls
            records = records[records["flags"][:,:,4].any(-1) & (records["header"][:,5] == 1)]
            count = len(records)
            if not count: continue
            self.ledger["FIFO_evicted"] += max(0,self.closed_count+count-len(self.closed))
            first = min(count,len(self.closed)-self.closed_cursor)
            self.closed[self.closed_cursor:self.closed_cursor+first] = records[:first]
            if first < count: self.closed[:count-first] = records[first:]
            self.closed_cursor = (self.closed_cursor+count)%len(self.closed)
            self.closed_count = min(len(self.closed),self.closed_count+count)

    def before_reset(self, ids):
        ids_np = ids.detach().cpu().numpy()
        require(self._true_end[ids_np].all() and not self._reset_admitted[ids_np].any(),
                "Unexpected bridge reset without this control's real termination/timeout")
        actual = self.raw.initialization_reset_generation[ids].detach().cpu().numpy()
        require(np.array_equal(actual,self.generation[ids_np].astype(np.int64)+1),
                "Reset generation did not advance exactly once at the original reset hook")
        require(self._endpoint_epoch == self.controls and not self.pending["flags"][:,ids_np,:,0].any(),
                "Reset would join or erase an unresolved bridge future")
        require(not bool((self.initialized[ids] & ~self.latched[ids]).any())
                and not bool((self.previous[ids] != 0).any()), "Reset would erase unpaid bridge debt")
        self._reset_admitted[ids_np] = True

    def after_reset(self, ids):
        ids_np = ids.detach().cpu().numpy()
        require(self._reset_admitted[ids_np].all(), "Bridge post-reset without admitted true endpoint")
        self.generation[ids_np] = self.raw.initialization_reset_generation[ids].detach().cpu().numpy()
        self._contact[ids_np] = False; self._seen_contact[ids_np] = False; self._air[ids_np] = 0
        self.initialized[ids] = False; self.latched[ids] = False; self.previous[ids] = 0
        self._reset_admitted[ids_np] = False; self._true_end[ids_np] = False
        self.data_epoch += 1

    def _train_state(self):
        return dict(learner=cpu(self.learner.state_dict()), service=cpu(self.service.state_dict()),
            optimizer=cpu(self.optimizer.state_dict()), private_rng=self.rng.get_state().clone(),
            Adam_commits=self.Adam_commits, serving_version=self.serving_version, armed=self.armed)

    def _restore_train(self, state):
        self.learner.load_state_dict(state["learner"],strict=True)
        self.service.load_state_dict(state["service"],strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["private_rng"])
        self.Adam_commits = state["Adam_commits"]; self.serving_version = state["serving_version"]
        self.armed = state["armed"]; self.optimizer.zero_grad(set_to_none=True)

    def _ordered_indices(self, heldout):
        ids = np.arange(self.closed_count)
        if self.closed_count == len(self.closed): ids = (ids+self.closed_cursor)%len(self.closed)
        group = self.closed["header"][ids,2]%8 == 0
        return ids[group if heldout else ~group]

    def _batch(self, ids):
        rec = self.closed[ids]
        x = torch.from_numpy(np.ascontiguousarray(rec["x"])).to(self.alg.device)
        y = torch.from_numpy(np.ascontiguousarray(rec["target"])).to(self.alg.device)
        m = torch.from_numpy(np.ascontiguousarray(rec["flags"][:,:,2:5].astype(bool))).to(self.alg.device)
        return x,y,m

    def update_before_ppo(self):
        require(self.controls == (self.alg.completed_updates+1)*48 and self._endpoint_epoch == self.controls
                and self._critic is None, "Auxiliary update is not a sealed48-control boundary")
        self._baseline = None
        rollback = self._train_state()
        attempts_before = self.Adam_attempts
        next_completed = self.alg.completed_updates+1
        eligible = self._ordered_indices(False)
        try:
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device_type=torch.device(self.alg.device).type, enabled=False):
                if len(eligible) >= 512:
                    for _ in range(self.contract["Adam_steps_per_rollout"]):
                        draw = torch.randint(len(eligible),(512,),generator=self.rng,device="cpu",dtype=torch.int64).numpy()
                        x,y,m = self._batch(eligible[draw])
                        self.optimizer.zero_grad(set_to_none=True)
                        self.learner_forwards += 1; pred = self.learner(x)
                        loss = masked_loss(pred,y,m)
                        require(bool(torch.isfinite(loss)), "Nonfinite bridge supervised loss")
                        self.backwards += 1; loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(self.learner.parameters(),1.)
                        require(bool(torch.isfinite(norm)) and all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in self.learner.parameters()), "Nonfinite bridge gradient")
                        self.Adam_attempts += 1; self.optimizer.step(); self.Adam_commits += 1
                        require(finite(self.learner.state_dict()) and finite(self.optimizer.state_dict()),
                                "Nonfinite bridge Adam state")
                        self.last_loss = float(loss.detach())
                        self._gpu_scratch = 512*(883+2*sum(self.contract["layers"][1:-1])+10+16)*4
                        del x,y,m,pred,loss
                else:
                    self.skipped_data_rollouts += 1
                if next_completed >= 40 and next_completed%10 == 0:
                    self.calibration = self._calibrate(next_completed)
                if next_completed >= 40 and next_completed%10 == 0:
                    require(self.calibration is not None and self.calibration["passed"],
                            "AUX_BRIDGE_ADMISSION_NEEDS_ATTENTION: no admitted calibration")
                    self.service.load_state_dict(self.learner.state_dict(),strict=True)
                    self.serving_version += 1; self.armed = True
            self.optimizer.zero_grad(set_to_none=True)
        except BaseException as error:
            rolled_back = self.Adam_commits-rollback["Adam_commits"]
            self.aux_rolled_back_commits += rolled_back
            self.aux_failed_Adam_attempts += self.Adam_attempts-attempts_before-rolled_back
            self._restore_train(rollback)
            self._journal("auxiliary_restored_stop",dict(error=repr(error), calibration=self.calibration,
                actual_rolled_back_commits=rolled_back,
                actual_Adam_attempts=self.Adam_attempts-attempts_before,
                physical_data_epoch_unchanged=self.data_epoch, snapshot=self.snapshot()))
            raise
        self.seal_id += 1; self._lease_epoch = self.data_epoch
        self._baseline = self._train_state(); self._seal_digest = digest(self._baseline)
        self._journal("pre_super_committed",dict(seal_id=self.seal_id, train_state_sha256=self._seal_digest,
            last_loss=self.last_loss, committed_steps=self.Adam_commits, calibration=self.calibration,
            skipped_data=len(eligible)<512, serving_version=self.serving_version,
            last_reward_model_version=self.last_reward_model_version))

    def _calibrate(self, completed):
        # This is the NEW closed FIFO, never the original4096 audit bank.
        ids = self._ordered_indices(True)[-4096:]
        masks = self.closed["flags"][ids,:,2:5].astype(bool)
        target = self.closed["target"][ids].copy()
        minima = dict(joint_labels=int(masks[...,2].sum()),
            joint_positive=int(((target[...,4]==1)&masks[...,2]).sum()),
            joint_negative=int(((target[...,4]==0)&masks[...,2]).sum()),
            contact_positive=int(((target[...,3]==1)&masks[...,1]).sum()),
            contact_negative=int(((target[...,3]==0)&masks[...,1]).sum()),
            contact_locations=int(masks[...,0].sum()))
        result = dict(completed=completed, record_count=len(ids), minima=minima, passed=False,
            strata={f"row{row}":dict(records=int((self.closed["header"][ids,3]==row).sum()),
                joint_positive_floor=int(((self.closed["header"][ids,3]==row)[:,None]
                    & (self.closed["header"][ids,10::8]==0) & masks[...,2] & (target[...,4]==1)).sum()),
                joint_positive_stair=int(((self.closed["header"][ids,3]==row)[:,None]
                    & (self.closed["header"][ids,10::8]>0) & masks[...,2] & (target[...,4]==1)).sum())) for row in range(3)})
        self.calibration = result
        missing = [k for k,v in self.contract["calibration_minimums"].items() if minima[k] < v]
        require(not missing, "AUX_BRIDGE_ADMISSION_NEEDS_ATTENTION: insufficient "+",".join(missing))
        predictions = []
        with torch.no_grad(), torch.autocast(device_type=torch.device(self.alg.device).type, enabled=False):
            for start in range(0,len(ids),512):
                x,_,_ = self._batch(ids[start:start+512])
                self.calibration_forwards += 1; out = self.learner(x)
                require(bool(torch.isfinite(out).all()), "Nonfinite bridge calibration forward")
                predictions.append(out.cpu().numpy()); del x,out
        pred = np.concatenate(predictions).reshape(-1,2,5)
        contact = 1/(1+np.exp(-pred[...,3].astype(np.float64)))
        joint = contact/(1+np.exp(-pred[...,4].astype(np.float64)))
        cm = binary_metrics(contact[masks[...,1]], target[...,3][masks[...,1]])
        jm = binary_metrics(joint[masks[...,2]], target[...,4][masks[...,2]])
        distance = np.linalg.norm(2*np.tanh(pred[...,:3])-target[...,:3],axis=-1)[masks[...,0]]
        censor = self.ledger["fully_censored"]/max(1,2*self.anchors)
        result.update(contact=cm,joint=jm,location_median=float(np.median(distance)),
            location_p90=float(np.quantile(distance,.9)), fully_censored_fraction=censor)
        threshold = self.contract["calibration_thresholds"]
        result["passed"] = (all(m["Brier"] <= threshold["Brier_max"] and m["ECE10"] <= threshold["ECE10_max"]
            and m["AUROC"] >= threshold["AUROC_min"] for m in (cm,jm))
            and result["location_median"] <= threshold["location_median_m_max"]
            and result["location_p90"] <= threshold["location_p90_m_max"]
            and censor <= threshold["fully_censored_fraction_max"])
        require(result["passed"], "AUX_BRIDGE_ADMISSION_NEEDS_ATTENTION: calibration thresholds failed")
        return result

    def proposal_seal(self):
        require(self._lease_epoch == self.data_epoch and self._baseline is not None,
                "Actor proposal has no immutable bridge data/model seal")
        return dict(seal_id=self.seal_id, data_epoch=self.data_epoch, train_state_sha256=self._seal_digest)

    def restore_proposal_seal(self, seal):
        require(seal == self.proposal_seal(), "Actor restore would rewind physical bridge facts")
        self._restore_train(self._baseline)
        require(digest(self._train_state()) == self._seal_digest, "Bridge sealed restore differs")
        self._journal("actor_proposal_bridge_restore",dict(seal=seal, physical_rewind=False))

    def snapshot(self):
        model_bytes = sum(t.numel()*t.element_size() for m in (self.learner,self.service)
                          for t in m.parameters())
        optimizer_bytes = sum(t.numel()*t.element_size() for state in self.optimizer.state.values()
                              for t in state.values() if torch.is_tensor(t))
        gpu = model_bytes+optimizer_bytes+self.parameter_count*4+self._gpu_scratch
        rings = self.pending.nbytes+self.closed.nbytes
        require(gpu <= 134217728 and rings+96468992 <= 536870912, "Bridge owned storage cap exceeded")
        return dict(contract=copy.deepcopy(self.contract), controls=self.controls,
            actual_environment_controls=int(self.raw.common_step_counter),
            pending_control=self.controls if self._endpoint_epoch != self.controls else None,
            parameters=self.parameter_count, actor_parameters_shared=False, CPU_ring_bytes=rings,
            CPU_owned_bytes=rings+96468992, CPU_scratch_reservation_bytes=96468992,
            GPU_owned_bytes=gpu, original_numeric_arena_unchanged=True,
            serving_forwards=self.serving_forwards, learner_forwards=self.learner_forwards,
            calibration_forwards=self.calibration_forwards, backwards=self.backwards,
            Adam_attempts=self.Adam_attempts, Adam_commits=self.Adam_commits,
            aux_steps_per_rollout=self.contract["Adam_steps_per_rollout"],
            aux_rolled_back_commits=self.aux_rolled_back_commits,
            aux_failed_Adam_attempts=self.aux_failed_Adam_attempts,
            skipped_data_rollouts=self.skipped_data_rollouts,
            serving_version=self.serving_version, last_reward_model_version=self.last_reward_model_version,
            seal_id=self.seal_id, data_epoch=self.data_epoch, anchors=self.anchors,
            pending_records=int(self.pending["flags"][...,0].any(-1).sum()),
            closed_records=self.closed_count, closed_cursor=self.closed_cursor,
            globally_armed=self.armed, per_env_initialized=int(self.initialized.sum()),
            potential_min=float(self.previous.min()), potential_max=float(self.previous.max()),
            last_added_event_min=self.last_added_min, last_added_event_max=self.last_added_max,
            last_raw_delta_min=self.last_delta_min, last_raw_delta_max=self.last_delta_max,
            last_calibration=copy.deepcopy(self.calibration), ledger=copy.deepcopy(self.ledger),
            actual_geometry=copy.deepcopy(self.geometry), seal_digest=self._seal_digest)

    def checkpoint_payload(self):
        require(self._endpoint_epoch == self.controls, "Checkpoint has an incomplete bridge control")
        return dict(snapshot=self.snapshot(), train=self._train_state(),
            pending_bytes=torch.from_numpy(self.pending.view(np.uint8).reshape(-1)),
            closed_bytes=torch.from_numpy(self.closed.view(np.uint8).reshape(-1)),
            previous=cpu(self.previous), initialized=cpu(self.initialized), latched=cpu(self.latched),
            quality=cpu(self.quality), generation=torch.from_numpy(self.generation.astype(np.int64)),
            contact=torch.from_numpy(self._contact.copy()), seen_contact=torch.from_numpy(self._seen_contact.copy()),
            air=torch.from_numpy(self._air.copy()), received_epoch=self._received_epoch,
            true_end=torch.from_numpy(self._true_end.copy()), reset_admitted=torch.from_numpy(self._reset_admitted.copy()),
            endpoint_epoch=self._endpoint_epoch, last_loss=self.last_loss,
            record_dtype=[list(x) for x in RECORD.descr], original_numeric_arena_unchanged=True)

    def close(self):
        if self.closed_down: return
        before = self.snapshot()
        self._close_feet(self.pending["flags"][...,0].astype(bool), known=None, cause="external_cut")
        self._enqueue_finished(); self.data_epoch += 1
        require(finite(self.previous), "Nonfinite unsettled bridge debt")
        # This cutoff is deliberately separate from the last accepted seal.
        # No paid reward is changed and no extra physical control is run.
        receipt = dict(before=before, after=self.snapshot(), unsettled_censored_debt=cpu(self.previous).tolist(),
            incomplete_control=self._endpoint_epoch != self.controls, physical_rewind=False)
        (self.directory/"external_cut.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        if self.controls and self.controls != 48000:
            payload = self.checkpoint_payload() if self._endpoint_epoch == self.controls else dict(
                snapshot=self.snapshot(), train=self._train_state(),
                pending_bytes=torch.from_numpy(self.pending.view(np.uint8).reshape(-1)),
                closed_bytes=torch.from_numpy(self.closed.view(np.uint8).reshape(-1)),
                previous=cpu(self.previous), incomplete_control=True)
            torch.save(payload,self.directory/"interrupted_bridge_state.pt")
        self.log.close(); self.closed_down = True


def validate_bridge_payload(payload, binding, snapshot, *, completed):
    """One typed CPU inspection of the actual new module, never a PT loader."""
    from stair_contact_bridge_binding import bridge_snapshot_errors
    require(not bridge_snapshot_errors(binding,snapshot,expected_controls=completed*48)
            and payload["snapshot"] == snapshot, "Typed actual bridge snapshot differs")
    require(not torch.cuda.is_initialized(), "Typed bridge CPU scope initialized CUDA")
    require(payload["endpoint_epoch"] == payload["received_epoch"] == completed*48,
            "Typed bridge contains a partial control")
    storages = []
    for name, shape in (("pending_bytes",(64,1024)),("closed_bytes",(49152,))):
        value = payload[name]
        require(value.dtype == torch.uint8 and value.device.type == "cpu" and value.is_contiguous()
                and value.numel() == math.prod(shape)*3840, "Typed bridge ring capacity/dtype differs")
        records = value.numpy().view(RECORD).reshape(shape)
        flat = records.reshape(-1)
        for start in range(0,len(flat),512):
            block = flat[start:start+512]
            require(all(np.isfinite(block[field]).all() for field in ("x","prediction","root","target","acc")),
                    "Nonfinite typed bridge record data")
            require((block["flags"] <= 1).all() and (block["header"][:,0] <= completed*48).all(),
                    "Typed bridge masks/control epoch invalid")
            for target_index,mask_index in ((3,3),(4,4)):
                known = block["flags"][:,:,mask_index].astype(bool)
                require(np.isin(block["target"][:,:,target_index][known],[0.,1.]).all(),
                        "Unknown bridge target fabricated as an observed class")
        storages.append(records)
    pending,closed = storages
    require(int(pending["flags"][...,0].any(-1).sum()) == snapshot["pending_records"],
            "Typed pending occupancy differs")
    train = payload["train"]
    require(digest(train) == snapshot["seal_digest"] and train["Adam_commits"] == snapshot["Adam_commits"]
            and train["serving_version"] == snapshot["serving_version"], "Typed bridge committed model seal differs")
    generator = torch.Generator(device="cpu").manual_seed(20261104)
    learner = ContactPredictor(generator,widths=binding["layers"])
    service = ContactPredictor(generator,widths=binding["layers"])
    for model, key in ((learner,"learner"),(service,"service")):
        model.load_state_dict(train[key],strict=True)
        require(sum(p.numel() for p in model.parameters()) == binding["parameters"] and finite(model.state_dict()),
                "Typed bridge strict finite parameter schema differs")
    optimizer = train["optimizer"]
    group = optimizer["param_groups"]
    require(len(group) == 1 and group[0]["lr"] == .0003 and len(group[0]["params"]) == 6
            and group[0]["foreach"] is False and group[0]["fused"] is False,
            "Typed independent auxiliary Adam schema differs")
    steps = train["Adam_commits"]
    require(0 <= steps <= completed*binding["Adam_steps_per_rollout"] and len(optimizer["state"]) == (6 if steps else 0),
            "Typed bridge actual optimizer counter differs")
    for index,param in zip(group[0]["params"],learner.parameters()):
        if steps:
            state = optimizer["state"][index]
            require(float(state["step"]) == steps and state["exp_avg"].shape == param.shape
                    and state["exp_avg_sq"].shape == param.shape and finite(state), "Typed bridge Adam moment differs")
    generator.set_state(train["private_rng"])
    x = torch.from_numpy(np.ascontiguousarray(pending["x"][(completed*48-1)%64,:2]))
    with torch.inference_mode():
        outputs = [model(x) for model in (learner,service)]
        require(all(tuple(y.shape) == (2,10) and bool(torch.isfinite(y).all()) for y in outputs),
                "Typed bridge actual-input CPU forward failed")
    require(snapshot["anchors"] == 1024*48*completed
            and snapshot["Adam_attempts"] >= steps and snapshot["backwards"] == snapshot["learner_forwards"],
            "Typed bridge actual execution denominators differ")
    if completed == 1:
        require(not snapshot["globally_armed"] and snapshot["serving_forwards"] == 0,
                "M0 bridge was paid/served during original critic warmup")
    if completed == 1000:
        require(snapshot["pending_records"] == 0 and not bool((payload["previous"] != 0).any()),
                "Natural bridge budget did not close actual debt/censor futures")
    return dict(actual_schema_passed=True, parameters=binding["parameters"], optimizer_parameter_tensors=6,
        actual_Adam_commits=steps, CPU_forward_batches=2, queue_records_inspected=64*1024+49152,
        unknown_masks_preserved=True, own_private_rng_valid=True, snapshot=snapshot,
        training_labels_are_not_formal_acceptance=True)
