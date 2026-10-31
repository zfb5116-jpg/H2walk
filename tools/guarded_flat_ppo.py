"""H2-local split PPO and finite-real-input whole-update rollback guard.

The rollout/storage/timeout/RND interfaces are the installed RSL PPO interfaces.
This module never imports an environment or launches a simulator.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from rsl_rl.algorithms.ppo import PPO

FORMAT = "h2_guarded_flat_ppo_v1"
MARGIN_FORMAT = "h2_guarded_flat_ppo_v2"
MARGIN_TRIAL = "overspeed_credit_action_margin_v1"
PROJECTION_FORMAT = "h2_guarded_flat_ppo_v3"
PROJECTION_TRIAL = "simultaneous_output_proposal_v1"
PROJECTION_KIND = "simultaneous_actual_bank_linear_v1"
RESERVED_FORMAT = "h2_guarded_flat_ppo_v4"
RESERVED_TRIAL = "reserved_simultaneous_output_proposal_v1"
RESERVED_KIND = "simultaneous_actual_bank_reserved_linear_v1"
RESOURCE_FORMAT = "h2_guarded_flat_ppo_v5"
RESOURCE_TRIAL = "resource_bounded_reserved_output_proposal_v1"
RESOURCE_PARENT_SHA = "e96a9fd29cc0d7457db742133345c0224f2dbcdde522692b52eba88ebdd6977d"
STRICT_FORMAT = "h2_guarded_flat_ppo_v6"
STRICT_TRIAL = "strict_primal_resource_output_proposal_v1"
STRICT_PARENT_SHA = "4cecd1fc24d618502402c1bd3ed5bcb0fdc7f44a0981a3bfe5855895715176b2"
PRIMAL_PRECISION_POLICY = "direct_fp64_primal_envelope_v1"
JACOBIAN_ARENA_BYTES = 1073741824
TRANSFER_PARENT_SHA = "01e1584898c5313ae996ae3c3c57ce23d4cc06ba51d7ab8decbae01c59e88a68"
RESTORED_PARENT_SHA = "319ff350be8c8e6359be978313f484262e2e1f9ddef03d675b7bf0a6ac7968c5"
RESERVED_PARENT_SHA = "ffcec4bbe62f63f77fcdfdac53d322fe19ef355dda40b7a2b1ee394015f81ed0"
MAX_ANGLE = [0.02, 0.015, 0.015, 0.02, 0.01, 0.0075] * 2
FIXED_BANK_ROWS, CURRENT_BANK_ROWS, ROLLOUT_STEPS = 4096, 1024, 24


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def cpu_state(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {key:cpu_state(item) for key,item in value.items()}
    if isinstance(value, list): return [cpu_state(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def evidence_json(value):
    """Keep failed-solver diagnostics serializable without repairing its data."""
    if isinstance(value, torch.Tensor): return evidence_json(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray): return evidence_json(value.tolist())
    if isinstance(value, np.generic): return evidence_json(value.item())
    if isinstance(value, dict): return {str(key):evidence_json(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)): return [evidence_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value): return repr(value)
    return value


def state_digest(value):
    digest = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([str(array.dtype),list(array.shape)]).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(json.dumps([str(item.dtype),list(item.shape)]).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x:(str(type(x)),str(x))):
                digest.update(repr(key).encode());walk(item[key])
        elif isinstance(item, (list,tuple)):
            digest.update(type(item).__name__.encode())
            for child in item: walk(child)
        else: digest.update(repr(item).encode())
    walk(value)
    return digest.hexdigest()


def save_packet(path, packet):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(packet, temporary)
    temporary.replace(path)


def create_fixed_real_bank(dataset_path, output, seed=42):
    """Copy actual TRAIN inputs only; never query a teacher or derive mean inputs."""
    root, output = Path(dataset_path), Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    require(manifest.get("immutable_after_collection") is True and manifest["status"] == "completed",
            "Functional bank requires the frozen completed parent dataset")
    fields = ("obs","lower","upper","stiffness","caps","phase","sample_uid","env_id","valid_label","clip")
    arrays = {name:np.load(root/"train"/(name+".npy"), mmap_mode="r", allow_pickle=False) for name in fields}
    for name,array in arrays.items():
        spec = manifest["train"]["files"][name]
        digest = hashlib.sha256()
        flat=array.reshape(-1);chunk=max(1,4*2**20//array.dtype.itemsize)
        for begin in range(0,len(flat),chunk): digest.update(flat[begin:begin+chunk].tobytes())
        require(digest.hexdigest()==spec["array_sha256"],"Parent functional-bank source changed: "+name)
    eligible = arrays["valid_label"] & ~arrays["clip"].any(1) & (arrays["env_id"]%8!=0)
    rng=np.random.default_rng(seed)
    selected=[]
    for phase in range(4):
        pool=np.flatnonzero(eligible & (arrays["phase"]==phase))
        require(len(pool)>=1024,"Insufficient distinct actual TRAIN phase rows for the fixed guard bank")
        selected.append(rng.choice(pool,1024,replace=False))
    rows=np.concatenate(selected)
    packet={name:np.asarray(arrays[name][rows]).copy() for name in ("obs","lower","upper","stiffness","caps","phase","sample_uid","env_id")}
    packet["source_row"]=rows
    require(len(np.unique(packet["sample_uid"]))==FIXED_BANK_ROWS,"Functional bank duplicated parent UIDs")
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**packet)
    metadata=dict(rows=FIXED_BANK_ROWS,source_dataset=str(root.resolve()),source_manifest_sha256=sha(root/"manifest.json"),
        source_array_sha256={name:manifest["train"]["files"][name]["array_sha256"] for name in fields},
        selected_row_digest=state_digest(rows),selected_UID_digest=state_digest(packet["sample_uid"]),
        bank_path=str(output.resolve()),bank_sha256=sha(output),per_phase_rows=1024,seed=seed,
        selection="Frozen TRAIN valid/no-clip; independent uniformly selected real rows per phase; no heldout labels or loss")
    write(output.with_suffix(".json"),metadata)
    return metadata


def validate_typed_policy_transfer(saved, checkpoint, plan, root):
    """A saved PPO policy may seed a new trial; no parent optimizer is resumed."""
    checkpoint,root=Path(checkpoint).resolve(strict=True),Path(root).resolve(strict=True)
    strict=plan.get("trial_protocol")==STRICT_TRIAL
    resource=plan.get("trial_protocol") in (RESOURCE_TRIAL,STRICT_TRIAL)
    reserved=resource or plan.get("trial_protocol")==RESERVED_TRIAL
    projection=plan.get("trial_protocol") in (PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL)
    if strict:
        kind,mode="saved_restored_PPO_v5_policy_transfer","transfer_restored_v5_policy_and_dualnorm_fresh_optimization"
        parent_sha,parent_format,parent_iteration,parent_updates=STRICT_PARENT_SHA,RESOURCE_FORMAT,182,183
        parent_trial=RESOURCE_TRIAL
        require(plan.get("initial_parent_kind")==kind and plan.get("initialization_mode")==mode
                and plan.get("parent_accepted_updates")==183 and plan.get("parent_last_complete_iteration")==182
                and plan.get("parent_runtime_rollback_completed") is True,
                "Typed restored183 transfer is not the reviewed strict-primal initialization")
    elif resource:
        kind,mode="saved_restored_PPO_v4_policy_transfer","transfer_restored_v4_policy_and_dualnorm_fresh_optimization"
        parent_sha,parent_format,parent_iteration,parent_updates=RESOURCE_PARENT_SHA,RESERVED_FORMAT,612,613
        parent_trial=RESERVED_TRIAL
        require(plan.get("initial_parent_kind")==kind and plan.get("initialization_mode")==mode
                and plan.get("parent_accepted_updates")==613 and plan.get("parent_last_complete_iteration")==612
                and plan.get("parent_runtime_rollback_completed") is True,
                "Typed restored613 transfer is not the reviewed resource-bounded initialization")
    elif reserved:
        kind,mode="saved_restored_PPO_v3_policy_transfer","transfer_restored_v3_policy_and_dualnorm_fresh_optimization"
        parent_sha,parent_format,parent_iteration,parent_updates=RESERVED_PARENT_SHA,PROJECTION_FORMAT,104,105
        parent_trial=PROJECTION_TRIAL
        require(plan.get("initial_parent_kind")==kind and plan.get("initialization_mode")==mode
                and plan.get("parent_accepted_updates")==105 and plan.get("parent_last_complete_iteration")==104
                and plan.get("parent_runtime_rollback_completed") is True,
                "Typed restored105 transfer is not the reviewed fixed-reserve initialization")
    elif projection:
        kind,mode="saved_restored_PPO_v2_policy_transfer","transfer_restored_policy_and_dualnorm_fresh_optimization"
        parent_sha,parent_format,parent_iteration,parent_updates=RESTORED_PARENT_SHA,MARGIN_FORMAT,100,101
        parent_trial=MARGIN_TRIAL
        require(plan.get("initial_parent_kind")==kind and plan.get("initialization_mode")==mode
                and plan.get("parent_accepted_updates")==101 and plan.get("parent_last_complete_iteration")==100
                and plan.get("parent_runtime_rollback_completed") is True,
                "Typed restored101 transfer is not the reviewed projection initialization")
    else:
        kind,mode="saved_pre_guard_PPO_v1_policy_transfer","transfer_pre_guard_policy_and_dualnorm_fresh_optimization"
        parent_sha,parent_format,parent_iteration,parent_updates=TRANSFER_PARENT_SHA,FORMAT,164,165
        require(plan.get("trial_protocol")==MARGIN_TRIAL and plan.get("initial_parent_kind")==kind,
                "Typed policy transfer is only declared for a reviewed trial")
    require(str(checkpoint)==str(Path(plan["initial_policy"]).resolve(strict=True))
            and sha(checkpoint)==plan["initial_policy_sha256"]==parent_sha,
            "Typed parent checkpoint identity differs")
    require(saved.get("format")==parent_format and saved.get("iter")==parent_iteration
            and saved["guarded_ppo_state"]["completed_updates"]==parent_updates
            and saved["guarded_ppo_state"]["uncommitted_rollout"] is True,
            "Typed parent is not the exact preserved committed policy")
    if projection:
        require(saved.get("trial_protocol")==parent_trial
                and saved["guarded_ppo_state"].get("trial_protocol")==parent_trial,
                "Restored parent does not carry the exact declared prior protocol")
    receipt_path=Path(plan["initial_policy_CPU_receipt"]).resolve(strict=True)
    require(receipt_path==checkpoint.with_suffix(".validation.json")
            and sha(receipt_path)==plan["initial_policy_CPU_receipt_sha256"],"Typed parent CPU receipt changed")
    receipt=json.loads(receipt_path.read_text())
    require(receipt.get("status")=="passed" and receipt.get("cpu_only") is True
            and receipt["sha256"]==parent_sha and receipt["checkpoint_format"]==parent_format
            and receipt["committed_updates"]==parent_updates,"Typed parent lacks its exact strict CPU receipt")
    parent_run_path=Path(plan["initial_parent_run"]).resolve(strict=True)
    require(sha(parent_run_path)==plan["initial_parent_run_sha256"]==receipt["producer_run_sha256"]
            and Path(receipt["producer_run"]).resolve(strict=True)==parent_run_path,"Typed producer run receipt changed")
    parent_run=json.loads(parent_run_path.read_text())
    expected_status="stopped_functional_guard" if projection else "failed"
    require(parent_run["status"]==expected_status and parent_run["completed_iterations"]==parent_updates,
            "The typed transfer must preserve parent stop status and actual accepted budget")
    rollback_evidence=None
    if projection:
        require(receipt.get("uncommitted_rollout") is True
                and Path(parent_run["final_checkpoint"]).resolve(strict=True)==checkpoint,
                "Restored parent receipt/run do not bind the restored checkpoint")
        rollback_path=Path(parent_run["functional_guard_evidence"]).resolve(strict=True)
        rollback=json.loads(rollback_path.read_text())
        require(rollback.get("complete_update_restored") is True
                and rollback["rollback_state_digest_before"]==rollback["rollback_state_digest_after"]
                and rollback.get("completed_updates")==parent_updates
                and rollback.get("restored_checkpoint_sha256")==parent_sha
                and Path(rollback["restored_checkpoint"]).resolve(strict=True)==checkpoint,
                "Restored parent lacks exact complete-state rollback evidence")
        rollback_evidence=dict(path=str(rollback_path),sha256=sha(rollback_path),
                               persistent_state_digest=rollback["rollback_state_digest_before"])
    parent_source=Path(plan["initial_parent_source_root"]).resolve(strict=True)
    require(parent_source==Path(receipt["source_root"]).resolve(strict=True),"Typed parent source binding differs")
    actual={"source/"+str(p.relative_to(parent_source)):sha(p) for p in parent_source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
    require(actual==receipt["source_sha256"],"Typed parent production source changed")
    protected=plan["protected_network_input_native_source_sha256"]
    require(len(protected)==14,"The reviewed14 protected network/input/native sources must be explicit")
    for key,digest in protected.items():
        require(key.startswith("source/") and actual.get(key)==digest and sha(root/key)==digest,
                "Protected model/input/native source differs across transfer: "+key)
    for path,digest in receipt["external_dependency_sha256"].items():
        require(sha(Path(path).resolve(strict=True))==digest,"Typed parent's physical external dependency changed")
    for key in ("obs_norm_state_dict","privileged_obs_norm_state_dict"):
        require(int(saved[key]["count"])==24576000,"Typed inherited own-normalizer count changed")
    result=dict(kind=kind,mode=mode,
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=parent_sha,
        parent_CPU_receipt=str(receipt_path),parent_CPU_receipt_sha256=sha(receipt_path),
        parent_run=str(parent_run_path),parent_run_sha256=sha(parent_run_path),parent_source_root=str(parent_source),
        parent_source_sha256=actual,parent_source_commit=parent_run["git_commit"],
        parent_external_dependency_sha256=receipt["external_dependency_sha256"],parent_accepted_updates=parent_updates,
        parent_last_complete_iteration=parent_iteration,parent_uncommitted_rollout=True,parent_runtime_rollback_completed=projection,
        protected_network_input_native_source_sha256=protected,model_and_dualnorm_only=True,
        fresh_actor_optimizer=True,fresh_critic_optimizer=True,fresh_RND=True,fresh_RND_optimizer=True,
        parent_RNG_loaded=False,new_starting_iteration=0,new_checkpoint=None,
        norm_counts={"actor":24576000,"critic":24576000})
    if rollback_evidence is not None:result["parent_rollback_evidence"]=rollback_evidence
    return result


def apply_typed_policy_transfer(algorithm, normalizers, saved, provenance):
    """Load only parameters/own norms; prove that no learning/RNG state moved."""
    require(algorithm.trial_protocol in (MARGIN_TRIAL,PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL) and algorithm.completed_updates==0
            and algorithm.actor_optimizer_steps==algorithm.critic_optimizer_steps==algorithm.rnd_optimizer_steps==0,
            "Typed transfer attempted after fresh optimization started")
    expected_kind={MARGIN_TRIAL:"saved_pre_guard_PPO_v1_policy_transfer",
                   PROJECTION_TRIAL:"saved_restored_PPO_v2_policy_transfer",
                   RESERVED_TRIAL:"saved_restored_PPO_v3_policy_transfer",
                   RESOURCE_TRIAL:"saved_restored_PPO_v4_policy_transfer",
                   STRICT_TRIAL:"saved_restored_PPO_v5_policy_transfer"}[algorithm.trial_protocol]
    require(provenance.get("kind")==expected_kind,"Typed transfer provenance does not match this trial")
    require(not algorithm.actor_optimizer.state and not algorithm.critic_optimizer.state
            and not algorithm.rnd_optimizer.state and algorithm.rnd.update_counter==0,
            "Typed transfer requires fresh independent Adam and RND")
    rnd_before=state_digest(algorithm.rnd.state_dict())
    rng_before=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    algorithm.policy.load_state_dict(saved["model_state_dict"],strict=True)
    for name,key in (("actor","obs_norm_state_dict"),("critic","privileged_obs_norm_state_dict")):
        normalizers[name].load_state_dict(saved[key],strict=True);normalizers[name].eval()
        require(state_digest(normalizers[name].state_dict())==state_digest(saved[key]),"Transferred own normalization differs")
    for key,value in algorithm.policy.state_dict().items():
        require(torch.equal(value.detach().cpu(),saved["model_state_dict"][key].detach().cpu()),"Transferred full model differs: "+key)
    require(rnd_before==state_digest(algorithm.rnd.state_dict()),"Typed initialization imported parent RND")
    rng_after=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    require(rng_before==rng_after,"Typed initialization changed or imported RNG state")
    provenance=copy.deepcopy(provenance)
    provenance.update(actor_and_critic_loaded_bitwise=True,dualnorm_loaded_bitwise=True,
                      fresh_RND_digest=rnd_before,loading_did_not_change_fresh_RNG=True)
    algorithm.initialization_transfer=provenance
    return provenance


class FunctionalUpdateRejected(RuntimeError):
    def __init__(self, evidence, restored_checkpoint, discarded_environment_steps, *, reason_kind="nonlinear_target_budget"):
        super().__init__("Whole PPO update rejected by finite-real-input "+reason_kind+"; all update state restored")
        self.evidence, self.restored_checkpoint = str(evidence), str(restored_checkpoint)
        self.discarded_environment_steps = int(discarded_environment_steps)
        self.reason_kind=reason_kind


class RealInputFunctionalGuard:
    def __init__(self, algorithm, normalizers, exploration, fixed_bank, directory, maximum_angle=MAX_ANGLE):
        self.algorithm, self.normalizers, self.exploration = algorithm, normalizers, exploration
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        with np.load(fixed_bank,allow_pickle=False) as saved: self.fixed={key:saved[key].copy() for key in saved.files}
        require(len(self.fixed["obs"])==FIXED_BANK_ROWS,"Expected4096 actual fixed guard inputs")
        self.maximum=np.asarray(maximum_angle,dtype="float32")
        require(np.array_equal(self.maximum,np.asarray(MAX_ANGLE,dtype="float32")),"Functional angular budget changed")
        self.normalizer_reference={name:cpu_state(norm.state_dict()) for name,norm in normalizers.items()}
        self.fixed_digest=state_digest(self.fixed)
        self.pending=None;self.current=[];self.last_consumed=None
        self.last_summary={};self.first_actor_baseline_saved=False
        self._bank=None;self._baseline_q=None

    def prepare_actual_raw(self, obs, lower, upper, stiffness, caps, control_step):
        # Called on the actual VecEnv raw output, before RSL normalization.
        require(obs.shape==(1024,1029) and lower.shape==upper.shape==stiffness.shape==caps.shape==(1024,12),
                "Actual rollout guard dimensions changed")
        self.pending=dict(obs=obs.detach().clone(),lower=lower.detach().clone(),upper=upper.detach().clone(),
                          stiffness=stiffness.detach().clone(),caps=caps.detach().clone(),control_step=int(control_step))

    def consume_actual_raw(self, normalized_obs, rollout_step):
        require(self.pending is not None,"PPO act did not receive an actual raw-input packet")
        p=self.pending;clock=(self.algorithm.completed_updates,int(rollout_step),p["control_step"])
        require(clock!=self.last_consumed,"Actual raw guard packet consumed twice")
        require(0<=rollout_step<ROLLOUT_STEPS,"Guard rollout sampling must cover the original24 control steps")
        with torch.no_grad():
            expected=self.normalizers["actor"](p["obs"])
            require(bool(torch.equal(expected,normalized_obs)),"PPO input is not exactly one own-frozen normalization of actual raw")
        amount=CURRENT_BANK_ROWS//ROLLOUT_STEPS+int(rollout_step<CURRENT_BANK_ROWS%ROLLOUT_STEPS)
        ids=(torch.arange(amount,device=p["obs"].device)*(1024//amount)
             +self.algorithm.completed_updates*73+rollout_step*17)%1024
        entry={name:p[name][ids].detach().cpu().numpy().copy() for name in ("obs","lower","upper","stiffness","caps")}
        entry.update(env_id=ids.cpu().numpy().astype("int32"),control_step=np.full(amount,p["control_step"],dtype="int64"),
                     rollout_step=np.full(amount,rollout_step,dtype="int16"))
        self.current.append(entry);self.last_consumed=clock;self.pending=None

    def assert_frozen(self):
        for name,norm in self.normalizers.items():
            require(not norm.training,"Guarded-flat inherited dual normalization must remain eval")
            require(state_digest(norm.state_dict())==state_digest(self.normalizer_reference[name]),"Inherited dualnorm changed")
        self.exploration.assert_valid()

    def _assemble(self):
        require(len(self.current)==ROLLOUT_STEPS,"Guard missed an actual rollout control step")
        current={name:np.concatenate([part[name] for part in self.current]) for name in self.current[0]}
        require(len(current["obs"])==CURRENT_BANK_ROWS,"Expected1024 time-stratified real current inputs")
        self._bank={name:np.concatenate([self.fixed[name],current[name]]) for name in ("obs","lower","upper","stiffness","caps")}
        self._bank["source_kind"]=np.r_[np.zeros(FIXED_BANK_ROWS,dtype="uint8"),np.ones(CURRENT_BANK_ROWS,dtype="uint8")]
        self._bank["source_uid"]=np.r_[self.fixed["sample_uid"],np.full(CURRENT_BANK_ROWS,-1,dtype="int64")]
        self._bank["source_row"]=np.r_[self.fixed["source_row"],np.full(CURRENT_BANK_ROWS,-1,dtype="int64")]
        self._bank["env_id"]=np.r_[self.fixed["env_id"],current["env_id"]]
        self._bank["control_step"]=np.r_[np.full(FIXED_BANK_ROWS,-1,dtype="int64"),current["control_step"]]
        self._bank["rollout_step"]=np.r_[np.full(FIXED_BANK_ROWS,-1,dtype="int16"),current["rollout_step"]]
        require(all(np.isfinite(self._bank[k]).all() for k in ("obs","lower","upper","stiffness","caps")),"Nonfinite actual guard data")
        require(np.all(self._bank["stiffness"]>0) and np.all(self._bank["caps"]>0)
                and np.all(self._bank["lower"]<self._bank["upper"]),"Invalid actual guard gain/cap/soft range")

    @torch.no_grad()
    def targets(self):
        result=[]
        for start in range(0,len(self._bank["obs"]),1024):
            def tensor(name):return torch.from_numpy(self._bank[name][start:start+1024]).to(self.algorithm.device)
            lo,hi=tensor("lower"),tensor("upper")
            action=self.algorithm.policy.act_inference(self.normalizers["actor"](tensor("obs"))).clamp(-100,100)
            q=(lo+hi)*.5+(hi-lo)*.5*torch.tanh(action)
            require(bool(torch.isfinite(q).all()),"Nonfinite deterministic functional-guard target")
            result.append(q.cpu().numpy().copy())
        return np.concatenate(result)

    def begin(self):
        self.assert_frozen();self._assemble();self._baseline_q=self.targets()
        self.budget=np.minimum(self.maximum[None],.10*self._bank["caps"]/self._bank["stiffness"])
        self.bank_digest=state_digest(self._bank)
        self.last_summary=dict(bank_rows=FIXED_BANK_ROWS+CURRENT_BANK_ROWS,fixed_rows=FIXED_BANK_ROWS,current_rows=CURRENT_BANK_ROWS,
            bank_digest=self.bank_digest,fixed_bank_digest=self.fixed_digest,maximum_ratio=0.,maximum_per_joint_delta_rad=[0.]*12,
            checked_actor_minibatches=0,rejected=False)

    def check(self):
        self.assert_frozen();candidate=self.targets();delta=np.abs(candidate-self._baseline_q)
        ratio=delta/self.budget
        self.last_summary["maximum_ratio"]=max(self.last_summary["maximum_ratio"],float(ratio.max()))
        self.last_summary["maximum_per_joint_delta_rad"]=np.maximum(self.last_summary["maximum_per_joint_delta_rad"],delta.max(0)).tolist()
        self.last_summary["checked_actor_minibatches"]+=1
        if np.any(delta>self.budget): return candidate,delta,ratio
        return None

    def margin_objective(self, weight, epsilon):
        """Full actual-bank forward value, one real-row max subgradient."""
        scanned=self.targets()
        delta=np.abs(scanned-self._baseline_q);ratio=delta/self.budget
        if np.any(delta>self.budget):
            return None,None,(scanned,delta,ratio)
        squared=np.square(ratio)
        row,joint=np.unravel_index(int(np.argmax(squared)),squared.shape)
        full_u=float(squared[row,joint])
        def tensor(name):return torch.from_numpy(self._bank[name][row:row+1]).to(self.algorithm.device)
        with torch.no_grad():normalized=self.normalizers["actor"](tensor("obs"))
        raw=self.algorithm.policy.act_inference(normalized).clamp(-100,100)
        lo,hi=tensor("lower"),tensor("upper")
        q=(lo+hi)*.5+(hi-lo)*.5*torch.tanh(raw)
        q0=torch.as_tensor(self._baseline_q[row,joint],device=q.device,dtype=q.dtype).detach()
        budget=torch.as_tensor(self.budget[row,joint],device=q.device,dtype=q.dtype)
        raw_row_u=((q[0,joint]-q0)/budget).square()
        scan_q=torch.as_tensor(scanned[row,joint],device=q.device,dtype=q.dtype)
        corrected_q=scan_q+(q[0,joint]-q[0,joint].detach())
        row_u=((corrected_q-q0)/budget).square()
        # Exact scanned forward value; gradient of this actual coordinate only.
        # No epsilon, roundoff correction, or penalty changes the hard check.
        full_u_tensor=torch.as_tensor(full_u,device=q.device,dtype=q.dtype)
        u=full_u_tensor+(row_u-row_u.detach())
        eps=torch.as_tensor(epsilon,device=q.device,dtype=q.dtype)
        penalty=weight*(torch.log1p(eps)-torch.log1p(eps-u))
        require(bool(torch.isfinite(penalty)),"Nonfinite actual-input action-margin penalty")
        record=dict(kind="worst_actual_input_log_barrier",weight=weight,epsilon=epsilon,
            full_scan_rows=len(scanned),worst_row=int(row),worst_joint=int(joint),
            full_scan_u=full_u,straight_through_u=float(u.detach()),row_autograd_u=float(row_u.detach()),
            raw_single_row_u=float(raw_row_u.detach()),squared_ratio_roundoff=float(raw_row_u.detach())-full_u,
            q_roundoff_rad=float(q[0,joint].detach())-float(scanned[row,joint]),
            single_row_q_rad=float(q[0,joint].detach()),single_row_raw_mean=float(raw[0,joint].detach()),
            full_scan_ratio=float(ratio[row,joint]),remaining_budget_ratio=1-float(ratio[row,joint]),
            budget_rad=float(self.budget[row,joint]),baseline_q_rad=float(q0),full_scan_q_rad=float(scanned[row,joint]),
            source_kind=int(self._bank["source_kind"][row]),source_uid=int(self._bank["source_uid"][row]),
            source_row=int(self._bank["source_row"][row]),env_id=int(self._bank["env_id"][row]),
            control_step=int(self._bank["control_step"][row]),rollout_step=int(self._bank["rollout_step"][row]),
            actual_raw_row_digest=state_digest(self._bank["obs"][row]),penalty=float(penalty.detach()))
        return penalty,record,None

    def persist_bank(self,path,**extra):
        np.savez_compressed(path,**self._bank,baseline_q=self._baseline_q,budget_rad=self.budget,**extra)

    def clear(self):
        self.current=[];self._bank=None;self._baseline_q=None


class H2GuardedPPO(PPO):
    def __init__(self, policy, *, actor_learning_rate=1e-5, critic_learning_rate=.001,
                 critic_only_iterations=40, actor_max_grad_norm=1., critic_max_grad_norm=1.,
                 trial_protocol="guarded_flat_balance_v1",actor_action_budget_penalty_weight=0.,
                 actor_action_budget_penalty_epsilon=1e-6, actor_proposal_projection=None,
                 actor_proposal_interior_fraction=.5,actor_proposal_max_active_rows=8,
                 actor_proposal_max_solver_sweeps=2048,actor_proposal_working_budget_fraction=None,
                 actor_proposal_jacobian_arena_budget_bytes=None,actor_proposal_primal_precision_policy=None, **kwargs):
        require(actor_learning_rate==1e-5 and critic_learning_rate==.001 and critic_only_iterations==40
                and actor_max_grad_norm==critic_max_grad_norm==1.,"Guarded-flat fixed optimization protocol changed")
        require(kwargs.get("schedule","fixed")=="fixed" and kwargs.get("symmetry_cfg") is None
                and kwargs.get("multi_gpu_cfg") is None and not policy.is_recurrent,
                "Guarded split PPO supports this fixed, feedforward, single-GPU, non-symmetry trial only")
        require(kwargs.get("learning_rate",actor_learning_rate)==actor_learning_rate,"Conflicting declared actor LR")
        kwargs["learning_rate"]=actor_learning_rate
        super().__init__(policy,**kwargs)
        require((trial_protocol=="guarded_flat_balance_v1" and actor_action_budget_penalty_weight==0.)
                or (trial_protocol==MARGIN_TRIAL and actor_action_budget_penalty_weight==.01
                    and self.gamma==.99 and self.lam==.95)
                or (trial_protocol==PROJECTION_TRIAL and actor_action_budget_penalty_weight==0.
                    and self.gamma==.99 and self.lam==.95)
                or (trial_protocol in (RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL) and actor_action_budget_penalty_weight==0.
                    and self.gamma==.99 and self.lam==.95),"Undeclared actor profile or discount change")
        require(actor_action_budget_penalty_epsilon==1e-6,"Actor-margin epsilon changed")
        self.strict_primal_enabled=trial_protocol==STRICT_TRIAL
        require((self.strict_primal_enabled and actor_proposal_primal_precision_policy==PRIMAL_PRECISION_POLICY)
                or (not self.strict_primal_enabled and actor_proposal_primal_precision_policy is None),
                "Direct FP64 primal certification belongs only to the reviewed strict-primal profile")
        self.projection_primal_precision_policy=actor_proposal_primal_precision_policy
        self.resource_projection_enabled=trial_protocol in (RESOURCE_TRIAL,STRICT_TRIAL)
        self.reserved_projection_enabled=trial_protocol in (RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL)
        self.projection_enabled=trial_protocol in (PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL)
        self.projection_kind=RESERVED_KIND if self.reserved_projection_enabled else PROJECTION_KIND
        require((self.projection_enabled and actor_proposal_projection==self.projection_kind
                 and actor_proposal_interior_fraction==.5
                 and actor_proposal_max_active_rows==(None if self.resource_projection_enabled else 8)
                 and actor_proposal_max_solver_sweeps==2048)
                or (not self.projection_enabled and actor_proposal_projection is None),
                "Undeclared simultaneous-proposal protocol")
        require((self.reserved_projection_enabled and actor_proposal_working_budget_fraction==.5)
                or (not self.reserved_projection_enabled and actor_proposal_working_budget_fraction is None),
                "Fixed whole-update working budget belongs only to the reviewed reserved profile")
        self.projection_working_budget_fraction=actor_proposal_working_budget_fraction
        require((self.resource_projection_enabled and actor_proposal_jacobian_arena_budget_bytes==JACOBIAN_ARENA_BYTES)
                or (not self.resource_projection_enabled and actor_proposal_jacobian_arena_budget_bytes is None),
                "Only the new reviewed resource profile may use the fixed1GiB CPU Jacobian arena")
        self.projection_arena_budget_bytes=actor_proposal_jacobian_arena_budget_bytes
        self.projection_interior_fraction=float(actor_proposal_interior_fraction)
        self.projection_mean_elements=sum(p.numel() for p in policy.actor.parameters())
        self.projection_bytes_per_row=12*self.projection_mean_elements*4
        self.projection_max_active_rows=(min(5120,JACOBIAN_ARENA_BYTES//self.projection_bytes_per_row)
            if self.resource_projection_enabled else int(actor_proposal_max_active_rows))
        self.projection_max_solver_sweeps=int(actor_proposal_max_solver_sweeps)
        self.projection_calls=self.projection_committed_minibatches=self.projection_std_hook_calls=0
        self.projection_fullbank_jvp_calls=self.projection_row_jacobian_calls=self.projection_solver_sweeps=0
        self.last_projection_summary={}
        self.trial_protocol=trial_protocol
        self.action_margin_weight=float(actor_action_budget_penalty_weight)
        self.action_margin_epsilon=float(actor_action_budget_penalty_epsilon)
        self.checkpoint_format=(STRICT_FORMAT if self.strict_primal_enabled else RESOURCE_FORMAT if self.resource_projection_enabled else RESERVED_FORMAT if self.reserved_projection_enabled else
            (PROJECTION_FORMAT if self.projection_enabled else (MARGIN_FORMAT if self.action_margin_weight>0 else FORMAT)))
        self.last_margin_summary={};self.initialization_transfer=None
        require(self.num_learning_epochs==5 and self.num_mini_batches==4 and self.entropy_coef==1e-4,
                "Guarded-flat PPO epochs/batches/entropy changed")
        require(self.rnd is not None and self.rnd.num_states==73 and self.rnd.initial_weight==0.,
                "Original73-state RND must be retained with zero reward weight")
        self.actor_names=["actor."+name for name,_ in policy.actor.named_parameters()]+["std"]
        self.critic_names=["critic."+name for name,_ in policy.critic.named_parameters()]
        require(len(self.actor_names)==45 and len(self.critic_names)==8 and hasattr(policy,"std"),"Original split parameter layout changed")
        named=dict(policy.named_parameters())
        self.actor_parameters=[named[name] for name in self.actor_names]
        self.critic_parameters=[named[name] for name in self.critic_names]
        # The new prospective formula is the single-tensor Adam path. Leave
        # legacy v1/v2 foreach=None unchanged instead of asserting GPU equivalence.
        actor_options={"foreach":False} if self.projection_enabled else {}
        self.actor_optimizer=torch.optim.Adam(self.actor_parameters,lr=actor_learning_rate,**actor_options)
        self.critic_optimizer=torch.optim.Adam(self.critic_parameters,lr=critic_learning_rate)
        # Compatibility alias for the unchanged std projection hook; no shared
        # optimizer is used or serialized in this task.
        self.optimizer=self.actor_optimizer
        self.learning_rate=self.actor_learning_rate=float(actor_learning_rate)
        self.critic_learning_rate=float(critic_learning_rate)
        self.critic_only_iterations=critic_only_iterations
        self.actor_max_grad_norm,self.critic_max_grad_norm=actor_max_grad_norm,critic_max_grad_norm
        self.completed_updates=self.actor_optimizer_steps=self.critic_optimizer_steps=self.rnd_optimizer_steps=0
        self.actor_updates_accepted=0;self.guard=None;self.normalizers=None;self.exploration=None
        self.uncommitted_rollout=False;self.last_guard_summary={};self.initial_policy_sha256=None

    def attach_guard(self, normalizers, exploration, fixed_bank, directory, initial_policy_sha256):
        self.normalizers,self.exploration=normalizers,exploration
        for norm in normalizers.values():norm.eval()
        self.guard=RealInputFunctionalGuard(self,normalizers,exploration,fixed_bank,directory)
        self.initial_policy_sha256=initial_policy_sha256

    def act(self,obs,critic_obs):
        require(self.guard is not None,"New flat PPO is missing its actual-raw functional guard")
        self.guard.consume_actual_raw(obs,self.storage.step)
        return super().act(obs,critic_obs)

    def learning_state(self):
        state=dict(completed_updates=self.completed_updates,actor_optimizer_steps=self.actor_optimizer_steps,
            critic_optimizer_steps=self.critic_optimizer_steps,rnd_optimizer_steps=self.rnd_optimizer_steps,
            actor_updates_accepted=self.actor_updates_accepted,critic_only_iterations=self.critic_only_iterations,
            actor_optimizer_parameter_count=len(self.actor_parameters),critic_optimizer_parameter_count=len(self.critic_parameters),
            critic_warmup_complete=self.completed_updates>=self.critic_only_iterations,
            actor_learning_rate=self.actor_optimizer.param_groups[0]["lr"],critic_learning_rate=self.critic_optimizer.param_groups[0]["lr"],
            rnd_learning_rate=self.rnd_optimizer.param_groups[0]["lr"],rnd_reward_weight=float(self.rnd.initial_weight),
            rnd_update_counter=int(self.rnd.update_counter),RND_predictor_updates_enabled=True,
            RND_predictor_actually_updated=self.rnd_optimizer_steps>0,
            rnd_optimizer_steps_per_accepted_iteration=20,rnd_initialization="fresh independent RND and Adam, never BC/imitation state",
            normalizers_frozen=True,normalizer_counts={k:int(v.count) for k,v in self.normalizers.items()} if self.normalizers else {},
            initial_policy_sha256=self.initial_policy_sha256,uncommitted_rollout=self.uncommitted_rollout,
            trial_protocol=self.trial_protocol,action_margin=dict(kind="worst_actual_input_log_barrier" if self.action_margin_weight else "disabled",
                weight=self.action_margin_weight,epsilon=self.action_margin_epsilon,whole_update_reference=True,
                hard_gate_unchanged=True,forward_value="full_actual_bank_scan_with_straight_through_single_real_row_gradient",
                committed_minibatches=self.actor_updates_accepted*20 if self.action_margin_weight else 0,
                latest_summary=self.last_margin_summary),
            finite_guard_limitation="5120 actual inputs; update-change bound is not unseen-state or physical-stability proof",
            last_guard=self.last_guard_summary)
        if self.projection_enabled:
            state["proposal_projection"]=dict(kind=self.projection_kind,protocol=self.trial_protocol,
                interior_fraction=self.projection_interior_fraction,max_active_rows=self.projection_max_active_rows,
                max_solver_sweeps=self.projection_max_solver_sweeps,whole_update_reference=True,linear_full_bank=True,
                hard_gate_unchanged=True,no_retry=True,adam_implementation="single_tensor_foreach_false",
                actor_optimizer_foreach=self.actor_optimizer.param_groups[0]["foreach"],
                committed_minibatches=self.projection_committed_minibatches,proposal_calls=self.projection_calls,
                std_hook_calls=self.projection_std_hook_calls,fullbank_jvp_calls=self.projection_fullbank_jvp_calls,
                row_jacobian_calls=self.projection_row_jacobian_calls,solver_sweeps=self.projection_solver_sweeps,
                latest_summary=copy.deepcopy(self.last_projection_summary),
                limitation="First-order finite5120-bank direction only; original strict nonlinear check remains authoritative")
            if self.reserved_projection_enabled:
                state["proposal_projection"].update(working_budget_fraction=self.projection_working_budget_fraction,
                    whole_update_fixed_working_bound=True,solver_tolerance_reserved_inward=True)
            if self.resource_projection_enabled:
                state["proposal_projection"].update(configured_max_active_rows=None,
                    jacobian_arena_budget_bytes=self.projection_arena_budget_bytes,
                    jacobian_bytes_per_row=self.projection_bytes_per_row,
                    jacobian_arena_capacity_bytes=self.projection_max_active_rows*self.projection_bytes_per_row,
                    jacobian_mean_parameter_elements=self.projection_mean_elements,
                    resource_kind="single_cpu_float32_jacobian_arena_v1")
            if self.strict_primal_enabled:
                state["proposal_projection"].update(primal_precision_policy=self.projection_primal_precision_policy,
                    primal_precision_summary=copy.deepcopy(self.last_projection_summary.get("primal_precision_summary",{})),
                    max_polish_per_QP=1,max_support_pivots=0,max_polish_per_projection=28,
                    solver_optimality_and_direct_primal_separate=True,
                    primal_envelope_is_machine_derived=True,
                    primal_envelope_scope="Direct FP64 dot/reconstruction arithmetic, not an NN or full-bank JVP input-error bound",
                    final_same_direction_cast_JVP_required=True)
        return state

    def _exploration_state(self):
        return dict(optimizer_steps=self.exploration.optimizer_steps,projected_entries=cpu_state(self.exploration.projected_entries),
            negative_proposals=cpu_state(self.exploration.negative_proposals),snapshot=self.exploration.snapshot())

    def packet(self,iteration,infos=None):
        self.guard.assert_frozen()
        np_rng=np.random.get_state()
        return dict(format=self.checkpoint_format,trial_protocol=self.trial_protocol,
            initialization_transfer=copy.deepcopy(self.initialization_transfer),model_state_dict=cpu_state(self.policy.state_dict()),
            actor_optimizer_state_dict=cpu_state(self.actor_optimizer.state_dict()),
            critic_optimizer_state_dict=cpu_state(self.critic_optimizer.state_dict()),
            optimizer_parameter_names={"actor":self.actor_names,"critic":self.critic_names},
            rnd_state_dict=cpu_state(self.rnd.state_dict()),rnd_optimizer_state_dict=cpu_state(self.rnd_optimizer.state_dict()),
            rnd_optimizer_parameter_names=[name for name,_ in self.rnd.predictor.named_parameters()],
            rnd_runtime=dict(update_counter=self.rnd.update_counter,initial_weight=self.rnd.initial_weight,
                             weight=getattr(self.rnd,"weight",self.rnd.initial_weight)),
            obs_norm_state_dict=cpu_state(self.normalizers["actor"].state_dict()),
            privileged_obs_norm_state_dict=cpu_state(self.normalizers["critic"].state_dict()),
            guarded_ppo_state=cpu_state(self.learning_state()),exploration_state=self._exploration_state(),
            rng_state=dict(torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else None,
                python=random.getstate(),numpy=dict(generator=np_rng[0],keys=torch.from_numpy(np_rng[1].astype("int64")),
                    position=int(np_rng[2]),has_gauss=int(np_rng[3]),cached_gaussian=float(np_rng[4]))),
            iter=int(iteration),infos=infos)

    def _snapshot(self):
        packet=self.packet(self.completed_updates-1,{"kind":"pre_whole_actor_update","uncommitted_rollout":True})
        digests=dict(state=state_digest({k:v for k,v in packet.items() if k!="infos"}),
            model=state_digest(packet["model_state_dict"]),actor_optimizer=state_digest(packet["actor_optimizer_state_dict"]),
            critic_optimizer=state_digest(packet["critic_optimizer_state_dict"]),rnd=state_digest(packet["rnd_state_dict"]),
            rnd_optimizer=state_digest(packet["rnd_optimizer_state_dict"]),
            normalizers=state_digest({k:packet[k] for k in ("obs_norm_state_dict","privileged_obs_norm_state_dict")}),
            exploration=state_digest(packet["exploration_state"]))
        return dict(packet=packet,digests=digests,
            torch_rng=torch.get_rng_state().clone(),cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith("cuda") else None,
            python_rng=random.getstate(),numpy_rng=np.random.get_state())

    def _restore(self,snapshot):
        packet=snapshot["packet"]
        self.policy.load_state_dict(packet["model_state_dict"],strict=True)
        self.actor_optimizer.load_state_dict(packet["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(packet["critic_optimizer_state_dict"])
        # RSL updates RND statistics inside rollout's inference_mode and
        # replaces _std with inference tensors. Their in-place state copy is
        # legal only in that mode. Keep Adam reconstruction outside this scope
        # so its moments remain ordinary tensors for any later explicit use.
        with torch.inference_mode():
            self.rnd.load_state_dict(packet["rnd_state_dict"],strict=True)
        self.rnd_optimizer.load_state_dict(packet["rnd_optimizer_state_dict"])
        for name,key in (("actor","obs_norm_state_dict"),("critic","privileged_obs_norm_state_dict")):
            self.normalizers[name].load_state_dict(packet[key],strict=True);self.normalizers[name].eval()
        state=packet["guarded_ppo_state"]
        for key in ("completed_updates","actor_optimizer_steps","critic_optimizer_steps","rnd_optimizer_steps","actor_updates_accepted","uncommitted_rollout"):
            setattr(self,key,state[key])
        self.learning_rate=self.actor_learning_rate=state["actor_learning_rate"]
        self.critic_learning_rate=state["critic_learning_rate"]
        if "action_margin" in state:
            self.last_margin_summary=copy.deepcopy(state["action_margin"]["latest_summary"])
        if self.projection_enabled:
            projection=state["proposal_projection"]
            for attribute,key in (("projection_calls","proposal_calls"),("projection_committed_minibatches","committed_minibatches"),
                    ("projection_std_hook_calls","std_hook_calls"),("projection_fullbank_jvp_calls","fullbank_jvp_calls"),
                    ("projection_row_jacobian_calls","row_jacobian_calls"),("projection_solver_sweeps","solver_sweeps")):
                setattr(self,attribute,projection[key])
            self.last_projection_summary=copy.deepcopy(projection["latest_summary"])
        for name,value in packet["rnd_runtime"].items():setattr(self.rnd,name,value)
        ex=packet["exploration_state"]
        self.exploration.optimizer_steps=ex["optimizer_steps"]
        self.exploration.projected_entries.copy_(ex["projected_entries"].to(self.device))
        self.exploration.negative_proposals.copy_(ex["negative_proposals"].to(self.device))
        for optimizer in (self.actor_optimizer,self.critic_optimizer,self.rnd_optimizer):optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(snapshot["torch_rng"])
        if snapshot["cuda_rng"] is not None:torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        random.setstate(snapshot["python_rng"]);np.random.set_state(snapshot["numpy_rng"])
        self.guard.assert_frozen()

    def _veto(self,snapshot,violation,minibatch):
        candidate,delta,ratio=violation;directory=self.guard.directory
        projection_summary=copy.deepcopy(self.last_projection_summary) if self.projection_enabled else None
        if projection_summary is not None:
            projection_summary.update(nonlinear_hard_check_passed=False,
                                      nonlinear_maximum_ratio=float(ratio.max()))
            self._write_projection_record(projection_summary)
        update=self.completed_updates
        prefix=directory/f"rejected_update_{update:04d}"
        pre=Path(str(prefix)+"_baseline.pt");proposal=Path(str(prefix)+"_candidate.pt")
        restored=directory.parent/f"model_{self.completed_updates-1}_guard_restored.pt"
        save_packet(pre,snapshot["packet"])
        save_packet(proposal,self.packet(self.completed_updates-1,{"kind":"rejected_uncommitted_candidate","minibatch":minibatch}))
        violation_file=Path(str(prefix)+"_actual_inputs.npz")
        self.guard.persist_bank(violation_file,candidate_q=candidate,absolute_delta_q=delta,violation_ratio=ratio,
            violating_row_joint=np.argwhere(delta>self.guard.budget))
        self._restore(snapshot)
        restored_packet=self.packet(self.completed_updates-1,{"kind":"restored_pre_update_no_retry","discard_current_rollout":True})
        # infos intentionally describe the restored stop; all actual train state
        # excluding that descriptive field must be exactly the pre-update state.
        before={k:v for k,v in snapshot["packet"].items() if k!="infos"}
        after={k:v for k,v in restored_packet.items() if k!="infos"}
        require(state_digest(before)==state_digest(after),"Whole-update rollback did not exactly restore persistent learning state")
        save_packet(restored,restored_packet)
        evidence=Path(str(prefix)+".json")
        sampled_environment_steps=int(self.storage.step)*int(self.storage.observations.shape[1])
        write(evidence,dict(status="stopped_functional_guard",update_index=update,minibatch=minibatch,
            reason_kind="nonlinear_target_budget",proposal_projection=projection_summary,
            completed_updates=self.completed_updates,actor_optimizer_steps=self.actor_optimizer_steps,
            critic_optimizer_steps=self.critic_optimizer_steps,rnd_optimizer_steps=self.rnd_optimizer_steps,
            maximum_ratio=float(ratio.max()),violating_row_joints=int((delta>self.guard.budget).sum()),
            per_joint_maximum_delta_rad=delta.max(0).tolist(),bank_digest=self.guard.bank_digest,
            actual_input_evidence=str(violation_file),actual_input_evidence_sha256=sha(violation_file),
            baseline=str(pre),baseline_sha256=sha(pre),candidate=str(proposal),candidate_sha256=sha(proposal),
            restored_checkpoint=str(restored),restored_checkpoint_sha256=sha(restored),
            rollback_state_digest_before=state_digest(before),rollback_state_digest_after=state_digest(after),
            complete_update_restored=True,normalizers_frozen=True,retry_allowed=False,
            discarded_uncommitted_rollout_steps=sampled_environment_steps,
            interpretation="Finite recorded-input functional veto, no physical state rewind or unseen-state guarantee"))
        raise FunctionalUpdateRejected(evidence,restored,sampled_environment_steps)

    def _write_projection_record(self,record):
        with (self.guard.directory/"proposal_projection_minibatches.jsonl").open("a") as stream:
            stream.write(json.dumps(evidence_json(record),allow_nan=False)+"\n")

    def _projection_veto(self,snapshot,metadata,minibatch,proposal,numerical_evidence=None):
        """A failed prospective solve has no fabricated nonlinear violation."""
        directory=self.guard.directory;update=self.completed_updates
        self.projection_fullbank_jvp_calls+=int(metadata.get("fullbank_jvp_calls",0))
        self.projection_row_jacobian_calls+=int(metadata.get("row_jacobian_calls",0))
        self.projection_solver_sweeps+=int(metadata.get("solver_sweeps",0))
        prefix=directory/f"rejected_update_{update:04d}"
        if numerical_evidence is not None:
            require(self.strict_primal_enabled and isinstance(numerical_evidence,dict),
                    "Unexpected numerical evidence outside the strict-primal protocol")
            arrays={}
            for key,value in numerical_evidence.items():
                require(isinstance(key,str) and isinstance(value,np.ndarray) and not value.dtype.hasobject,
                        "Numerical diagnostics require named non-object NumPy arrays")
                arrays[key]=value
            numerical_path=Path(str(prefix)+"_numerical_evidence.npz")
            temporary=numerical_path.with_name(numerical_path.name+".tmp")
            with temporary.open("wb") as stream:np.savez_compressed(stream,**arrays)
            temporary.replace(numerical_path)
            # Only this scalar receipt enters infos/state/JSON. The arrays
            # stay in their original NumPy dtype in a separate forensic file.
            metadata=dict(metadata,numerical_diagnostic_evidence=dict(path=str(numerical_path),
                sha256=sha(numerical_path),arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype))
                    for key,value in arrays.items()},deployable_checkpoint=False))
        pre=Path(str(prefix)+"_baseline.pt");candidate=Path(str(prefix)+"_candidate.pt")
        restored=directory.parent/f"model_{self.completed_updates-1}_guard_restored.pt"
        detail=evidence_json(dict(metadata,kind=self.projection_kind,update_index=update,minibatch=minibatch,
            committed=False,nonlinear_hard_check_passed=None,bank_rows=5120,linear_output_coordinates=61440,
            actor_mean_parameter_tensors=44,adam_parameter_tensors=45,hard_gate_unchanged=True,no_retry=True))
        self.last_projection_summary=copy.deepcopy(detail)
        self._write_projection_record(detail)
        save_packet(pre,snapshot["packet"])
        save_packet(candidate,self.packet(self.completed_updates-1,
            {"kind":"rejected_uncommitted_projection_failure_state","minibatch":minibatch,
             "proposal_committed":False,"projection_failure":detail}))
        proposal_path=Path(str(prefix)+"_prospective_adam_audit.pt")
        save_packet(proposal_path,dict(format="h2_prospective_adam_failure_audit_v1",deployable_checkpoint=False,
            parameter_names=self.actor_names,deltas=None if proposal is None else cpu_state(proposal.deltas),
            new_states=None if proposal is None else cpu_state(proposal.new_states),
            proposal_committed=False,projection_failure=detail))
        actual=Path(str(prefix)+"_actual_inputs.npz")
        self.guard.persist_bank(actual)
        # Restore every whole-update state even if earlier minibatches committed.
        # The failed prospective proposal itself never changes live Adam state.
        self._restore(snapshot)
        restored_packet=self.packet(self.completed_updates-1,
            {"kind":"restored_pre_update_projection_failure_no_retry","discard_current_rollout":True})
        before={key:value for key,value in snapshot["packet"].items() if key!="infos"}
        after={key:value for key,value in restored_packet.items() if key!="infos"}
        require(state_digest(before)==state_digest(after),"Projection-failure rollback did not exactly restore persistent learning state")
        save_packet(restored,restored_packet)
        evidence=Path(str(prefix)+".json")
        sampled_environment_steps=int(self.storage.step)*int(self.storage.observations.shape[1])
        write(evidence,dict(status="stopped_functional_guard",reason_kind="proposal_projection_failed",
            update_index=update,minibatch=minibatch,completed_updates=self.completed_updates,
            actor_optimizer_steps=self.actor_optimizer_steps,critic_optimizer_steps=self.critic_optimizer_steps,
            rnd_optimizer_steps=self.rnd_optimizer_steps,projection_failure=detail,
            bank_digest=self.guard.bank_digest,actual_input_evidence=str(actual),actual_input_evidence_sha256=sha(actual),
            baseline=str(pre),baseline_sha256=sha(pre),candidate=str(candidate),candidate_sha256=sha(candidate),
            prospective_adam_audit=str(proposal_path),prospective_adam_audit_sha256=sha(proposal_path),
            restored_checkpoint=str(restored),restored_checkpoint_sha256=sha(restored),
            rollback_state_digest_before=state_digest(before),rollback_state_digest_after=state_digest(after),
            complete_update_restored=True,normalizers_frozen=True,retry_allowed=False,
            discarded_uncommitted_rollout_steps=sampled_environment_steps,
            interpretation="Prospective first-order finite-bank certificate failed; no physical-fault claim or nonlinear violating row is invented"))
        raise FunctionalUpdateRejected(evidence,restored,sampled_environment_steps,reason_kind="proposal_projection_failed")

    def _projected_actor_step(self,snapshot,minibatch):
        from functional_proposal import adam_proposal,project_actual_bank,ProposalProjectionError
        proposal=None;self.projection_calls+=1
        started=time.perf_counter()
        def runtime_resources():
            result=dict(elapsed_s=time.perf_counter()-started,
                process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                RSS_scope="Whole Isaac process lifetime peak; not the isolated CPU diagnostic3GiB cap")
            if str(self.device).startswith("cuda"):
                result["GPU_process_allocator"]=dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    lifetime_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    scope="Whole training process PyTorch allocator; peaks not reset; excludes non-Torch GPU memory")
            return result
        try:
            proposal=adam_proposal(self.actor_optimizer)
            if len(proposal.parameters)!=45 or any(p is not q for p,q in zip(proposal.parameters,self.actor_parameters)):
                raise ProposalProjectionError(dict(reason="Prospective Adam parameter identities/order changed"))
            if len(tuple(self.policy.actor.parameters()))!=44:
                raise ProposalProjectionError(dict(reason="Original44 actor mean parameters changed"))
            projection_options=dict(interior_fraction=self.projection_interior_fraction,
                max_active_rows=self.projection_max_active_rows,max_solver_sweeps=self.projection_max_solver_sweeps)
            if self.reserved_projection_enabled:
                projection_options["working_budget_fraction"]=self.projection_working_budget_fraction
            if self.resource_projection_enabled:
                projection_options.update(max_active_rows=None,jacobian_arena_budget_bytes=self.projection_arena_budget_bytes)
            if self.strict_primal_enabled:
                projection_options["primal_precision_policy"]=self.projection_primal_precision_policy
            projected,metadata=project_actual_bank(self.policy.actor,self.normalizers["actor"],self.guard._bank,
                self.guard._baseline_q,self.guard.budget,proposal.deltas[:44],
                **projection_options)
            if metadata.get("kind")!=self.projection_kind or metadata.get("feasible") is not True or len(projected)!=44:
                raise ProposalProjectionError(dict(metadata,reason="Malformed prospective projection certificate"))
            if self.reserved_projection_enabled and not (
                    metadata.get("working_budget_fraction")==.5
                    and metadata.get("whole_update_fixed_working_bound") is True
                    and metadata.get("solver_tolerance_reserved_inward") is True):
                raise ProposalProjectionError(dict(metadata,reason="Missing fixed working-bound inward-reserve certificate"))
            if self.resource_projection_enabled:
                row_bytes=self.projection_bytes_per_row;used=metadata.get("jacobian_used_rows",-1)
                if not (metadata.get("resource_kind")=="single_cpu_float32_jacobian_arena_v1"
                        and metadata.get("configured_max_active_rows","missing") is None
                        and metadata.get("jacobian_storage_location")=="cpu_numpy"
                        and metadata.get("jacobian_storage_dtype")=="float32"
                        and metadata.get("jacobian_arena_budget_bytes")==self.projection_arena_budget_bytes
                        and metadata.get("jacobian_bytes_per_row")==row_bytes
                        and metadata.get("jacobian_derived_max_active_rows")==metadata.get("max_active_rows")==self.projection_max_active_rows
                        and metadata.get("jacobian_arena_capacity_bytes")==row_bytes*self.projection_max_active_rows
                        and metadata.get("jacobian_used_bytes")==metadata.get("jacobian_storage_bytes")==used*row_bytes
                        and 0<=used<=self.projection_max_active_rows
                        and metadata.get("jacobian_arena_allocated_bytes") in (0,row_bytes*self.projection_max_active_rows)
                        and metadata.get("signed_midpoint_bounds_verified_coordinates")==61440):
                    raise ProposalProjectionError(dict(metadata,reason="Incomplete or inconsistent CPU arena coverage certificate"))
                metadata["runtime_resources"]=runtime_resources()
            if self.strict_primal_enabled:
                precision=metadata.get("primal_precision_summary")
                if not (metadata.get("primal_precision_policy")==self.projection_primal_precision_policy
                        and isinstance(precision,dict) and precision.get("direct_primal_certified") is True
                        and precision.get("polish_call_limit_per_subproblem")==1
                        and precision.get("polish_call_limit_per_projection")==28
                        and precision.get("support_pivot_limit")==0
                        and precision.get("final_same_direction_row_JVP_checked") is True):
                    raise ProposalProjectionError(dict(metadata,reason="Missing direct FP64 primal precision certificate"))
            # Std keeps its actual Adam proposal. commit dispatches no optimizer
            # hook; run the unchanged exploration post-hook exactly once here.
            proposal.commit(tuple(projected)+(proposal.deltas[44],))
        except ProposalProjectionError as error:
            if self.resource_projection_enabled:error.metadata["runtime_resources"]=runtime_resources()
            self._projection_veto(snapshot,error.metadata,minibatch,proposal,
                numerical_evidence=getattr(error,"diagnostic_evidence",None) if self.strict_primal_enabled else None)
        hook_before=self.exploration.optimizer_steps
        self.exploration._after_step(self.actor_optimizer,(),{})
        require(self.exploration.optimizer_steps==hook_before+1,"Prospective Adam requires exactly one explicit std hook")
        self.projection_std_hook_calls+=1;self.projection_committed_minibatches+=1
        self.projection_fullbank_jvp_calls+=int(metadata.get("fullbank_jvp_calls",0))
        self.projection_row_jacobian_calls+=int(metadata.get("row_jacobian_calls",0))
        self.projection_solver_sweeps+=int(metadata.get("solver_sweeps",0))
        record=evidence_json(dict(metadata,update_index=self.completed_updates,minibatch=minibatch,
            committed=True,nonlinear_hard_check_passed=None,bank_rows=5120,linear_output_coordinates=61440,
            actor_mean_parameter_tensors=44,adam_parameter_tensors=45,adam_implementation="single_tensor_foreach_false",
            std_proposal_unchanged=True,std_hook_calls_this_minibatch=1,whole_update_reference=True,
            hard_gate_unchanged=True,no_retry=True))
        self.last_projection_summary=copy.deepcopy(record)
        return record

    def update(self):
        require(self.guard is not None,"Guarded-flat PPO runtime was not bound")
        self.guard.assert_frozen()
        require(all(g["lr"]==1e-5 for g in self.actor_optimizer.param_groups)
                and all(g["lr"]==.001 for g in self.critic_optimizer.param_groups),"Fixed split PPO LRs changed")
        actor_enabled=self.completed_updates>=self.critic_only_iterations
        self.uncommitted_rollout=True
        snapshot=None
        if actor_enabled:
            self.guard.begin();snapshot=self._snapshot()
            if not self.guard.first_actor_baseline_saved:
                save_packet(self.guard.directory/"first_actor_preupdate.pt",snapshot["packet"])
                self.guard.persist_bank(self.guard.directory/"first_actor_actual_bank.npz")
                self.guard.first_actor_baseline_saved=True
        losses={key:0. for key in ("value_function","surrogate","entropy","rnd","actor_gradient_norm","critic_gradient_norm","KL",
                                  "action_budget_margin","action_budget_margin_gradient_norm")}
        margin_records=[]
        generator=self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
        batches=0
        for batch in generator:
            (obs,critic_obs,actions,target_values,advantages,returns,old_logp,old_mu,old_sigma,hid,masks,rnd_state)=batch
            if self.normalize_advantage_per_mini_batch:
                advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            self.policy.act(obs,masks=masks,hidden_states=hid[0])
            logp=self.policy.get_actions_log_prob(actions);entropy=self.policy.entropy
            mu,sigma=self.policy.action_mean,self.policy.action_std
            values=self.policy.evaluate(critic_obs,masks=masks,hidden_states=hid[1])
            ratio=torch.exp(logp-old_logp.squeeze())
            surrogate=torch.maximum(-advantages.squeeze()*ratio,
                -advantages.squeeze()*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            if self.use_clipped_value_loss:
                clipped=target_values+(values-target_values).clamp(-self.clip_param,self.clip_param)
                value_loss=torch.maximum((values-returns).square(),(clipped-returns).square()).mean()
            else:value_loss=(values-returns).square().mean()
            actor_loss=surrogate-self.entropy_coef*entropy.mean()
            margin_loss=torch.zeros((),device=self.device)
            margin_gradient_norm=torch.zeros((),device=self.device)
            margin_record=None
            if actor_enabled and self.action_margin_weight:
                margin_loss,margin_record,violation=self.guard.margin_objective(self.action_margin_weight,self.action_margin_epsilon)
                if violation is not None:self._veto(snapshot,violation,batches+1)
                components=torch.autograd.grad(margin_loss,self.actor_parameters,retain_graph=True,allow_unused=True)
                margin_gradient_norm=torch.sqrt(sum(g.detach().square().sum() for g in components if g is not None))
                del components
                require(bool(torch.isfinite(margin_gradient_norm)),"Nonfinite action-margin gradient")
                margin_record.update(update_index=self.completed_updates,minibatch=batches+1,
                    margin_gradient_norm=float(margin_gradient_norm),base_actor_loss=float(actor_loss.detach()))
                margin_records.append(margin_record)
                with (self.guard.directory/"action_margin_minibatches.jsonl").open("a") as stream:
                    stream.write(json.dumps(margin_record)+"\n")
                actor_loss=actor_loss+margin_loss
            with torch.no_grad():
                kl=(torch.log(sigma/old_sigma+1e-5)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
            rnd_loss=(self.rnd.predictor(rnd_state)-self.rnd.target(rnd_state).detach()).square().mean()
            require(all(bool(torch.isfinite(x)) for x in (actor_loss,value_loss,rnd_loss,kl)),"Nonfinite guarded-flat optimization loss")
            self.critic_optimizer.zero_grad(set_to_none=True)
            (self.value_loss_coef*value_loss).backward()
            critic_grad=torch.nn.utils.clip_grad_norm_(self.critic_parameters,self.critic_max_grad_norm)
            require(bool(torch.isfinite(critic_grad)),"Nonfinite separate critic gradient")
            self.critic_optimizer.step();self.critic_optimizer_steps+=1
            actor_grad=torch.zeros((),device=self.device)
            projection_record=None
            if actor_enabled:
                self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
                actor_grad=torch.nn.utils.clip_grad_norm_(self.actor_parameters,self.actor_max_grad_norm)
                require(bool(torch.isfinite(actor_grad)),"Nonfinite separate actor gradient")
                if self.projection_enabled:
                    projection_record=self._projected_actor_step(snapshot,batches+1)
                else:self.actor_optimizer.step()
                self.actor_optimizer_steps+=1
            self.rnd_optimizer.zero_grad(set_to_none=True);rnd_loss.backward();self.rnd_optimizer.step();self.rnd_optimizer_steps+=1
            batches+=1
            if actor_enabled:
                violation=self.guard.check()
                if violation is not None:self._veto(snapshot,violation,batches)
                if projection_record is not None:
                    projection_record.update(nonlinear_hard_check_passed=True,
                        nonlinear_whole_update_maximum_ratio=float(self.guard.last_summary["maximum_ratio"]))
                    self.last_projection_summary=copy.deepcopy(projection_record)
                    self._write_projection_record(projection_record)
            for key,value in dict(value_function=value_loss,surrogate=surrogate,entropy=entropy.mean(),rnd=rnd_loss,
                                   actor_gradient_norm=actor_grad,critic_gradient_norm=critic_grad,KL=kl,
                                   action_budget_margin=margin_loss,action_budget_margin_gradient_norm=margin_gradient_norm).items():losses[key]+=float(value.detach())
        require(batches==20,"Original PPO minibatch/epoch count changed")
        self.completed_updates+=1;self.actor_updates_accepted+=int(actor_enabled);self.uncommitted_rollout=False
        if self.projection_enabled:
            require(self.projection_committed_minibatches==self.actor_optimizer_steps==self.actor_updates_accepted*20
                    and self.projection_calls==self.projection_committed_minibatches==self.projection_std_hook_calls
                    ==self.exploration.optimizer_steps,
                    "Committed prospective Adam/std-hook/minibatch counters diverged")
        self.last_guard_summary=dict(self.guard.last_summary) if actor_enabled else dict(critic_only=True,actor_steps=0)
        if margin_records:
            self.last_margin_summary=dict(minibatches=len(margin_records),weight=self.action_margin_weight,epsilon=self.action_margin_epsilon,
                max_full_scan_ratio=max(r["full_scan_ratio"] for r in margin_records),
                max_abs_q_roundoff_rad=max(abs(r["q_roundoff_rad"]) for r in margin_records),
                max_abs_u_roundoff=max(abs(r["squared_ratio_roundoff"]) for r in margin_records),
                max_margin_gradient_norm=max(r["margin_gradient_norm"] for r in margin_records),
                mean_weighted_penalty=sum(r["penalty"] for r in margin_records)/len(margin_records))
        record=dict(update_index=self.completed_updates-1,completed_updates=self.completed_updates,
            actor_enabled=actor_enabled,learning_state=self.learning_state(),guard=self.last_guard_summary,
            before_whole_actor_update=None if snapshot is None else snapshot["digests"],
            after_whole_update_state_digest=state_digest({k:v for k,v in self.packet(self.completed_updates-1).items() if k!="infos"}),
            model_digest=state_digest(self.policy.state_dict()),actor_optimizer_digest=state_digest(self.actor_optimizer.state_dict()),
            critic_optimizer_digest=state_digest(self.critic_optimizer.state_dict()),rnd_digest=state_digest(self.rnd.state_dict()),
            rnd_optimizer_digest=state_digest(self.rnd_optimizer.state_dict()))
        with (self.guard.directory/"accepted_updates.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
        self.guard.clear();self.storage.clear()
        losses={key:value/batches for key,value in losses.items()}
        losses.update(actor_learning_rate=self.actor_learning_rate,critic_learning_rate=self.critic_learning_rate,
            critic_only=float(not actor_enabled),actor_optimizer_steps=float(self.actor_optimizer_steps),
            critic_optimizer_steps=float(self.critic_optimizer_steps),rnd_optimizer_steps=float(self.rnd_optimizer_steps),
            functional_guard_maximum_ratio=float(self.last_guard_summary.get("maximum_ratio",0.)))
        return losses
