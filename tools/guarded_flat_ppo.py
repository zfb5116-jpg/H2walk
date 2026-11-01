"""H2-local split PPO and finite-real-input whole-update rollback guard.

The rollout/storage/timeout/RND interfaces are the installed RSL PPO interfaces.
This module never imports an environment or launches a simulator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
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
COST_FORMAT = "h2_guarded_flat_ppo_v7"
COST_TRIAL = "command_scaled_velocity_cost_v1"
OUTWARD_FORMAT = "h2_guarded_flat_ppo_v8"
OUTWARD_TRIAL = "outward_tilt_growth_v1"
CLEAN_FORCE_FORMAT = "h2_guarded_flat_ppo_v9"
CLEAN_FORCE_TRIAL = "episode_clean_force_eligibility_v1"
CONTEXT_FORMAT = "h2_guarded_flat_ppo_v10"
CONTEXT_TRIAL = "explicit_task_context_v1"
ACTIVE_CONTEXT_TRIAL = "explicit_task_context_active_support_v1"
ACTIVE_CONTEXT_FORMAT = "h2_guarded_flat_ppo_v11"
CONTEXT_TRIALS = (CONTEXT_TRIAL, ACTIVE_CONTEXT_TRIAL)
SUPPORT_POLISH_POLICY = "bounded_dual_active_set_v1"
COST_PARENT_SHA = "fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa"
COST_PARENT_RECEIPT_SHA = "e7016cd9f64cf81e7eb9562bbde371caf44c0d63b8e6bc308ae882bf2d77ee4b"
COST_PARENT_RUN_SHA = "797aecdb0c137d57f396fd7705940921b0ea9624c7b75323f8bcde58caf9e33c"
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
    if plan.get("trial_protocol")==ACTIVE_CONTEXT_TRIAL:
        return validate_typed_active_context_transfer(saved,checkpoint,plan,root)
    context=plan.get("trial_protocol")==CONTEXT_TRIAL
    natural=plan.get("trial_protocol") in (COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL)
    strict=plan.get("trial_protocol")==STRICT_TRIAL
    resource=plan.get("trial_protocol") in (RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL)
    reserved=resource or plan.get("trial_protocol")==RESERVED_TRIAL
    projection=plan.get("trial_protocol") in (PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL)
    restored=projection and not natural
    if natural:
        kind,mode=("saved_complete_PPO_v6_context_extension_transfer",
                   "extend_completed_v6_policy_and_frozen_prefix_norm_fresh_optimization") if context else (
                   "saved_complete_PPO_v6_policy_transfer","transfer_completed_v6_policy_and_dualnorm_fresh_optimization")
        parent_sha,parent_format,parent_iteration,parent_updates=COST_PARENT_SHA,STRICT_FORMAT,999,1000
        parent_trial=STRICT_TRIAL
        require(plan.get("initial_parent_kind")==kind and plan.get("initialization_mode")==mode
                and plan.get("parent_accepted_updates")==1000 and plan.get("parent_last_complete_iteration")==999
                and plan.get("parent_natural_completion") is True
                and plan.get("parent_runtime_rollback_completed") is False
                and plan.get("initial_policy_CPU_receipt_sha256")==COST_PARENT_RECEIPT_SHA
                and plan.get("initial_parent_run_sha256")==COST_PARENT_RUN_SHA,
                "Typed natural1000 transfer is not the reviewed completed-v6 initialization")
    elif strict:
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
            and saved["guarded_ppo_state"]["uncommitted_rollout"] is (not natural),
            "Typed parent is not the exact preserved committed policy")
    if projection:
        require(saved.get("trial_protocol")==parent_trial
                and saved["guarded_ppo_state"].get("trial_protocol")==parent_trial,
                "Typed parent does not carry the exact declared prior protocol")
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
    expected_status="completed" if natural else "stopped_functional_guard" if projection else "failed"
    require(parent_run["status"]==expected_status and parent_run["completed_iterations"]==parent_updates,
            "The typed transfer must preserve parent stop status and actual accepted budget")
    rollback_evidence=None
    if natural:
        require(receipt.get("uncommitted_rollout") is False
                and Path(parent_run["final_checkpoint"]).resolve(strict=True)==checkpoint
                and parent_run.get("total_timesteps")==24576000
                and parent_run.get("discarded_uncommitted_rollout_steps")==0
                and parent_run["guarded_ppo_state"].get("uncommitted_rollout") is False
                and all(saved["guarded_ppo_state"].get(key)==value
                        and parent_run["guarded_ppo_state"].get(key)==value
                        for key,value in (("completed_updates",1000),("actor_optimizer_steps",19200),
                                          ("critic_optimizer_steps",20000),("rnd_optimizer_steps",20000)))
                and all(receipt["optimizers"][key]["steps"]==value
                        for key,value in (("actor",19200),("critic",20000),("rnd",20000))),
                "Natural parent lacks its exact completed1000 final-checkpoint/counter evidence")
    if restored:
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
    context_deltas={"source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py",
                    "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/direct_initialization.py"}
    if context:
        reported=plan.get("reported_source_delta_sha256",{})
        require(set(reported)==context_deltas|{"tools/functional_proposal.py"}
                and all(sha(root/key)==value for key,value in reported.items())
                and plan.get("network_dimensions")==[1034,871,73,12]
                and plan.get("network_or_input_structure_change") is True,
                "Context extension lacks its exact reported source/dimension admission")
    for key,digest in protected.items():
        child_digest=reported[key] if context and key in context_deltas else digest
        require(key.startswith("source/") and actual.get(key)==digest and sha(root/key)==child_digest,
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
        parent_last_complete_iteration=parent_iteration,parent_uncommitted_rollout=not natural,parent_runtime_rollback_completed=restored,
        protected_network_input_native_source_sha256=protected,model_and_dualnorm_only=True,
        fresh_actor_optimizer=True,fresh_critic_optimizer=True,fresh_RND=True,fresh_RND_optimizer=True,
        parent_RNG_loaded=False,new_starting_iteration=0,new_checkpoint=None,
        norm_counts={"actor":24576000,"critic":24576000})
    if rollback_evidence is not None:result["parent_rollback_evidence"]=rollback_evidence
    if natural:result.update(parent_natural_completion=True,parent_rollback_evidence=None)
    if context:
        result.pop("norm_counts");result.pop("model_and_dualnorm_only")
        result.update(model_prefix_and_dualnorm_prefix_only=True,extra_context_dim=5,
            parent_network_dimensions=[1029,866,73,12],network_dimensions=[1034,871,73,12],
            normalizer_prefix_counts={"actor":24576000,"critic":24576000},
            reported_source_delta_sha256=copy.deepcopy(reported))
    return result


def apply_typed_policy_transfer(algorithm, normalizers, saved, provenance):
    """Load only parameters/own norms; prove that no learning/RNG state moved."""
    if algorithm.trial_protocol==ACTIVE_CONTEXT_TRIAL:
        return apply_typed_active_context_transfer(algorithm,normalizers,saved,provenance)
    if algorithm.trial_protocol==CONTEXT_TRIAL:
        return apply_typed_context_policy_transfer(algorithm,normalizers,saved,provenance)
    require(algorithm.trial_protocol in (MARGIN_TRIAL,PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL) and algorithm.completed_updates==0
            and algorithm.actor_optimizer_steps==algorithm.critic_optimizer_steps==algorithm.rnd_optimizer_steps==0,
            "Typed transfer attempted after fresh optimization started")
    expected_kind={MARGIN_TRIAL:"saved_pre_guard_PPO_v1_policy_transfer",
                   PROJECTION_TRIAL:"saved_restored_PPO_v2_policy_transfer",
                   RESERVED_TRIAL:"saved_restored_PPO_v3_policy_transfer",
                   RESOURCE_TRIAL:"saved_restored_PPO_v4_policy_transfer",
                   STRICT_TRIAL:"saved_restored_PPO_v5_policy_transfer",
                   COST_TRIAL:"saved_complete_PPO_v6_policy_transfer",
                   OUTWARD_TRIAL:"saved_complete_PPO_v6_policy_transfer",
                   CLEAN_FORCE_TRIAL:"saved_complete_PPO_v6_policy_transfer"}[algorithm.trial_protocol]
    require(provenance.get("kind")==expected_kind,"Typed transfer provenance does not match this trial")
    if algorithm.trial_protocol in (COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL):
        require(provenance.get("parent_natural_completion") is True
                and provenance.get("parent_uncommitted_rollout") is False
                and provenance.get("parent_runtime_rollback_completed") is False
                and provenance.get("parent_rollback_evidence") is None,
                "Natural-completion initialization must not masquerade as a restored rollout")
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


def apply_typed_context_policy_transfer(algorithm, normalizers, saved, provenance):
    """Explicitly extend three input matrices; all optimization stays fresh."""
    from task_context_normalization import normalization_metadata
    require(algorithm.trial_protocol==CONTEXT_TRIAL
            and provenance.get("kind")=="saved_complete_PPO_v6_context_extension_transfer"
            and provenance.get("mode")=="extend_completed_v6_policy_and_frozen_prefix_norm_fresh_optimization"
            and provenance.get("parent_natural_completion") is True
            and provenance.get("parent_uncommitted_rollout") is False
            and provenance.get("parent_runtime_rollback_completed") is False
            and provenance.get("parent_rollback_evidence") is None,
            "Context transfer is not the reviewed natural-v6 extension")
    require(algorithm.completed_updates==0 and algorithm.actor_optimizer_steps==0
            and algorithm.critic_optimizer_steps==0 and algorithm.rnd_optimizer_steps==0
            and not algorithm.actor_optimizer.state and not algorithm.critic_optimizer.state
            and not algorithm.rnd_optimizer.state and algorithm.rnd.update_counter==0,
            "Context transfer requires fresh PPO/Adam/RND state")
    require(algorithm.policy.actor.extra_context_dim==5 and algorithm.policy.actor.obs_dim==1034,
            "Context transfer requires the explicitly declared extended actor")
    rnd_before=state_digest(algorithm.rnd.state_dict())
    rng_before=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    original=saved["model_state_dict"];current=algorithm.policy.state_dict()
    require(set(original)==set(current),"Context extension may not add parameter names")
    columns={"actor.fusion.net.0.weight":(256,224),"actor.actor.net.0.weight":(256,224),
             "critic.net.0.weight":(512,866)}
    extended={}
    for key,value in current.items():
        parent=original[key]
        if key in columns:
            rows,width=columns[key]
            require(tuple(parent.shape)==(rows,width) and tuple(value.shape)==(rows,width+5),
                    "Unexpected context extension shape: "+key)
            target=torch.zeros_like(value)
            target[:,:width].copy_(parent.to(device=target.device,dtype=target.dtype))
            extended[key]=target
        else:
            require(tuple(value.shape)==tuple(parent.shape),"An undeclared model tensor changed: "+key)
            extended[key]=parent
    algorithm.policy.load_state_dict(extended,strict=True)
    for key,value in algorithm.policy.state_dict().items():
        if key in columns:
            width=columns[key][1]
            require(torch.equal(value[:,:width].detach().cpu(),original[key].detach().cpu())
                    and bool((value[:,width:]==0).all()),"Copied context prefix/new zero columns differ: "+key)
        else:
            require(torch.equal(value.detach().cpu(),original[key].detach().cpu()),
                    "Copied nonextended model differs: "+key)
    for name,key in (("actor","obs_norm_state_dict"),("critic","privileged_obs_norm_state_dict")):
        normalizers[name].load_prefix_state_dict(saved[key])
        require(state_digest(normalizers[name].prefix.state_dict())==state_digest(saved[key]),
                "Copied empirical prefix normalization differs")
    abi=normalization_metadata(normalizers)
    require(abi["prefix_empirical_counts"]=={"actor":24576000,"critic":24576000}
            and abi["context_empirical_count"] is None and abi["context_statistics_fitted"] is False
            and abi["frozen"] is True,"Context normalization is not a real frozen prefix plus fixed block")
    require(rnd_before==state_digest(algorithm.rnd.state_dict()),"Context transfer imported parent RND")
    rng_after=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    require(rng_before==rng_after,"Context transfer changed/imported the fresh RNG")
    result=copy.deepcopy(provenance)
    result.update(actor_and_critic_prefix_loaded_bitwise=True,dualnorm_prefix_loaded_bitwise=True,
        new_context_columns_zero=True,normalizer_prefix_counts=abi["prefix_empirical_counts"],
        normalization_abi=abi,context_extension_columns={key:dict(prefix_columns=shape[1],new_zero_columns=5)
            for key,shape in columns.items()},fresh_RND_digest=rnd_before,loading_did_not_change_fresh_RNG=True)
    algorithm.initialization_transfer=result
    return result


class ActualInputBankAcquisitionRejected(RuntimeError):
    def __init__(self,evidence,reason):
        super().__init__("First-rollout actual input bank rejected before optimization: "+reason)
        self.evidence=str(evidence)
        self.reason_kind="actual_input_bank_acquisition"


class FunctionalUpdateRejected(RuntimeError):
    def __init__(self, evidence, restored_checkpoint, discarded_environment_steps, *, reason_kind="nonlinear_target_budget"):
        super().__init__("Whole PPO update rejected by finite-real-input "+reason_kind+"; all update state restored")
        self.evidence, self.restored_checkpoint = str(evidence), str(restored_checkpoint)
        self.discarded_environment_steps = int(discarded_environment_steps)
        self.reason_kind=reason_kind


def validate_typed_active_context_transfer(saved, checkpoint, plan, root):
    """Validate the exact complete restored-v10 parent; do not resume state."""
    from v11_parent_custody import verify_parent_files, PARENT_KIND, INITIALIZATION_MODE
    custody=verify_parent_files(plan,root)
    require(saved.get("format")==CONTEXT_FORMAT and saved.get("trial_protocol")==CONTEXT_TRIAL
            and saved.get("iter")==371, "v11 initializer is not the exact restored-v10 model")
    gs=saved["guarded_ppo_state"]
    require(gs["completed_updates"]==372 and gs["uncommitted_rollout"] is True
            and gs["actor_optimizer_steps"]==6640 and gs["critic_optimizer_steps"]==gs["rnd_optimizer_steps"]==7440
            and gs["rnd_update_counter"]==saved["rnd_runtime"]["update_counter"]==8952
            and gs["proposal_projection"]["committed_minibatches"]==6640,
            "Actualv10 parent packet history differs from its complete CPU receipt")
    protected=plan["protected_network_input_native_source_sha256"]
    require(len(protected)==14 and all(custody["parent_source_sha256"].get(key)==value
            and sha(Path(root)/key)==value for key,value in protected.items()),
            "v11 may not change the inherited network/input/native14 source files")
    result=dict(kind=PARENT_KIND,mode=INITIALIZATION_MODE,
        parent_checkpoint=str(Path(checkpoint).resolve(strict=True)),parent_checkpoint_sha256=plan["initial_policy_sha256"],
        parent_CPU_receipt=plan["initial_policy_CPU_receipt"],parent_CPU_receipt_sha256=plan["initial_policy_CPU_receipt_sha256"],
        parent_run=plan["initial_parent_run"],parent_run_sha256=plan["initial_parent_run_sha256"],
        parent_source_root=plan["initial_parent_source_root"],parent_source_sha256=custody["parent_source_sha256"],
        parent_source_commit="18840a7e87558e5f664f4ee0996cacc30821b8ba",
        parent_external_dependency_sha256=json.loads(Path(plan["initial_policy_CPU_receipt"]).read_text())["external_dependency_sha256"],
        parent_accepted_updates=372,parent_last_complete_iteration=371,parent_uncommitted_rollout=True,
        parent_natural_completion=False,parent_runtime_rollback_completed=True,parent_rollback_evidence=custody["parent_rollback_evidence"],
        parent_persistent_state_digest=custody["parent_persistent_state_digest"],
        parent_formal_review=custody["parent_formal_review"],parent_formal_review_sha256=custody["parent_formal_review_sha256"],
        parent_formal_result=custody["parent_formal_result"],parent_formal_result_sha256=custody["parent_formal_result_sha256"],
        parent_formal_successful_envs=842,parent_formal_acceptance_passed=False,
        protected_network_input_native_source_sha256=protected,model_and_dualnorm_only=True,
        same_width_context_transfer=True,network_or_input_structure_changed=False,raw_context_semantics_unchanged=True,
        parent_network_dimensions=[1034,871,73,12],network_dimensions=[1034,871,73,12],
        fresh_actor_optimizer=True,fresh_critic_optimizer=True,fresh_RND=True,fresh_RND_optimizer=True,
        parent_RNG_loaded=False,new_starting_iteration=0,new_checkpoint=None,
        normalizer_prefix_counts={"actor":24576000,"critic":24576000})
    return result


def apply_typed_active_context_transfer(algorithm, normalizers, saved, provenance):
    """Copy same-width model and complete own block norms; reset no hidden state."""
    from task_context_normalization import normalization_metadata
    from v11_parent_custody import PARENT_KIND, INITIALIZATION_MODE
    require(algorithm.trial_protocol==ACTIVE_CONTEXT_TRIAL and provenance["kind"]==PARENT_KIND
            and provenance["mode"]==INITIALIZATION_MODE and provenance["parent_uncommitted_rollout"] is True
            and provenance["parent_natural_completion"] is False and provenance["parent_runtime_rollback_completed"] is True,
            "v11 transfer must preserve restored-parent semantics")
    require(algorithm.completed_updates==algorithm.actor_optimizer_steps==algorithm.critic_optimizer_steps
            ==algorithm.rnd_optimizer_steps==0 and not algorithm.actor_optimizer.state
            and not algorithm.critic_optimizer.state and not algorithm.rnd_optimizer.state
            and algorithm.rnd.update_counter==0, "v11 must start all optimization/RND counters fresh")
    rnd_before=state_digest(algorithm.rnd.state_dict())
    rng_before=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    require(set(algorithm.policy.state_dict())==set(saved["model_state_dict"])
            and all(a.shape==saved["model_state_dict"][key].shape for key,a in algorithm.policy.state_dict().items()),
            "v11 must not extend, truncate or rename any model tensor")
    algorithm.policy.load_state_dict(saved["model_state_dict"],strict=True)
    for key,value in algorithm.policy.state_dict().items():
        require(torch.equal(value.detach().cpu(),saved["model_state_dict"][key].detach().cpu()),
                "v11 full inherited model differs: "+key)
    for name,key in (("actor","obs_norm_state_dict"),("critic","privileged_obs_norm_state_dict")):
        normalizers[name].load_state_dict(saved[key],strict=True);normalizers[name].eval()
        require(state_digest(normalizers[name].state_dict())==state_digest(saved[key]),
                "v11 full own prefix/fixed-context normalization differs")
    abi=normalization_metadata(normalizers)
    require(abi==saved["normalization_abi"] and abi["prefix_empirical_counts"]=={"actor":24576000,"critic":24576000},
            "v11 own normalization ABI/counts changed")
    require(rnd_before==state_digest(algorithm.rnd.state_dict()),"v11 imported parentRND state")
    rng_after=state_digest(dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),
        cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    require(rng_before==rng_after,"v11 changed/imported fresh RNG during parameter transfer")
    result=copy.deepcopy(provenance)
    result.update(actor_and_critic_loaded_bitwise=True,dualnorm_loaded_bitwise=True,
        normalization_abi=abi,fresh_RND_digest=rnd_before,loading_did_not_change_fresh_RNG=True)
    algorithm.initialization_transfer=result
    return result


class RealInputFunctionalGuard:
    def __init__(self, algorithm, normalizers, exploration, fixed_bank, directory, maximum_angle=MAX_ANGLE,
                 *, acquisition_spec=None, on_bank_sealed=None):
        self.algorithm, self.normalizers, self.exploration = algorithm, normalizers, exploration
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.context_mode=getattr(algorithm,"trial_protocol",None) in CONTEXT_TRIALS
        self.fixed=None;self.fixed_digest=None;self._sealed_file_hashes={}
        if not self.context_mode:
            with np.load(fixed_bank,allow_pickle=False) as saved: self.fixed={key:saved[key].copy() for key in saved.files}
            require(len(self.fixed["obs"])==FIXED_BANK_ROWS,"Expected4096 actual fixed guard inputs")
            self.fixed_digest=state_digest(self.fixed)
        self.maximum=np.asarray(maximum_angle,dtype="float32")
        require(np.array_equal(self.maximum,np.asarray(MAX_ANGLE,dtype="float32")),"Functional angular budget changed")
        self.normalizer_reference={name:cpu_state(norm.state_dict()) for name,norm in normalizers.items()}
        self.normalization_abi_reference=None
        if self.context_mode:
            from task_context_normalization import normalization_metadata
            require(fixed_bank is None and isinstance(acquisition_spec,dict) and callable(on_bank_sealed),
                    "New context bank must be acquired once from this run, with a seal callback")
            require(acquisition_spec.get("kind")=="new_run_first_rollout_actual_preaction_v1"
                    and acquisition_spec.get("fixed_rows")==4096 and acquisition_spec.get("current_rows")==1024
                    and acquisition_spec.get("wrapper_clip_actions")==100.
                    and isinstance(acquisition_spec.get("bindings"),dict) and acquisition_spec["bindings"],
                    "Actual bank acquisition lacks its declared real-window/source binding")
            require(algorithm.completed_updates==0 and algorithm.actor_optimizer_steps==0
                    and algorithm.critic_optimizer_steps==0 and algorithm.rnd_optimizer_steps==0,
                    "Real bank acquisition cannot start after optimization")
            self.normalization_abi_reference=normalization_metadata(normalizers)
            require(self.normalization_abi_reference["prefix_empirical_counts"]=={"actor":24576000,"critic":24576000}
                    and self.normalization_abi_reference["frozen"] is True,
                    "Actual context acquisition requires the loaded frozen normalization ABI")
            self.acquisition_spec=copy.deepcopy(acquisition_spec);self.on_bank_sealed=on_bank_sealed
            self._fixed_path=self.directory/"fixed_real_bank.npz"
            self._receipt_path=self.directory/"fixed_real_bank.receipt.json"
            self._acquisition_log=self.directory/"fixed_bank_acquisition.jsonl"
            require(not any(path.exists() for path in (self._fixed_path,self._receipt_path,self._acquisition_log)),
                    "Actual bank acquisition would reuse an existing output")
            self._acquisition_rows=[];self._acquisition_controls=[]
            self._initial_actor_digest=state_digest(algorithm.policy.actor.state_dict())
            self._initial_std_digest=state_digest(algorithm.policy.std)
            self.bank_receipt=dict(status="collecting_first_rollout",rows=0,target_rows=4096,
                bank_path=str(self._fixed_path.resolve()),bank_sha256=None,
                receipt_path=str(self._receipt_path.resolve()),receipt_sha256=None,completed_controls=0,
                stratum_counts=[0,0,0,0])
            with self._acquisition_log.open("x") as stream:
                stream.write(json.dumps(dict(event="acquisition_started",kind=acquisition_spec["kind"],
                    bindings=acquisition_spec["bindings"],quotas_per_stratum=[171,171,171,171,170,170]))+"\n")
        self.pending=None;self.current=[];self.last_consumed=None
        self.last_summary={};self.first_actor_baseline_saved=False
        self._bank=None;self._baseline_q=None

    def prepare_actual_raw(self, obs, lower, upper, stiffness, caps, control_step,
                           *, reset_generation=None, episode_length_buf=None):
        # Called on the actual VecEnv raw output, before RSL normalization.
        require(obs.shape==(1024,1034 if self.context_mode else 1029)
                and lower.shape==upper.shape==stiffness.shape==caps.shape==(1024,12),
                "Actual rollout guard dimensions changed")
        self.pending=dict(obs=obs.detach().clone(),lower=lower.detach().clone(),upper=upper.detach().clone(),
                          stiffness=stiffness.detach().clone(),caps=caps.detach().clone(),control_step=int(control_step))
        if self.context_mode:
            require(reset_generation is not None and episode_length_buf is not None
                    and reset_generation.shape==episode_length_buf.shape==(1024,),
                    "Actual context input lacks real reset-generation/episode-age metadata")
            self.pending.update(reset_generation=reset_generation.detach().clone(),
                                episode_length_buf=episode_length_buf.detach().clone())

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
        if self.context_mode:
            for name in ("reset_generation","episode_length_buf"):
                entry[name]=p[name][ids].detach().cpu().numpy().astype("int64",copy=True)
        self.current.append(entry);self.last_consumed=clock;self.pending=None
        if self.context_mode and self.bank_receipt["status"]=="collecting_first_rollout":
            return dict(packet=p,rollout_step=int(rollout_step),clock=clock,normalization_verified=True)

    def acquisition_state(self):
        require(self.context_mode,"Only the explicit-context profile acquires a new fixed bank")
        return copy.deepcopy(self.bank_receipt)

    def acquire_after_act(self, token, actual_actions, *, action_mean, action_sigma):
        """Use the already sampled command; do not query the actor a second time."""
        require(self.context_mode and self.bank_receipt["status"]=="collecting_first_rollout",
                "Actual bank acquisition is not active")
        step=token["rollout_step"];p=token["packet"]
        require(token["normalization_verified"] is True and token["clock"]==self.last_consumed
                and step==len(self._acquisition_controls)==self.algorithm.storage.step
                and self.algorithm.completed_updates==0 and self.algorithm.actor_optimizer_steps==0
                and actual_actions.shape==action_mean.shape==action_sigma.shape==(1024,12),
                "Actual bank action/input/clock does not match the original PPO act")
        stratum,within=divmod(step,6);quotas=(171,171,171,171,170,170)
        begin=sum(quotas[:within]);amount=quotas[within]
        ids=(torch.arange(begin,begin+amount,device=p["obs"].device)+stratum*73)%1024
        native_finite=torch.stack([torch.isfinite(p[key]).all(-1) for key in
            ("lower","upper","stiffness","caps")]).all(0)
        input_valid=(torch.isfinite(p["obs"]).all(-1) & native_finite
            & (p["lower"]<p["upper"]).all(-1) & (p["stiffness"]>0).all(-1) & (p["caps"]>0).all(-1))
        eligible_bit=p["obs"][:,-1]
        input_valid &= (eligible_bit==0)|(eligible_bit==1)
        finite_action=torch.isfinite(actual_actions).all(-1)
        unclipped=finite_action & (actual_actions==actual_actions.clamp(-100,100)).all(-1)
        midpoint=torch.stack((p["lower"],p["upper"]),dim=-1).mean(-1)
        decoded=midpoint+(p["upper"]-p["lower"])*.5*torch.tanh(actual_actions)
        decode_valid=(torch.isfinite(decoded).all(-1) & (decoded>=p["lower"]).all(-1)
                      & (decoded<=p["upper"]).all(-1))
        stats_valid=torch.isfinite(action_mean).all(-1)&torch.isfinite(action_sigma).all(-1)&(action_sigma>0).all(-1)
        valid=input_valid&unclipped&decode_valid&stats_valid
        accepted=ids[valid[ids]]
        entry={key:p[key][accepted].detach().cpu().numpy().copy() for key in
               ("obs","lower","upper","stiffness","caps")}
        for key,value in (("actual_raw_action",actual_actions),("action_mean",action_mean),
                          ("action_sigma",action_sigma),("native_q_request",decoded)):
            entry[key]=value[accepted].detach().cpu().numpy().copy()
        count=int(accepted.numel())
        entry.update(env_id=accepted.cpu().numpy().astype("int32"),
            sample_uid=(p["control_step"]*1024+accepted).cpu().numpy().astype("int64"),
            source_row=(stratum*1024+torch.arange(begin,begin+amount,device=ids.device)[valid[ids]])
                .cpu().numpy().astype("int64"),
            control_step=np.full(count,p["control_step"],dtype="int64"),
            rollout_step=np.full(count,step,dtype="int16"),stratum_index=np.full(count,stratum,dtype="int16"),
            valid_input=input_valid[accepted].cpu().numpy().copy(),
            action_unclipped=unclipped[accepted].cpu().numpy().copy())
        for key in ("reset_generation","episode_length_buf"):
            entry[key]=p[key][accepted].detach().cpu().numpy().astype("int64",copy=True)
        self._acquisition_rows.append(entry)
        record=dict(event="actual_control_action",rollout_step=step,stratum_index=stratum,
            control_step=p["control_step"],planned_rows=amount,admitted_rows=count,
            planned_env_ids=ids.cpu().tolist(),
            rejected_input=int((~input_valid[ids]).sum()),rejected_nonfinite_action=int((~finite_action[ids]).sum()),
            rejected_raw_wrapper_clip=int((finite_action[ids]&~unclipped[ids]).sum()),
            rejected_native_decode=int((~decode_valid[ids]).sum()),
            rejected_action_statistics=int((~stats_valid[ids]).sum()),
            actor_optimizer_steps=self.algorithm.actor_optimizer_steps,extra_NN_forward=False)
        self._acquisition_controls.append(record)
        with self._acquisition_log.open("a") as stream:stream.write(json.dumps(record,allow_nan=False)+"\n")
        self.bank_receipt["rows"]+=count;self.bank_receipt["completed_controls"]+=1
        self.bank_receipt["stratum_counts"][stratum]+=count

    def _acquisition_reject(self,reason):
        partial={key:np.concatenate([entry[key] for entry in self._acquisition_rows])
                 for key in self._acquisition_rows[0]} if self._acquisition_rows else {}
        partial_path=self.directory/"fixed_bank_acquisition_partial.npz"
        with partial_path.open("xb") as stream:np.savez_compressed(stream,**partial)
        self.bank_receipt.update(status="rejected",reason=reason,partial_path=str(partial_path.resolve()),
                                 partial_sha256=sha(partial_path))
        evidence=self.directory/"fixed_bank_acquisition_rejected.json"
        write(evidence,dict(self.bank_receipt,controls=self._acquisition_controls,
            completed_updates=self.algorithm.completed_updates,actor_optimizer_steps=self.algorithm.actor_optimizer_steps,
            bindings=self.acquisition_spec["bindings"],retry_allowed=False,physical_acceptance_claimed=False))
        raise ActualInputBankAcquisitionRejected(evidence,reason)

    @staticmethod
    def _publish_new(temporary,final):
        # The hard link publishes finished bytes without overwriting a prior
        # output; only this task's temporary filename is removed afterwards.
        os.link(temporary,final)
        temporary.unlink()

    def seal_first_rollout(self,storage_step,completed_updates):
        require(self.context_mode,"Only the new context profile seals a rollout bank")
        if self.bank_receipt["status"]=="sealed":return self.acquisition_state()
        require(self.bank_receipt["status"]=="collecting_first_rollout","Actual bank is no longer collecting")
        if storage_step!=24 or completed_updates!=0 or len(self._acquisition_controls)!=24:
            self._acquisition_reject("The first real24-control rollout did not complete before sealing")
        counters=[entry["control_step"] for entry in self._acquisition_controls]
        if counters!=list(range(counters[0],counters[0]+24)) or self.bank_receipt["rows"]!=4096 \
                or self.bank_receipt["stratum_counts"]!=[1024]*4:
            self._acquisition_reject("Real input admission/counter/stratum quota shortfall; no replacement or extension")
        self.assert_frozen()
        require(self.algorithm.completed_updates==0 and self.algorithm.actor_optimizer_steps==0
                and self.algorithm.critic_optimizer_steps==0 and self.algorithm.rnd_optimizer_steps==0
                and not self.algorithm.actor_optimizer.state and not self.algorithm.critic_optimizer.state
                and not self.algorithm.rnd_optimizer.state
                and self._initial_actor_digest==state_digest(self.algorithm.policy.actor.state_dict())
                and self._initial_std_digest==state_digest(self.algorithm.policy.std),
                "Fixed bank was not captured with the unchanged preoptimization policy/std")
        fixed={key:np.concatenate([entry[key] for entry in self._acquisition_rows]) for key in self._acquisition_rows[0]}
        require(fixed["obs"].shape==(4096,1034) and len(np.unique(fixed["sample_uid"]))==4096
                and np.array_equal(fixed["source_row"],np.arange(4096))
                and all(np.array_equal(np.sort(fixed["env_id"][fixed["stratum_index"]==s]),np.arange(1024)) for s in range(4)),
                "Fixed bank UID/env/row schema is not the exact planned real collection")
        temporary=self._fixed_path.with_name(self._fixed_path.name+".pending")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream,**fixed);stream.flush();os.fsync(stream.fileno())
        self._publish_new(temporary,self._fixed_path)
        with self._acquisition_log.open("a") as stream:
            stream.write(json.dumps(dict(event="collection_closed",rows=4096,controls=24,bank_sha256=sha(self._fixed_path)))+"\n")
            stream.flush();os.fsync(stream.fileno())
        receipt=dict(status="sealed",kind="new_run_first_rollout_actual_preaction_v1",rows=4096,
            bank_path=str(self._fixed_path.resolve()),bank_sha256=sha(self._fixed_path),
            acquisition_log_path=str(self._acquisition_log.resolve()),acquisition_log_sha256=sha(self._acquisition_log),
            arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                    array_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()) for key,value in fixed.items()},
            bindings=copy.deepcopy(self.acquisition_spec["bindings"]),normalization_abi=copy.deepcopy(self.normalization_abi_reference),
            normalizer_prefix_counts=copy.deepcopy(self.normalization_abi_reference["prefix_empirical_counts"]),
            completed_controls=24,control_steps=counters,rows_per_stratum=[1024]*4,
            quotas_per_stratum=[171,171,171,171,170,170],source_actor_optimizer_steps=0,
            source_critic_optimizer_steps=0,source_rnd_optimizer_steps=0,
            rnd_rollout_update_counter=int(self.algorithm.rnd.update_counter),
            source_actor_digest=self._initial_actor_digest,source_std_digest=self._initial_std_digest,
            actor_and_std_unchanged=True,all_rows_valid_input=True,all_rows_action_unclipped=True,
            action_semantics="actual sampled12raw and native full_soft_tanh position request; not a teacher label or measured torque",
            selection="four chronological6-control strata; deterministic disjoint env partitions; no replacement",
            initial_partial_is_input_not_completed_episode=True,qualification_zero_allowed=True,
            extra_NN_forward=False,formal_or_heldout_data_used=False,physical_acceptance_claimed=False)
        temp_receipt=self._receipt_path.with_name(self._receipt_path.name+".pending")
        with temp_receipt.open("x") as stream:
            stream.write(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
            stream.flush();os.fsync(stream.fileno())
        self._publish_new(temp_receipt,self._receipt_path)
        self.fixed=fixed;self.fixed_digest=state_digest(fixed)
        self._sealed_file_hashes={self._fixed_path:receipt["bank_sha256"],self._receipt_path:sha(self._receipt_path),
                                 self._acquisition_log:receipt["acquisition_log_sha256"]}
        self.bank_receipt=dict(receipt,receipt_path=str(self._receipt_path.resolve()),
                              receipt_sha256=self._sealed_file_hashes[self._receipt_path])
        self._acquisition_rows=[]
        self.on_bank_sealed(copy.deepcopy(self.bank_receipt))
        self.assert_frozen()
        return self.acquisition_state()

    def assert_frozen(self):
        for name,norm in self.normalizers.items():
            require(not norm.training,"Guarded-flat inherited dual normalization must remain eval")
            require(state_digest(norm.state_dict())==state_digest(self.normalizer_reference[name]),"Inherited dualnorm changed")
        if self.context_mode:
            from task_context_normalization import normalization_metadata
            require(state_digest(normalization_metadata(self.normalizers))==state_digest(self.normalization_abi_reference),
                    "Frozen context normalization ABI changed")
            if self.bank_receipt["status"]=="sealed":
                require(state_digest(self.fixed)==self.fixed_digest
                        and all(path.is_file() and sha(path)==digest for path,digest in self._sealed_file_hashes.items()),
                        "Sealed actual bank/receipt/acquisition bytes changed")
            else:
                require(self.bank_receipt["status"]=="collecting_first_rollout"
                        and self.algorithm.completed_updates==0 and self.algorithm.actor_optimizer_steps==0
                        and self.algorithm.critic_optimizer_steps==0 and self.algorithm.rnd_optimizer_steps==0,
                        "Unsealed actual bank may exist only before the first real24-control optimization")
        self.exploration.assert_valid()

    def _assemble(self):
        if self.context_mode:require(self.bank_receipt["status"]=="sealed","Actual bank was not sealed before actor update")
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
        if self.context_mode:
            for key in ("control_step","rollout_step","reset_generation","episode_length_buf"):
                self._bank[key]=np.r_[self.fixed[key],current[key]]
            self._bank["stratum_index"]=np.r_[self.fixed["stratum_index"],np.full(CURRENT_BANK_ROWS,-1,dtype="int16")]
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
                 actor_proposal_jacobian_arena_budget_bytes=None,actor_proposal_primal_precision_policy=None,
                 actor_proposal_support_polish_policy=None, **kwargs):
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
                or (trial_protocol in (RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL,ACTIVE_CONTEXT_TRIAL) and actor_action_budget_penalty_weight==0.
                    and self.gamma==.99 and self.lam==.95),"Undeclared actor profile or discount change")
        require(actor_action_budget_penalty_epsilon==1e-6,"Actor-margin epsilon changed")
        self.strict_primal_enabled=trial_protocol in (STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL,ACTIVE_CONTEXT_TRIAL)
        require((self.strict_primal_enabled and actor_proposal_primal_precision_policy==PRIMAL_PRECISION_POLICY)
                or (not self.strict_primal_enabled and actor_proposal_primal_precision_policy is None),
                "Direct FP64 primal certification belongs only to the reviewed strict-primal profile")
        self.support_polish_enabled=trial_protocol==ACTIVE_CONTEXT_TRIAL
        require(actor_proposal_support_polish_policy==(SUPPORT_POLISH_POLICY if self.support_polish_enabled else None),
                "Active-support polish belongs only to the reviewedv11 profile")
        self.projection_support_polish_policy=actor_proposal_support_polish_policy
        self.projection_primal_precision_policy=actor_proposal_primal_precision_policy
        self.resource_projection_enabled=trial_protocol in (RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL,ACTIVE_CONTEXT_TRIAL)
        self.reserved_projection_enabled=trial_protocol in (RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL,ACTIVE_CONTEXT_TRIAL)
        self.projection_enabled=trial_protocol in (PROJECTION_TRIAL,RESERVED_TRIAL,RESOURCE_TRIAL,STRICT_TRIAL,COST_TRIAL,OUTWARD_TRIAL,CLEAN_FORCE_TRIAL,CONTEXT_TRIAL,ACTIVE_CONTEXT_TRIAL)
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
        self.checkpoint_format=(ACTIVE_CONTEXT_FORMAT if trial_protocol==ACTIVE_CONTEXT_TRIAL else CONTEXT_FORMAT if trial_protocol==CONTEXT_TRIAL else CLEAN_FORCE_FORMAT if trial_protocol==CLEAN_FORCE_TRIAL else OUTWARD_FORMAT if trial_protocol==OUTWARD_TRIAL else COST_FORMAT if trial_protocol==COST_TRIAL else STRICT_FORMAT if self.strict_primal_enabled else RESOURCE_FORMAT if self.resource_projection_enabled else RESERVED_FORMAT if self.reserved_projection_enabled else
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
        if trial_protocol in CONTEXT_TRIALS:
            require(policy.actor.extra_context_dim==5 and policy.actor.obs_dim==1034
                    and self.projection_mean_elements==782988 and self.projection_bytes_per_row==37583424
                    and self.projection_max_active_rows==28,
                    "New context policy/resource declaration differs from the reviewed actual layout")

    def attach_guard(self, normalizers, exploration, fixed_bank, directory, initial_policy_sha256,
                     *, acquisition_spec=None,on_bank_sealed=None):
        self.normalizers,self.exploration=normalizers,exploration
        for norm in normalizers.values():norm.eval()
        self.initial_policy_sha256=initial_policy_sha256
        self.guard=RealInputFunctionalGuard(self,normalizers,exploration,fixed_bank,directory,
            acquisition_spec=acquisition_spec,on_bank_sealed=on_bank_sealed)

    def act(self,obs,critic_obs):
        require(self.guard is not None,"New flat PPO is missing its actual-raw functional guard")
        token=self.guard.consume_actual_raw(obs,self.storage.step)
        actions=super().act(obs,critic_obs)
        if token is not None:
            self.guard.acquire_after_act(token,actions,action_mean=self.transition.action_mean,
                                         action_sigma=self.transition.action_sigma)
        return actions

    def compute_returns(self,last_critic_obs):
        # The original RSL loop reaches here only after24real env.step and
        # process_env_step returns. Seal before its ordinary critic bootstrap.
        if self.trial_protocol in CONTEXT_TRIALS:
            require(self.guard is not None,"Context rollout has no real-input bank guard")
            if self.guard.bank_receipt["status"]=="collecting_first_rollout":
                self.guard.seal_first_rollout(self.storage.step,self.completed_updates)
            require(self.guard.bank_receipt["status"]=="sealed","Context bank did not seal before PPO update0")
        return super().compute_returns(last_critic_obs)

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
            normalizers_frozen=True,normalizer_counts={k:int(v.count) for k,v in self.normalizers.items()}
                if self.normalizers and self.trial_protocol not in CONTEXT_TRIALS else {},
            initial_policy_sha256=self.initial_policy_sha256,uncommitted_rollout=self.uncommitted_rollout,
            trial_protocol=self.trial_protocol,action_margin=dict(kind="worst_actual_input_log_barrier" if self.action_margin_weight else "disabled",
                weight=self.action_margin_weight,epsilon=self.action_margin_epsilon,whole_update_reference=True,
                hard_gate_unchanged=True,forward_value="full_actual_bank_scan_with_straight_through_single_real_row_gradient",
                committed_minibatches=self.actor_updates_accepted*20 if self.action_margin_weight else 0,
                latest_summary=self.last_margin_summary),
            finite_guard_limitation="5120 actual inputs; update-change bound is not unseen-state or physical-stability proof",
            last_guard=self.last_guard_summary)
        if self.trial_protocol in CONTEXT_TRIALS:
            from task_context_normalization import normalization_metadata
            state.pop("normalizer_counts")
            abi=normalization_metadata(self.normalizers) if self.normalizers else None
            state.update(normalization_abi=abi,
                normalizer_prefix_counts=abi["prefix_empirical_counts"] if abi else {},
                audit_bank_acquisition=self.guard.acquisition_state() if self.guard else None)
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
                    max_polish_per_QP=1345 if self.support_polish_enabled else 1,
                    max_support_pivots=1344 if self.support_polish_enabled else 0,
                    max_polish_per_projection=37660 if self.support_polish_enabled else 28,
                    solver_optimality_and_direct_primal_separate=True,
                    primal_envelope_is_machine_derived=True,
                    primal_envelope_scope="Direct FP64 dot/reconstruction arithmetic, not an NN or full-bank JVP input-error bound",
                    final_same_direction_cast_JVP_required=True)
                if self.support_polish_enabled:
                    state["proposal_projection"].update(support_polish_policy=self.projection_support_polish_policy,
                        support_pivot_limit_rule="2*m",polish_call_limit_per_subproblem_rule="2*m+1")
        return state

    def _exploration_state(self):
        return dict(optimizer_steps=self.exploration.optimizer_steps,projected_entries=cpu_state(self.exploration.projected_entries),
            negative_proposals=cpu_state(self.exploration.negative_proposals),snapshot=self.exploration.snapshot())

    def packet(self,iteration,infos=None):
        self.guard.assert_frozen()
        normalization_fields={}
        if self.trial_protocol in CONTEXT_TRIALS:
            from task_context_normalization import normalization_metadata
            require(self.guard.bank_receipt["status"]=="sealed","An unsealed input bank cannot produce a deployable PPO packet")
            abi=normalization_metadata(self.normalizers)
            normalization_fields=dict(normalization_abi=abi,normalizer_prefix_counts=abi["prefix_empirical_counts"])
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
            iter=int(iteration),infos=infos,**normalization_fields)

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
            if self.support_polish_enabled:
                projection_options["support_polish_policy"]=self.projection_support_polish_policy
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
                        and precision.get("polish_call_limit_per_subproblem")== (1345 if self.support_polish_enabled else 1)
                        and precision.get("polish_call_limit_per_projection")== (37660 if self.support_polish_enabled else 28)
                        and precision.get("support_pivot_limit")== (1344 if self.support_polish_enabled else 0)
                        and precision.get("final_same_direction_row_JVP_checked") is True):
                    raise ProposalProjectionError(dict(metadata,reason="Missing direct FP64 primal precision certificate"))
            if self.support_polish_enabled:
                from v11_parent_custody import validate_support_projection_metadata
                try:
                    validate_support_projection_metadata(
                        {"actor_proposal_support_polish_policy":self.projection_support_polish_policy},
                        dict(support_polish_policy=self.projection_support_polish_policy,max_polish_per_QP=1345,
                            max_support_pivots=1344,max_polish_per_projection=37660,
                            support_pivot_limit_rule="2*m",polish_call_limit_per_subproblem_rule="2*m+1"),metadata)
                except (RuntimeError,TypeError,KeyError) as error:
                    raise ProposalProjectionError(dict(metadata,reason="Invalid bounded support certificate: "+str(error))) from error
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
