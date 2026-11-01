"""Custody manager for one reviewed bounded direct-flat repair and its replay.

Default is a read-only preflight. Starting or replaying requires --start.
No completion path automatically dispatches another simulator or optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from launch_stair_evaluation import (active_simulators, birth, digest, gpu_snapshot,
                                    immutable, process_check, write)
from launch_stair_learning import parse_training_log, fatal_lines, tail_jsonl, stop_policy, LEGS
from evaluate_flat_balance_policy import (validate_outward_reward_plan, validate_outward_reward_metadata,
                                         validate_episode_force_reward_plan, validate_episode_force_reward_metadata)
from task_context_metadata import (CONTEXT_PROTOCOL, CONTEXT_FORMAT, ACTIVE_CONTEXT_PROTOCOL, ACTIVE_CONTEXT_FORMAT,
    CONTEXT_PROTOCOLS, ACTIVE_CONTEXT_PROTOCOLS, FULL_OUTWARD_CONTEXT_PROTOCOL, FULL_OUTWARD_CONTEXT_FORMAT,
    FULL_OUTWARD_REWARD_PROFILE, FORWARD_EXCESS_CONTEXT_PROTOCOL, FORWARD_EXCESS_CONTEXT_FORMAT,
    FORWARD_EXCESS_REWARD_PROFILE, validate_task_context_plan,
    validate_task_context_metadata, validate_normalization_metadata)

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
POINTER = ROOT / "logs/launches/current_main_training.json"
SLOT = ROOT / "logs/stair_learning/flat_balance.slot.lock"
TASK = "Unitree-H2-Flat-Balance-Policy"
KIND = "guarded_flat_balance_ppo_learning"
EVAL_KIND = "guarded_flat_balance_frozen_validation"
V10_RESTORED372_KIND = "single_frozen_validation_of_controlled_guard_restored_v10_context_372"
BRANCH = "codex/h2-stair-ascent-20261030"
INIT_SHA = "ce822132385de7b9c70315087bbcc4da1739d0da8460cf656c39e14aa1424688"
TRANSFER_SHA = "01e1584898c5313ae996ae3c3c57ce23d4cc06ba51d7ab8decbae01c59e88a68"
PROPOSAL_PROTOCOL = "simultaneous_output_proposal_v1"
RESTORED_SHA = "319ff350be8c8e6359be978313f484262e2e1f9ddef03d675b7bf0a6ac7968c5"
RESERVED_PROTOCOL = "reserved_simultaneous_output_proposal_v1"
RESERVED_SHA = "ffcec4bbe62f63f77fcdfdac53d322fe19ef355dda40b7a2b1ee394015f81ed0"
RESOURCE_PROTOCOL = "resource_bounded_reserved_output_proposal_v1"
RESOURCE_SHA = "e96a9fd29cc0d7457db742133345c0224f2dbcdde522692b52eba88ebdd6977d"
PRECISION_PROTOCOL = "strict_primal_resource_output_proposal_v1"
PRECISION_SHA = "4cecd1fc24d618502402c1bd3ed5bcb0fdc7f44a0981a3bfe5855895715176b2"
VELOCITY_COST_PROTOCOL = "command_scaled_velocity_cost_v1"
VELOCITY_COST_SHA = "fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa"
OUTWARD_TILT_PROTOCOL = "outward_tilt_growth_v1"
CLEAN_FORCE_PROTOCOL = "episode_clean_force_eligibility_v1"
TERMINAL = {"completed", "failed", "stopped_no_improvement", "stopped_functional_guard"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def inside(path):
    value = Path(path).resolve(strict=True)
    require(value.is_relative_to(ROOT), f"Outside H2walk: {value}")
    return value


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hashes(directory):
    directory = Path(directory)
    return {str(p): digest(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def root_hashes():
    return {**hashes(ROOT / "source"), **hashes(ROOT / "tools")}


def dataset_stats(directory):
    return {str(p): {k: getattr(p.stat(), k) for k in
                    ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def prior_finished(pointer):
    require(pointer.get("status") in TERMINAL, "Current job is not terminal")
    require(read(pointer["supervision_path"]).get("status") in TERMINAL, "Previous supervision is active")
    case = inside(pointer["active_stair_job"])
    require(read(case / "manager_state.json").get("status") == "finished", "Previous manager still owns a job")
    for row in read(pointer["launch_record"]).get("processes", []):
        try:
            observed = birth(row["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not (observed["start_ticks"] == row["start_ticks"]
                     and observed["pgid"] == row["pgid"]
                     and observed["process_state"] not in ("Z", "X")), "Previous birth is still active")
    return {"case": str(case), "status": "finished"}


def freeze_transfer_lineage(receipt_path):
    """Freeze the already-validated policy ancestry without reloading tensors."""
    frozen, seen = {}, set()
    while receipt_path is not None:
        receipt_path = inside(receipt_path)
        require(str(receipt_path) not in seen, "Cyclic policy transfer receipt")
        seen.add(str(receipt_path))
        receipt = read(receipt_path)
        require(receipt.get("status") == "passed" and receipt.get("cpu_only") is True,
                "An ancestral policy lacks a passed actual CPU receipt")
        frozen[str(receipt_path)] = digest(receipt_path)
        for key, sha_key in (("checkpoint", "sha256"), ("producer_run", "producer_run_sha256")):
            path = inside(receipt[key])
            require(digest(path) == receipt[sha_key], "Ancestral policy/run bytes changed")
            frozen[str(path)] = receipt[sha_key]
        source = inside(receipt["source_root"])
        actual = {"source/" + str(Path(path).relative_to(source)): value for path, value in hashes(source).items()}
        require(actual == receipt["source_sha256"], "Ancestral training source changed")
        frozen.update(hashes(source))
        agent = inside(source.parent.parent / "params/agent.yaml")
        frozen[str(agent)] = digest(agent)
        for name, expected in receipt.get("producer_training_tool_sha256", {}).items():
            path = inside(source.parent / name)
            require(digest(path) == expected, "Ancestral training tool changed")
            frozen[str(path)] = expected
        for key, sha_key in (("initial_policy", "initial_policy_sha256"), ("plan_path", "plan_sha256")):
            if key in receipt:
                path = inside(receipt[key])
                require(digest(path) == receipt[sha_key], "Ancestral initialization or plan changed")
                frozen[str(path)] = receipt[sha_key]
        transfer = receipt.get("typed_initialization_transfer")
        if transfer and transfer.get("parent_rollback_evidence"):
            proof = transfer["parent_rollback_evidence"]
            path = inside(proof["path"])
            require(digest(path) == proof["sha256"], "Ancestral complete rollback evidence changed")
            frozen[str(path)] = proof["sha256"]
        receipt_path = transfer.get("parent_CPU_receipt") if transfer else None
    return frozen


def preflight(args):
    if args.postmortem_review:
        require(not args.validate, "Choose natural completion or explicit postmortem validation")
        return preflight_postmortem(args)
    pointer = read(POINTER)
    require(not pointer.get("stair_experiments_paused_by_user"), "User paused experiments")
    previous = prior_finished(pointer)
    require(not gpu_snapshot()["compute_processes"] and not active_simulators(), "Another Isaac/GPU task is active")
    require(git("branch", "--show-current") == BRANCH, "Wrong development branch")
    require(not git("status", "--porcelain=v1"), "Commit all reviewed source/tools before launch")
    plan = inside(args.plan or pointer.get("next_repair_plan") or pointer["guarded_flat_balance_plan"])
    declared = read(plan)
    require(declared.get("status") == "reviewed_ready", "Flat repair implementation has not been reviewed")
    require([declared[k] for k in ("task", "num_envs", "num_steps_per_env", "max_iterations", "save_interval")]
            == [TASK, 1024, 24, 1000, 500], "Plan sampling/budget changed")
    forward_excess=declared.get("trial_protocol")==FORWARD_EXCESS_CONTEXT_PROTOCOL
    full_outward=declared.get("trial_protocol")==FULL_OUTWARD_CONTEXT_PROTOCOL
    active=declared.get("trial_protocol") in ACTIVE_CONTEXT_PROTOCOLS
    context=declared.get("trial_protocol") in CONTEXT_PROTOCOLS
    if context:
        validate_task_context_plan(declared)
        require(declared.get("source_commit")==git("rev-parse","HEAD"),
                "Explicit context launch is not the reviewed unified source commit")
        reviewed_roots=declared.get("reviewed_root_source_sha256",{})
        require(reviewed_roots and reviewed_roots=={str(Path(path).relative_to(ROOT)):value
                                                    for path,value in root_hashes().items()},
                "Actual ROOT source/tools set differs from the completed independent review")
        deltas=declared.get("reported_source_delta_sha256",{})
        expected_deltas=({"tools/functional_proposal.py"} if active else {"source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py",
                         "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/direct_initialization.py",
                         "tools/functional_proposal.py"})
        require(set(deltas)==expected_deltas and all(digest(ROOT/k)==v for k,v in deltas.items()),
                "Reported context source adapters are not exactly reviewed/pinned")
    else:
        require(declared["network_or_input_structure_change"] is False, "Network/input change requires prior reporting")
    initial = inside(declared["initial_policy"])
    clean_force = context or declared.get("trial_protocol") == CLEAN_FORCE_PROTOCOL
    outward_tilt = clean_force or declared.get("trial_protocol") == OUTWARD_TILT_PROTOCOL
    velocity_cost = declared.get("trial_protocol") == VELOCITY_COST_PROTOCOL
    natural_parent = forward_excess or ((outward_tilt or velocity_cost) and not active)
    precision = outward_tilt or velocity_cost or declared.get("trial_protocol") == PRECISION_PROTOCOL
    resource = precision or declared.get("trial_protocol") == RESOURCE_PROTOCOL
    reserved = resource or declared.get("trial_protocol") == RESERVED_PROTOCOL
    projected = reserved or declared.get("trial_protocol") == PROPOSAL_PROTOCOL
    transfer = projected or declared.get("trial_protocol") == "overspeed_credit_action_margin_v1"
    expected_initial_sha = "f27165c7c3ed0f1033e2ef02b1cfbd4d6120af3d4eac070d8fda7c720d53eb83" if forward_excess else "b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce" if active else VELOCITY_COST_SHA if natural_parent else PRECISION_SHA if precision else RESOURCE_SHA if resource else RESERVED_SHA if reserved else RESTORED_SHA if projected else TRANSFER_SHA if transfer else INIT_SHA
    require(digest(initial) == declared["initial_policy_sha256"] == expected_initial_sha, "Wrong reviewed actual initialization")
    receipt_path = inside(declared["initial_policy_CPU_receipt"]) if transfer else initial.with_suffix(".validation.json")
    receipt = read(receipt_path)
    require(receipt.get("status") == "passed" and receipt["sha256"] == expected_initial_sha, "Initializer CPU integrity receipt is missing")
    dependencies = {str(ROOT / "assets/isaaclab/unitree_robots/h2/h2.usd"):
                    "7f3432d01756efaeb6d4337536a3ead5a9d0c2f4130654bd43a353012b6c2dbd",
                    str(ROOT / "logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt"):
                    "d260001c9850941c95d1ba83196076f27e118e687e72586d0a2bff84f1674c50"}
    for path, expected in dependencies.items():
        require(digest(path) == expected, "Native USD or frozen LiDAR changed")
    baseline = pointer["accepted_flat_baseline"]
    require(digest(inside(baseline["final_checkpoint"])) == baseline["checkpoint_sha256"], "Protected accepted flat PT changed")
    for path, expected in baseline["network_hashes"].items():
        baseline_source=(ROOT / "artifacts/flat_forward/training_source/source" / path
                         if context else ROOT / "source" / path)
        require(digest(baseline_source) == expected, "Protected accepted flat own-source changed")
        if context and not path.endswith("networks/actor_critic_our.py"):
            require(digest(ROOT / "source" / path)==expected,"Unreported LiDAR/network source changed")
    kind, source, parent, checkpoint = KIND, ROOT / "source", None, None
    entry = ROOT / "tools/train.py"
    frozen = {str(initial): expected_initial_sha, str(receipt_path): digest(receipt_path),
              str(plan): digest(plan), **dependencies}
    if context:
        frozen.update(hashes(ROOT/"artifacts/flat_forward"))
    if transfer:
        require(declared.get("initial_parent_kind") == ("saved_complete_PPO_v12_context_policy_transfer" if forward_excess else "saved_guard_restored_v10_context_policy_transfer" if active else "saved_complete_PPO_v6_context_extension_transfer" if context else "saved_complete_PPO_v6_policy_transfer" if natural_parent else "saved_restored_PPO_v5_policy_transfer" if precision else "saved_restored_PPO_v4_policy_transfer" if resource else "saved_restored_PPO_v3_policy_transfer" if reserved else "saved_restored_PPO_v2_policy_transfer" if projected else "saved_pre_guard_PPO_v1_policy_transfer")
                and declared.get("initialization_mode") == ("transfer_completed_v12_policy_and_frozen_dualnorm_fresh_optimization" if forward_excess else "transfer_verified_guard_restored_v10_policy_and_frozen_dualnorm_fresh_optimization" if active else "extend_completed_v6_policy_and_frozen_prefix_norm_fresh_optimization" if context else "transfer_completed_v6_policy_and_dualnorm_fresh_optimization" if natural_parent else "transfer_restored_v5_policy_and_dualnorm_fresh_optimization" if precision else "transfer_restored_v4_policy_and_dualnorm_fresh_optimization" if resource else "transfer_restored_v3_policy_and_dualnorm_fresh_optimization" if reserved else "transfer_restored_policy_and_dualnorm_fresh_optimization" if projected else "transfer_pre_guard_policy_and_dualnorm_fresh_optimization"),
                "New trial must explicitly transfer model/norm only and start fresh optimization")
        require(receipt.get("checkpoint_format") == (FULL_OUTWARD_CONTEXT_FORMAT if forward_excess else CONTEXT_FORMAT if active else "h2_guarded_flat_ppo_v6" if natural_parent else "h2_guarded_flat_ppo_v5" if precision else "h2_guarded_flat_ppo_v4" if resource else "h2_guarded_flat_ppo_v3" if reserved else "h2_guarded_flat_ppo_v2" if projected else "h2_guarded_flat_ppo_v1")
                and receipt.get("cpu_only") is True and receipt.get("committed_updates") == (1000 if forward_excess else 372 if active else 1000 if natural_parent else 183 if precision else 613 if resource else 105 if reserved else 101 if projected else 165)
                and digest(receipt_path) == declared["initial_policy_CPU_receipt_sha256"], "Wrong typed parent checkpoint receipt")
        parent_run = inside(declared["initial_parent_run"])
        require(str(parent_run) == receipt["producer_run"] and digest(parent_run) == receipt["producer_run_sha256"]
                == declared["initial_parent_run_sha256"] and read(parent_run)["status"] == ("completed" if natural_parent else "stopped_functional_guard" if projected else "failed"),
                "Reviewed parent terminal provenance changed or was relabeled")
        if velocity_cost:
            calibration=declared.get("reward_velocity_cost_calibration",{})
            require(declared.get("parent_natural_completion") is True
                    and declared.get("parent_runtime_rollback_completed") is False
                    and declared.get("parent_accepted_updates")==1000
                    and declared.get("parent_last_complete_iteration")==999
                    and calibration.get("kind")=="command_scaled_existing_world_xy_error_v1"
                    and calibration.get("new_weight")==-(10./.3)
                    and calibration.get("command_m_s")==.3
                    and calibration.get("maximum_positive_forward_rate")==3.,
                    "Reviewed complete-parent/command-scaled existing-cost profile changed")
        if outward_tilt and not active:
            require(declared.get("parent_natural_completion") is True
                    and declared.get("parent_runtime_rollback_completed") is False
                    and declared.get("parent_accepted_updates")==1000
                    and declared.get("parent_last_complete_iteration")==999,
                    "Outward-growth trial requires the exact reviewed natural completed-v6 parent")
            if clean_force:
                validate_episode_force_reward_plan(declared)
            else:
                validate_outward_reward_plan(declared)
        if active:
            from v11_parent_custody import verify_parent_files, verify_complete_v12_parent_files
            active_parent=(verify_complete_v12_parent_files(declared,ROOT) if forward_excess
                           else verify_parent_files(declared,ROOT))
            frozen.update(active_parent["frozen_files"])
            validate_episode_force_reward_plan(declared)
        parent_source = inside(declared["initial_parent_source_root"])
        actual_source = {"source/" + str(Path(path).relative_to(parent_source)): value for path, value in hashes(parent_source).items()}
        require(actual_source == receipt["source_sha256"], "Parent model's training source changed")
        for name, expected in declared["protected_network_input_native_source_sha256"].items():
            require(expected == receipt["source_sha256"][name],"Protected parent source changed: "+name)
            child_expected=declared["reported_source_delta_sha256"].get(name,expected) if context and not active else expected
            require(digest(ROOT/name)==child_expected,"Child source differs from the exact reviewed adapter: "+name)
        require(declared["learning"]["actor_action_budget_penalty"]["weight"] == (0. if projected else .01)
                and declared["learning"]["actor_action_budget_penalty"]["epsilon"] == 1e-6,
                "Reviewed fixed target-margin protocol changed")
        if projected:
            projection=declared["learning"]["actor_proposal_projection"]
            require(projection["kind"] == ("simultaneous_actual_bank_reserved_linear_v1" if reserved else "simultaneous_actual_bank_linear_v1")
                    and projection["interior_fraction"] == .5 and projection["max_active_rows"] == (None if resource else 8)
                    and projection["max_solver_sweeps"] == 2048,
                    "Reviewed simultaneous proposal geometry/bounded solver changed")
            if reserved:
                require(projection["working_budget_fraction"]==.5
                        and projection["whole_update_fixed_working_bound"] is True
                        and projection["solver_tolerance_reserved_inward"] is True,
                        "Reviewed fixed whole-update reserve changed")
            if resource:
                require(projection["jacobian_arena_budget_bytes"]==1073741824
                        and projection["derived_max_active_rows"]==28
                        and projection["resource_kind"]=="single_cpu_float32_jacobian_arena_v1",
                        "Reviewed CPU-byte-derived coverage changed")
            if precision:
                require(projection.get("primal_precision_policy")=="direct_fp64_primal_envelope_v1"
                        and projection.get("max_polish_per_QP")== (1345 if active else 1)
                        and projection.get("max_support_pivots")== (1344 if active else 0)
                        and projection.get("max_polish_per_projection")== (37660 if active else 28)
                        and (not active or projection.get("support_polish_policy")=="bounded_dual_active_set_v1"),
                        "Reviewed strict primal feasibility policy changed")
        frozen[str(parent_run)] = digest(parent_run)
        frozen[str(parent_run.parent / "params/agent.yaml")] = digest(parent_run.parent / "params/agent.yaml")
        if resource:
            frozen.update(freeze_transfer_lineage(receipt_path))
    for value in declared.get("evidence", {}).values():
        path = inside(value)
        frozen[str(path)] = digest(path)
    contracts = declared.get("implementation_contracts", [])
    require(len(contracts) >= 2, "Physical bookkeeping and actual RSL optimizer/rollback CPU contracts are required")
    for value in contracts:
        path = inside(value)
        require(read(path).get("status") == "passed", "Necessary implementation contract failed: " + str(path))
        frozen[str(path)] = digest(path)
    boundary = inside(plan.parent / "startup_monitor_boundary_contract.json")
    boundary_receipt = read(boundary)
    require(boundary_receipt.get("status") == "passed"
            and boundary_receipt.get("manager_sha256") == digest(SELF), "Current monitor startup/checkpoint boundary contract is missing")
    frozen[str(boundary)] = digest(boundary)
    dataset = (inside(declared["historical_frozen_real_audit_dataset"]) if context else
               inside(declared["fixed_real_audit_dataset"]) if transfer else initial.parent / "dataset")
    if args.validate:
        parent = inside(args.validate)
        parent_request = request_for(parent)[1]
        require(parent_request["job_kind"] == KIND, "Validation needs this flat PPO run")
        require(read(parent / "manager_state.json").get("status") == "finished", "Training manager has not finished")
        parent_state = read(parent / "supervision.json")
        require(parent_state.get("status") == "completed", "This validation requires natural completed training")
        run = inside(parent_state["run_dir"])
        metadata = read(run / "run.json")
        require(metadata["status"] == "completed" and metadata["completed_iterations"] == 1000,
                "Training did not finish the bounded1000 iterations")
        checkpoint = inside(run / "model_999.pt")
        source = inside(run / "training_source/source")
        entry, kind = ROOT / "tools/evaluate_flat_balance_policy.py", EVAL_KIND
        require(not check_inputs(parent_request), "Training frozen inputs changed before validation")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        checked = subprocess.run(["bash", str(entry.with_suffix(".sh")), "--cpu-check-only",
                                  "--checkpoint", str(checkpoint), "--source_root", str(source)],
                                 cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=180)
        require(checked.returncode == 0, "Strict split-optimizer CPU checkpoint check failed: " + checked.stdout[-4000:])
        validation = checkpoint.with_suffix(".validation.json")
        data = read(validation)
        require(data.get("status") == "passed" and data.get("cpu_only") is True
                and data["sha256"] == digest(checkpoint), "Actual final CPU receipt is incomplete")
        frozen.update({str(checkpoint): digest(checkpoint), str(validation): digest(validation),
                       str(run / "run.json"): digest(run / "run.json")})
        if context:
            bank=data["fixed_context_bank_custody"]
            for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),("seal_path","seal_sha256"),("acquisition_log_path","acquisition_log_sha256")):
                require(digest(inside(bank[key]))==bank[sha_key],"Sealed bank changed after finalCPU")
                frozen[str(inside(bank[key]))]=bank[sha_key]
        dataset = None
    review = inside(declared["reward_review_path"])
    frozen[str(review)] = digest(review)
    roots = root_hashes()
    key = hashlib.sha256(json.dumps(dict(kind=kind, root=roots, source=hashes(source),
        initial=expected_initial_sha, checkpoint=digest(checkpoint) if checkpoint else None,
        plan=digest(plan)), sort_keys=True).encode()).hexdigest()[:20]
    guard = ROOT / "logs/stair_learning" / ("flat_balance_" + key + ".launch.lock")
    require(not guard.exists(), "Identical inputs/protocol already dispatched; no blind retry")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer,
                source_branch=BRANCH, source_commit=git("rev-parse", "HEAD"),
                job_kind=kind, entrypoint=str(entry), source_root=str(source),
                source_hashes=hashes(source), root_hashes=roots, frozen_files=frozen,
                initial_policy=str(initial), initial_policy_sha256=expected_initial_sha,
                initial_parent_kind=declared.get("initial_parent_kind", "direct_imitation_initializer_v1"),
                trial_protocol=declared.get("trial_protocol", "guarded_flat_balance_v1"),
                **({"parent_excluded_prior_active_ROOT_hashes":active_parent["excluded_prior_active_ROOT_hashes"]} if active else {}),
                plan_path=str(plan), plan_sha256=digest(plan), duplicate_guard=str(guard),
                parent_case=str(parent) if parent else None,
                checkpoint=str(checkpoint) if checkpoint else None,
                dataset=str(dataset) if dataset else None,
                dataset_role="historical_frozen_not_new_width_bank" if context else "fixed_real_audit_source",
                task_context_acquisition=declared.get("audit_bank_source") if context else None,
                dataset_stats=dataset_stats(dataset) if dataset else {},
                accepted_flat_baseline=baseline, GPU_soft_budget_MiB=11264)


def v10_restored372_custody(review, parent, parent_request, state):
    """Admit this one restored context policy using its single complete CPU audit."""
    expected_case=ROOT/"logs/stair_learning/flat_balance_20261101_101628_48cecc2f"
    expected_run=ROOT/"logs/rsl_rl/h2_main_policy_manager_based/2026-11-01_10-16-33_guarded_flat_balance_48cecc2f"
    model_commit="18840a7e87558e5f664f4ee0996cacc30821b8ba"
    model_sha="b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce"
    state_sha="f1c5ae30855147af9b12c757f89831fa8af2913406d2ac3145df09e112a12f91"
    producer_plan_sha="560c0b630ece3407ea7a82dc0a3010518291248eab91adee41c698b457faa045"
    postend_tools={"tools/launch_flat_balance_learning.py","tools/evaluate_flat_balance_policy.py",
                   "tools/verify_guard_restored_flat.py"}
    require(review.get("kind")==V10_RESTORED372_KIND and review.get("status")=="reviewed_ready"
            and parent==expected_case
            and parent_request["job_kind"]==KIND and parent_request["trial_protocol"]==CONTEXT_PROTOCOL
            and parent_request["source_commit"]==model_commit==review["model_source_commit"]
            and state.get("status")=="stopped_functional_guard", "Wrong restored-v10 producer")
    tools=review["postend_tool_sha256"]
    require(set(tools)==postend_tools and all(digest(ROOT/name)==value for name,value in tools.items()),
            "Unreviewed postend tool change")
    current,old=root_hashes(),parent_request["root_hashes"]
    allowed={str(ROOT/name) for name in postend_tools}
    require(set(current)==set(old)|{str(ROOT/"tools/verify_guard_restored_flat.py")}
            and all(current.get(name)==value for name,value in old.items() if name not in allowed),
            "A producer source/tool changed outside the three postend tools")
    require(hashes(parent_request["source_root"])==parent_request["source_hashes"],
            "Original producer full source changed")
    for path,value in parent_request["frozen_files"].items():
        require(digest(path)==value,"Original frozen producer input changed: "+path)
    require(digest(parent/"launch.json")==read(parent/"launch.identity.sha256.json")["sha256"],
            "Original producer launch identity changed")
    ended=read(parent/"manager_state.json")
    require(ended.get("status")=="finished" and ended.get("exit_code")==0 and ended.get("error") is None,
            "Producer manager has not finished cleanly")
    run=inside(state["run_dir"]);checkpoint=inside(review["checkpoint"])
    require(run==expected_run and checkpoint==run/"model_371_guard_restored.pt"
            and checkpoint==inside(state["final_checkpoint"])
            and digest(checkpoint)==review["checkpoint_sha256"]==model_sha,
            "Only the recorded restored-v10 policy is eligible")
    guard_path=inside(review["rollback_evidence"]);guard=read(guard_path)
    require(guard_path==run/"functional_guard/rejected_update_0372.json"
            and guard_path==inside(state["functional_guard_evidence"])
            and digest(guard_path)==review["rollback_evidence_sha256"]
            and guard.get("status")=="stopped_functional_guard"
            and guard.get("complete_update_restored") is True and guard.get("retry_allowed") is False
            and guard["rollback_state_digest_before"]==guard["rollback_state_digest_after"]
            ==review["rollback_state_digest"]==state_sha
            and guard["completed_updates"]==guard["update_index"]==state["completed_iterations"]
            ==review["committed_updates"]==372 and guard["minibatch"]==19,
            "Actual restored-v10 rollback identity differs")
    baseline=inside(review["original_baseline_snapshot"])
    require(baseline==run/"functional_guard/rejected_update_0372_baseline.pt"
            and baseline==inside(guard["baseline"]) and digest(baseline)==guard["baseline_sha256"]
            and inside(guard["restored_checkpoint"])==checkpoint
            and guard["restored_checkpoint_sha256"]==model_sha,
            "Baseline/candidate/restored policy identities were mixed")
    terminal_path=inside(review["terminal_review"]);terminal=read(terminal_path)
    require(terminal_path==parent/"terminal_v10_guardstop_review.json"
            and digest(terminal_path)==review["terminal_review_sha256"]
            and terminal.get("manager_finished") is True and terminal.get("GPU_released") is True
            and terminal.get("natural_completion") is False and terminal.get("complete_update_restored") is True
            and terminal.get("checkpoint_sha256")==model_sha
            and terminal.get("rollback_evidence_sha256")==digest(guard_path)
            and terminal.get("pre_tool_change_frozen_differences")==[]
            and terminal.get("terminal_commit_journal_verified") is True,
            "Independent ended-producer terminal review changed")
    receipt_path=inside(review["CPU_receipt"]);receipt=read(receipt_path)
    require(receipt_path==checkpoint.with_suffix(".validation.json")
            and digest(receipt_path)==review["CPU_receipt_sha256"]
            and receipt.get("status")=="passed" and receipt.get("cpu_only") is True
            and inside(receipt["checkpoint"])==checkpoint and receipt["sha256"]==model_sha
            and receipt["checkpoint_format"]==CONTEXT_FORMAT and receipt["trial_protocol"]==CONTEXT_PROTOCOL
            and receipt["checkpoint_iteration"]==371 and receipt["committed_updates"]==372
            and receipt["finite_tensors"]==255 and receipt["uncommitted_rollout"] is True
            and receipt["rnd_rollout_update_counter"]==8952
            and receipt["network_dimensions"]==[1034,871,73,12]
            and receipt.get("normalizer_prefix_frozen_and_parent_bitwise_equal") is True
            and receipt.get("normalizers_frozen_and_parent_bitwise_equal") is False,
            "Restored-v10 complete CPU receipt differs")
    require(receipt["validator_sha256"]==tools["tools/evaluate_flat_balance_policy.py"]
            and receipt["physical_validator_sha256"]==digest(ROOT/"tools/evaluate_direct_initialization.py"),
            "Actual integrity/physical validator changed after the CPU receipt")
    proof=receipt.get("rollback_equivalence",{})
    require(proof.get("status")=="passed" and proof.get("state_excluded_fields")==["infos"]
            and proof.get("all_state_fields_equal") is True
            and proof.get("baseline_state_digest")==proof.get("restored_state_digest")==state_sha
            and proof.get("cuda_rng_equal") is True and type(proof.get("cuda_rng_tensor_count")) is int
            and proof["cuda_rng_tensor_count"]>0 and inside(proof["baseline"])==baseline
            and proof["baseline_sha256"]==guard["baseline_sha256"]
            and proof["evidence_sha256"]==digest(guard_path)
            and proof["inspector_sha256"]==tools["tools/verify_guard_restored_flat.py"]
            and proof.get("restored_packet_reused_from_complete_validator") is True
            and proof.get("restored_checkpoint_loads")==proof.get("baseline_checkpoint_loads")==1
            and proof.get("candidate_checkpoint_loads")==proof.get("NNforward_calls")==proof.get("optimizer_updates")==0
            and proof.get("no_simulation") is True and proof.get("CUDA_runtime_initialized") is False,
            "Independent full-state/GPU-RNG comparison is missing or differs")
    claim_path=inside(checkpoint.with_suffix(".CPU_once.json"));claim=read(claim_path)
    require(claim.get("status")=="passed" and inside(claim["checkpoint"])==checkpoint
            and claim["sha256"]==model_sha and inside(claim["receipt"])==receipt_path
            and claim["receipt_sha256"]==digest(receipt_path)
            and claim["inspector_sha256"]==proof["inspector_sha256"],
            "The exclusive restored-policy CPU claim did not pass")
    source=inside(receipt["source_root"])
    require(source==run/"training_source/source", "Restored-v10 must use its own imported source")
    actual={"source/"+str(Path(path).relative_to(source)):value for path,value in hashes(source).items()}
    require(actual==receipt["source_sha256"],"Restored-v10 imported source changed")
    producer_path=inside(receipt["producer_run"]);producer=read(producer_path)
    require(producer_path==run/"run.json" and digest(producer_path)==receipt["producer_run_sha256"]
            and producer["status"]=="stopped_functional_guard" and producer["git_commit"]==model_commit
            and inside(producer["final_checkpoint"])==checkpoint
            and inside(producer["functional_guard_evidence"])==guard_path
            and producer["completed_iterations"]==372 and producer["total_timesteps"]==9142272
            and producer["actual_sampled_environment_steps"]==9166848
            and producer["discarded_uncommitted_rollout_steps"]==24576,
            "Stopped producer sampling/status/source evidence changed")
    require(receipt["plan_sha256"]==producer_plan_sha==digest(inside(receipt["plan_path"]))
            ==parent_request["plan_sha256"]==review["producer_plan_sha256"],"Producer plan changed")
    validate_task_context_metadata(producer["runtime_flat_geometry"])
    validate_normalization_metadata(receipt["normalization_abi"],receipt["normalizer_prefix_counts"])
    require(receipt["normalization_abi"]==producer["guarded_flat_learning"]["normalization_abi"]
            ==review["normalization_abi"] and receipt["task_context"]==producer["runtime_flat_geometry"]["task_context"],
            "Restored-v10 own normalization/context ABI changed")
    for name,count,steps in (("actor",45,6640),("critic",8,7440),("rnd",4,7440)):
        require(receipt["optimizers"][name]["parameter_count"]==count
                and receipt["optimizers"][name]["steps"]==steps,"Restored-v10 optimizer history differs")
    require(receipt["actor_proposal_projection"]["committed_minibatches"]==6640,
            "Restored-v10 projection commits differ from actual actor Adam")
    frozen=dict(parent_request["frozen_files"])
    frozen.update({str(path):digest(path) for path in (checkpoint,baseline,guard_path,terminal_path,receipt_path,
        claim_path,producer_path,run/"params/agent.yaml",parent/"request.json",parent/"launch.json",
        parent/"launch.identity.sha256.json",parent/"launch.lock",parent/"completion_audit.json")})
    for key in ("candidate","prospective_adam_audit","actual_input_evidence"):
        path=inside(guard[key])
        require(path.parent==run/"functional_guard" and digest(path)==guard[key+"_sha256"],
                "Preserved rejected-update evidence changed: "+key)
        frozen[str(path)]=guard[key+"_sha256"]
    numerical=guard["projection_failure"]["numerical_diagnostic_evidence"]
    path=inside(numerical["path"])
    require(path==run/"functional_guard/rejected_update_0372_numerical_evidence.npz"
            and digest(path)==numerical["sha256"],"Preserved numerical failure evidence changed")
    frozen[str(path)]=numerical["sha256"]
    require(receipt["producer_training_tool_sha256"]==producer["training_tool_sha256"],"Producer tool manifest differs")
    for name,value in producer["training_tool_sha256"].items():
        path=inside(source.parent/name)
        require(digest(path)==value,"Frozen producer tool changed: "+name);frozen[str(path)]=value
    require(proof["producer_state_digest_source_sha256"]==producer["training_tool_sha256"]["guarded_flat_ppo.py"],
            "State comparison did not use the frozen producer digest implementation")
    bank=receipt["fixed_context_bank_custody"]
    for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),
                        ("acquisition_log_path","acquisition_log_sha256"),("seal_path","seal_sha256")):
        path=inside(bank[key]);require(digest(path)==bank[sha_key],"Frozen context bank changed: "+key)
        frozen[str(path)]=bank[sha_key]
    frozen.update({str(ROOT/name):value for name,value in tools.items()})
    return dict(checkpoint=checkpoint,source=source,source_hashes=actual,receipt=receipt,frozen_files=frozen)


def postmortem_custody(review, parent, parent_request, state):
    """Bind a preserved baseline or a completely restored policy without loading it."""
    if review.get("kind")==V10_RESTORED372_KIND:
        return v10_restored372_custody(review,parent,parent_request,state)
    legacy = review.get("kind") == "single_frozen_validation_of_preserved_pre_guard_baseline_v1"
    restored = review.get("kind") == "single_frozen_validation_of_controlled_guard_restored_v4"
    require(legacy or restored, "Unknown postmortem custody protocol")
    expected_status = "failed" if legacy else "stopped_functional_guard"
    require(parent_request["job_kind"] == KIND and state.get("status") == expected_status,
            "Postmortem producer has the wrong terminal state")
    if legacy:
        require(state.get("failure_kind") == "functional_guard_rollback_inference_tensor_error",
                "Legacy preserved baseline requires its recorded rollback failure")
    else:
        require(parent_request.get("trial_protocol") == RESERVED_PROTOCOL,
                "Restored v4 policy requires its actual reserved-proposal producer")
        require(digest(parent / "launch.json") == read(parent / "launch.identity.sha256.json")["sha256"],
                "Producer birth record changed")
    checkpoint = inside(review["checkpoint"])
    run = inside(state["run_dir"])
    require(checkpoint == inside(state["final_checkpoint"]) and checkpoint.parent == run,
            "Only the recorded trusted terminal checkpoint can be replayed")
    original = inside(review["original_baseline_snapshot"])
    require(original.parent == run / "functional_guard" and original.name.endswith("_baseline.pt"),
            "Missing original whole-update baseline")
    checkpoint_sha = digest(checkpoint)
    require(checkpoint_sha == review["checkpoint_sha256"], "Reviewed checkpoint changed")
    extra = []
    if legacy:
        require(checkpoint_sha == digest(original), "Rejected candidate cannot replace the preserved baseline")
        extra.append(parent / "guard_rollback_failure_reconciliation.json")
    else:
        guard_path = inside(review["rollback_evidence"])
        guard = read(guard_path)
        require(guard_path == inside(state["functional_guard_evidence"])
                and guard_path.parent == run / "functional_guard"
                and digest(guard_path) == review["rollback_evidence_sha256"], "Whole-update rollback evidence changed")
        require(guard.get("status") == expected_status and guard.get("complete_update_restored") is True
                and guard.get("retry_allowed") is False
                and guard["rollback_state_digest_before"] == guard["rollback_state_digest_after"]
                == review["rollback_state_digest"], "Actual complete learning-state rollback is not established")
        require(inside(guard["restored_checkpoint"]) == checkpoint
                and guard["restored_checkpoint_sha256"] == checkpoint_sha
                and inside(guard["baseline"]) == original and guard["baseline_sha256"] == digest(original),
                "Candidate, baseline, and restored policy identities were mixed")
        require(guard["completed_updates"] == guard["update_index"] == state["completed_iterations"]
                and checkpoint.name == f"model_{guard['completed_updates'] - 1}_guard_restored.pt",
                "Restored checkpoint iteration differs from committed history")
        terminal_path = inside(review["terminal_review"])
        terminal = read(terminal_path)
        require(digest(terminal_path) == review["terminal_review_sha256"]
                and terminal.get("complete_update_restored") is True
                and terminal.get("checkpoint_sha256") == checkpoint_sha
                and terminal.get("rollback_evidence_sha256") == digest(guard_path),
                "Independent terminal review is not bound to this restored policy")
        extra.extend([guard_path, terminal_path])
    receipt_path = inside(review["CPU_receipt"])
    receipt = read(receipt_path)
    require(digest(receipt_path) == review["CPU_receipt_sha256"] and receipt.get("status") == "passed"
            and receipt.get("cpu_only") is True and receipt["sha256"] == checkpoint_sha
            and receipt["checkpoint_format"] == ("h2_guarded_flat_ppo_v1" if legacy else "h2_guarded_flat_ppo_v4")
            and receipt["committed_updates"] == state["completed_iterations"] == review["committed_updates"],
            "Complete split-optimizer checkpoint lacks its actual strict CPU receipt")
    source = inside(receipt["source_root"])
    require(source == inside(run / "training_source/source"), "Replay must use this weight's own training source")
    actual = {"source/" + str(Path(path).relative_to(source)): value for path, value in hashes(source).items()}
    require(actual == receipt["source_sha256"], "Frozen imported model/plant source changed")
    producer_path = inside(receipt["producer_run"])
    producer = read(producer_path)
    require(producer_path == run / "run.json" and digest(producer_path) == receipt["producer_run_sha256"],
            "Reconciled producer evidence changed")
    require(producer["status"] == expected_status, "Do not relabel a stopped producer as natural completion")
    if restored:
        count = review["committed_updates"]
        require(inside(receipt["checkpoint"]) == checkpoint
                and inside(producer["final_checkpoint"]) == checkpoint
                and inside(producer["functional_guard_evidence"]) == guard_path,
                "Actual producer and integrity receipt target another checkpoint")
        require(receipt["checkpoint_iteration"] == count - 1 and receipt["finite_tensors"] == 253
                and receipt["trial_protocol"] == RESERVED_PROTOCOL
                and receipt["normalizers_frozen_and_parent_bitwise_equal"] is True,
                "Restored policy lacks its complete typed v4 integrity audit")
        require(digest(ROOT / "tools/evaluate_flat_balance_policy.py") == receipt["validator_sha256"]
                and digest(ROOT / "tools/evaluate_direct_initialization.py") == receipt["physical_validator_sha256"],
                "Restored-policy integrity or physical acceptance validator changed")
        require(producer["completed_iterations"] == count and producer["total_timesteps"] == count * 24576
                and producer["actual_sampled_environment_steps"] == (count + 1) * 24576
                and producer["discarded_uncommitted_rollout_steps"] == 24576,
                "Committed and discarded producer sampling counts disagree")
        require(producer["git_commit"] == parent_request["source_commit"] == review["model_source_commit"],
                "Model source commit differs from immutable producer request")
        for name, parameters, steps in (("actor", 45, (count - 40) * 20),
                                        ("critic", 8, count * 20), ("rnd", 4, count * 20)):
            entry = receipt["optimizers"][name]
            require(entry["parameter_count"] == parameters and entry["steps"] == steps,
                    "Restored split optimizer counters disagree")
    frozen = {str(path): digest(path) for path in
              (checkpoint, original, receipt_path, producer_path, run / "params/agent.yaml", *extra)}
    return dict(checkpoint=checkpoint, source=source, source_hashes=actual,
                receipt=receipt, frozen_files=frozen)


def preflight_postmortem(args):
    """Measure one reviewed trusted terminal policy without resuming PPO."""
    pointer = read(POINTER)
    require(not pointer.get("stair_experiments_paused_by_user"), "User paused experiments")
    previous = prior_finished(pointer)
    require(not gpu_snapshot()["compute_processes"] and not active_simulators(), "Another Isaac/GPU task is active")
    require(git("branch", "--show-current") == BRANCH and not git("status", "--porcelain=v1"),
            "Commit the reviewed tools before freezing a validation")
    review_path = inside(args.postmortem_review)
    review = read(review_path)
    require(review.get("status") == "reviewed_ready" and
            review.get("kind") in ("single_frozen_validation_of_preserved_pre_guard_baseline_v1",
                                   "single_frozen_validation_of_controlled_guard_restored_v4",
                                   V10_RESTORED372_KIND),
            "A specific reviewed postmortem measurement plan is required")
    require(review.get("protocol") == dict(num_envs=1024, duration_s=20, speed_m_s=.3, seed=42,
        minimum_successful_envs=922, minimum_forward_m=4.8, maximum_forward_m=7.2,
        minimum_valid_landings_each_foot=2, teacher_interventions=0, learning_updates=0),
        "Independent physical acceptance protocol changed")
    parent = inside(review["parent_case"])
    require(str(parent) == pointer["active_stair_job"], "Postmortem plan targets another previous job")
    parent_request = request_for(parent)[1]
    state = read(parent / "supervision.json")
    custody = postmortem_custody(review, parent, parent_request, state)
    checkpoint, source = custody["checkpoint"], custody["source"]
    actual, receipt = custody["source_hashes"], custody["receipt"]
    frozen = {str(review_path): digest(review_path), **custody["frozen_files"]}
    for value in review.get("evidence", []):
        path = inside(value)
        frozen[str(path)] = digest(path)
    for name, expected in receipt["external_dependency_sha256"].items():
        path = inside(name)
        require(digest(path) == expected, "Training USD/LiDAR dependency changed")
        frozen[str(path)] = expected
    baseline = pointer["accepted_flat_baseline"]
    require(digest(baseline["final_checkpoint"]) == baseline["checkpoint_sha256"], "Protected flat baseline changed")
    for name, expected in baseline["network_hashes"].items():
        published=(ROOT/"artifacts/flat_forward/training_source/source"/name
                   if review["kind"]==V10_RESTORED372_KIND else ROOT/"source"/name)
        require(digest(published) == expected, "Original network/input source changed")
    # Use model/source/protocol identity, not tooling revisions, for deduplication.
    key = hashlib.sha256(json.dumps(dict(kind=EVAL_KIND, checkpoint=digest(checkpoint),
        source=actual, protocol=review["protocol"]), sort_keys=True).encode()).hexdigest()[:20]
    guard = ROOT / "logs/stair_learning" / ("flat_postmortem_" + key + ".launch.lock")
    require(not guard.exists(), "This frozen baseline/protocol was already dispatched")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer,
        source_branch=BRANCH, source_commit=parent_request["source_commit"], launcher_commit=git("rev-parse", "HEAD"),
        job_kind=EVAL_KIND, entrypoint=str(ROOT / "tools/evaluate_flat_balance_policy.py"),
        source_root=str(source), source_hashes=hashes(source), root_hashes=root_hashes(), frozen_files=frozen,
        initial_policy=parent_request["initial_policy"], initial_policy_sha256=parent_request["initial_policy_sha256"],
        plan_path=str(review_path), plan_sha256=digest(review_path), duplicate_guard=str(guard),
        parent_case=str(parent), checkpoint=str(checkpoint), dataset=None, dataset_stats={},
        producer_completed_iterations=review["committed_updates"],
        **(dict(trial_protocol=CONTEXT_PROTOCOL,producer_status="stopped_functional_guard",
                producer_uncommitted_rollout=True) if review["kind"]==V10_RESTORED372_KIND else {}),
        postmortem_measurement=True, accepted_flat_baseline=baseline, GPU_soft_budget_MiB=11264)


def request_for(case):
    case = inside(case)
    request = read(case / "request.json")
    expected = read(case / "launch.lock")["request_sha256"]
    require(digest(case / "request.json") == expected and request["case"] == str(case), "Request/lock changed")
    require(read(request["duplicate_guard"])["request_sha256"] == expected, "Independent deduplication guard changed")
    return case, request


def identities(case):
    path = Path(case) / "launch.json"
    if not path.exists():
        return []
    require(digest(path) == read(Path(case) / "launch.identity.sha256.json")["sha256"], "Immutable launch identity changed")
    return read(path)["processes"]


def owned(row):
    return process_check(row, entry=row["entrypoint"])


def check_inputs(request):
    changes = [path for path, value in request["frozen_files"].items()
               if not Path(path).is_file() or digest(path) != value]
    if root_hashes() != request["root_hashes"]:
        changes.append("ROOT source/tools file set or bytes")
    if hashes(request["source_root"]) != request["source_hashes"]:
        changes.append("Actual imported source snapshot file set or bytes")
    if request.get("dataset") and dataset_stats(request["dataset"]) != request["dataset_stats"]:
        changes.append("Fixed real audit dataset identity changed")
    if request.get("trial_protocol") in CONTEXT_PROTOCOLS and request.get("job_kind")==KIND:
        seal_path=Path(request["case"])/"actual_bank.sealed.json"
        state=read(Path(request["case"])/"supervision.json")
        if seal_path.is_file():
            seal=read(seal_path)
            if seal.get("case")!=request["case"] or seal.get("plan_sha256")!=request["plan_sha256"]:
                changes.append("Actual first-rollout bank seal identity")
            bank=seal.get("fixed_real_bank",{})
            for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),("acquisition_log_path","acquisition_log_sha256")):
                path=bank.get(key)
                if not path or not Path(path).is_file() or digest(path)!=bank.get(sha_key):
                    changes.append("Sealed actual bank changed: "+key)
            current=state.get("guarded_flat_learning",{}).get("fixed_real_bank",{})
            if current and current!=bank:
                changes.append("Sealed bank/run supervision binding changed")
        elif state.get("completed_iterations",0)>0:
            changes.append("Actual first-rollout bank was not sealed before update0")
    if (git("branch", "--show-current") != request["source_branch"]
            or git("rev-parse", "HEAD") != request.get("launcher_commit", request["source_commit"])):
        changes.append("Runtime branch/commit changed")
    return changes


def startup_contract(metadata, request):
    if metadata.get("status") in (None, "initializing"):
        return dict(status="pending", errors=[])
    errors = []
    def expect(ok, reason):
        if not ok:
            errors.append(reason)
    expect(metadata.get("task") == TASK, "Wrong physical task")
    expect(metadata.get("git_commit") == request["source_commit"], "Wrong runtime source commit")
    expect(metadata.get("checkpoint") is None and metadata.get("starting_iteration") == 0, "Unexpected PPO resume")
    expect(metadata.get("initial_policy_sha256") == request["initial_policy_sha256"], "Wrong actual initializer")
    expect([metadata.get(k) for k in ("num_envs", "num_steps_per_env", "max_iterations", "save_interval", "num_actions")]
           == [1024, 24, 1000, 500, 12], "Wrong sampling/action/iteration budget")
    forward_excess=request.get("trial_protocol")==FORWARD_EXCESS_CONTEXT_PROTOCOL
    full_outward=request.get("trial_protocol")==FULL_OUTWARD_CONTEXT_PROTOCOL
    active=request.get("trial_protocol") in ACTIVE_CONTEXT_PROTOCOLS
    context=request.get("trial_protocol") in CONTEXT_PROTOCOLS
    expect(metadata.get("policy_shape") == [1024, 1034 if context else 1029], "Actor layout changed")
    shapes = metadata.get("observation_shapes", {})
    expect(shapes.get("critic") == [1024, 871 if context else 866] and shapes.get("rnd_state") == [1024, 73], "Critic/RND layout changed")
    action = metadata.get("action_contract", {})
    expect(action.get("class") == "DirectLegPositionAction" and action.get("joint_names") == LEGS,
           "Original direct12 action contract missing")
    control = metadata.get("control_settings", {})
    expect(all(control.get(k) is None for k in ("contact_gait_reference", "stair_reference", "stance_orientation_feedback", "lateral_com_feedback"))
           and control.get("stance_preload_torques") == {}, "Reference/IMU/CoM/preload controller is attached")
    guarded = metadata.get("guarded_flat_learning", {})
    expect(guarded.get("plan_sha256") == request["plan_sha256"], "Guarded flat plan binding missing")
    expect(guarded.get("normalizers_frozen") is True and guarded.get("normalizer_prefix_counts" if context else "normalizer_counts") == {"actor": 24576000, "critic": 24576000},
           "Inherited dual normalization contract changed")
    bank=guarded.get("fixed_real_bank",{})
    collecting=(context and bank.get("status")=="collecting_first_rollout"
                and bank.get("rows")==0 and bank.get("target_rows")==4096 and bank.get("sealed") is False)
    expect((bank.get("rows")==4096 and (not context or bank.get("status")=="sealed")) or collecting,
           "Actual first-rollout acquisition is neither explicitly collecting nor sealed")
    expect(guarded.get("current_rollout_real_rows") == 1024 and guarded.get("total_guard_rows") == 5120,
           "Actual finite input audit bank size changed")
    learning = guarded.get("state", {})
    expect(all(learning.get(k) == 0 for k in ("completed_updates", "actor_optimizer_steps", "critic_optimizer_steps", "rnd_optimizer_steps")),
           "Split PPO/RND optimization did not start fresh")
    expect(learning.get("critic_only_iterations") == 40 and learning.get("actor_learning_rate") == 1e-5
           and learning.get("critic_learning_rate") == .001 and learning.get("rnd_reward_weight") == 0,
           "Actual split optimizer protocol changed")
    clean_force = context or request.get("trial_protocol") == CLEAN_FORCE_PROTOCOL
    outward_tilt = clean_force or request.get("trial_protocol") == OUTWARD_TILT_PROTOCOL
    velocity_cost = request.get("trial_protocol") == VELOCITY_COST_PROTOCOL
    precision = outward_tilt or velocity_cost or request.get("trial_protocol") == PRECISION_PROTOCOL
    resource = precision or request.get("trial_protocol") == RESOURCE_PROTOCOL
    reserved = resource or request.get("trial_protocol") == RESERVED_PROTOCOL
    projected = reserved or request.get("trial_protocol") == PROPOSAL_PROTOCOL
    transfer = projected or request.get("trial_protocol") == "overspeed_credit_action_margin_v1"
    actor_flag = ("initial_actor_weights_match_parent_exactly" if active else "initial_actor_prefix_and_zero_columns_verified" if context else
                  "initial_actor_weights_match_parent_exactly" if transfer else "initial_actor_weights_match_ce822_exactly")
    source_flag = ("initial_network_source_matches_parent_exactly" if active else "initial_context_source_matches_reviewed_delta" if context else
                   "initial_network_source_matches_parent_exactly" if transfer else "initial_network_source_matches_ce822_exactly")
    expect(metadata.get(actor_flag) is True and metadata.get(source_flag) is True,
           "Actual initialized actor/network does not match the reviewed candidate")
    if transfer:
        expect(metadata.get("initial_critic_prefix_and_zero_columns_verified" if context and not active else "initial_critic_weights_match_parent_exactly") is True,
               "Transferred critic weights do not match the declared parent")
        expect(guarded.get("optimizer_format") == (FORWARD_EXCESS_CONTEXT_FORMAT if forward_excess else FULL_OUTWARD_CONTEXT_FORMAT if full_outward else ACTIVE_CONTEXT_FORMAT if active else CONTEXT_FORMAT if context else "h2_guarded_flat_ppo_v9" if clean_force else "h2_guarded_flat_ppo_v8" if outward_tilt else "h2_guarded_flat_ppo_v7" if velocity_cost else "h2_guarded_flat_ppo_v6" if precision else "h2_guarded_flat_ppo_v5" if resource else "h2_guarded_flat_ppo_v4" if reserved else "h2_guarded_flat_ppo_v3" if projected else "h2_guarded_flat_ppo_v2"), "Typed optimizer format differs from its declared protocol")
        margin = learning.get("action_margin", {})
        expect(margin.get("weight") == (0. if projected else .01) and margin.get("epsilon") == 1e-6,
               "Actual actor target-margin coefficient differs from reviewed fixed protocol")
        expect(metadata.get("algorithm_settings", {}).get("gamma") == .99,
               "Temporal discount was changed without support from the selected repair")
        if projected:
            proposal=learning.get("proposal_projection",{})
            expect(proposal.get("kind")==("simultaneous_actual_bank_reserved_linear_v1" if reserved else "simultaneous_actual_bank_linear_v1")
                   and proposal.get("interior_fraction")==.5 and proposal.get("max_active_rows")== (28 if resource else 8)
                   and proposal.get("max_solver_sweeps")==2048
                   and all(proposal.get(k) is True for k in ("whole_update_reference","linear_full_bank","hard_gate_unchanged","no_retry")),
                   "Runtime simultaneous proposal coverage/fixed geometry differs")
            if reserved:
                expect(proposal.get("working_budget_fraction")==.5
                       and proposal.get("whole_update_fixed_working_bound") is True
                       and proposal.get("solver_tolerance_reserved_inward") is True,
                       "Runtime working reserve is not fixed for the entire update")
            if resource:
                expect(proposal.get("resource_kind")=="single_cpu_float32_jacobian_arena_v1"
                       and proposal.get("configured_max_active_rows","missing") is None
                       and proposal.get("jacobian_arena_budget_bytes")==1073741824
                       and proposal.get("jacobian_mean_parameter_elements")== (782988 if context else 780428)
                       and proposal.get("jacobian_bytes_per_row")== (37583424 if context else 37460544)
                       and proposal.get("jacobian_arena_capacity_bytes")== (1052335872 if context else 1048895232),
                       "Runtime compute coverage is not the reviewed fixed CPU-byte-derived resource protocol")
            if precision:
                expect(proposal.get("primal_precision_policy")=="direct_fp64_primal_envelope_v1"
                       and proposal.get("max_polish_per_QP")== (1345 if active else 1)
                       and proposal.get("max_support_pivots")== (1344 if active else 0)
                       and proposal.get("max_polish_per_projection")== (37660 if active else 28)
                       and (not active or proposal.get("support_polish_policy")=="bounded_dual_active_set_v1")
                       and proposal.get("solver_optimality_and_direct_primal_separate") is True
                       and proposal.get("primal_envelope_is_machine_derived") is True
                       and proposal.get("final_same_direction_cast_JVP_required") is True,
                       "Runtime strict primal precision policy is missing")
            expect("functional_proposal.py" in metadata.get("training_tool_sha256",{}),
                   "Actual producer snapshot omitted its prospective optimizer/projector")
        if clean_force:
            expect({"evaluate_flat_balance_policy.py", "evaluate_direct_initialization.py"}.issubset(metadata.get("training_tool_sha256",{})),
                   "Episode force-credit admission source dependencies are absent")
    runtime = metadata.get("runtime_flat_geometry", {})
    if velocity_cost:
        cost=runtime.get("velocity_cost_scaling",{})
        expect(cost.get("kind")=="command_scaled_existing_world_xy_error_v1"
               and cost.get("actual_world_xy_velocity_error_weight")==-(10./.3)
               and cost.get("actual_signed_forward_weight")==10.
               and cost.get("command_m_s")==.3 and cost.get("maximum_positive_forward_rate")==3.
               and abs(cost.get("one_command_speed_error_cost_per_s",0.)+3.)<1e-12
               and cost.get("function_unchanged") is True and cost.get("no_new_reward_information") is True,
               "Runtime actual RewardManager velocity-cost scale differs from its fixed reviewed calibration")
    if context:
        try:
            validate_task_context_metadata(runtime)
            validate_normalization_metadata(guarded.get("normalization_abi"),guarded.get("normalizer_prefix_counts"))
        except (RuntimeError,TypeError,ValueError) as error:
            errors.append(str(error))
        expect(({"task_context_normalization.py","task_context_metadata.py"}|({"v11_parent_custody.py"} if active else set())).issubset(metadata.get("training_tool_sha256",{})),
               "Explicit context producer omitted normalization/metadata source dependencies")
    if active:
        transfer_record=metadata.get("initial_policy_transfer",{})
        expect(guarded.get("trial_protocol")==learning.get("trial_protocol")==request.get("trial_protocol") and request.get("trial_protocol") in ACTIVE_CONTEXT_PROTOCOLS,
               "v11 runtime state protocol differs")
        expect(transfer_record.get("kind")==("saved_complete_PPO_v12_context_policy_transfer" if forward_excess else "saved_guard_restored_v10_context_policy_transfer")
               and transfer_record.get("mode")==("transfer_completed_v12_policy_and_frozen_dualnorm_fresh_optimization" if forward_excess else "transfer_verified_guard_restored_v10_policy_and_frozen_dualnorm_fresh_optimization")
               and transfer_record.get("parent_accepted_updates")== (1000 if forward_excess else 372)
               and transfer_record.get("parent_last_complete_iteration")== (999 if forward_excess else 371)
               and transfer_record.get("parent_uncommitted_rollout") is (not forward_excess)
               and transfer_record.get("parent_natural_completion") is forward_excess
               and transfer_record.get("parent_runtime_rollback_completed") is (not forward_excess)
               and (not forward_excess or (transfer_record.get("parent_rollback_evidence") is None
                    and transfer_record.get("parent_persistent_state_digest") is None
                    and transfer_record.get("parent_checkpoint_sha256")==request.get("initial_policy_sha256")))
               and transfer_record.get("same_width_context_transfer") is True
               and transfer_record.get("actor_and_critic_loaded_bitwise") is True
               and transfer_record.get("dualnorm_loaded_bitwise") is True
               and transfer_record.get("loading_did_not_change_fresh_RNG") is True
               and transfer_record.get("model_and_dualnorm_only") is True
               and all(transfer_record.get(key) is True for key in ("fresh_actor_optimizer","fresh_critic_optimizer","fresh_RND","fresh_RND_optimizer"))
               and transfer_record.get("parent_RNG_loaded") is False
               and transfer_record.get("new_starting_iteration")==0
               and transfer_record.get("new_checkpoint") is None
               and transfer_record.get("normalizer_prefix_counts")=={"actor":24576000,"critic":24576000},
               "Actual same-width initializer must match its explicit restored372 or naturalv12 protocol")
    if outward_tilt:
        try:
            if clean_force:
                validate_episode_force_reward_metadata(runtime)
            else:
                validate_outward_reward_metadata(runtime)
        except (RuntimeError,TypeError,ValueError) as error:
            errors.append(str(error))
    expect(runtime.get("domain_bounds_local_m") == [-7.9, 7.9, -2.9, 2.9]
           and runtime.get("below_floor_tolerance_m") == .05, "Actual flat physical domain changed")
    expect(runtime.get("generator_size_m") == [16., 6.] and runtime.get("terrain_rows") == 1
           and runtime.get("terrain_columns") == 1 and runtime.get("terrain_subtypes") == {"flat": "MeshPlaneTerrainCfg"},
           "Actual flat mesh differs from the reviewed finite tile")
    expect(runtime.get("command_ranges") == {"lin_vel_x": [.3, .3], "lin_vel_y": [0., 0.],
        "ang_vel_z": [0., 0.], "heading": [0., 0.]}, "Actual forward command ranges changed")
    expect(runtime.get("scripted_reference_installed") is False and runtime.get("action_dimension") == 12
           and runtime.get("held_upper_count") == 17, "Direct action/upper-body authority changed")
    expect({"stair_below_floor", "stair_out_of_tile"}.issubset(runtime.get("termination_terms", []))
           and len(runtime.get("termination_terms", [])) == 7, "Original five plus two physical guard terms required")
    expect(bool(runtime.get("episode_evidence_directory")), "Pre-reset physical episode recorder is absent")
    if transfer:
        expect(runtime.get("trial_protocol") == request.get("trial_protocol")
               and runtime.get("reward_profile") == (FORWARD_EXCESS_REWARD_PROFILE if forward_excess else FULL_OUTWARD_REWARD_PROFILE if full_outward else "overspeed_outward_tilt_growth_clean_force_v1" if clean_force else "overspeed_outward_tilt_growth_v1" if outward_tilt else "overspeed_positive_credit_v1"),
               "Actual declared flat reward profile is missing")
        speed = runtime.get("positive_credit_speed_weight", {})
        expect(speed.get("command_m_s") == speed.get("full_at_or_below_m_s") == .3
               and speed.get("zero_at_or_above_m_s") == .6
               and speed.get("actual_peak_consumed_unweighted") is True
               and speed.get("negative_debt_unweighted") is True
               and speed.get("physical_or_action_speed_limit_changed") is False,
               "Actual credit weighting changed the reviewed reward or physical authority")
    return dict(status=("failed" if errors else "pending_bank_acquisition" if collecting else "passed"), errors=errors)


def overdue_checkpoint(iteration, starting_iteration, interval, saved_iterations):
    """RSL saves global zero-based multiples; no save is due before learning."""
    require(interval > 0 and starting_iteration >= 0, "Invalid checkpoint schedule")
    if iteration < starting_iteration:
        return None
    due = iteration // interval * interval
    if due < starting_iteration:
        return None
    return due if iteration > due + 250 and due not in saved_iterations else None


def sample(case, commit=False, reviewed_pending=None):
    case, request = request_for(case)
    with (case / "monitor.write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read(case / "supervision.monitor.json")
        log = case / "job.log"
        size = log.stat().st_size if log.exists() else 0
        offset = int(previous.get("log_checked_offset", 0))
        require(size >= offset, "Active log was truncated")
        with log.open("rb") if log.exists() else open(os.devnull, "rb") as stream:
            stream.seek(offset)
            data = stream.read(4 * 1024 * 1024)
        raw = previous.get("raw_log_fragment", "") + data.decode(errors="replace")
        newline = raw.rfind("\n")
        complete = raw[:newline + 1] if newline >= 0 else ""
        fragment = raw[newline + 1:] if newline >= 0 else raw
        text = previous.get("incomplete_iteration_block", "") + complete
        parsed = parse_training_log(text, previous)
        evaluation_progress = previous.get("evaluation_progress")
        advances = list(re.finditer(r"\[DIRECT_INIT_VALIDATION\] t=([0-9.]+)s first_active=(\d+)/(\d+)", text))
        if advances:
            match = advances[-1]
            evaluation_progress = dict(simulated_time_s=float(match[1]), first_episode_active=int(match[2]), num_envs=int(match[3]))
        fatal, known = fatal_lines(text)
        state = read(case / "supervision.json")
        active_job = read(case / "manager_state.json").get("status") != "finished"
        if state.get("status") == "stopped_functional_guard":
            fatal = [line for line in fatal if "FunctionalUpdateRejected" not in line]
        run = Path(state["run_dir"]) if state.get("run_dir") else None
        metadata = read(run / "run.json") if run and (run / "run.json").exists() else {}
        startup = startup_contract(metadata, request) if request["job_kind"] == KIND else dict(status="not_training", errors=[])
        files = []
        if run:
            for path in run.glob("model_*.pt"):
                try:
                    number = int(path.stem.split("_")[1])
                except ValueError:
                    continue
                files.append(dict(path=str(path), iteration=number, bytes=path.stat().st_size, mtime_epoch=path.stat().st_mtime))
        files.sort(key=lambda x: (x["iteration"], x["mtime_epoch"]))
        checks = [owned(row) for row in identities(case)]
        advanced = (time.time() if parsed["iteration"] > previous.get("iteration", -1)
                    else previous.get("advanced_at_epoch", time.time()))
        if evaluation_progress != previous.get("evaluation_progress"):
            advanced = time.time()
        changes = check_inputs(request)
        reasons = ((["new fatal runtime evidence"] if fatal else [])
                   + (["frozen input changed"] if changes else [])) if active_job else []
        if active_job and startup["errors"]:
            reasons.append("runtime startup contract failed")
        if active_job and request["job_kind"] == KIND:
            latest = parsed["iteration"]
            due = overdue_checkpoint(latest, request["starting_iteration"], request["save_interval"],
                                     {x["iteration"] for x in files})
            if due is not None:
                reasons.append("expected500-iteration checkpoint missing for over250 iterations")
            if latest > 999:
                reasons.append("bounded1000 iteration budget exceeded")
        if (active_job and state.get("status") not in TERMINAL and time.time() - advanced >= 600
                and log.exists() and time.time() - log.stat().st_mtime >= 600):
            reasons.append("log and iteration jointly stalled10min")
        point = dict(checked_at=now(), checked_at_epoch=time.time(), case=str(case),
                     **parsed, log_checked_offset=offset + len(data), log_size=size,
                     raw_log_fragment=fragment,
                     log_mtime_epoch=log.stat().st_mtime if log.exists() else None,
                     log_caught_up=offset + len(data) == size, advanced_at_epoch=advanced,
                     state_status=state.get("status"), checkpoints=files,
                     active_job_monitoring=active_job,
                     evaluation_progress=evaluation_progress,
                     new_checkpoints=[x for x in files if x["path"] not in {c["path"] for c in previous.get("checkpoints", [])}],
                     processes=checks, gpu=gpu_snapshot(), changed_frozen_inputs=changes,
                     startup_contract=startup, fatal_log_candidates=list(dict.fromkeys(fatal))[-30:],
                     known_nonfatal=list(dict.fromkeys(known))[-5:], confirmed_stop_reasons=reasons,
                     latest_motion=(tail_jsonl(run / "motion_metrics.jsonl") or [None])[-1] if run else None,
                     latest_effect=(tail_jsonl(run / "effect_assessments.jsonl") or [None])[-1] if run else None,
                     reviewed_pending=reviewed_pending)
        pending = case / ("monitor.pending_" + uuid.uuid4().hex[:12] + ".json")
        immutable(pending, point)
        if commit:
            write(case / "supervision.monitor.json", point)
            with (case / "supervision.jsonl").open("a") as stream:
                stream.write(json.dumps(point, ensure_ascii=False) + "\n")
            state = read(case / "supervision.json")
            state["manager_monitor"] = dict(checked_at=point["checked_at"], iteration=point["iteration"],
                ETA=point["ETA"], decision=reasons or ["progress_checked"], pending=str(pending))
            write(case / "supervision.json", state)
            if request["job_kind"] == KIND and startup["status"] in ("passed", "failed") and not (case / "startup_verification.json").exists():
                immutable(case / "startup_verification.json", dict(**startup, checked_at=now(), metadata=metadata))
            pointer = read(POINTER)
            if pointer.get("active_stair_job") == str(case):
                pointer.update(last_monitor_check_at=point["checked_at"], last_monitor_iteration=point["iteration"],
                               last_monitor_ETA=point["ETA"], run_dir=str(run) if run else None,
                               last_monitor_pending=str(pending))
                if metadata.get("source_snapshot"):
                    pointer["source_snapshot"] = metadata["source_snapshot"]
                if files:
                    pointer["latest_checkpoint"] = files[-1]["path"]
                write(POINTER, pointer)
        return point, pending


def descendants(pid):
    found, queue = {pid}, [pid]
    while queue:
        current = queue.pop()
        try:
            children = (Path("/proc") / str(current) / "task" / str(current) / "children").read_text().split()
        except FileNotFoundError:
            continue
        for child in map(int, children):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


def safe_stop(case, reasons):
    case, request = request_for(case)
    sent, limitation = [], None
    for sig, grace in ((signal.SIGINT, 12), (signal.SIGTERM, 6)):
        for role in ("learner", "wrapper"):
            for row in [r for r in identities(case) if r["role"] == role]:
                request_for(case)
                identities(case)
                if read(POINTER).get("active_stair_job") != str(case):
                    limitation = "Current pointer changed; unsafe to signal previous job"
                    break
                checked = owned(row)
                if checked["identity_matches"]:
                    try:
                        os.kill(row["pid"], sig)
                        sent.append(dict(signal=sig.name, identity=row, observed=checked["observed"], at=now()))
                    except (ProcessLookupError, PermissionError) as error:
                        sent.append(dict(signal=sig.name, signal_error=str(error), identity=row, at=now()))
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and any(owned(r)["identity_matches"] for r in identities(case)):
                time.sleep(.25)
            if limitation or not any(owned(r)["identity_matches"] for r in identities(case)):
                break
        if limitation or not any(owned(r)["identity_matches"] for r in identities(case)):
            break
    evidence = dict(at=now(), reasons=reasons, signals=sent, limitation=limitation,
                    remaining=[owned(r) for r in identities(case)])
    immutable(case / ("stop_evidence_" + uuid.uuid4().hex[:10] + ".json"), evidence)
    return evidence


def sync_campaign(case, status, **details):
    request = read(Path(case) / "request.json")
    for path in (ROOT / "logs/stairs/campaign.json", ROOT / "logs/stairs_work/current_stair_work_state.json"):
        value = read(path)
        jobs = value.setdefault("flat_balance_jobs", [])
        record = next((item for item in jobs if item["case"] == str(case)), None)
        if record is None:
            record = dict(case=str(case), kind=request["job_kind"], plan=request["plan_path"],
                          plan_sha256=request["plan_sha256"], source_commit=request["source_commit"])
            jobs.append(record)
        record.update(status=status, updated_at=now(), **details)
        value.update(status=status, active_job=str(case) if status in ("starting", "running") else None,
                     current_mode="guarded_flat_balance_learning" if request["job_kind"] == KIND else "guarded_flat_balance_frozen_validation",
                     stairs_acceptance_passed=False,
                     guarded_flat_balance_case=str(case), **details)
        write(path, value)


def manage(case, slot_fd):
    case, request = request_for(case)
    require(os.fstat(slot_fd).st_ino == SLOT.stat().st_ino, "Missing exclusive simulator slot lease")
    process, rows, code, error, stopped = None, [], -1, None, False
    manager = birth(os.getpid())
    try:
        require(not check_inputs(request) and not gpu_snapshot()["compute_processes"], "Inputs/GPU changed before dispatch")
        require(not active_simulators(exclude=[request["parent_launcher_pid"]]), "Another simulator started before dispatch")
        environment = {**os.environ, "CTRLLAB_H2_USD_PATH": str(ROOT / "assets/isaaclab/unitree_robots/h2/h2.usd")}
        with (case / "job.log").open("xb") as stream:
            process = subprocess.Popen(request["command"], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            immutable(case / "spawn_identity.json", birth(process.pid))
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                found = []
                for pid in descendants(process.pid):
                    try:
                        row = birth(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if request["entrypoint"] in row["argv"]:
                        found.append({**row, "entrypoint": request["entrypoint"],
                                      "role": "learner" if Path(row["argv"][0]).name.startswith("python") else "wrapper"})
                if found:
                    rows = found
                if any(r["role"] == "learner" for r in rows) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(case / "launch.json", dict(request_case=str(case), manager=manager, processes=rows))
            immutable(case / "launch.identity.sha256.json", dict(sha256=digest(case / "launch.json")))
            require(process.poll() is not None or any(r["role"] == "learner" for r in rows), "Cannot bind learner birth")
            write(case / "manager_state.json", dict(status="running", manager=manager, processes=rows))
            state = read(case / "supervision.json")
            state.update(processes=rows, manager=manager)
            write(case / "supervision.json", state)
            pointer = read(POINTER)
            if pointer.get("active_stair_job") == str(case):
                pointer.update(launch_record_sha256=digest(case / "launch.json"), status="initializing")
                write(POINTER, pointer)
            sync_campaign(case, "running")
            while True:
                point, _ = sample(case, commit=True)
                reasons = list(point["confirmed_stop_reasons"])
                if process.poll() is not None and any(x["identity_matches"] for x in point["processes"]):
                    reasons.append("Wrapper exited with recorded learner still active")
                for line in point["gpu"]["compute_processes"].splitlines():
                    columns = [x.strip() for x in line.split(",")]
                    if columns and columns[-1].isdigit() and int(columns[-1]) > request["GPU_soft_budget_MiB"]:
                        reasons.append("11GiB GPU soft budget exceeded")
                if reasons and not stopped:
                    stopped, error = True, "; ".join(reasons)
                    safe_stop(case, reasons)
                if process.poll() is not None and not any(owned(r)["identity_matches"] for r in rows) and point["log_caught_up"]:
                    break
                if stopped and any(owned(r)["identity_matches"] for r in rows):
                    write(case / "manager_state.json", dict(status="attention_required", manager=manager,
                        error=error, limitation="Owned processes remain; no automatic retry", processes=rows))
                try:
                    process.wait(timeout=60 if point["log_caught_up"] and not stopped else 1)
                except subprocess.TimeoutExpired:
                    pass
            code = process.wait()
    except BaseException as exception:
        error = repr(exception)
        if (case / "launch.identity.sha256.json").exists():
            safe_stop(case, [error])
        while process is not None and (process.poll() is None or any(owned(r)["identity_matches"] for r in rows)):
            write(case / "manager_state.json", dict(status="attention_required", manager=manager, error=error, processes=rows))
            time.sleep(5)
        if process is not None:
            code = process.wait()
    finally:
        deadline = time.monotonic() + 30
        gpu = gpu_snapshot()
        while gpu["compute_processes"] and time.monotonic() < deadline:
            time.sleep(1)
            gpu = gpu_snapshot()
        state = read(case / "supervision.json")
        remaining = [owned(r) for r in rows]
        changes = check_inputs(request)
        result = read(Path(request["output"]) / "result.json") if request.get("output") and (Path(request["output"]) / "result.json").exists() else None
        if changes or gpu["compute_processes"] or any(r["identity_matches"] for r in remaining):
            error = (error + "; " if error else "") + "Frozen input/process/GPU release audit failed"
        if request["job_kind"] == KIND:
            run = Path(state["run_dir"]) if state.get("run_dir") else None
            metadata = read(run / "run.json") if run and (run / "run.json").exists() else {}
            if state.get("status") == "completed":
                if not (metadata.get("status") == "completed" and metadata.get("completed_iterations") == 1000
                        and metadata.get("total_timesteps") == 24576000 and state.get("iteration") == 999
                        and state.get("completed_iterations") == 1000 and (run / "model_999.pt").is_file()):
                    error = (error + "; " if error else "") + "Incomplete final training metadata/checkpoint"
            if state.get("status") not in TERMINAL:
                error = (error + "; " if error else "") + "Process exited before a recorded terminal status"
        else:
            if not result or result.get("status") != "completed":
                error = (error + "; " if error else "") + "Frozen validation did not complete"
            if result:
                state["flat_balance_ready"] = bool(result.get("flat_balance_ready"))
                state["successful_envs"] = result.get("successful_envs", 0)
        if code != 0 or error:
            state["status"] = "failed"
        elif request["job_kind"] == EVAL_KIND:
            state["status"] = "completed"
        state.update(ended_at=now(), wrapper_exit_code=code, manager_error=error,
                     GPU_released=not gpu["compute_processes"], stairs_acceptance_passed=False)
        write(case / "supervision.json", state)
        audit = dict(checked_at=now(), wrapper_exit_code=code, state_status=state["status"], error=error,
                     final_processes=remaining, GPU=gpu, changed_frozen_inputs=changes, result=result,
                     independent_stair_acceptance_passed=False)
        immutable(case / "completion_audit.json", audit)
        write(case / "manager_state.json", dict(status="finished", outcome=state["status"], exit_code=code,
            error=error, manager=manager, processes=rows, GPU_released=not gpu["compute_processes"], ended_at=now()))
        pointer = read(POINTER)
        if pointer.get("active_stair_job") == str(case):
            pointer.update(status=state["status"], runtime_state_status="guarded_flat_balance_" + state["status"],
                active_flat_balance_learning=None, active_flat_balance_validation=None,
                last_flat_balance_completion_audit=str(case / "completion_audit.json"),
                next_action="Review actual terminal evidence; never automatically extend/retry or launch stair PPO.")
            if request["job_kind"] == KIND:
                pointer.update(run_dir=state.get("run_dir"), final_checkpoint=state.get("final_checkpoint") or state.get("latest_checkpoint"),
                               completed_iterations=state.get("completed_iterations", 0))
            else:
                pointer.update(last_flat_balance_result=str(Path(request["output"]) / "result.json"),
                               direct_flat_balance_ready=bool(state.get("flat_balance_ready")),
                               successful_envs=state.get("successful_envs", 0))
            write(POINTER, pointer)
        sync_campaign(case, "guarded_flat_balance_" + state["status"], completion_audit=str(case / "completion_audit.json"))
        os.close(slot_fd)


def launch(args):
    slot_fd = None
    try:
        if args.start:
            SLOT.parent.mkdir(parents=True, exist_ok=True)
            slot_fd = os.open(SLOT, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = preflight(args)
        if not args.start:
            print(json.dumps(dict(mode="preflight_only", **checked), ensure_ascii=False))
            return
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        case = ROOT / "logs/stair_learning" / ("flat_balance_" + stamp + "_" + uuid.uuid4().hex[:8])
        case.mkdir(exist_ok=False)
        for name in ("source", "tools"):
            shutil.copytree(ROOT / name, case / "snapshot" / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        previous = checked.pop("previous_pointer")
        write(case / "previous_pointer.json", previous)
        policy = stop_policy()
        policy.update(bounded_run_note="The flat1000 cap ends before full original patience; never infer convergence or extend automatically.",
            acceptance="Independent own-source1024 first20s, >=922 joint physical survival,4.8..7.2m,eachfoot>=2valid alternating landings and matched ground/raw footforce evidence. Separate stair90s acceptance is unchanged.")
        immutable(case / "stop_policy.json", policy)
        state_path = case / "supervision.json"
        if checked["job_kind"] == KIND:
            command = ["bash", str(ROOT / "tools/train.sh"), "--task", TASK, "--headless", "--device", "cuda:0",
                "--num_envs", "1024", "--max_iterations", "1000", "--save_interval", "500", "--seed", "42",
                "--run_name", "guarded_flat_balance_" + case.name[-8:], "--initial_policy", checked["initial_policy"],
                "--guarded_flat_plan", checked["plan_path"], "--supervision_state", str(state_path),
                "--stop_policy", str(case / "stop_policy.json"), "--expected_git_commit", checked["source_commit"]]
            output = None
        else:
            output = case / "evaluation"
            command = ["bash", str(ROOT / "tools/evaluate_flat_balance_policy.sh"), "--headless", "--device", "cuda:0",
                "--num_envs", "1024", "--checkpoint", checked["checkpoint"], "--source_root", checked["source_root"],
                "--duration", "20", "--speed", ".3", "--seed", "42", "--video", "--output", str(output)]
        request = dict(**checked, case=str(case), command=command, output=str(output) if output else None,
                       supervision_path=str(state_path), log=str(case / "job.log"),
                       launch_record=str(case / "launch.json"), single_launch_guard=str(case / "launch.lock"),
                       monitor_record=str(case / "supervision.monitor.json"), stop_policy_path=str(case / "stop_policy.json"),
                       monitor_snapshot_tool=str(SELF), monitor_snapshot_command=[sys.executable, str(SELF), "--monitor", str(case)],
                       parent_launcher_pid=os.getpid(), num_envs=1024,
                       num_steps_per_env=24 if checked["job_kind"] == KIND else 0,
                       max_iterations=1000 if checked["job_kind"] == KIND else 0,
                       starting_iteration=0, save_interval=500 if checked["job_kind"] == KIND else None,
                       expected_final_iteration=999 if checked["job_kind"] == KIND else -1,
                       expected_total_timesteps=24576000 if checked["job_kind"] == KIND else 0,
                       automatic_restart_allowed=False)
        immutable(case / "request.json", request)
        seal = dict(request_sha256=digest(case / "request.json"), case=str(case), at=now())
        immutable(case / "launch.lock", seal)
        immutable(request["duplicate_guard"], seal)
        write(state_path, dict(**request, status="launching", run_dir=None, stairs_acceptance_passed=False,
                              walking_acceptance_passed=False, initialization_ready=False, completed_iterations=0))
        write(case / "supervision.monitor.json", dict(iteration=-1, log_checked_offset=0, advanced_at_epoch=time.time()))
        write(case / "manager_state.json", dict(status="starting", created_at=now()))
        pointer = dict(previous)
        pointer.update({k: request[k] for k in ("supervision_path", "launch_record", "single_launch_guard", "monitor_record",
            "stop_policy_path", "monitor_snapshot_tool", "monitor_snapshot_command", "source_commit", "source_branch")})
        pointer.update(status="launching", runtime_state_status="guarded_flat_balance_starting", active_stair_job=str(case),
            active_stair_job_kind=request["job_kind"], active_stair_learning_case=str(case) if request["job_kind"] == KIND else None,
            active_flat_balance_learning=str(case) if request["job_kind"] == KIND else None,
            active_flat_balance_validation=str(case) if request["job_kind"] == EVAL_KIND else None,
            run_dir=None, final_checkpoint=None, latest_checkpoint=None, checkpoint=request["checkpoint"],
            checkpoint_sha256=digest(request["checkpoint"]) if request["checkpoint"] else None,
            source_snapshot=None, final_checkpoint_validation=None,
            checkpoint_validation_tool=str(ROOT / "tools/evaluate_flat_balance_policy.py"),
            max_iterations=1000 if request["job_kind"] == KIND else 0,
            starting_iteration=0, completed_iterations=0, last_monitor_iteration=-1,
            expected_final_iteration=request["expected_final_iteration"],
            expected_total_timesteps=request["expected_total_timesteps"], last_monitor_ETA=None,
            launch_record_sha256=None, single_launch_guard_sha256=digest(case / "launch.lock"),
            stairs_acceptance_passed=False, direct_flat_balance_ready=False, initialization_ready=False,
            diagnostic_only=False, successful_envs=0, previous_pointer_record=str(case / "previous_pointer.json"),
            next_action="Supervise this unique bounded task with frozen source and parameters; review real terminal evidence before any later dispatch.")
        if request["job_kind"] == KIND:
            pointer.update(guarded_flat_balance_plan=request["plan_path"], guarded_flat_balance_plan_sha256=request["plan_sha256"])
        else:
            pointer.update(active_flat_validation_review=request["plan_path"],
                           frozen_checkpoint_iteration=request.get("producer_completed_iterations", 1000) - 1,
                           frozen_checkpoint_completed_iterations=request.get("producer_completed_iterations", 1000),
                           launcher_commit=request.get("launcher_commit", request["source_commit"]))
        write(POINTER, pointer)
        sync_campaign(case, "starting", guarded_flat_balance_plan=request["plan_path"])
        with (case / "manager.log").open("xb") as stream:
            manager = subprocess.Popen([sys.executable, str(SELF), "--manage", str(case), str(slot_fd)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(slot_fd,))
        immutable(case / "manager_parent_receipt.json", dict(manager=birth(manager.pid), created_at=now()))
        print(json.dumps(dict(mode="started", case=str(case), manager_pid=manager.pid,
                             job_kind=request["job_kind"], monitor=request["monitor_snapshot_command"]), ensure_ascii=False))
    finally:
        if slot_fd is not None:
            os.close(slot_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--plan")
    parser.add_argument("--validate", help="Completed flat PPO case; requires --start to dispatch physical validation")
    parser.add_argument("--postmortem-review", help="Explicitly reviewed unique frozen measurement of a failed run's saved baseline; never resumes PPO")
    parser.add_argument("--monitor")
    parser.add_argument("--commit-monitor", action="store_true")
    parser.add_argument("--pending", help="Reviewed pending record; a fresh monotonic point is committed under lock")
    parser.add_argument("--manage", nargs=2)
    args = parser.parse_args()
    if args.manage:
        manage(Path(args.manage[0]), int(args.manage[1]))
    elif args.monitor:
        case = inside(args.monitor)
        if args.pending:
            pending = inside(args.pending)
            require(pending.parent == case and read(pending)["case"] == str(case), "Pending point targets another case")
        point, pending = sample(case, commit=args.commit_monitor, reviewed_pending=args.pending)
        print(json.dumps(dict(pending=str(pending), committed=args.commit_monitor,
            iteration=point["iteration"], ETA=point["ETA"], status=point["state_status"],
            stop_reasons=point["confirmed_stop_reasons"], startup_contract=point["startup_contract"]), ensure_ascii=False))
    else:
        launch(args)


if __name__ == "__main__":
    main()
