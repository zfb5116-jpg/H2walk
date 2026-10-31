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

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
POINTER = ROOT / "logs/launches/current_main_training.json"
SLOT = ROOT / "logs/stair_learning/flat_balance.slot.lock"
TASK = "Unitree-H2-Flat-Balance-Policy"
KIND = "guarded_flat_balance_ppo_learning"
EVAL_KIND = "guarded_flat_balance_frozen_validation"
BRANCH = "codex/h2-stair-ascent-20261030"
INIT_SHA = "ce822132385de7b9c70315087bbcc4da1739d0da8460cf656c39e14aa1424688"
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
    plan = inside(args.plan or pointer["guarded_flat_balance_plan"])
    declared = read(plan)
    require(declared.get("status") == "reviewed_ready", "Flat repair implementation has not been reviewed")
    require([declared[k] for k in ("task", "num_envs", "num_steps_per_env", "max_iterations", "save_interval")]
            == [TASK, 1024, 24, 1000, 500], "Plan sampling/budget changed")
    require(declared["network_or_input_structure_change"] is False, "Network/input change requires prior reporting")
    initial = inside(declared["initial_policy"])
    require(digest(initial) == declared["initial_policy_sha256"] == INIT_SHA, "Wrong reviewed actual initialization")
    receipt_path = initial.with_suffix(".validation.json")
    receipt = read(receipt_path)
    require(receipt.get("status") == "passed" and receipt["sha256"] == INIT_SHA, "Initializer CPU integrity receipt is missing")
    dependencies = {str(ROOT / "assets/isaaclab/unitree_robots/h2/h2.usd"):
                    "7f3432d01756efaeb6d4337536a3ead5a9d0c2f4130654bd43a353012b6c2dbd",
                    str(ROOT / "logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt"):
                    "d260001c9850941c95d1ba83196076f27e118e687e72586d0a2bff84f1674c50"}
    for path, expected in dependencies.items():
        require(digest(path) == expected, "Native USD or frozen LiDAR changed")
    baseline = pointer["accepted_flat_baseline"]
    require(digest(inside(baseline["final_checkpoint"])) == baseline["checkpoint_sha256"], "Protected accepted flat PT changed")
    for path, expected in baseline["network_hashes"].items():
        require(digest(ROOT / "source" / path) == expected, "Original network architecture/source changed")
    kind, source, parent, checkpoint = KIND, ROOT / "source", None, None
    entry = ROOT / "tools/train.py"
    frozen = {str(initial): INIT_SHA, str(receipt_path): digest(receipt_path),
              str(plan): digest(plan), **dependencies}
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
    dataset = initial.parent / "dataset"
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
        dataset = None
    review = inside(declared["reward_review_path"])
    frozen[str(review)] = digest(review)
    roots = root_hashes()
    key = hashlib.sha256(json.dumps(dict(kind=kind, root=roots, source=hashes(source),
        initial=INIT_SHA, checkpoint=digest(checkpoint) if checkpoint else None,
        plan=digest(plan)), sort_keys=True).encode()).hexdigest()[:20]
    guard = ROOT / "logs/stair_learning" / ("flat_balance_" + key + ".launch.lock")
    require(not guard.exists(), "Identical inputs/protocol already dispatched; no blind retry")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer,
                source_branch=BRANCH, source_commit=git("rev-parse", "HEAD"),
                job_kind=kind, entrypoint=str(entry), source_root=str(source),
                source_hashes=hashes(source), root_hashes=roots, frozen_files=frozen,
                initial_policy=str(initial), initial_policy_sha256=INIT_SHA,
                plan_path=str(plan), plan_sha256=digest(plan), duplicate_guard=str(guard),
                parent_case=str(parent) if parent else None,
                checkpoint=str(checkpoint) if checkpoint else None,
                dataset=str(dataset) if dataset else None,
                dataset_stats=dataset_stats(dataset) if dataset else {},
                accepted_flat_baseline=baseline, GPU_soft_budget_MiB=11264)


def preflight_postmortem(args):
    """A reviewed saved baseline may be measured without resuming failed PPO."""
    pointer = read(POINTER)
    require(not pointer.get("stair_experiments_paused_by_user"), "User paused experiments")
    previous = prior_finished(pointer)
    require(not gpu_snapshot()["compute_processes"] and not active_simulators(), "Another Isaac/GPU task is active")
    require(git("branch", "--show-current") == BRANCH and not git("status", "--porcelain=v1"),
            "Commit the reviewed tools before freezing a validation")
    review_path = inside(args.postmortem_review)
    review = read(review_path)
    require(review.get("status") == "reviewed_ready" and
            review.get("kind") == "single_frozen_validation_of_preserved_pre_guard_baseline_v1",
            "A specific reviewed postmortem measurement plan is required")
    require(review.get("protocol") == dict(num_envs=1024, duration_s=20, speed_m_s=.3, seed=42,
        minimum_successful_envs=922, minimum_forward_m=4.8, maximum_forward_m=7.2,
        minimum_valid_landings_each_foot=2, teacher_interventions=0, learning_updates=0),
        "Independent physical acceptance protocol changed")
    parent = inside(review["parent_case"])
    require(str(parent) == pointer["active_stair_job"], "Postmortem plan targets another previous job")
    parent_request = request_for(parent)[1]
    state = read(parent / "supervision.json")
    require(parent_request["job_kind"] == KIND and state.get("status") == "failed"
            and state.get("failure_kind") == "functional_guard_rollback_inference_tensor_error",
            "Postmortem source is not the reviewed failed bounded PPO")
    checkpoint = inside(review["checkpoint"])
    run = inside(state["run_dir"])
    require(checkpoint == inside(state["final_checkpoint"]) and checkpoint.parent == run,
            "Only the preserved pre-update baseline can be replayed")
    original = inside(review["original_baseline_snapshot"])
    require(original.parent == run / "functional_guard" and original.name.endswith("_baseline.pt")
            and digest(checkpoint) == digest(original) == review["checkpoint_sha256"],
            "Rejected candidate or altered checkpoint cannot be substituted for the baseline")
    receipt_path = inside(review["CPU_receipt"])
    receipt = read(receipt_path)
    require(digest(receipt_path) == review["CPU_receipt_sha256"] and receipt.get("status") == "passed"
            and receipt.get("cpu_only") is True and receipt["sha256"] == digest(checkpoint)
            and receipt["checkpoint_format"] == "h2_guarded_flat_ppo_v1"
            and receipt["committed_updates"] == state["completed_iterations"] == review["committed_updates"],
            "Preserved split optimizer checkpoint has not passed actual CPU validation")
    source = inside(receipt["source_root"])
    require(source == inside(run / "training_source/source"), "Replay must use this weight's actual training source")
    actual = {"source/" + str(Path(path).relative_to(source)): value for path, value in hashes(source).items()}
    require(actual == receipt["source_sha256"], "Frozen imported model/plant source changed")
    require(digest(receipt["producer_run"]) == receipt["producer_run_sha256"], "Reconciled producer evidence changed")
    require(read(receipt["producer_run"])["status"] == "failed", "Do not relabel failed PPO as natural completion")
    frozen = {str(path): digest(path) for path in (review_path, checkpoint, original, receipt_path,
        inside(receipt["producer_run"]), run / "params/agent.yaml", parent / "guard_rollback_failure_reconciliation.json")}
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
        require(digest(ROOT / "source" / name) == expected, "Original network/input source changed")
    # Use model/source/protocol identity, not tooling revisions, for deduplication.
    key = hashlib.sha256(json.dumps(dict(kind=EVAL_KIND, checkpoint=digest(checkpoint),
        source=actual, protocol=review["protocol"]), sort_keys=True).encode()).hexdigest()[:20]
    guard = ROOT / "logs/stair_learning" / ("flat_postmortem_" + key + ".launch.lock")
    require(not guard.exists(), "This frozen baseline/protocol was already dispatched")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer,
        source_branch=BRANCH, source_commit=parent_request["source_commit"], launcher_commit=git("rev-parse", "HEAD"),
        job_kind=EVAL_KIND, entrypoint=str(ROOT / "tools/evaluate_flat_balance_policy.py"),
        source_root=str(source), source_hashes=hashes(source), root_hashes=root_hashes(), frozen_files=frozen,
        initial_policy=parent_request["initial_policy"], initial_policy_sha256=INIT_SHA,
        plan_path=str(review_path), plan_sha256=digest(review_path), duplicate_guard=str(guard),
        parent_case=str(parent), checkpoint=str(checkpoint), dataset=None, dataset_stats={},
        producer_completed_iterations=review["committed_updates"],
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
    expect(metadata.get("initial_policy_sha256") == INIT_SHA, "Wrong actual initializer")
    expect([metadata.get(k) for k in ("num_envs", "num_steps_per_env", "max_iterations", "save_interval", "num_actions")]
           == [1024, 24, 1000, 500, 12], "Wrong sampling/action/iteration budget")
    expect(metadata.get("policy_shape") == [1024, 1029], "Actor layout changed")
    shapes = metadata.get("observation_shapes", {})
    expect(shapes.get("critic") == [1024, 866] and shapes.get("rnd_state") == [1024, 73], "Critic/RND layout changed")
    action = metadata.get("action_contract", {})
    expect(action.get("class") == "DirectLegPositionAction" and action.get("joint_names") == LEGS,
           "Original direct12 action contract missing")
    control = metadata.get("control_settings", {})
    expect(all(control.get(k) is None for k in ("contact_gait_reference", "stair_reference", "stance_orientation_feedback", "lateral_com_feedback"))
           and control.get("stance_preload_torques") == {}, "Reference/IMU/CoM/preload controller is attached")
    guarded = metadata.get("guarded_flat_learning", {})
    expect(guarded.get("plan_sha256") == request["plan_sha256"], "Guarded flat plan binding missing")
    expect(guarded.get("normalizers_frozen") is True and guarded.get("normalizer_counts") == {"actor": 24576000, "critic": 24576000},
           "Inherited dual normalization contract changed")
    expect(guarded.get("fixed_real_bank", {}).get("rows") == 4096
           and guarded.get("current_rollout_real_rows") == 1024 and guarded.get("total_guard_rows") == 5120,
           "Actual finite input audit bank size changed")
    learning = guarded.get("state", {})
    expect(all(learning.get(k) == 0 for k in ("completed_updates", "actor_optimizer_steps", "critic_optimizer_steps", "rnd_optimizer_steps")),
           "Split PPO/RND optimization did not start fresh")
    expect(learning.get("critic_only_iterations") == 40 and learning.get("actor_learning_rate") == 1e-5
           and learning.get("critic_learning_rate") == .001 and learning.get("rnd_reward_weight") == 0,
           "Actual split optimizer protocol changed")
    expect(metadata.get("initial_actor_weights_match_ce822_exactly") is True
           and metadata.get("initial_network_source_matches_ce822_exactly") is True,
           "Actual initialized actor/network does not match the reviewed candidate")
    runtime = metadata.get("runtime_flat_geometry", {})
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
    return dict(status="passed" if not errors else "failed", errors=errors)


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
