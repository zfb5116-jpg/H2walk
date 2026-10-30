"""Explicitly launch one bounded direct-policy initialization job or validation.

The collector never launches its validation or PPO. Every process has an
immutable birth record; a confirmed fault stops only that exact owned process.
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
                                    immutable, old_job_finished, process_check, write)

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
POINTER = ROOT / "logs/launches/current_main_training.json"
SLOT = ROOT / "logs/stair_learning/closed_loop_initialization.slot.lock"
COLLECTOR = ROOT / "tools/collect_closed_loop_initialization.py"
VALIDATOR = ROOT / "tools/evaluate_direct_initialization.py"
BRANCH = "codex/h2-stair-ascent-20261030"
PROTOCOL = dict(num_envs=1024, seed=42, interface_controls=150,
                round_controls=400, rounds=4, teacher_probabilities=[.75, .5, .25, 0.],
                maximum_physics_controls=1750, maximum_actor_updates=3200,
                GPU_soft_budget_MiB=11264, validation_duration_s=20,
                initialization_ready_requires_independent_validation=True)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inside(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT), f"Outside H2walk: {path}")
    return path


def root_hashes():
    return {str(p): digest(p) for name in ("source", "tools")
            for p in sorted((ROOT/name).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def source_hashes(source):
    return {"source/"+str(p.relative_to(source)): digest(p)
            for p in sorted(source.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def check_inputs(request):
    changed = [name for name, value in request["frozen_files"].items()
               if not Path(name).is_file() or digest(name) != value]
    if root_hashes() != request["root_hashes"]:
        changed.append("ROOT source/tools file set or bytes")
    if source_hashes(Path(request["source_root"])) != request["source_hashes"]:
        changed.append(request["source_root"])
    return changed


def owned(identity):
    return process_check(identity, entry=identity["entrypoint"])


def preflight(args):
    pointer = read(POINTER)
    require(not pointer.get("stair_experiments_paused_by_user"), "User paused stair experiments")
    previous = old_job_finished(pointer)
    gpu = gpu_snapshot()
    require(not gpu["compute_processes"] and not active_simulators(), "Another Isaac/GPU job is active")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    require(branch == BRANCH, "Wrong project branch")
    require(not subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip(),
            "Commit the reviewed source/tools before freezing a new run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline = pointer["accepted_flat_baseline"]
    teacher = inside(args.teacher_checkpoint or baseline["final_checkpoint"])
    require(digest(teacher) == baseline["checkpoint_sha256"], "Accepted teacher checkpoint changed")
    dependencies = {str(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd"):
                        digest(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd"),
                    str(ROOT/"logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt"):
                        digest(ROOT/"logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt")}
    require(dependencies[str(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd")] ==
            "7f3432d01756efaeb6d4337536a3ead5a9d0c2f4130654bd43a353012b6c2dbd", "Original USD changed")
    require(dependencies[str(ROOT/"logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt")] ==
            "d260001c9850941c95d1ba83196076f27e118e687e72586d0a2bff84f1674c50", "Frozen LiDAR changed")
    if args.validate:
        parent = inside(args.validate)
        request = read(parent/"request.json")
        result = read(parent/"output/result.json")
        require(read(parent/"manager_state.json")["status"] == "finished", "Collection manager is active")
        require(result.get("status") == "completed", "Collection did not complete fitting")
        initial = inside(parent/"output/candidate_initial_policy.pt")
        validation = inside(initial.with_suffix(".validation.json"))
        receipt = read(validation)
        require(receipt.get("status") == "passed" and receipt.get("sha256") == digest(initial),
                "Candidate requires strict CPU validation before physical validation")
        source = inside(receipt["source_root"])
        require(source == inside(parent/"output/training_source/source"), "Candidate source must be its exported production snapshot")
        require(source_hashes(source) == request["source_hashes"], "Exported candidate source differs from collected source")
        frozen = {str(initial): digest(initial), str(validation): digest(validation),
                  str(parent/"request.json"): digest(parent/"request.json"),
                  str(parent/"output/result.json"): digest(parent/"output/result.json")}
        entry, kind = VALIDATOR, "direct_initialization_validation"
        source_commit = request["source_commit"]
    else:
        if args.initial_policy:
            initial = inside(args.initial_policy)
        elif pointer.get("initialization_seed_policy"):
            initial = inside(pointer["initialization_seed_policy"])
        else:
            initial = inside(read(inside(pointer["run_dir"])/"run.json")["initial_policy"])
        source, parent = ROOT/"source", None
        frozen = {str(initial): digest(initial), str(teacher): digest(teacher)}
        entry, kind = COLLECTOR, "closed_loop_initialization_learning"
        source_commit = commit
    require(entry.is_file() and entry.with_suffix(".sh").is_file(), "Job entry tools are not ready")
    roots = root_hashes()
    hashes = source_hashes(source)
    frozen.update(dependencies)
    key = hashlib.sha256(json.dumps(dict(kind=kind, files=roots, initial=digest(initial),
        teacher=digest(teacher), source=hashes, protocol=PROTOCOL), sort_keys=True).encode()).hexdigest()[:20]
    guard = ROOT/"logs/stair_learning"/("closed_loop_"+key+".launch.lock")
    require(not guard.exists(), "Identical inputs/protocol already dispatched; no blind retry")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer, job_kind=kind,
                entrypoint=str(entry), source_commit=source_commit, launcher_commit=commit, source_branch=branch,
                source_root=str(source), source_hashes=hashes, root_hashes=roots,
                frozen_files=frozen, initial_policy=str(initial), teacher_checkpoint=str(teacher),
                accepted_flat_baseline=baseline, protocol=PROTOCOL, duplicate_guard=str(guard),
                parent_collection=str(parent) if parent else None, gpu=gpu)


def request_for(case):
    case = inside(case)
    request, guard = read(case/"request.json"), read(case/"launch.lock")
    require(request["case"] == str(case) and digest(case/"request.json") == guard["request_sha256"],
            "Immutable request/lock changed")
    require(read(request["duplicate_guard"])["request_sha256"] == guard["request_sha256"],
            "Independent duplicate guard changed")
    return case, request


def identities(case):
    record = read(case/"launch.json")
    require(read(case/"launch.identity.sha256.json")["sha256"] == digest(case/"launch.json"),
            "Immutable process identities changed")
    return record["processes"]


def descendants(pid):
    pending, result = [pid], []
    while pending:
        value = pending.pop()
        if value in result:
            continue
        result.append(value)
        try:
            pending.extend(map(int, Path(f"/proc/{value}/task/{value}/children").read_text().split()))
        except FileNotFoundError:
            pass
    return result


def snapshot(case):
    case, request = request_for(case)
    previous = read(case/"supervision.monitor.json") if (case/"supervision.monitor.json").exists() else {}
    log = case/"job.log"
    size = log.stat().st_size if log.exists() else 0
    offset = int(previous.get("log_checked_offset", 0))
    require(offset <= size, "Job log truncated or replaced")
    data = b""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(offset); data = stream.read(2*1024*1024)
    text = previous.get("fragment", "")+data.decode(errors="replace")
    complete, separator, fragment = text.rpartition("\n")
    if not separator:
        complete, fragment = "", text
    lines = ANSI.sub("", complete+"\n"+fragment).splitlines()
    fatal = list(previous.get("fatal_log_candidates", []))
    for line in lines:
        if "cuDeviceGetUuid" in line or re.search(r"Warp CUDA error\s*:?\s*36\b", line):
            continue
        if re.search(r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA out of memory|"
                     r"illegal memory access|device-side assert|non[- ]finite|No space left on device|"
                     r"PytorchStreamWriter failed|CUDA error:(?!\s*36\b)", line, re.I):
            fatal.append(line)
    state = read(case/"supervision.json")
    progress = [state.get("physics_controls", 0), state.get("actor_updates", 0), state.get("phase")]
    validation_progress = previous.get("validation_progress")
    if request["job_kind"] == "direct_initialization_validation":
        matches = re.findall(r"\[DIRECT_INIT_VALIDATION\] t=([0-9.]+)s first_active=(\d+)/1024", complete)
        if matches:
            seconds, active = matches[-1]
            validation_progress = dict(simulated_time_s=float(seconds), first_episode_active=int(active), num_envs=1024)
        if validation_progress:
            progress = [round(validation_progress["simulated_time_s"]/.02), 0, "frozen_student_validation"]
    advanced = previous.get("advanced_at_epoch", time.time())
    if progress != previous.get("progress"):
        advanced = time.time()
    record = identities(case) if (case/"launch.identity.sha256.json").exists() else []
    # Constant source/identity blobs already live in the immutable request.
    # Keep repeated history focused on measured progress and conclusions.
    state_view = {name:value for name,value in state.items()
                  if name not in request or name in ("status", "source_commit", "job_kind")}
    point = dict(checked_at=now(), checked_at_epoch=time.time(), progress=progress,
        advanced_at_epoch=advanced, state=state_view, processes=[owned(row) for row in record],
        validation_progress=validation_progress,
        gpu=gpu_snapshot(), changed_frozen_inputs=check_inputs(request),
        fatal_log_candidates=list(dict.fromkeys(fatal))[-30:], log_checked_offset=offset+len(data),
        log_size=size, fragment=fragment, log_excerpt=ANSI.sub("", complete)[-6000:])
    write(case/"supervision.monitor.json", point)
    with (case/"supervision.jsonl").open("a") as stream:
        stream.write(json.dumps(point, ensure_ascii=False)+"\n"); stream.flush()
    return point


def safe_stop(case, reasons):
    case, request = request_for(case)
    sent, limitation = [], None
    for sig, grace in ((signal.SIGINT, 8), (signal.SIGTERM, 5)):
        for role in ("learner", "wrapper"):
            # Give the actual learner its shutdown grace before touching its
            # wrapper. Both roles are re-bound immediately before each signal.
            for row in [value for value in identities(case) if value["role"] == role]:
                request_for(case); identities(case)
                if read(POINTER).get("active_stair_job") != str(case):
                    limitation = "Current pointer changed; cannot signal the previous job"
                    break
                checked = owned(row)
                if checked["identity_matches"]:
                    try:
                        os.kill(row["pid"], sig)
                        sent.append(dict(signal=sig.name, identity=row, observed=checked["observed"], at=now()))
                    except (ProcessLookupError, PermissionError) as error:
                        sent.append(dict(signal=sig.name, identity=row, signal_error=str(error), at=now()))
            deadline = time.monotonic()+grace
            while time.monotonic() < deadline and any(owned(row)["identity_matches"] for row in identities(case)):
                time.sleep(.2)
            if limitation or not any(owned(row)["identity_matches"] for row in identities(case)):
                break
        if limitation or not any(owned(row)["identity_matches"] for row in identities(case)):
            break
    evidence = dict(at=now(), reasons=reasons, signals=sent, limitation=limitation,
                    remaining=[owned(row) for row in identities(case)])
    immutable(case/("stop_evidence_"+uuid.uuid4().hex[:8]+".json"), evidence)
    return evidence


def manage(case, slot_fd):
    case, request = request_for(case)
    require(os.fstat(slot_fd).st_ino == SLOT.stat().st_ino, "Missing independent slot lease")
    process, records, error, code = None, [], None, -1
    started = time.monotonic()
    try:
        require(not check_inputs(request) and not gpu_snapshot()["compute_processes"], "Inputs/GPU changed before startup")
        with (case/"job.log").open("xb") as stream:
            process = subprocess.Popen(request["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            immutable(case/"spawn_identity.json", birth(process.pid))
            while time.monotonic()-started < 30:
                found = []
                for pid in descendants(process.pid):
                    try:
                        item = birth(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if request["entrypoint"] in item["argv"]:
                        found.append({**item, "entrypoint":request["entrypoint"],
                                      "role":"learner" if Path(item["argv"][0]).name.startswith("python") else "wrapper"})
                if found:
                    records = found
                if any(row["role"]=="learner" for row in records) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(case/"launch.json", dict(request_case=str(case), manager=birth(os.getpid()), processes=records))
            immutable(case/"launch.identity.sha256.json", dict(sha256=digest(case/"launch.json")))
            require(process.poll() is not None or any(row["role"]=="learner" for row in records),
                    "Cannot bind owned learner identity")
            write(case/"manager_state.json", dict(status="running", manager=birth(os.getpid()), processes=records))
            pointer = read(POINTER)
            if pointer.get("active_stair_job") == str(case):
                pointer.update(status="initializing", launch_record_sha256=digest(case/"launch.json"),
                    single_launch_guard_sha256=digest(case/"launch.lock"))
                write(POINTER, pointer)
            state = read(case/"supervision.json"); state.update(status="initializing", processes=records)
            write(case/"supervision.json", state)
            stopped = False
            while True:
                point = snapshot(case)
                reasons = (["fatal runtime log"] if point["fatal_log_candidates"] else [])
                reasons += (["frozen source/input changed"] if point["changed_frozen_inputs"] else [])
                if point["progress"][0] > 1750 or point["progress"][1] > 3200:
                    reasons.append("fixed collection/optimization budget exceeded")
                for line in point["gpu"]["compute_processes"].splitlines():
                    fields = [x.strip() for x in line.split(",")]
                    if len(fields) >= 3 and fields[-1].isdigit() and int(fields[-1]) > 11264:
                        reasons.append("11GiB GPU soft budget exceeded")
                if time.monotonic()-started > request["max_wall_seconds"]-20:
                    reasons.append("bounded job wall-time reserve reached")
                if (time.time()-point["advanced_at_epoch"] >= 600 and log_stale(case, 600)):
                    reasons.append("log and progress jointly stalled10min")
                live = any(v["identity_matches"] for v in point["processes"])
                if process.poll() is not None and live:
                    reasons.append("wrapper exited while an owned learner remains")
                if reasons and not stopped:
                    error, stopped = "; ".join(reasons), True
                    if live:
                        evidence = safe_stop(case, reasons)
                        if any(row["identity_matches"] for row in evidence["remaining"]):
                            write(case/"manager_state.json", dict(status="attention_required", error=error,
                                processes=records, limitation="Matching processes remain after INT/TERM; no retry or new job"))
                if process.poll() is not None and not any(owned(row)["identity_matches"] for row in records):
                    if point["log_checked_offset"] == point["log_size"]:
                        break
                time.sleep(2)
            code = process.wait(timeout=5)
    except BaseException as exception:
        error = repr(exception)
        if (case/"launch.identity.sha256.json").exists() and process is not None:
            if process.poll() is None or any(owned(row)["identity_matches"] for row in records):
                safe_stop(case, [error])
        while process is not None and (process.poll() is None or any(owned(row)["identity_matches"] for row in records)):
            write(case/"manager_state.json", dict(status="attention_required", error=error, processes=records))
            time.sleep(2)
        if process is not None:
            code = process.wait()
    finally:
        deadline = time.monotonic()+30
        gpu = gpu_snapshot()
        while gpu["compute_processes"] and time.monotonic() < deadline:
            time.sleep(1); gpu = gpu_snapshot()
        result_file = Path(request["output"])/"result.json"
        result = read(result_file) if result_file.exists() else {}
        if result.get("status") == "failed" and not error:
            # Kit shutdown can leave the wrapper exit code at zero even when
            # the job recorded a failed runtime result. Preserve that evidence.
            error = result.get("error") or "Job recorded a failed runtime result"
        final_changes = check_inputs(request)
        if final_changes:
            error = (error+"; " if error else "")+"Frozen input changed: "+repr(final_changes)
        good = code == 0 and not error and not gpu["compute_processes"] and result.get("status") in ("completed", "rejected_interface")
        physically_ready = bool(good and request["job_kind"] == "direct_initialization_validation"
            and result.get("initialization_ready") is True and result.get("num_envs") == 1024
            and result.get("required_successful_envs") == 922 and result.get("successful_envs", 0) >= 922
            and result.get("simulated_duration_s") == 20.0
            and result.get("policy_and_both_normalizers_unchanged") is True
            and result.get("frozen_artifacts_unchanged") is True
            and result.get("teacher_constructed") is False and result.get("teacher_interventions") == 0)
        state = read(case/"supervision.json")
        state.update(status="completed" if good else "failed", ended_at=now(), wrapper_exit_code=code,
                     error=error, result=str(result_file), outcome=result.get("status"), GPU_released=not gpu["compute_processes"],
                     initialization_ready=physically_ready, stairs_acceptance_passed=False)
        write(case/"supervision.json", state)
        immutable(case/"exit.json", dict(wrapper_exit_code=code, error=error, at=now()))
        write(case/"manager_state.json", dict(status="finished", exit_code=code, outcome=state["status"],
            error=error, processes=records, GPU_released=not gpu["compute_processes"], ended_at=now()))
        pointer = read(POINTER)
        if pointer.get("active_stair_job") == str(case):
            pointer.update(status=state["status"], runtime_state_status="closed_loop_initialization_"+state["status"],
                initialization_ready=physically_ready,
                last_initialization_result=str(result_file),
                next_action="Review this result and evidence; no automatic retry, validation or PPO.")
            write(POINTER, pointer)
        os.close(slot_fd)


def log_stale(case, seconds):
    path = case/"job.log"
    return not path.exists() or time.time()-path.stat().st_mtime >= seconds


def launch(args):
    slot, case, process = None, None, None
    try:
        if args.start:
            SLOT.parent.mkdir(parents=True, exist_ok=True)
            slot = os.open(SLOT, os.O_CREAT|os.O_RDWR, 0o600)
            fcntl.flock(slot, fcntl.LOCK_EX|fcntl.LOCK_NB)
        checked = preflight(args)
        if not args.start:
            print(json.dumps({key:value for key,value in checked.items()
                  if key not in ("previous_pointer", "root_hashes", "source_hashes", "frozen_files")}, indent=2)); return
        previous = checked.pop("previous_pointer")
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        case = ROOT/"logs/stair_learning"/("closed_loop_"+stamp+"_"+uuid.uuid4().hex[:8])
        case.mkdir(exist_ok=False)
        snapshot_dir = case/"snapshot"; snapshot_dir.mkdir()
        for name in ("source", "tools"):
            shutil.copytree(ROOT/name, snapshot_dir/name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if not args.validate:
            checked["source_root"] = str(snapshot_dir/"source")
        output = case/"output"
        command = ["bash", str(Path(checked["entrypoint"]).with_suffix(".sh")), "--headless", "--device", "cuda:0",
                   "--num_envs", "1024", "--seed", "42", "--source_root", checked["source_root"], "--output", str(output)]
        if args.validate:
            command += ["--checkpoint", checked["initial_policy"], "--duration", "20", "--speed", ".3", "--video"]
        else:
            command += ["--initial_policy", checked["initial_policy"], "--teacher_checkpoint", checked["teacher_checkpoint"],
                        "--state", str(case/"supervision.json")]
        request = {**checked, "case":str(case), "created_at":now(), "command":command,
                   "output":str(output), "max_wall_seconds":3600 if not args.validate else 1800,
                   "source_hashes":source_hashes(Path(checked["source_root"])), "automatic_restart_allowed":False}
        immutable(case/"request.json", request)
        guard = dict(request_sha256=digest(case/"request.json"), created_at=now())
        immutable(case/"launch.lock", guard); immutable(request["duplicate_guard"], guard)
        immutable(case/"previous_pointer.json", previous)
        immutable(case/"stop_policy.json", dict(max_wall_seconds=request["max_wall_seconds"], protocol=PROTOCOL,
            signal="Re-read exact birth identities; learner INT, then TERM", automatic_retry_allowed=False))
        write(case/"supervision.json", {**request, "status":"starting", "physics_controls":0, "actor_updates":0})
        write(case/"manager_state.json", dict(status="starting", created_at=now()))
        pointer = dict(previous)
        pointer.update(status="starting", active_stair_job=str(case), active_stair_job_kind=request["job_kind"],
            supervision_path=str(case/"supervision.json"), launch_record=str(case/"launch.json"),
            single_launch_guard=str(case/"launch.lock"), monitor_record=str(case/"supervision.monitor.json"),
            stop_policy_path=str(case/"stop_policy.json"), monitor_snapshot_tool=str(SELF),
            monitor_snapshot_command=[sys.executable, str(SELF), "--monitor", str(case)],
            runtime_state_status=request["job_kind"]+"_starting", run_dir=str(output),
            active_closed_loop_initialization=str(case), initialization_seed_policy=checked["initial_policy"],
            source_commit=checked["source_commit"], source_snapshot=checked["source_root"],
            launch_record_sha256=None, single_launch_guard_sha256=digest(case/"launch.lock"),
            initialization_ready=False, stairs_acceptance_passed=False, automatic_restart_allowed=False)
        write(POINTER, pointer)
        with (case/"manager.log").open("xb") as stream:
            process = subprocess.Popen([sys.executable, str(SELF), "--manage", str(case), "--slot-fd", str(slot)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(slot,))
        immutable(case/"manager.launch.json", birth(process.pid))
        print(json.dumps(dict(status="started", case=str(case), manager_pid=process.pid)))
    except BaseException as exception:
        if case is not None and process is None:
            write(case/"dispatch_error.json", dict(error=repr(exception), at=now(), automatic_retry_allowed=False))
            if (case/"supervision.json").exists():
                state = read(case/"supervision.json")
                state.update(status="failed", error=repr(exception), ended_at=now())
                write(case/"supervision.json", state)
                write(case/"manager_state.json", dict(status="finished", exit_code=-1, error=repr(exception), ended_at=now()))
            pointer = read(POINTER)
            if pointer.get("active_stair_job") == str(case):
                pointer.update(status="failed", runtime_state_status="initialization_dispatch_failed")
                write(POINTER, pointer)
        raise
    finally:
        if slot is not None:
            os.close(slot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--initial_policy", type=Path)
    parser.add_argument("--teacher_checkpoint", type=Path)
    parser.add_argument("--monitor", type=Path)
    parser.add_argument("--manage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--slot-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.manage:
        require(args.slot_fd is not None, "Manager requires inherited slot")
        manage(args.manage, args.slot_fd)
    elif args.monitor:
        case, request = request_for(args.monitor)
        point = read(case/"supervision.monitor.json") if (case/"supervision.monitor.json").exists() else {}
        records = identities(case) if (case/"launch.identity.sha256.json").exists() else []
        manager = read(case/"manager_state.json")
        changes = check_inputs(request)
        if manager.get("status") == "finished":
            changes = [value for value in changes if value != "ROOT source/tools file set or bytes"]
        print(json.dumps(dict(checked_at=now(), read_only=True, manager=read(case/"manager_state.json"),
            state=read(case/"supervision.json"), latest_manager_check=point,
            processes=[owned(row) for row in records], gpu=gpu_snapshot(),
            changed_frozen_inputs=changes)))
    else:
        launch(args)


if __name__ == "__main__":
    main()
