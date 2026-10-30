"""Preflight, explicitly start, or monitor one frozen H2 stair evaluation.

No arguments perform preflight only. --start is required to launch. --monitor
creates an incremental pending inspection; --commit-monitor commits that exact
pending inspection and never sends signals. Only the detached manager may stop
its own identity-matched evaluator after a confirmed fault. It never retries.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import tempfile
import time
import uuid
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
POINTER = ROOT / "logs/launches/current_main_training.json"
SLOT = ROOT / "logs/stair_learning/formal_evaluation.slot.lock"
ENTRY = str(ROOT / "tools/evaluate_stair_policy.py")
SHELL = ROOT / "tools/evaluate_stair_policy.sh"
KIND = "stair_policy_formal_evaluation"
TERMINAL = {"completed", "failed", "finished", "accepted", "rejected", "stopped_no_improvement"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def immutable(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def locked(path):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def inside(path, base=ROOT):
    value = Path(path).resolve(strict=True)
    require(value.is_relative_to(base.resolve()), f"Outside H2walk scope: {value}")
    return value


def birth(pid):
    proc = Path("/proc") / str(int(pid))
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [value.decode(errors="replace") for value in (proc/"cmdline").read_bytes().split(b"\0") if value]
    return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]), ppid=int(fields[1]),
                argv=argv, fullargv=" ".join(argv), process_state=fields[0])


def process_check(identity, entry=ENTRY):
    observed = None
    try:
        observed = birth(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        pass
    matches = bool(observed and observed["process_state"] not in ("Z", "X")
        and observed["start_ticks"] == identity["start_ticks"] and observed["pgid"] == identity["pgid"]
        and observed["argv"] == identity["argv"] and entry in observed["argv"])
    return dict(recorded_identity=identity, observed=observed, identity_matches=matches)


def active_simulators(exclude=()):
    """Inventory only; never signal processes discovered by a broad search."""
    excluded = {os.getpid(), *exclude}
    jobs = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            item = birth(int(proc.name))
            if item["process_state"] in ("Z", "X"):
                continue
            args = item["argv"]
            scripts = []
            for token in args[1:]:
                if not token.endswith((".py", ".sh")):
                    continue
                path = Path(token)
                path = path if path.is_absolute() else (proc/"cwd").resolve()/path
                path = path.resolve()
                if path.name in {"train.py", "train.sh", "bootstrap_direct_policy.py", "bootstrap_direct_policy.sh",
                                 "evaluate_main_walk.py", "evaluate_main_walk.sh", "evaluate_stair_policy.py",
                                 "evaluate_stair_policy.sh"} and (path.is_relative_to(ROOT) or "IsaacLab" in str(path)):
                    scripts.append(str(path))
            kit = any("isaac-sim" in token and Path(token).name in ("kit", "isaac-sim.sh") for token in args)
            kit = kit or bool(args and "isaac-sim" in args[0]
                              and ("--headless" in args or "--enable_cameras" in args))
            manager = "--manage" in args and any(str(ROOT/"tools") in token for token in args)
            if scripts or kit or manager:
                jobs.append(dict(pid=item["pid"], argv=args, scripts=scripts, manager=manager))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return jobs


def gpu_snapshot():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    return dict(compute_processes=apps, devices=devices)


def source_hashes(source):
    source = Path(source)
    return {"source/"+str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def old_job_finished(pointer):
    require(pointer.get("status") in TERMINAL, "Current job has not entered a terminal state")
    state = read(pointer["supervision_path"])
    require(state.get("status") in TERMINAL, "Previous supervision still indicates an active job")
    previous = inside(pointer["active_stair_job"], ROOT/"logs")
    manager = read(previous/"manager_state.json")
    require(manager.get("status") == "finished", "Previous manager has not confirmed finish")
    for identity in read(pointer["launch_record"]).get("processes", []):
        try:
            observed = birth(identity["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        same_birth = observed["start_ticks"] == identity["start_ticks"] and observed["pgid"] == identity["pgid"]
        require(not same_birth or observed["process_state"] in ("Z", "X"), "A recorded previous process is still alive")
    return dict(case=str(previous), supervision=str(pointer["supervision_path"]),
                manager_state=str(previous/"manager_state.json"), manager_status="finished")


def preflight(checkpoint=None, validation=None):
    pointer = read(POINTER)
    require(not pointer.get("active_formal_evaluation"), "A formal evaluation is already recorded as active")
    previous = old_job_finished(pointer)
    checkpoint = inside(checkpoint or pointer["final_checkpoint"], ROOT/"logs/rsl_rl")
    match = re.fullmatch(r"model_(\d+)(?:_early_stop)?\.pt", checkpoint.name)
    require(match is not None, "Expected an iteration-labelled PPO checkpoint from its preserved run")
    iteration = int(match.group(1))
    checkpoint_sha = digest(checkpoint)
    receipt_path = inside(validation or checkpoint.with_suffix(".validation.json"))
    receipt = read(receipt_path)
    require(receipt.get("status") == "passed" and receipt.get("sha256") == checkpoint_sha,
            "Strict CPU checkpoint validation did not pass for these exact bytes")
    require(Path(receipt["checkpoint"]).resolve() == checkpoint and receipt.get("iteration") == iteration,
            "Validation receipt is for a different checkpoint/iteration")
    require(receipt.get("completed_iterations") == iteration+1 and receipt.get("size") == checkpoint.stat().st_size,
            "Validation receipt iteration count or checkpoint size differs")
    require(all(receipt.get(key) is True for key in ("weights_only", "all_tensors_finite", "model_strict_load",
        "optimizer_state_shapes_valid", "normalizers_strict_load", "rnd_strict_load", "rnd_optimizer_state_shapes_valid")),
        "Validation receipt is incomplete")
    require([receipt.get(key) for key in ("actor_dim", "critic_dim", "rnd_dim", "actions")] == [1029, 866, 73, 12],
            "Checkpoint dimensions changed")
    run_file = inside(checkpoint.parent/"run.json")
    metadata = read(run_file)
    require(metadata.get("task") == "Unitree-H2-Stair-Ascent-Policy" and metadata.get("status") in TERMINAL,
            "Checkpoint source run is not a terminal direct stair PPO job")
    source_commit = metadata.get("git_commit", "")
    require(isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
            "Checkpoint run has no recorded training commit")
    require(int(metadata.get("starting_iteration", 0)) <= iteration
            and int(metadata.get("completed_iterations", -1)) >= iteration+1,
            "Checkpoint iteration lies outside the source run's completed training")
    source = inside(checkpoint.parent/"training_source/source")
    require(Path(metadata["source_snapshot"]).resolve() == source, "Source must be run/training_source/source")
    manifest = inside(source.parent/"main_source_sha256.json")
    hashes = source_hashes(source)
    require(hashes == read(manifest), "Preserved source fingerprint changed")
    # The evaluator also reads these run-local files. Bind their exact bytes so
    # a later pointer/checkout change cannot silently retarget this request.
    run_files = [run_file, manifest, inside(checkpoint.parent/"params/agent.yaml"),
                 inside(checkpoint.parent/"params/env.yaml")]
    run_hashes = {str(path): digest(path) for path in run_files}
    dependencies = metadata.get("external_dependency_sha256", {})
    require(bool(dependencies), "The training-time external dependency manifest is missing")
    for path, expected in dependencies.items():
        require(digest(inside(path)) == expected, f"Training dependency changed: {path}")
    tools = {str(path): digest(path) for path in (SELF, Path(ENTRY), SHELL)}
    jobs = active_simulators()
    require(not jobs, f"An Isaac/training/evaluation/manager process is active: {jobs}")
    gpu = gpu_snapshot()
    require(not gpu["compute_processes"], "GPU compute is occupied; no parallel Isaac Sim is permitted")
    for line in gpu["devices"].splitlines():
        fields = [x.strip() for x in line.split(",")]
        if fields[0] == "0":
            require(int(fields[1]) <= 512, "cuda:0 is not empty enough for this1024-env formal replay")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer, checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha, checkpoint_validation=str(receipt_path),
                checkpoint_validation_sha256=digest(receipt_path), checkpoint_iteration=iteration,
                checkpoint_completed_iterations=iteration+1, training_run=str(checkpoint.parent),
                training_run_metadata=str(run_file), training_run_files_sha256=run_hashes,
                training_source=str(source), source_commit=source_commit,
                source_branch=metadata.get("source_branch"),
                source_hashes=hashes, external_dependency_sha256=dependencies, evaluator_tools_sha256=tools,
                accepted_flat_baseline=pointer.get("accepted_flat_baseline"), gpu=gpu)


def request_for(case):
    case = inside(case, ROOT/"logs/stair_learning")
    request = read(case/"request.json")
    guard = read(case/"launch.lock")
    require(request.get("job_kind") == KIND and request.get("case") == str(case), "Not a formal evaluation case")
    require(guard["request_sha256"] == digest(case/"request.json") and guard["launch_id"] == request["launch_id"],
            "Immutable request/independent launch guard changed")
    return case, request


def current_case(value=None):
    if value and value != "current":
        return inside(value, ROOT/"logs/stair_learning")
    pointer = read(POINTER)
    path = pointer.get("active_formal_evaluation") or pointer.get("formal_evaluation_case")
    require(path, "Current pointer has no formal evaluation case")
    return inside(path, ROOT/"logs/stair_learning")


def launch_identity(case):
    case, request = request_for(case)
    record = read(case/"launch.json")
    receipt = read(case/"launch.identity.sha256.json")
    require(receipt["sha256"] == digest(case/"launch.json") and record["launch_id"] == request["launch_id"],
            "Immutable launch identity fingerprint changed; do not signal any PID")
    return record


def fatal_lines(text):
    fatal, known = [], []
    for line in ANSI.sub("", text).splitlines():
        if "cuDeviceGetUuid" in line or re.search(r"Warp CUDA error\s*:?\s*36\b", line):
            known.append(line)
        elif re.search(r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA out of memory|"
                       r"illegal memory access|device-side assert|non[- ]finite|No space left on device|"
                       r"PytorchStreamWriter failed|CUDA error:(?!\s*36\b)", line, re.I):
            fatal.append(line)
    return fatal, known


def changed_inputs(request):
    changed = []
    files = {request["checkpoint"]: request["checkpoint_sha256"],
             request["checkpoint_validation"]: request["checkpoint_validation_sha256"],
             **request.get("training_run_files_sha256", {}),
             **request["external_dependency_sha256"], **request["evaluator_tools_sha256"]}
    for path, expected in files.items():
        if not Path(path).is_file() or digest(path) != expected:
            changed.append(path)
    if source_hashes(request["training_source"]) != request["source_hashes"]:
        changed.append(request["training_source"])
    return changed


def snapshot(case):
    case, request = request_for(case)
    with locked(case/"monitor.write.lock"):
        previous = read(case/"supervision.monitor.json") if (case/"supervision.monitor.json").exists() else {}
        log = Path(request["log"])
        size = log.stat().st_size if log.exists() else 0
        offset = int(previous.get("log_checked_offset", 0))
        require(offset <= size, "Evaluation log was truncated or replaced")
        data = b""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(4*1024*1024)
        text = previous.get("log_fragment", "") + data.decode(errors="replace")
        if text.endswith("\n"):
            complete, fragment = text, ""
        elif "\n" in text:
            complete, fragment = text.rsplit("\n", 1)
            complete += "\n"
        else:
            complete, fragment = "", text
        clean = ANSI.sub("", complete)
        new_fatal, known = fatal_lines(clean + ANSI.sub("", fragment))
        fatal = list(dict.fromkeys(previous.get("fatal_log_candidates", []) + new_fatal))[-30:]
        progress = dict(previous.get("progress", {}))
        for found in re.finditer(r"\[STAIR_POLICY\]\s+t=([\d.]+)s first_active=(\d+)/(\d+) six_levels=(\d+)", clean):
            value = float(found.group(1))
            if value > progress.get("simulated_time_s", -1):
                progress = dict(simulated_time_s=value, first_episode_active=int(found.group(2)),
                                num_envs=int(found.group(3)), all_six_levels=int(found.group(4)), advanced_at_epoch=time.time())
        launch = launch_identity(case) if (case/"launch.identity.sha256.json").exists() else {"processes": []}
        processes = [process_check(item) for item in launch["processes"]]
        manager = read(case/"manager_state.json") if (case/"manager_state.json").exists() else {}
        result_path = Path(request["output"])/"result.json"
        result = read(result_path) if result_path.exists() else None
        exit_record = read(case/"exit.json") if (case/"exit.json").exists() else None
        changes = changed_inputs(request)
        reasons = (["fatal evaluation log"] if fatal else []) + (["frozen input changed"] if changes else [])
        stamp = time.time()
        if result and result.get("status") == "failed":
            reasons.append("evaluator result is failed: "+str(result.get("error", "unspecified")))
        live = any(item["identity_matches"] for item in processes)
        anchor = progress.get("advanced_at_epoch", request["created_at_epoch"])
        if live and stamp-anchor >= 600 and log.exists() and stamp-log.stat().st_mtime >= 600:
            reasons.append("at least10minutes with neither log nor simulation-time advance")
        point = dict(check_id=uuid.uuid4().hex, launch_id=request["launch_id"], case=str(case), checked_at=now(),
            checked_at_epoch=stamp, base_check_id=previous.get("check_id"), log_read_from=offset,
            log_checked_offset=offset+len(data), log_size=size, log_caught_up=offset+len(data)==size,
            log_fragment=fragment, last_log_excerpt=clean[-6000:], fatal_log_candidates=fatal,
            known_startup_warnings=known[-10:], progress=progress, processes=processes, manager=manager,
            gpu=gpu_snapshot(), changed_frozen_inputs=changes, confirmed_stop_reasons=reasons,
            result=result, wrapper_exit=exit_record,
            decision="finished" if manager.get("status")=="finished" else "stop_required" if reasons else "evaluating_or_startup",
            training_updates=0, automatic_restart_allowed=False)
        pending_dir = case/"monitor_pending"
        pending_dir.mkdir(exist_ok=True)
        pending = pending_dir/(str(time.time_ns())+"_"+point["check_id"]+".json")
        immutable(pending, point)
        write(case/"supervision.monitor.pending.json", dict(path=str(pending), sha256=digest(pending)))
        return point, pending


def commit_snapshot(case, point, pending):
    case, request = request_for(case)
    require(Path(pending).is_relative_to(case/"monitor_pending"), "Pending inspection is outside this case")
    require(read(pending) == point and point["launch_id"] == request["launch_id"], "Pending inspection changed")
    with locked(case/"monitor.write.lock"):
        old = read(case/"supervision.monitor.json") if (case/"supervision.monitor.json").exists() else {}
        if old.get("check_id") == point["check_id"]:
            return True
        if old.get("check_id") != point["base_check_id"]:
            # A manual commit may race the manager. Do not stop a healthy
            # evaluator because another valid inspection advanced the cursor.
            return False
        with (case/"supervision.jsonl").open("a") as stream:
            stream.write(json.dumps(point, ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        write(case/"supervision.monitor.json", point)
        state = read(case/"supervision.json")
        state.update(last_monitor_check_at=point["checked_at"], last_monitor_progress=point["progress"],
                     monitor_current_conclusion=point["decision"], monitor_confirmed_stop_reasons=point["confirmed_stop_reasons"])
        write(case/"supervision.json", state)
        with locked(POINTER.with_suffix(".evaluation.write.lock")):
            pointer = read(POINTER)
            if pointer.get("supervision_path") == str(case/"supervision.json"):
                pointer.update(last_monitor_check_at=point["checked_at"], last_monitor_progress=point["progress"],
                               status=state["status"])
                if state["status"] not in TERMINAL:
                    pointer["runtime_state_status"] = point["decision"]
                write(POINTER, pointer)
        return True


def descendants(pid):
    pending, found = [pid], []
    while pending:
        value = pending.pop()
        if value in found:
            continue
        found.append(value)
        try:
            pending.extend(map(int, Path(f"/proc/{value}/task/{value}/children").read_text().split()))
        except FileNotFoundError:
            pass
    return found


def safe_stop(case, reasons):
    """INT evaluator first, then TERM; never signal a group or searched PID."""
    case, request = request_for(case)
    signals = []
    limitation = None
    for sig, grace in ((signal.SIGINT, 15), (signal.SIGTERM, 10)):
        # Re-read the immutable launch and current pointer before each signal.
        launch = launch_identity(case)
        for item in sorted(launch["processes"], key=lambda row: row["role"] != "evaluator"):
            _, fresh_request = request_for(case)
            pointer = read(POINTER)
            if pointer.get("supervision_path") != fresh_request["supervision_path"]:
                limitation = "Current pointer changed; no further signals are allowed from this case"
                break
            require(launch_identity(case) == launch, "Immutable launch identity changed")
            if sig == signal.SIGINT and item["role"] != "evaluator":
                continue
            checked = process_check(item)
            if not checked["identity_matches"]:
                continue
            try:
                os.kill(item["pid"], sig)
                signals.append(dict(at=now(), signal=sig.name, identity=item, observed=checked["observed"]))
            except (ProcessLookupError, PermissionError) as error:
                signals.append(dict(at=now(), signal=sig.name, identity=item, error=str(error)))
        if limitation:
            break
        deadline = time.monotonic()+grace
        while time.monotonic() < deadline:
            if not any(process_check(item)["identity_matches"] for item in launch_identity(case)["processes"]):
                break
            time.sleep(.25)
        if not any(process_check(item)["identity_matches"] for item in launch_identity(case)["processes"]):
            break
    evidence = dict(at=now(), reasons=reasons, signals=signals,
                    remaining=[process_check(item) for item in launch_identity(case)["processes"]],
                    limitation=limitation, automatic_restart_allowed=False)
    immutable(case/("stop_evidence_"+uuid.uuid4().hex[:8]+".json"), evidence)
    return evidence


def finish(case, exit_code, manager_birth, error=None, stopped=False):
    case, request = request_for(case)
    launch = launch_identity(case) if (case/"launch.identity.sha256.json").exists() else {"processes": []}
    remaining = [process_check(item) for item in launch["processes"]]
    require(not any(item["identity_matches"] for item in remaining), "Cannot mark a live evaluator finished")
    result_path = Path(request["output"])/"result.json"
    result = read(result_path) if result_path.exists() else None
    good = bool(not error and not stopped and exit_code == 0 and result and result.get("status")=="completed"
                and result.get("training_updates")==0 and result.get("diagnostic_only") is False
                and result.get("checkpoint_sha256")==request["checkpoint_sha256"]
                and result.get("checkpoint_unchanged") is True and result.get("frozen_source_unchanged") is True
                and result.get("external_asset_unchanged") is True
                and result.get("policy_and_both_normalizers_unchanged") is True
                and result.get("no_reference_controller") is True and not changed_inputs(request))
    passed = bool(good and result.get("formal_acceptance_passed") is True
                  and result.get("acceptance", {}).get("successful_envs", 0)>=922 and result.get("num_envs")==1024)
    status = "completed" if good else "failed"
    # Serialize with manual/manager monitor commits so an older cache merge
    # cannot restore an evaluating status after the child has actually ended.
    with locked(case/"monitor.write.lock"):
        state = read(case/"supervision.json")
        state.update(status=status, phase="formal_evaluation", ended_at=now(), wrapper_exit_code=exit_code,
            result=str(result_path) if result else None, formal_acceptance_passed=passed, stairs_acceptance_passed=passed,
            successful_envs=result.get("acceptance", {}).get("successful_envs", 0) if result else 0,
            manager_error=error, manager_stopped_on_fault=stopped, automatic_restart_allowed=False)
        write(case/"supervision.json", state)
        write(case/"manager_state.json", dict(status="finished", outcome="accepted" if passed else "rejected" if good else "failed",
            exit_code=exit_code, ended_at=now(), manager=manager_birth, error=error, processes=launch["processes"]))
        with locked(POINTER.with_suffix(".evaluation.write.lock")):
            pointer = read(POINTER)
            if pointer.get("supervision_path") == str(case/"supervision.json"):
                pointer.update(status=status, phase="formal_evaluation", active_formal_evaluation=None,
                    runtime_state_status="formal_evaluation_accepted" if passed else "formal_evaluation_rejected" if good else "formal_evaluation_failed",
                    stairs_acceptance_passed=passed, formal_acceptance_passed=passed, last_stair_result=state["result"],
                    formal_evaluation_successful_envs=state["successful_envs"], next_action="Wait for user/parent review; no automatic retry or training.")
                write(POINTER, pointer)


def manage(case, slot_fd):
    case, request = request_for(case)
    require(os.fstat(slot_fd).st_ino == SLOT.stat().st_ino, "Missing independent inherited evaluation slot lock")
    fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not (case/"launch.json").exists() and read(case/"supervision.json").get("status")=="starting",
            "This immutable evaluation request was already dispatched; never retry it")
    identity = birth(os.getpid())
    process, processes, stopped, error = None, [], False, None
    try:
        write(case/"manager_state.json", dict(status="prelaunch_verification", manager=identity))
        require(not changed_inputs(request), "Frozen inputs changed after preparation")
        require(not active_simulators(exclude=[request["parent_launcher_pid"]]), "Another simulator/manager started before dispatch")
        require(not gpu_snapshot()["compute_processes"], "GPU became occupied before dispatch")
        with Path(request["log"]).open("xb") as stream:
            process = subprocess.Popen(request["command"], cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(150):
                found = []
                for pid in descendants(process.pid):
                    try:
                        record = birth(pid)
                        if ENTRY in record["argv"]:
                            evaluator = Path(record["argv"][0]).name.startswith("python")
                            found.append({**record, "role": "evaluator" if evaluator else "wrapper", "also_wrapper": pid==process.pid})
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                if found:
                    processes = found
                if any(item["role"]=="evaluator" for item in processes) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(case/"launch.json", {**request, "manager": identity, "processes": processes,
                "wrapper_pid": process.pid,
                "wrapper_identity": next((item for item in processes if item["also_wrapper"]), None),
                "evaluator_identity": next((item for item in processes if item["role"]=="evaluator"), None),
                "identity_recorded_at": now()})
            immutable(case/"launch.identity.sha256.json", dict(sha256=digest(case/"launch.json"),
                                                               launch_id=request["launch_id"]))
            state = read(case/"supervision.json")
            state.update(status="evaluating", processes=processes, manager=identity,
                         launch_record_sha256=digest(case/"launch.json"), wrapper_started_at=now())
            write(case/"supervision.json", state)
            with locked(POINTER.with_suffix(".evaluation.write.lock")):
                pointer = read(POINTER)
                if pointer.get("supervision_path")==str(case/"supervision.json"):
                    pointer.update(launch_record_sha256=digest(case/"launch.json"),
                                   single_launch_guard_sha256=digest(case/"launch.lock"))
                    write(POINTER, pointer)
            write(case/"manager_state.json", dict(status="running", manager=identity, processes=processes))
            if process.poll() is None:
                require(any(item["role"]=="evaluator" for item in processes), "Could not bind evaluator birth identity")
                require(any(item["also_wrapper"] for item in processes), "Could not bind wrapper birth identity")
            while True:
                code = process.poll()
                point, pending = snapshot(case)
                commit_snapshot(case, point, pending)
                live = any(item["identity_matches"] for item in point["processes"])
                if (point["confirmed_stop_reasons"] or (code is not None and live)) and live and not stopped:
                    stopped = True
                    evidence = safe_stop(case, point["confirmed_stop_reasons"] or ["wrapper exited with evaluator still alive"])
                    state = read(case/"supervision.json")
                    state.update(status="stopping" if not any(x["identity_matches"] for x in evidence["remaining"])
                                 else "attention_required", stop_evidence=evidence, automatic_restart_allowed=False)
                    write(case/"supervision.json", state)
                live = any(process_check(item)["identity_matches"] for item in processes)
                if code is not None and not live:
                    break
                if not point["log_caught_up"]:
                    continue
                if code is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        pass
                else:
                    time.sleep(5)
            code = process.wait()
        immutable(case/"exit.json", dict(wrapper_exit_code=code, at=now(), wrapper_pid=process.pid))
        # Drain remaining bounded log chunks before concluding a zero exit is good.
        while True:
            point, pending = snapshot(case)
            commit_snapshot(case, point, pending)
            if point["fatal_log_candidates"] or point["changed_frozen_inputs"]:
                error = "Fatal log or changed frozen input"
            if point["log_caught_up"]:
                break
        finish(case, code, identity, error=error, stopped=stopped)
        point, pending = snapshot(case)
        commit_snapshot(case, point, pending)
    except BaseException as exception:
        error = repr(exception)
        if process is not None and (case/"launch.identity.sha256.json").exists():
            if any(process_check(item)["identity_matches"] for item in launch_identity(case)["processes"]):
                safe_stop(case, ["manager exception: "+error])
            while any(process_check(item)["identity_matches"] for item in launch_identity(case)["processes"]):
                state = read(case/"supervision.json")
                state.update(status="attention_required", manager_error=error, automatic_restart_allowed=False)
                write(case/"supervision.json", state)
                time.sleep(5)
            code = process.wait(timeout=15)
        elif process is not None:
            # Without a durable birth identity no signal is allowed. Retain
            # the slot and report attention instead of claiming a live child ended.
            while process.poll() is None:
                state = read(case/"supervision.json")
                state.update(status="attention_required", manager_error=error,
                             limitation="No durable launch identity; cannot safely signal", automatic_restart_allowed=False)
                write(case/"supervision.json", state)
                time.sleep(5)
            code = process.wait()
        else:
            code = -1
        if not (case/"exit.json").exists():
            immutable(case/"exit.json", dict(wrapper_exit_code=code, at=now(), manager_exception=error))
        finish(case, code, identity, error=error, stopped=True)
        raise
    finally:
        os.close(slot_fd)


def launch(args):
    slot_fd = None
    case, manager = None, None
    try:
        if args.start:
            SLOT.parent.mkdir(parents=True, exist_ok=True)
            slot_fd = os.open(SLOT, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = preflight(args.checkpoint, args.validation)
        if not args.start:
            print(json.dumps(dict(mode="preflight_only", **checked), ensure_ascii=False, indent=2))
            return
        previous = checked.pop("previous_pointer")
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        case = ROOT/"logs/stair_learning"/("formal_"+stamp+"_"+args.label+"_"+uuid.uuid4().hex[:8])
        case.mkdir(exist_ok=False)
        immutable(case/"previous_pointer.json", previous)
        immutable(case/"stopped_training_record.json", dict(pointer=previous,
            run=read(Path(checked["checkpoint"]).parent/"run.json"), previous_manager=read(Path(checked["previous"]["case"])/"manager_state.json")))
        immutable(case/"preflight.json", checked)
        immutable(case/"training_source_sha256.json", checked["source_hashes"])
        shutil.copy2(checked["checkpoint_validation"], case/"checkpoint.validation.json")
        (case/"tools_snapshot").mkdir()
        for path in (SELF, Path(ENTRY), SHELL):
            shutil.copy2(path, case/"tools_snapshot"/path.name)
        output = case/"evaluation"
        command = ["bash", str(SHELL), "--headless", "--device", "cuda:0", "--num_envs", "1024",
                   "--checkpoint", checked["checkpoint"], "--source_root", checked["training_source"],
                   "--output", str(output), "--video", "--seed", str(args.seed)]
        policy = dict(monitor_period_seconds=60, no_log_and_progress_timeout_seconds=600,
                      known_nonfatal=["Warp cuDeviceGetUuid", "Warp CUDA error36"],
                      on_confirmed_fault="SIGINT matching evaluator, then SIGTERM only re-read matching identities",
                      training_updates=0, automatic_restart_allowed=False, automatic_next_experiment_allowed=False)
        immutable(case/"stop_policy.json", policy)
        request = {**checked, "launch_id": uuid.uuid4().hex, "created_at": now(), "created_at_epoch": time.time(),
            "case": str(case), "job_kind": KIND, "phase": "formal_evaluation", "entrypoint": ENTRY,
            "command": command, "output": str(output), "log": str(case/"evaluation.log"),
            "supervision_path": str(case/"supervision.json"), "launch_record": str(case/"launch.json"),
            "single_launch_guard": str(case/"launch.lock"), "monitor_record": str(case/"supervision.monitor.json"),
            "stop_policy_path": str(case/"stop_policy.json"), "stop_policy": policy,
            "parent_launcher_pid": os.getpid(), "seed": args.seed, "num_envs":1024, "duration_s":90,
            "acceptance":dict(levels=6,height_m=.15,depth_m=.35,width_m=2.4,approach_m=1.4,
                              required_successful_envs=922, first_episodes_only=True),
            "training_updates":0, "automatic_restart_allowed":False,
            "scope":"Frozen checkpoint with its own validated bytes and preserved training source; no training or current ROOT fallback"}
        immutable(case/"request.json", request)
        immutable(case/"launch.lock", dict(launch_id=request["launch_id"], request_sha256=digest(case/"request.json"),
            created_at=now(), checkpoint_sha256=checked["checkpoint_sha256"], command=command, independent_duplicate_guard=True))
        write(case/"supervision.json", {**request, "status":"starting", "stairs_acceptance_passed":False})
        write(case/"manager_state.json", dict(status="starting", created_at=now()))
        with locked(POINTER.with_suffix(".evaluation.write.lock")):
            pointer = read(POINTER)
            require(pointer == previous, "Current pointer changed during preflight; do not dispatch")
            pointer.update(status="starting", phase="formal_evaluation", runtime_state_status="formal_evaluation_starting",
                active_formal_evaluation=str(case), formal_evaluation_case=str(case), active_stair_job=str(case),
                active_stair_job_kind=KIND, active_stair_learning_case=None,
                stopped_stair_training_record=str(case/"stopped_training_record.json"),
                previous_pointer_record=str(case/"previous_pointer.json"), evaluation_output=str(output),
                supervision_path=request["supervision_path"], launch_record=request["launch_record"],
                single_launch_guard=request["single_launch_guard"], monitor_record=request["monitor_record"],
                stop_policy_path=request["stop_policy_path"], monitor_snapshot_tool=str(SELF),
                monitor_snapshot_command=[sys.executable,str(SELF),"--monitor",str(case)],
                checkpoint=checked["checkpoint"], final_checkpoint=checked["checkpoint"],
                checkpoint_sha256=checked["checkpoint_sha256"],
                final_checkpoint_validation=checked["checkpoint_validation"], run_dir=checked["training_run"],
                frozen_checkpoint_iteration=checked["checkpoint_iteration"],
                frozen_checkpoint_completed_iterations=checked["checkpoint_completed_iterations"],
                source_snapshot=checked["training_source"], source_commit=checked["source_commit"],
                source_branch=checked["source_branch"],
                stairs_acceptance_passed=False, formal_acceptance_passed=False,
                accepted_flat_baseline=checked["accepted_flat_baseline"], automatic_restart_allowed=False)
            write(POINTER, pointer)
        with (case/"manager.log").open("xb") as stream:
            manager = subprocess.Popen([sys.executable,str(SELF),"--manage",str(case),"--slot-fd",str(slot_fd)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(slot_fd,))
        immutable(case/"manager.launch.json", birth(manager.pid))
        print(json.dumps(dict(mode="started_manager", case=str(case), manager_pid=manager.pid,
                             output=str(output), command=command), ensure_ascii=False, indent=2))
    except BaseException as error:
        if case is not None and (case/"supervision.json").exists() and manager is None:
            state = read(case/"supervision.json")
            state.update(status="failed", ended_at=now(), dispatch_error=repr(error), automatic_restart_allowed=False)
            write(case/"supervision.json", state)
            write(case/"manager_state.json", dict(status="finished", exit_code=-1,
                                                  ended_at=now(), dispatch_error=repr(error)))
            with locked(POINTER.with_suffix(".evaluation.write.lock")):
                pointer = read(POINTER)
                if pointer.get("supervision_path")==str(case/"supervision.json"):
                    pointer.update(status="failed", active_formal_evaluation=None,
                                   runtime_state_status="formal_evaluation_dispatch_failed")
                    write(POINTER, pointer)
        raise
    finally:
        if slot_fd is not None:
            os.close(slot_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--start", action="store_true", help="Explicitly dispatch exactly one formal frozen evaluation")
    modes.add_argument("--monitor", nargs="?", const="current", help="Write a bounded incremental pending inspection; no signals")
    modes.add_argument("--manage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help="Default: current final_checkpoint; use this checkpoint's own run.json/source snapshot")
    parser.add_argument("--validation", type=Path, help="Default: checkpoint.with_suffix('.validation.json')")
    parser.add_argument("--label", default="frozen_training_source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commit-monitor", action="store_true", help="Commit an existing pending inspection exactly; no signals")
    parser.add_argument("--pending", type=Path, help="Exact pending JSON returned by --monitor; avoids racing the manager")
    parser.add_argument("--slot-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.label):
        parser.error("label must contain1–64 letters,digits,underscores,or hyphens")
    if args.commit_monitor and args.monitor is None:
        parser.error("--commit-monitor requires --monitor [CASE]")
    if args.pending and not args.commit_monitor:
        parser.error("--pending requires --commit-monitor")
    if args.manage:
        require(args.slot_fd is not None, "Manager requires the inherited slot descriptor")
        manage(args.manage, args.slot_fd)
    elif args.monitor is not None:
        case = current_case(args.monitor)
        if args.commit_monitor:
            pending = inside(args.pending or read(case/"supervision.monitor.pending.json")["path"], case/"monitor_pending")
            point = read(pending)
            require(commit_snapshot(case, point, pending), "Pending inspection is stale; take a new snapshot")
            print(json.dumps(dict(mode="monitor_committed", pending=str(pending), decision=point["decision"]),ensure_ascii=False))
        else:
            point, pending = snapshot(case)
            print(json.dumps(dict(mode="pending_monitor", pending=str(pending), decision=point["decision"],
                progress=point["progress"], confirmed_stop_reasons=point["confirmed_stop_reasons"]),ensure_ascii=False,indent=2))
    else:
        launch(args)


if __name__ == "__main__":
    main()
