"""Preflight or explicitly dispatch one serial BC/PT0 diagnostic pair, never learning.

No arguments only read/preflight. --start is required to dispatch. Each arm has
its own immutable request, guard and process birth record. A physical first
episode failure is diagnostic data; a tool fault aborts the pair without retry.
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
ENTRY = ROOT/"tools/diagnose_stair_initialization.py"
SHELL = ROOT/"tools/diagnose_stair_initialization.sh"
POINTER = ROOT/"logs/launches/current_main_training.json"
SLOT = ROOT/"logs/stair_learning/initialization_check.slot.lock"
COMMIT = "36ac622054e083850565905bab0f5d56a7ceb64e"
KIND = "stair_initialization_diagnostic_pair"
TERMINAL = {"completed", "failed", "finished", "stopped_no_improvement"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(path, base=ROOT):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(base.resolve()), f"Outside H2walk scope: {path}")
    return path


def write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def immutable(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def locked(path):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def birth(pid):
    proc = Path("/proc")/str(int(pid))
    fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
    argv = [item.decode(errors="replace") for item in (proc/"cmdline").read_bytes().split(b"\0") if item]
    return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]), ppid=int(fields[1]),
                argv=argv, fullargv=" ".join(argv), process_state=fields[0])


def process_check(identity):
    try:
        observed = birth(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        observed = None
    matches = bool(observed and observed["process_state"] not in ("Z", "X")
        and observed["start_ticks"] == identity["start_ticks"] and observed["pgid"] == identity["pgid"]
        and observed["argv"] == identity["argv"] and identity["entrypoint"] in observed["argv"]
        and identity["entrypoint"] in (str(ENTRY), str(SHELL)))
    return dict(recorded_identity=identity, observed=observed, identity_matches=matches)


def source_hashes(source):
    source = Path(source)
    return {"source/"+str(path.relative_to(source)): digest(path) for path in sorted(source.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def gpu_snapshot():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    return dict(compute_processes=apps, devices=devices)


def require_gpu_empty():
    gpu = gpu_snapshot()
    require(not gpu["compute_processes"], "GPU compute is occupied; never overlap Isaac arms")
    rows = [row.split(",") for row in gpu["devices"].splitlines()]
    cuda0 = next((row for row in rows if row[0].strip() == "0"), None)
    require(cuda0 is not None and int(cuda0[1].strip()) <= 512, "cuda:0 memory has not been released")
    return gpu


def active_jobs(exclude=()):
    excluded = {os.getpid(), *exclude}
    names = {"train.py", "train.sh", "bootstrap_direct_policy.py", "bootstrap_direct_policy.sh",
             "evaluate_main_walk.py", "evaluate_main_walk.sh", "evaluate_stair_policy.py",
             "evaluate_stair_policy.sh", ENTRY.name, SHELL.name, "validate_stair_domain.py"}
    result = []
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
                if Path(token).name not in names:
                    continue
                path = Path(token)
                path = path if path.is_absolute() else (proc/"cwd").resolve()/path
                if path.resolve().is_relative_to(ROOT) or "IsaacLab" in str(path):
                    scripts.append(str(path.resolve()))
            manager = "--manage" in args and any(str(ROOT/"tools") in arg for arg in args)
            kit = any("isaac-sim" in arg and Path(arg).name in {"kit", "isaac-sim.sh"} for arg in args)
            kit = kit or bool(args and "isaac-sim" in args[0] and "--headless" in args)
            if scripts or manager or kit:
                result.append(dict(pid=item["pid"], argv=args, scripts=scripts, manager=manager))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def previous_finished(pointer):
    require(pointer.get("status") == "completed" and not pointer.get("active_formal_evaluation"),
            "Previous formal job has not completed")
    case = inside(pointer["active_stair_job"], ROOT/"logs/stair_learning")
    require(read(pointer["supervision_path"]).get("status") == "completed", "Previous supervision is not complete")
    manager = read(case/"manager_state.json")
    require(manager.get("status") == "finished" and manager.get("exit_code") == 0,
            "Previous manager has not confirmed a zero-exit finish")
    require(read(case/"exit.json").get("wrapper_exit_code") == 0, "Previous wrapper did not exit successfully")
    for identity in read(pointer["launch_record"]).get("processes", []):
        try:
            observed = birth(identity["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        same = observed["start_ticks"] == identity["start_ticks"] and observed["pgid"] == identity["pgid"]
        require(not same or observed["process_state"] in ("Z", "X"), "A previous formal process is still alive")
    return dict(case=str(case), manager_state=str(case/"manager_state.json"), status="finished", exit_code=0)


def preflight():
    pointer = read(POINTER)
    previous = previous_finished(pointer)
    require(not pointer.get("active_stair_initialization_check"), "An initialization pair is already active")
    run = inside(pointer["run_dir"], ROOT/"logs/rsl_rl")
    metadata = read(run/"run.json")
    require(metadata.get("task") == "Unitree-H2-Stair-Ascent-Policy" and metadata.get("status") == "completed"
            and metadata.get("git_commit") == COMMIT, "Require this completed36ac direct stair run")
    source = inside(metadata["source_snapshot"])
    require(source == inside(run/"training_source/source"), "Never use ROOT source or another snapshot")
    hashes = source_hashes(source)
    manifest = inside(source.parent/"main_source_sha256.json")
    require(hashes == read(manifest), "Preserved source fingerprint changed")
    bc, pt0 = inside(metadata["initial_policy"]), inside(run/"model_0.pt")
    bc_receipt = inside(bc.parent/"result.json")
    require(read(bc_receipt).get("initialization_ready") is True, "BC initialization receipt is not ready")
    require(digest(bc) == metadata.get("initial_policy_sha256"), "BC bytes differ from this run's initialization")
    dependencies = metadata.get("external_dependency_sha256", {})
    require(bool(dependencies), "Missing USD/LiDAR dependencies")
    files = {str(path): digest(path) for path in (SELF, inside(ENTRY), inside(SHELL), run/"run.json",
             manifest, run/"params/agent.yaml", run/"params/env.yaml", bc, pt0, bc_receipt)}
    for path, expected in dependencies.items():
        require(digest(inside(path)) == expected, f"Training dependency changed: {path}")
        files[path] = expected
    guard = ROOT/"logs/stair_learning"/("initialization_check_"+hashlib.sha256(str(run).encode()).hexdigest()[:16]+".launch.lock")
    require(not guard.exists(), "This training run already dispatched its diagnostic pair; no retry")
    jobs = active_jobs()
    require(not jobs, f"Another Isaac/training/manager job is active: {jobs}")
    return dict(checked_at=now(), previous=previous, previous_pointer=pointer, training_run=str(run),
        source_commit=COMMIT, training_source=str(source), source_hashes=hashes, frozen_files_sha256=files,
        duplicate_guard=str(guard), arms={"bc":dict(checkpoint=str(bc), checkpoint_sha256=files[str(bc)]),
                                            "pt0":dict(checkpoint=str(pt0), checkpoint_sha256=files[str(pt0)])},
        gpu=require_gpu_empty(), num_envs=1024, duration_s=3.0, seed=42, deterministic=True,
        wall_limit_per_arm_seconds=300, shutdown_reserve_seconds=30, diagnostic_only=True,
        training_updates=0, formal_acceptance_passed=False, automatic_retry_allowed=False,
        accepted_flat_baseline=pointer.get("accepted_flat_baseline"))


def pair_request(case):
    case = inside(case, ROOT/"logs/stair_learning")
    request = read(case/"request.json")
    guard = read(case/"launch.lock")
    require(request.get("job_kind") == KIND and request.get("case") == str(case), "Not this diagnostic pair")
    require(guard["request_sha256"] == digest(case/"request.json") and guard["launch_id"] == request["launch_id"],
            "Immutable pair request/guard changed")
    require(read(request["duplicate_guard"])["request_sha256"] == guard["request_sha256"], "Duplicate guard changed")
    return case, request


def arm_request(case, arm):
    case, pair = pair_request(case)
    folder = case/arm
    request = read(folder/"request.json")
    guard = read(folder/"launch.lock")
    require(arm in ("bc", "pt0") and request["pair_launch_id"] == pair["launch_id"] and request["arm"] == arm,
            "Wrong arm identity")
    require(guard["request_sha256"] == digest(folder/"request.json")
            == pair["arm_request_sha256"][arm], "Immutable arm request/guard changed")
    return folder, request


def frozen_changes(request):
    changed = [path for path, expected in request["frozen_files_sha256"].items()
               if not Path(path).is_file() or digest(path) != expected]
    if source_hashes(request["training_source"]) != request["source_hashes"]:
        changed.append(request["training_source"])
    return changed


def launch_record(case, arm):
    folder, request = arm_request(case, arm)
    record = read(folder/"launch.json")
    require(record["arm_launch_id"] == request["arm_launch_id"]
            and read(folder/"launch.identity.sha256.json")["sha256"] == digest(folder/"launch.json"),
            "Immutable arm birth record changed; do not signal")
    return record


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


def fatal_lines(text):
    result = []
    for line in ANSI.sub("", text).splitlines():
        if "cuDeviceGetUuid" in line or re.search(r"Warp CUDA error\s*:?\s*36\b", line):
            continue
        if re.search(r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA out of memory|"
                     r"illegal memory access|device-side assert|non[- ]finite|No space left on device|"
                     r"PytorchStreamWriter failed|CUDA error:(?!\s*36\b)|"
                     r"(?:^|[\s:=,])(?:nan|[+-]?inf(?:inity)?)(?:$|[\s,;)])", line, re.I):
            result.append(line)
    return result


def inspect_arm(case, arm, previous=None):
    folder, request = arm_request(case, arm)
    previous = previous or {}
    log = folder/"diagnostic.log"
    size = log.stat().st_size if log.exists() else 0
    offset = int(previous.get("log_checked_offset", 0))
    require(size >= offset, "Diagnostic log was truncated or replaced")
    data = b""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(offset); data = stream.read(2*1024*1024)
    text = previous.get("log_fragment", "")+data.decode(errors="replace")
    if text.endswith("\n"):
        complete, fragment = text, ""
    elif "\n" in text:
        complete, fragment = text.rsplit("\n", 1)
    else:
        complete, fragment = "", text
    fatal = list(dict.fromkeys(previous.get("fatal_log_candidates", [])+fatal_lines(complete+fragment)))[-30:]
    records = launch_record(case, arm)["processes"] if (folder/"launch.identity.sha256.json").exists() else []
    result_path = Path(request["output"])/"result.json"
    result = read(result_path) if result_path.exists() else None
    changed = frozen_changes(pair_request(case)[1])
    point = dict(checked_at=now(), arm=arm, log_checked_offset=offset+len(data), log_size=size,
        log_caught_up=offset+len(data)==size, log_fragment=fragment, log_excerpt=ANSI.sub("",complete)[-4000:],
        fatal_log_candidates=fatal, frozen_input_changes=changed,
        processes=[process_check(item) for item in records], result=result, gpu=gpu_snapshot())
    write(folder/"monitor.json", point)
    with (folder/"monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(point, ensure_ascii=False)+"\n"); stream.flush(); os.fsync(stream.fileno())
    write(case/"supervision.monitor.json", point)
    return point


def safe_stop(case, arm, reasons):
    folder, request = arm_request(case, arm)
    signals, limitation = [], None
    for sig, grace in ((signal.SIGINT, 8), (signal.SIGTERM, 5)):
        record = launch_record(case, arm)
        for item in sorted(record["processes"], key=lambda value:value["role"] != "evaluator"):
            arm_request(case, arm); launch_record(case, arm)
            pointer = read(POINTER)
            if pointer.get("active_stair_initialization_check") != str(case) or pointer.get("active_initialization_arm") != arm:
                limitation = "Current arm pointer changed; no further signals are allowed"
                break
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
        while time.monotonic() < deadline and any(process_check(item)["identity_matches"] for item in record["processes"]):
            time.sleep(.2)
        if not any(process_check(item)["identity_matches"] for item in record["processes"]):
            break
    evidence = dict(at=now(), reasons=reasons, signals=signals, limitation=limitation,
                    remaining=[process_check(item) for item in launch_record(case, arm)["processes"]])
    immutable(folder/("stop_evidence_"+uuid.uuid4().hex[:8]+".json"), evidence)
    return evidence


def run_arm(case, arm, manager):
    folder, request = arm_request(case, arm)
    require(not (folder/"launch.json").exists(), "This arm already dispatched; never retry")
    pair = pair_request(case)[1]
    require(not frozen_changes(pair), "Frozen diagnostic inputs changed")
    require(not active_jobs(exclude=[pair["parent_launcher_pid"]]), "Another simulator/manager is active")
    require_gpu_empty()
    state = {**request, "status":"starting", "started_at":now(), "manager":manager}
    write(folder/"supervision.json", state)
    with locked(POINTER.with_suffix(".initialization.write.lock")):
        pointer = read(POINTER)
        require(pointer.get("active_stair_initialization_check") == str(case), "Pair pointer changed before arm dispatch")
        pointer.update(status="starting", active_initialization_arm=arm, supervision_path=str(folder/"supervision.json"),
            launch_record=str(folder/"launch.json"), single_launch_guard=str(folder/"launch.lock"),
            launch_record_sha256=None, single_launch_guard_sha256=digest(folder/"launch.lock"),
            monitor_record=str(folder/"monitor.json"), diagnostic_checkpoint=request["checkpoint"],
            runtime_state_status="initialization_arm_starting")
        write(POINTER, pointer)
    process, records, stopped, error, previous = None, [], False, None, None
    started = time.monotonic()
    try:
        with (folder/"diagnostic.log").open("xb") as stream:
            process = subprocess.Popen(request["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            spawn = birth(process.pid)
            immutable(folder/"spawn_identity.json", spawn)
            while time.monotonic()-started < 30:
                found = []
                for pid in descendants(process.pid):
                    try:
                        item = birth(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    entry = str(ENTRY) if str(ENTRY) in item["argv"] else str(SHELL) if str(SHELL) in item["argv"] else None
                    if entry:
                        found.append({**item, "entrypoint":entry,
                            "role":"evaluator" if Path(item["argv"][0]).name.startswith("python") else "wrapper"})
                if found:
                    records = found
                if any(item["role"] == "evaluator" for item in records) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(folder/"launch.json", {**request, "manager":manager, "processes":records,
                "wrapper_pid":process.pid, "recorded_at":now(), "spawn_identity":spawn})
            immutable(folder/"launch.identity.sha256.json", dict(sha256=digest(folder/"launch.json")))
            if process.poll() is None:
                require(any(item["role"] == "evaluator" for item in records), "Could not bind evaluator birth identity")
                require(any(item["pid"] == process.pid for item in records), "Could not bind wrapper birth identity")
            state.update(status="evaluating", launch_record_sha256=digest(folder/"launch.json"))
            write(folder/"supervision.json", state)
            with locked(POINTER.with_suffix(".initialization.write.lock")):
                pointer = read(POINTER)
                require(pointer.get("active_stair_initialization_check") == str(case)
                        and pointer.get("active_initialization_arm") == arm, "Current arm pointer changed during startup")
                pointer.update(status="evaluating", runtime_state_status="initialization_arm_evaluating",
                    launch_record_sha256=digest(folder/"launch.json"),
                    single_launch_guard_sha256=digest(folder/"launch.lock"))
                write(POINTER, pointer)
            write(case/"manager_state.json", dict(status="running", active_arm=arm, manager=manager))
            while True:
                previous = inspect_arm(case, arm, previous)
                reasons = (["fatal diagnostic log"] if previous["fatal_log_candidates"] else [])
                reasons += (["frozen diagnostic input changed"] if previous["frozen_input_changes"] else [])
                if previous["result"] and previous["result"].get("status") == "failed":
                    reasons.append("diagnostic tool reported failure")
                if time.monotonic()-started >= pair["wall_limit_per_arm_seconds"]-pair["shutdown_reserve_seconds"]:
                    reasons.append("arm reached its wall-time shutdown reserve")
                live = any(row["identity_matches"] for row in previous["processes"])
                if process.poll() is not None and live:
                    reasons.append("wrapper ended while an identity-matched child is alive")
                if reasons:
                    error = "; ".join(reasons)
                if reasons and live and not stopped:
                    stopped = True
                    evidence = safe_stop(case, arm, reasons)
                    if any(item["identity_matches"] for item in evidence["remaining"]):
                        state.update(status="attention_required", error=error, stop_evidence=evidence,
                                     limitation="Matching processes remain after INT/TERM; pair cannot advance")
                        write(folder/"supervision.json", state)
                if process.poll() is not None and not any(process_check(item)["identity_matches"] for item in records):
                    if previous["log_caught_up"]:
                        break
                    continue
                time.sleep(2)
            code = process.wait(timeout=5)
    except BaseException as exception:
        error = repr(exception)
        if process is not None and (folder/"launch.identity.sha256.json").exists():
            if process.poll() is None or any(process_check(item)["identity_matches"] for item in records):
                stopped = True; safe_stop(case, arm, [error])
        # If identity binding failed, do not broaden signal scope to an unverified
        # PID. Retain the pair slot and report attention while any child remains.
        while process is not None and (process.poll() is None or any(process_check(item)["identity_matches"] for item in records)):
            write(folder/"supervision.json", {**state, "status":"attention_required", "error":error,
                  "limitation":"Cannot yet prove every spawned process ended; pair will not advance"})
            time.sleep(2)
        code = process.wait() if process is not None else -1
    immutable(folder/"exit.json", dict(wrapper_exit_code=code, at=now(), stopped_on_fault=stopped, error=error))
    result_path = Path(request["output"])/"result.json"
    result = read(result_path) if result_path.exists() else None
    release = None
    deadline = started+pair["wall_limit_per_arm_seconds"]
    while time.monotonic() < deadline:
        release = gpu_snapshot()
        if not release["compute_processes"]:
            try:
                release = require_gpu_empty(); break
            except RuntimeError:
                pass
        time.sleep(1)
    released = bool(release and not release["compute_processes"]
                    and any(row.split(",")[0].strip()=="0" and int(row.split(",")[1].strip())<=512
                            for row in release["devices"].splitlines()))
    good = bool(code == 0 and not error and not stopped and released and result
        and result.get("status") == "completed" and result.get("diagnostic_only") is True
        and result.get("training_updates") == 0 and result.get("formal_acceptance_passed") is False
        and result.get("arm") == arm and result.get("checkpoint_sha256") == request["checkpoint_sha256"]
        and result.get("checkpoint_unchanged") is True and result.get("frozen_source_unchanged") is True
        and result.get("external_dependencies_unchanged") is True
        and result.get("policy_and_both_normalizers_unchanged") is True
        and result.get("num_envs") == 1024 and result.get("seed") == 42
        and result.get("duration_limit_s") == 3.0 and 0 <= result.get("simulated_duration_s", -1) <= 3.0+1e-6
        and result.get("source_root") == pair["training_source"]
        and result.get("network_dimensions") == [1029, 866, 73, 12]
        and not frozen_changes(pair))
    state.update(status="completed" if good else "failed", ended_at=now(), wrapper_exit_code=code,
        result=str(result_path) if result else None, error=error, gpu_released=released, gpu=release,
        remaining=[process_check(item) for item in records], diagnostic_only=True,
        formal_acceptance_passed=False, automatic_retry_allowed=False)
    write(folder/"supervision.json", state)
    write(folder/"manager_state.json", dict(status="finished", exit_code=code, outcome=state["status"],
        processes=records, gpu_released=released, error=error, ended_at=now()))
    return dict(arm=arm, status=state["status"], result=result, result_path=str(result_path),
                exit_record=str(folder/"exit.json"), manager_state=str(folder/"manager_state.json"),
                gpu_released=released, error=error)


def pairing(case, outcomes):
    paths = {arm:str(case/arm/"evidence/pairing_manifest.json") for arm in ("bc", "pt0")}
    if not all(Path(path).is_file() for path in paths.values()):
        return dict(status="unavailable", actual_initial_states_matched=None, manifests=paths)
    try:
        documents = {arm:read(path) for arm, path in paths.items()}
        request = pair_request(case)[1]
        required = {"initial_state", "initial_physical_parameters", "first_raw_observations"}
        malformed = []
        for arm, document in documents.items():
            if document.get("schema_version") != 1 or document.get("arm") != arm:
                malformed.append(arm+":schema_or_arm")
            groups = document.get("required_groups", {})
            if set(groups) != required:
                malformed.append(arm+":required_groups")
            for group in groups.values():
                if not isinstance(group, dict) or not group:
                    malformed.append(arm+":empty_group"); continue
                for entry in group.values():
                    if (not isinstance(entry, dict) or not isinstance(entry.get("shape"), list)
                            or not isinstance(entry.get("dtype"), str)
                            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))) is None):
                        malformed.append(arm+":invalid_array_descriptor")
        if malformed:
            return dict(status="invalid_manifest", actual_initial_states_matched=None, manifests=paths, reasons=malformed)
        fields = ["seed", "num_envs", "source_root", "source_sha256", "external_dependency_sha256", "required_groups"]
        differences = [field for field in fields if documents["bc"].get(field) != documents["pt0"].get(field)]
        for arm, document in documents.items():
            if document.get("seed") != 42 or document.get("num_envs") != 1024:
                differences.append(arm+":seed_or_num_envs")
            if document.get("source_root") != request["training_source"] or document.get("source_sha256") != request["source_hashes"]:
                differences.append(arm+":source_binding")
            dependencies = {path:request["frozen_files_sha256"][path]
                            for path in read(Path(request["training_run"])/"run.json")["external_dependency_sha256"]}
            if document.get("external_dependency_sha256") != dependencies:
                differences.append(arm+":dependency_binding")
        matched = not differences and all(outcomes.get(arm, {}).get("status") == "completed" for arm in ("bc", "pt0"))
        return dict(status="matched" if matched else "mismatched", actual_initial_states_matched=matched,
            manifests=paths, mismatching_fields=differences,
            comparison="Required contiguous array bytes, shape and dtype; ZIP metadata and normalized own observations excluded")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        return dict(status="invalid_manifest", actual_initial_states_matched=None, manifests=paths, error=repr(error))


def manage(case, slot_fd):
    case, request = pair_request(case)
    require(os.fstat(slot_fd).st_ino == SLOT.stat().st_ino, "Missing inherited independent pair slot")
    fcntl.flock(slot_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(read(case/"manager_state.json").get("status") == "starting", "Pair already dispatched; never retry")
    manager = birth(os.getpid())
    immutable(case/"manager.identity.json", manager)
    outcomes, error = {}, None
    try:
        for arm in ("bc", "pt0"):
            outcome = run_arm(case, arm, manager)
            outcomes[arm] = outcome
            if outcome["status"] != "completed":
                error = "Tool/process/input/GPU-release failure in "+arm+"; pair aborted without retry"
                break
    except BaseException as exception:
        error = repr(exception)
    finally:
        good = not error and len(outcomes) == 2 and all(row["status"]=="completed" for row in outcomes.values())
        combined = dict(status="completed" if good else "failed", ended_at=now(), arms=outcomes,
            initial_state_pairing=pairing(case, outcomes), error=error, diagnostic_only=True,
            formal_acceptance_passed=False, training_updates=0, automatic_retry_allowed=False,
            limitation="3s initializer diagnostic is neither formal stair acceptance nor learning")
        write(case/"result.json", combined)
        write(case/"supervision.json", {**request, **combined})
        write(case/"manager_state.json", dict(status="finished", exit_code=0 if good else 1,
            manager=manager, outcome=combined["status"], error=error, ended_at=now()))
        immutable(case/"exit.json", dict(wrapper_exit_code=0 if good else 1, manager_exit_code=0 if good else 1, at=now()))
        with locked(POINTER.with_suffix(".initialization.write.lock")):
            pointer = read(POINTER)
            if pointer.get("active_stair_initialization_check") == str(case):
                pointer.update(status=combined["status"], active_stair_initialization_check=None,
                    active_initialization_arm=None, runtime_state_status="initialization_diagnostic_"+combined["status"],
                    supervision_path=str(case/"supervision.json"), monitor_record=str(case/"supervision.monitor.json"),
                    last_initialization_diagnostic=str(case/"result.json"), diagnostic_only=True,
                    next_action="Review paired initializer data; no automatic training, restart or acceptance claim.")
                write(POINTER, pointer)
        os.close(slot_fd)
    if error:
        raise RuntimeError(error)


def launch(args):
    slot_fd, case, process = None, None, None
    try:
        if args.start:
            SLOT.parent.mkdir(parents=True, exist_ok=True)
            slot_fd = os.open(SLOT, os.O_CREAT|os.O_RDWR, 0o600)
            fcntl.flock(slot_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        checked = preflight()
        if not args.start:
            print(json.dumps(dict(mode="preflight_only", **checked), ensure_ascii=False, indent=2)); return
        previous = checked.pop("previous_pointer")
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        case = ROOT/"logs/stair_learning"/("initialization_"+stamp+"_"+args.label+"_"+uuid.uuid4().hex[:8])
        case.mkdir(exist_ok=False)
        launch_id = uuid.uuid4().hex
        immutable(case/"previous_pointer.json", previous)
        policy_path = case/"stop_policy.json"
        immutable(policy_path, dict(maximum_wall_seconds_per_arm=300, shutdown_reserve_seconds=30,
            monitor_period_seconds=2, bounded_log_read_bytes=2*1024*1024,
            on_fault="Re-read this arm's immutable birth record; exact PID SIGINT then SIGTERM",
            physical_first_episode_failure_is_tool_failure=False, tool_failure_aborts_pair=True,
            automatic_retry_allowed=False, automatic_learning_allowed=False))
        checked["frozen_files_sha256"][str(policy_path)] = digest(policy_path)
        (case/"tools_snapshot").mkdir()
        for tool in (SELF, ENTRY, SHELL):
            shutil.copy2(tool, case/"tools_snapshot"/tool.name)
        arm_hashes = {}
        for arm in ("bc", "pt0"):
            folder = case/arm; folder.mkdir()
            output = folder/"evidence"
            item = checked["arms"][arm]
            command = ["bash", str(SHELL), "--headless", "--device", "cuda:0", "--num_envs", "1024",
                "--duration", "3", "--seed", "42", "--arm", arm, "--checkpoint", item["checkpoint"],
                "--source_root", checked["training_source"], "--training_run", checked["training_run"],
                "--output", str(output)]
            request = {**item, "arm":arm, "arm_launch_id":uuid.uuid4().hex, "pair_launch_id":launch_id,
                "pair_case":str(case), "command":command, "output":str(output), "diagnostic_only":True,
                "num_envs":1024, "duration_s":3.0, "seed":42, "deterministic":True,
                "training_updates":0, "formal_acceptance_passed":False, "automatic_retry_allowed":False}
            immutable(folder/"request.json", request)
            arm_hashes[arm] = digest(folder/"request.json")
            immutable(folder/"launch.lock", dict(request_sha256=arm_hashes[arm], arm_launch_id=request["arm_launch_id"]))
        request = {**checked, "launch_id":launch_id, "job_kind":KIND, "case":str(case),
            "created_at":now(), "parent_launcher_pid":os.getpid(), "arm_request_sha256":arm_hashes,
            "arm_order":["bc","pt0"], "automatic_next_experiment_allowed":False, "stop_policy_path":str(policy_path)}
        immutable(case/"request.json", request)
        guard = dict(request_sha256=digest(case/"request.json"), launch_id=launch_id, created_at=now())
        immutable(case/"launch.lock", guard); immutable(request["duplicate_guard"], guard)
        write(case/"supervision.json", {**request, "status":"starting"})
        write(case/"manager_state.json", dict(status="starting", created_at=now()))
        with locked(POINTER.with_suffix(".initialization.write.lock")):
            pointer = read(POINTER)
            require(pointer == previous, "Current pointer changed during preflight; do not dispatch")
            pointer.update(status="starting", phase="initialization_diagnostic", diagnostic_only=True,
                active_stair_initialization_check=str(case), initialization_check_case=str(case),
                active_initialization_arm=None, active_stair_job=str(case), active_stair_job_kind=KIND,
                supervision_path=str(case/"supervision.json"), single_launch_guard=str(case/"launch.lock"),
                launch_record=None, launch_record_sha256=None, single_launch_guard_sha256=digest(case/"launch.lock"),
                monitor_record=str(case/"supervision.monitor.json"), monitor_snapshot_tool=str(SELF),
                stop_policy_path=str(policy_path),
                monitor_snapshot_command=[sys.executable,str(SELF),"--monitor",str(case)],
                previous_pointer_record=str(case/"previous_pointer.json"), automatic_restart_allowed=False)
            write(POINTER, pointer)
        with (case/"manager.log").open("xb") as stream:
            process = subprocess.Popen([sys.executable,str(SELF),"--manage",str(case),"--slot-fd",str(slot_fd)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(slot_fd,))
        immutable(case/"manager.launch.json", birth(process.pid))
        print(json.dumps(dict(mode="started_manager", case=str(case), manager_pid=process.pid), indent=2))
    except BaseException as error:
        if case is not None and process is None:
            write(case/"dispatch_error.json", dict(error=repr(error), at=now(), automatic_retry_allowed=False))
            if (case/"supervision.json").exists():
                state = read(case/"supervision.json")
                state.update(status="failed", error=repr(error), ended_at=now(), automatic_retry_allowed=False)
                write(case/"supervision.json", state)
                write(case/"manager_state.json", dict(status="finished", exit_code=-1, error=repr(error), ended_at=now()))
                with locked(POINTER.with_suffix(".initialization.write.lock")):
                    pointer = read(POINTER)
                    if pointer.get("active_stair_initialization_check") == str(case):
                        pointer.update(status="failed", active_stair_initialization_check=None, active_initialization_arm=None,
                            runtime_state_status="initialization_dispatch_failed",
                            next_action="Review dispatch failure; immutable duplicate guard forbids automatic retry.")
                        write(POINTER, pointer)
        raise
    finally:
        if slot_fd is not None:
            os.close(slot_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--start", action="store_true")
    modes.add_argument("--monitor", nargs="?", const="current")
    modes.add_argument("--manage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--slot-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--label", default="bc_vs_pt0")
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.label), "Invalid case label")
    if args.manage:
        require(args.slot_fd is not None, "Manager requires an inherited slot")
        manage(args.manage, args.slot_fd)
    elif args.monitor:
        pointer = read(POINTER)
        case = pointer.get("active_stair_initialization_check") or pointer.get("initialization_check_case")
        case, request = pair_request(case if args.monitor == "current" else args.monitor)
        print(json.dumps(dict(case=str(case), manager=read(case/"manager_state.json"),
            arms={arm:read(case/arm/"supervision.json") if (case/arm/"supervision.json").exists() else {"status":"not_started"}
                  for arm in ("bc","pt0")}, frozen_input_changes=frozen_changes(request), gpu=gpu_snapshot()),
            ensure_ascii=False, indent=2))
    else:
        launch(args)


if __name__ == "__main__":
    main()
