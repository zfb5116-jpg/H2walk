"""Prepare/start one bounded direct-policy stair PPO job, or monitor its record.

Preflight is the default. Actual launch requires --start. With no arguments,
the tool collects the current stair-learning monitor into a pending record.
Neither monitoring nor preflight sends signals or starts Isaac Sim.
"""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from launch_residual_learning import birth, now, write


ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "logs/launches/current_main_training.json"
BRANCH = "codex/h2-stair-ascent-20261030"
TASK = "Unitree-H2-Stair-Ascent-Policy"
KIND = "stair_policy_ppo_learning"
ENTRY = str(ROOT / "tools/train.py")
SLOT = ROOT / "logs/stair_learning/direct_ppo.slot.lock"
TEACHER_SHA = "2b06db2e09061cc660d46578b112ea29d0e601c3e75ac2163f8425aa24553fcb"
TERMINAL = {"completed", "failed", "stopped_no_improvement", "accepted", "finished", "rejected_development"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LEGS = [f"{side}_{joint}_joint" for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_roll", "ankle_pitch")]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def immutable(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def inside(path, base=ROOT):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(base.resolve()), f"Path is outside this H2walk scope: {path}")
    return path


def gpu_snapshot():
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True).strip()
    return {"compute_processes": apps, "devices": gpu}


def process_check(identity):
    observed = None
    try:
        observed = birth(int(identity["pid"]))
        fields = (Path("/proc")/str(identity["pid"])/"stat").read_text().rsplit(")", 1)[1].split()
        observed["process_state"] = fields[0]
    except (FileNotFoundError, ProcessLookupError):
        pass
    matches = bool(observed and observed["start_ticks"] == identity["start_ticks"]
                   and observed["pgid"] == identity["pgid"] and observed["argv"] == identity["argv"]
                   and ENTRY in observed["argv"] and observed["process_state"] not in ("Z", "X"))
    return {"recorded_identity": identity, "observed": observed, "identity_matches": matches}


def active_project_jobs(exclude=()):
    """Read-only inventory; recognize script argv tokens, not arbitrary shell text."""
    excluded = {os.getpid(), *exclude}
    jobs = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in ("Z", "X"):
                continue
            args = [s.decode(errors="replace") for s in (proc/"cmdline").read_bytes().split(b"\0") if s]
            for token in args[1:]:
                if not token.endswith((".py", ".sh")):
                    continue
                path = Path(token)
                if not path.is_absolute():
                    path = (proc/"cwd").resolve()/path
                path = path.resolve()
                if path.parent != ROOT/"tools":
                    continue
                name = path.name
                blocking = (name in ("train.py", "train.sh", "bootstrap_direct_policy.py", "bootstrap_direct_policy.sh",
                                     "evaluate_main_walk.py", "evaluate_main_walk.sh", "evaluate_stair_policy.py", "evaluate_stair_policy.sh")
                            or name.startswith("manage_") or "--manage" in args)
                if blocking:
                    jobs.append({"pid": int(proc.name), "script": str(path), "argv": args})
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return jobs


def no_active_record(previous):
    if previous.get("active_stair_job"):
        state = read(previous["supervision_path"])
        require(state.get("status") in TERMINAL, f"Current H2 job is still {state.get('status')}")
        manager_path = Path(previous["active_stair_job"])/"manager_state.json"
        if manager_path.exists():
            require(read(manager_path).get("status") == "finished", "Previous H2 manager has not finished")
    if previous.get("active_stair_trial"):
        case = Path(previous["active_stair_trial"])
        if (case/"state.json").exists():
            require(read(case/"state.json").get("status") in TERMINAL, "A stair replay is still active")


def flat_baseline():
    base = ROOT/"artifacts/flat_forward"
    manifest = read(base/"manifest.sha256.json")
    checkpoint, acceptance = base/"model_999.pt", read(base/"acceptance.json")
    require(manifest.get("model_999.pt") == TEACHER_SHA and digest(checkpoint) == TEACHER_SHA,
            "The independently accepted flat checkpoint hash changed")
    require(acceptance.get("successful_envs", 0) >= 922 and acceptance.get("num_envs") == 1024,
            "Accepted flat baseline evidence is incomplete")
    for name in ("acceptance.json", "verification/result.json", "verification/source_manifest.json"):
        require(digest(base/name) == manifest[name], f"Accepted flat evidence changed: {name}")
    source = Path("source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks")
    expected = {str(p.relative_to(base/"training_source/source")): digest(p)
                for p in (base/"training_source"/source).rglob("*")
                if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts}
    actual = {str(p.relative_to(ROOT/"source")): digest(p) for p in (ROOT/source).rglob("*")
              if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts}
    require(actual == expected, "Network source differs from the accepted flat architecture; report before changing it")
    return {"walking_acceptance_passed": True, "successful_envs": acceptance["successful_envs"],
            "num_envs": 1024, "final_checkpoint": str(checkpoint), "checkpoint_sha256": TEACHER_SHA,
            "acceptance": str(base/"acceptance.json"), "verification_result": str(base/"verification/result.json"),
            "network_hashes": expected, "scope": "Accepted flat reference-controller system; not acceptance of this direct stair policy"}


def inspect_bootstrap(path, receipt_path):
    """CPU, weights_only strict model/norm validation, never SimulationApp."""
    import torch
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    path, receipt_path = inside(path), inside(receipt_path)
    receipt = read(receipt_path)
    require(receipt.get("status") == "completed" and receipt.get("initialization_ready") is True,
            "Bootstrap is not completed and initialization_ready")
    require(receipt.get("ppo_updates") == 0, "Bootstrap receipt contains PPO updates")
    teacher = inside(receipt["teacher_checkpoint"])
    require(digest(teacher) == TEACHER_SHA, "Bootstrap used a different original teacher checkpoint")
    declared = inside(receipt["initial_policy"])
    require(digest(declared) == digest(path), "Bootstrap receipt and requested initial policy differ")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("format") == "h2_direct_policy_initialization_v1"
            and payload.get("ppo_completed_iterations") == 0
            and payload.get("policy_dimensions") == [1029, 866, 73, 12]
            and payload.get("action_semantics") == "full_soft_tanh_position_v1"
            and payload.get("teacher_checkpoint_sha256") == TEACHER_SHA,
            "Bootstrap payload violates the fresh direct-action architecture/teacher contract")
    forbidden = {"optimizer_state_dict", "rnd_state_dict", "rnd_optimizer_state_dict", "iter"} & payload.keys()
    require(not forbidden, f"Initialization must not carry resume/Adam/RND states: {sorted(forbidden)}")
    count = 0
    def finite(value):
        nonlocal count
        if torch.is_tensor(value):
            count += 1
            require(bool(torch.isfinite(value).all()), "Bootstrap contains nonfinite tensors")
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value: finite(child)
        elif isinstance(value, float):
            require(math.isfinite(value), "Bootstrap contains nonfinite metadata")
    finite(payload)
    spec = importlib.util.spec_from_file_location("stair_initial_actor", ROOT/
        "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.H2ActorCriticOur(1029, 866, 12, num_joints=29)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    require(bool((model.std > 0).all()), "Initial policy noise must be positive")
    normalizers = {}
    for label, key, width in (("actor", "obs_norm_state_dict", 1029),
                               ("critic", "privileged_obs_norm_state_dict", 866)):
        norm = EmpiricalNormalization([width])
        norm.load_state_dict(payload[key], strict=True)
        normalizers[label] = {k: float(v) for k, v in norm.state_dict().items() if "count" in k}
    quality = receipt["initialization_quality"]
    require(payload["metadata"] == quality, "Receipt quality does not describe the selected bootstrap weights")
    for key, threshold in (("mean_normalized_torque_error", .05), ("standing_mean_normalized_error", .05),
                           ("walking_mean_normalized_error", .05), ("p95_normalized_torque_error", .15),
                           ("teacher_label_clipped_fraction", .05)):
        require(0 <= float(quality[key]) < threshold, f"Bootstrap initialization quality failed: {key}")
    return {"status": "passed", "initial_policy": str(path), "initial_policy_sha256": digest(path),
            "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path), "quality": quality,
            "teacher_checkpoint": str(teacher), "teacher_checkpoint_sha256": TEACHER_SHA,
            "dimensions": [1029, 866, 73, 12], "tensor_count": count,
            "model_strict_load": True, "dual_normalizer_strict_load": True,
            "normalizer_counts": normalizers, "nonzero_BC_normalizer_counts_allowed": True,
            "PPO_iteration": 0, "resume_optimizer_or_RND_states_present": False,
            "fresh_Adam_and_RND_required_at_runtime": True, "formal_walking_or_stair_acceptance": False}


def bootstrap_check(path, receipt):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
                       PYTHONPATH=os.environ.get("H2_RL_PACKAGES", "/home/nsgm/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages"))
    command = [str(Path(os.environ.get("H2_ISAAC_SIM", "/home/nsgm/isaac-sim"))/"python.sh"),
               str(Path(__file__).resolve()), "--inspect-bootstrap", str(path), str(receipt)]
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=90)
    require(result.returncode == 0, "CPU bootstrap validation failed:\n"+result.stdout[-3000:]+result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def stop_policy():
    return {"warmup_iterations": 2000, "assessment_interval_iterations": 1000,
            "patience_iterations": 3000, "patience_seconds": 7200, "required_stagnant_assessments": 3,
            "episode_window_size": 5000,
            "absolute_improvement_thresholds": {
                "survival20s_fraction": .05, "survival_and_progress_fraction": .05,
                "locomotion_success_fraction": .02, "terrain_level": .25,
                "mean_completed_levels": .25, "first_level_bilateral_fraction": .05,
                "all_levels_bilateral_fraction": .02, "stair_success_fraction": .02,
                "mean_verified_feet": .5, "mean_transferred_levels": .25},
            "median_episode_improvement": "max(1 second, 10 percent)", "progress_ratio_improvement": .1,
            "preserve_episode_stability_for_progress_gain": True, "reward_alone_does_not_count": True,
            "automatic_retry_allowed": False, "automatic_parameter_change_allowed": False,
            "physical_learning_improvements_are_not_stair_acceptance": True,
            "bounded_run_note": "A4000-iteration cap can finish before the earliest full stagnation patience; never extend it automatically.",
            "acceptance": "Independent1024-env first90s replay; >=922 same environments complete every level with bilateral80-600ms valid flight/80ms load/8mm footprint, net0.9m rise±0.1m and2s top dwell"}


def preflight(initial, receipt, iterations):
    require(1 <= iterations <= 4000, "This run is capped at4000 PPO iterations")
    previous = read(POINTER)
    require(not previous.get("stair_experiments_paused_by_user"), "Stair work is paused by the user")
    no_active_record(previous)
    require(git("branch", "--show-current") == BRANCH, f"Required branch is {BRANCH}")
    jobs = active_project_jobs()
    require(not jobs, f"An H2 job is still active: {jobs}")
    gpu = gpu_snapshot()
    require(not gpu["compute_processes"], f"GPU still occupied: {gpu['compute_processes']}")
    baseline = flat_baseline()
    bootstrap = bootstrap_check(initial, receipt)
    return {"checked_at": now(), "source_branch": BRANCH, "source_commit": git("rev-parse", "HEAD"),
            "working_tree_status": git("status", "--porcelain=v1"), "bootstrap": bootstrap,
            "accepted_flat_baseline": baseline, "gpu": gpu, "active_project_jobs": jobs,
            "task": TASK, "num_envs": 1024, "num_steps_per_env": 24, "max_iterations": iterations,
            "save_interval": 500, "checkpoint": None, "starting_iteration": 0,
            "initialization_mode": "behavior_cloned_direct_policy", "network_structure_changed": False,
            "expected_total_timesteps": iterations*1024*24, "will_start_without_explicit_start": False}


def tail_jsonl(path, limit=2*1024*1024):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as stream:
        offset = max(0, path.stat().st_size-limit)
        stream.seek(offset)
        data = stream.read(limit)
    lines = data.decode(errors="replace").splitlines()
    if offset:
        lines = lines[1:]
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A final line can still be in the writer's buffer.
    return result


def parse_training_log(text, previous):
    """A Learning header becomes complete only after its complete ETA footer."""
    clean = ANSI.sub("", text)
    starts = list(re.finditer(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", clean))
    result = {"iteration": previous.get("iteration", -1), "ETA": previous.get("ETA"),
              "scalars": previous.get("scalars", {}), "total_timesteps": previous.get("total_timesteps"),
              "incomplete_iteration_block": ""}
    for index, match in enumerate(starts):
        block = clean[match.start():starts[index+1].start() if index+1 < len(starts) else len(clean)]
        eta = re.search(r"(?m)^\s*ETA:\s*([^\n]+)\n", block)
        if eta is None:
            result["incomplete_iteration_block"] = block[-20000:]
            continue
        iteration = int(match.group(1))
        if iteration < result["iteration"]:
            continue
        scalars = {}
        for line in block.splitlines():
            scalar = re.match(r"\s*([^:]+):\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$", line)
            if scalar:
                scalars[scalar.group(1).strip()] = float(scalar.group(2))
        steps = re.search(r"Total timesteps:\s*([0-9,]+)", block)
        result.update(iteration=iteration, ETA=eta.group(1).strip(), scalars=scalars)
        if steps:
            result["total_timesteps"] = int(steps.group(1).replace(",", ""))
    return result


def fatal_lines(text):
    result, known = [], []
    for line in ANSI.sub("", text).splitlines():
        if ("cuDeviceGetUuid" in line or re.search(r"Warp CUDA error\s*:?\s*36\b", line)):
            known.append(line)
            continue
        if re.search(r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA out of memory|"
                     r"illegal memory access|device-side assert|non[- ]finite|No space left on device|"
                     r"PytorchStreamWriter failed|failed (?:to )?(?:write|save).*checkpoint", line, re.I):
            result.append(line)
        elif re.search(r"(?:loss|reward|noise|entropy|gradient|value function|action.*std)[^:]*[:=].*"
                       r"(?<![a-z])(?:nan|[-+]?inf)(?![a-z])", line, re.I):
            result.append(line)
    return result, known


def startup_contract(metadata, request):
    if not metadata or metadata.get("status") == "initializing":
        return {"status": "pending", "errors": []}
    errors = []
    def expect(ok, reason):
        if not ok: errors.append(reason)
    expect(metadata.get("task") == TASK, "wrong task")
    expect(metadata.get("source_branch") == request["source_branch"], "wrong runtime branch")
    expect(metadata.get("git_commit") == request["source_commit"], "wrong runtime commit")
    expect(metadata.get("checkpoint") is None and metadata.get("starting_iteration") == 0, "checkpoint resume detected")
    expect(metadata.get("initialization_mode") == "behavior_cloned_direct_policy", "BC initialization not recorded")
    expect(metadata.get("initial_policy_sha256") == request["initial_policy_sha256"], "initial policy hash changed")
    expect(metadata.get("num_envs") == 1024 and metadata.get("num_steps_per_env") == 24, "wrong sampling count")
    expect(metadata.get("max_iterations") == request["max_iterations"] and metadata.get("save_interval") == 500,
           "wrong iteration/save budget")
    expect(metadata.get("num_actions") == 12 and metadata.get("policy_shape") == [1024, 1029], "wrong actor/action dimensions")
    shapes = metadata.get("observation_shapes", {})
    expect(shapes.get("critic") == [1024, 866] and shapes.get("rnd_state") == [1024, 73], "wrong critic/RND dimensions")
    action = metadata.get("action_contract", {})
    expect(action.get("class") == "DirectLegPositionAction" and action.get("joint_names") == LEGS,
           "wrong direct-action class/order")
    control = metadata.get("control_settings", {})
    expect(control.get("contact_gait_reference") is None and control.get("stance_preload_torques") == {}
           and control.get("stair_reference") is None and control.get("stance_orientation_feedback") is None
           and control.get("lateral_com_feedback") is None,
           "legacy controller/reference is attached")
    geometry = metadata.get("task_geometry") or {}
    terrain = geometry.get("sub_terrains", {}).get("ascending_stairway", {})
    expect(geometry.get("rows") == 8 and geometry.get("curriculum") is True
           and set(geometry.get("sub_terrains", {})) == {"ascending_stairway"}, "stair generator was replaced/mixed with another terrain")
    expect(str(terrain.get("function", "")).endswith("stair_ascent_terrain.ascent_stairs")
           and terrain.get("num_steps") == 6 and terrain.get("height_levels") == 8,
           "wrong actual stair generator/level count")
    for key, value in (("minimum_riser_height", .03), ("maximum_riser_height", .15),
                       ("tread_depth", .35), ("stair_width", 2.4), ("approach_distance", 1.4)):
        expect(isinstance(terrain.get(key), (int, float)) and math.isclose(terrain[key], value, abs_tol=1e-6),
               "wrong authored stair geometry: "+key)
    runtime = metadata.get("runtime_stair_geometry") or {}
    expect(runtime.get("terrain_rows") == 8 and runtime.get("levels") == 6
           and runtime.get("scripted_gait_installed") is False, "runtime is not a six-level direct-policy stair task")
    expect({"stair_below_floor", "stair_out_of_tile"}.issubset(runtime.get("termination_terms", [])),
           "physical-domain termination terms are absent")
    expect(runtime.get("domain_bounds_local") == [-1.9, 9.65, -2.9, 2.9]
           and runtime.get("below_floor_tolerance_m") == .05, "physical-domain geometry disagrees with the reviewed task")
    expect(bool(runtime.get("episode_evidence_directory")), "per-episode physical evidence recorder is absent")
    for key, value in (("riser_min_m", .03), ("riser_max_m", .15), ("current_riser_min_m", .03),
                       ("current_riser_max_m", .03), ("depth_m", .35), ("width_m", 2.4), ("approach_m", 1.4)):
        expect(isinstance(runtime.get(key), (int, float)) and math.isclose(runtime[key], value, abs_tol=1e-6),
               "wrong initialized runtime stair geometry: "+key)
    initial = metadata.get("initial_learning_state", {})
    expect(initial.get("checkpoint_loaded") is False and initial.get("behavior_policy_initialized") is True,
           "not a fresh BC-initialized PPO state")
    expect(initial.get("iteration") == 0 and initial.get("policy_optimizer_state_entries") == 0
           and initial.get("rnd_optimizer_state_entries") == 0, "PPO/Adam/RND state was resumed")
    expect(set(initial.get("normalizer_counts", {})) == {"actor", "critic"}, "dual BC normalizers are missing")
    expect(all(math.isfinite(float(v)) and float(v) >= 0 for group in initial.get("normalizer_counts", {}).values()
               for v in group.values()), "invalid BC normalization counts")
    exploration=metadata.get("stair_exploration_budget", {})
    std=initial.get("policy_std", [])
    low,high=exploration.get("minimum_raw_std", []),exploration.get("maximum_raw_std", [])
    expect(len(std)==len(low)==len(high)==12 and all(
        math.isfinite(s) and 0<lo<=s<=hi for s,lo,hi in zip(std,low,high)),
        "actual initial std is outside the per-joint physical exploration budget")
    expect(exploration.get("policy_mean_and_soft_range_unrestricted") is True
           and exploration.get("network_structure_changed") is False,
           "exploration projection changed mean authority or network design")
    expect(metadata.get("algorithm_settings",{}).get("entropy_coef")==1e-4,
           "raw-coordinate entropy coefficient does not match the angular exploration scale")
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "normalizer_counts": initial.get("normalizer_counts"), "nonzero_BC_counts_are_expected": True,
            "fresh_optimizer_and_RND": not errors, "network_architecture_unchanged": not errors}


def snapshot(case=None, commit=False, _locked=False):
    pointer = read(POINTER)
    case = inside(case or Path(pointer["supervision_path"]).parent, ROOT/"logs/stair_learning")
    if commit and not _locked:
        with (case/"monitor.write.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return snapshot(case, commit=True, _locked=True)
    request = read(case/"request.json")
    require(request.get("job_kind") == KIND, "Current pointer is not a direct-policy PPO job")
    state_path = Path(request["supervision_path"])
    state = read(state_path)
    monitor_path = Path(request["monitor_record"])
    old = read(monitor_path) if monitor_path.exists() else {}
    log = Path(request["log"])
    stamp = time.time()
    size = log.stat().st_size if log.exists() else 0
    offset = min(int(old.get("log_checked_offset", 0)), size)
    data = b""
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(4*1024*1024)
    new = old.get("log_fragment", "") + data.decode(errors="replace")
    if new.endswith("\n"):
        complete, fragment = new, ""
    elif "\n" in new:
        complete, fragment = new.rsplit("\n", 1)
        complete += "\n"
    else:
        complete, fragment = "", new
    parsed = parse_training_log(old.get("incomplete_iteration_block", "") + complete, old)
    new_fatal, known = fatal_lines(complete)
    fatal = list(dict.fromkeys(old.get("fatal_log_candidates", []) + new_fatal))[-30:]
    launch_path = Path(request["launch_record"])
    launch = read(launch_path) if launch_path.exists() else {"processes": []}
    processes = [process_check(identity) for identity in launch["processes"]]
    manager = read(case/"manager_state.json") if (case/"manager_state.json").exists() else {}
    run = Path(state["run_dir"]) if state.get("run_dir") else None
    metadata, metadata_read_error = None, None
    if run and (run/"run.json").exists():
        try:
            metadata = read(run/"run.json")
        except json.JSONDecodeError as error:
            metadata_read_error = str(error)
    startup = startup_contract(metadata, request)
    checkpoints = []
    if run:
        for path in run.glob("model_*.pt"):
            match = re.match(r"model_(\d+)(?:_early_stop)?\.pt$", path.name)
            if match:
                stat = path.stat()
                checkpoints.append({"path": str(path), "iteration": int(match.group(1)), "bytes": stat.st_size,
                                    "mtime_epoch": stat.st_mtime, "early_stop": "early_stop" in path.name})
    checkpoints.sort(key=lambda item: (item["iteration"], item["early_stop"]))
    interval = int(request["save_interval"])
    missing = []
    for due in range(0, max(0, parsed["iteration"]-250)+1, interval):
        if parsed["iteration"] > due+250 and not any(c["iteration"] == due and c["bytes"] > 0 and not c["early_stop"] for c in checkpoints):
            missing.append(due)
    grew = parsed["iteration"] > int(old.get("iteration", -1))
    last_progress = stamp if grew else old.get("last_iteration_advance_epoch", stamp)
    caught_up = offset+len(data) == size
    stall = bool(state.get("status") == "training" and caught_up and parsed["iteration"] >= 0
                 and stamp-last_progress >= 600 and log.exists() and stamp-log.stat().st_mtime >= 600)
    telemetry = tail_jsonl(run/"motion_metrics.jsonl") if run else []
    assessments = tail_jsonl(run/"effect_assessments.jsonl") if run else []
    required = []
    if fatal: required.append("new_fatal_log")
    if missing: required.append("checkpoint_overdue_more_than250_iterations")
    if stall: required.append("log_and_iteration_stalled_at_least10minutes")
    if startup["status"] == "failed": required.append("startup_contract_failed")
    changed = [name for name, expected in request["prelaunch_source_hashes"].items()
               if not (ROOT/name).exists() or digest(ROOT/name) != expected]
    if changed: required.append("frozen_source_or_tool_changed")
    if manager.get("status") == "finished" and state.get("status") not in ("completed", "stopped_no_improvement", "failed"):
        required.append("process_ended_before_declared_training_completion")
    patience = assessments[-1] if assessments else None
    if patience and patience.get("stop_required") and state.get("status") == "training":
        if (patience.get("stagnant_assessments", 0) >= 3
                and patience.get("iterations_without_improvement", 0) >= 3000
                and patience.get("seconds_without_improvement", 0) >= 7200
                and stamp-patience.get("checked_at_epoch", stamp) >= 30):
            required.append("recorded_continuous_stagnation_rule_not_exited")
    final_checks = None
    if state.get("status") == "completed" and metadata:
        final_iteration = request["max_iterations"]-1
        final = run/f"model_{final_iteration}.pt"
        final_checks = {"last_complete_log_iteration": parsed["iteration"], "expected_last_iteration": final_iteration,
                        "run_completed_iterations": metadata.get("completed_iterations"),
                        "expected_total_timesteps": request["expected_total_timesteps"],
                        "log_total_timesteps": parsed.get("total_timesteps"),
                        "final_checkpoint": str(final), "final_checkpoint_exists": final.exists() and final.stat().st_size > 0,
                        "weight_validation_and_frozen_replay_required": True,
                        "process_exit_and_gpu_release_required": True, "stair_acceptance_passed": False}
        if caught_up and (parsed["iteration"] != final_iteration or metadata.get("completed_iterations") != request["max_iterations"]
                          or parsed.get("total_timesteps") != request["expected_total_timesteps"]
                          or not final_checks["final_checkpoint_exists"]):
            required.append("completed_run_count_or_final_checkpoint_mismatch")
    point = {"checked_at": now(), "checked_at_epoch": stamp, "job_kind": KIND, "case": str(case),
             "state_status": state.get("status"), "run_dir": str(run) if run else None,
             **parsed, "last_iteration_advance_epoch": last_progress,
             "log": str(log), "log_checked_offset": offset+len(data), "log_bytes": size,
             "log_fragment": fragment, "log_caught_up": caught_up, "log_unread_bytes": size-offset-len(data),
             "log_mtime_epoch": log.stat().st_mtime if log.exists() else None,
             "log_age_s": stamp-log.stat().st_mtime if log.exists() else None,
             "last_log_excerpt": ANSI.sub("", complete)[-6000:], "fatal_log_candidates": fatal,
             "known_startup_warnings": known[-10:], "processes": processes, "manager": manager,
             "gpu": gpu_snapshot(), "startup_contract": startup, "metadata_read_error": metadata_read_error,
             "checkpoints": checkpoints, "latest_checkpoint": checkpoints[-1] if checkpoints else None,
             "new_checkpoints": [c for c in checkpoints if c["path"] not in {v["path"] for v in old.get("checkpoints", [])}],
             "missing_periodic_checkpoint_iterations": missing,
             "latest_motion": telemetry[-1] if telemetry else None, "latest_effect_assessment": patience,
             "continuous_assessment_tail": assessments[-4:], "source_or_tool_changed": changed,
             "confirmed_stop_reasons": required, "final_checks": final_checks,
             "decision": "stop_required" if required else "complete_pending_weight_and_stair_acceptance" if state.get("status") == "completed"
                         else "stopped_pending_analysis" if state.get("status") in ("failed", "stopped_no_improvement") else "progressing_or_startup",
             "stair_acceptance_passed": False, "reward_is_not_walking_evidence": True}
    write(case/"supervision.monitor.pending.json", point)
    if commit:
        write(monitor_path, point)
        with state_path.with_suffix(".jsonl").open("a") as stream:
            stream.write(json.dumps(point, ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Re-read, merge only monitor conclusions. Never erase birth identities
        # or the entry point's iteration-boundary stop-policy configuration.
        latest = read(state_path)
        latest.update(last_monitor_check_at=point["checked_at"], last_monitor_iteration=point["iteration"],
                      last_monitor_ETA=point["ETA"], monitor_current_conclusion=point["decision"],
                      monitor_confirmed_stop_reasons=required)
        write(state_path, latest)
        current = read(POINTER)
        if current.get("supervision_path") == str(state_path):
            current.update(status=state.get("status"), run_dir=state.get("run_dir"),
                           last_monitor_check_at=point["checked_at"], last_monitor_iteration=point["iteration"],
                           last_monitor_ETA=point["ETA"], runtime_state_status=point["decision"],
                           completed_iterations=state.get("completed_iterations", max(0, point["iteration"]+1)))
            if point["latest_checkpoint"]:
                current["latest_checkpoint"] = point["latest_checkpoint"]["path"]
            if metadata and state.get("status") in ("completed", "stopped_no_improvement"):
                final = Path(metadata.get("final_checkpoint", ""))
                if final.is_file():
                    current.update(final_checkpoint=str(final), checkpoint_sha256=digest(final))
            write(POINTER, current)
    return point


def descendants(pid):
    result, pending = [], [pid]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.append(current)
        try:
            pending.extend(int(p) for p in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except FileNotFoundError:
            pass
    return result


def safe_stop(case, reasons):
    """Only signal newly re-read, exact birth/PGID/argv identities of this train.py."""
    request = read(case/"request.json")
    signals = []
    for sig, grace in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        current = read(POINTER)
        if current.get("launch_record") != request["launch_record"] or current.get("supervision_path") != request["supervision_path"]:
            evidence = {"at": now(), "reasons": reasons, "signals": signals,
                        "limitation": "Current pointer changed; no further signals are authorized from an old record",
                        "remaining": [process_check(i) for i in read(request["launch_record"])["processes"]],
                        "automatic_restart_allowed": False}
            immutable(case/f"stop_evidence_{uuid.uuid4().hex[:8]}.json", evidence)
            return evidence
        launch = read(request["launch_record"])
        identities = sorted(launch["processes"], key=lambda i: i.get("role") != "learner")
        live = [process_check(i) for i in identities]
        for item in live:
            identity = item["recorded_identity"]
            if sig == signal.SIGINT and identity.get("role") != "learner":
                continue
            # Check once more immediately before each signal, never kill a PGID
            # wholesale and never trust the supervision entry point's PID cache.
            fresh = process_check(identity)
            if not fresh["identity_matches"]:
                continue
            try:
                os.kill(identity["pid"], sig)
                signals.append({"at": now(), "signal": sig.name, "identity": identity,
                                "observed_before_signal": fresh["observed"]})
            except ProcessLookupError:
                pass
            except PermissionError as error:
                signals.append({"at": now(), "signal": sig.name, "identity": identity,
                                "signal_failed": str(error)})
        until = time.monotonic()+grace
        while time.monotonic() < until:
            if not any(process_check(i)["identity_matches"] for i in read(request["launch_record"])["processes"]):
                break
            time.sleep(.25)
        if not any(process_check(i)["identity_matches"] for i in read(request["launch_record"])["processes"]):
            break
    evidence = {"at": now(), "reasons": reasons, "signals": signals,
                "remaining": [process_check(i) for i in read(request["launch_record"])["processes"]],
                "automatic_restart_allowed": False}
    immutable(case/f"stop_evidence_{uuid.uuid4().hex[:8]}.json", evidence)
    return evidence


def manage(case, slot_fd):
    case = inside(case, ROOT/"logs/stair_learning")
    request = read(case/"request.json")
    require(request.get("job_kind") == KIND, "Invalid learning manager request")
    require(os.fstat(slot_fd).st_ino == SLOT.stat().st_ino, "Missing inherited independent learning slot lock")
    fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manager_birth = birth(os.getpid())
    write(case/"manager_state.json", {"status": "prelaunch_verification", "manager": manager_birth})
    process = None
    try:
        require(git("branch", "--show-current") == request["source_branch"]
                and git("rev-parse", "HEAD") == request["source_commit"], "Branch/commit changed after launch preparation")
        require(digest(request["initial_policy"]) == request["initial_policy_sha256"], "Frozen bootstrap weights changed")
        changed = [name for name, expected in request["prelaunch_source_hashes"].items()
                   if not (ROOT/name).exists() or digest(ROOT/name) != expected]
        require(not changed, f"Source/tools changed after preparation: {changed}")
        require(not active_project_jobs(exclude=[request["parent_launcher_pid"]]), "Another H2 job started before the manager")
        require(not gpu_snapshot()["compute_processes"], "GPU became occupied before training launch")
        environment = os.environ.copy()
        environment["CTRLLAB_H2_USD_PATH"] = str(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd")
        with Path(request["log"]).open("xb") as stream:
            process = subprocess.Popen(request["command"], cwd=ROOT, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            identities = []
            for _ in range(100):
                identities = []
                for pid in descendants(process.pid):
                    try:
                        identity = birth(pid)
                        if ENTRY in identity["argv"]:
                            is_python = Path(identity["argv"].split()[0]).name.startswith("python")
                            role = "wrapper" if pid == process.pid and not is_python else "learner"
                            identities.append({**identity, "role": role, "also_wrapper": pid == process.pid})
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                if any(i["role"] == "learner" for i in identities) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(request["launch_record"], {**request, "manager": manager_birth, "processes": identities})
            state_path = Path(request["supervision_path"])
            state = read(state_path)
            state.update(processes=identities, wrapper_started_at=now(), manager=manager_birth)
            write(state_path, state)
            pointer = read(POINTER)
            if pointer.get("supervision_path") == str(state_path):
                pointer.update(launch_record_sha256=digest(request["launch_record"]),
                               single_launch_guard_sha256=digest(request["single_launch_guard"]))
                write(POINTER, pointer)
            write(case/"manager_state.json", {"status": "running", "manager": manager_birth, "processes": identities})
            stopped = False
            while True:
                code = process.poll()
                point = snapshot(case, commit=True)
                if code is None and point["confirmed_stop_reasons"] and not stopped:
                    evidence = safe_stop(case, point["confirmed_stop_reasons"])
                    stopped = True
                    state = read(state_path)
                    state.update(status="failed", manager_stop_reason=point["confirmed_stop_reasons"],
                                 manager_stop_evidence=evidence, automatic_restart_allowed=False)
                    write(state_path, state)
                if code is not None:
                    break
                # Drain a large unread segment without skipping fatal evidence.
                wait = 0 if not point["log_caught_up"] else 60
                if wait:
                    try:
                        process.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        pass
            code = process.wait()
        state = read(request["supervision_path"])
        state.update(wrapper_exit_code=code, wrapper_ended_at=now())
        if code or state.get("status") not in ("completed", "stopped_no_improvement", "failed"):
            state.update(status="failed", wrapper_failure=True)
        write(Path(request["supervision_path"]), state)
        write(case/"manager_state.json", {"status": "finished", "exit_code": code, "ended_at": now(),
                                         "manager": manager_birth, "processes": identities})
        snapshot(case, commit=True)
    except BaseException as error:
        if process is not None and process.poll() is None and Path(request["launch_record"]).exists():
            safe_stop(case, ["manager_exception: "+repr(error)])
            process.wait(timeout=10)
        state = read(request["supervision_path"])
        state.update(status="failed", manager_error=repr(error), automatic_restart_allowed=False)
        write(Path(request["supervision_path"]), state)
        write(case/"manager_state.json", {"status": "finished", "exit_code": -1, "manager": manager_birth,
                                         "ended_at": now(), "error": repr(error)})
        raise
    finally:
        os.close(slot_fd)


def launch(args):
    initial = Path(args.initial_policy or args.bootstrap_output).resolve(strict=True)
    if initial.is_dir():
        if (initial/"output/initial_policy.pt").exists():
            initial = initial/"output/initial_policy.pt"
        else:
            initial = initial/"initial_policy.pt"
    initial = inside(initial)
    receipt = inside(args.bootstrap_result or initial.parent/"result.json")
    slot_fd = None
    try:
        if args.start:
            SLOT.parent.mkdir(parents=True, exist_ok=True)
            slot_fd = os.open(SLOT, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(slot_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = preflight(initial, receipt, args.iterations)
        if not args.start:
            print(json.dumps({"mode": "preflight_only", **checked}, ensure_ascii=False))
            return
        previous = read(POINTER)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:8]
        case = ROOT/"logs/stair_learning"/f"ppo_{stamp}_{args.label}_{token}"
        case.mkdir(exist_ok=False)
        source = case/"snapshot"
        for name in ("source", "tools"):
            shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        files = {str(path.relative_to(ROOT)): digest(path) for name in ("source", "tools")
                 for path in (ROOT/name).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
        usd = ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd"
        files[str(usd.relative_to(ROOT))] = digest(usd)
        write(case/"source_manifest.json", files)
        (case/"git_working_tree.patch").write_bytes(subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary"]))
        write(case/"preflight.json", checked)
        write(case/"previous_pointer.json", previous)
        frozen_init = case/"initialization"
        frozen_init.mkdir()
        shutil.copy2(initial, frozen_init/"initial_policy.pt")
        shutil.copy2(receipt, frozen_init/"result.json")
        state_path, policy_path = case/"supervision.json", case/"stop_policy.json"
        policy = stop_policy()
        write(policy_path, policy)
        command = ["bash", str(ROOT/"tools/train.sh"), "--task", TASK, "--headless", "--device", "cuda:0",
                   "--num_envs", "1024", "--max_iterations", str(args.iterations), "--save_interval", "500",
                   "--run_name", f"direct_stair_{args.label}_{token}", "--initial_policy", str(frozen_init/"initial_policy.pt"),
                   "--supervision_state", str(state_path), "--stop_policy", str(policy_path),
                   "--expected_git_commit", checked["source_commit"], "--seed", str(args.seed)]
        baseline = {**previous.get("accepted_flat_baseline", {}), **checked["accepted_flat_baseline"]}
        request = {"created_at": now(), "case": str(case), "job_kind": KIND, "command": command,
                   "entrypoint": ENTRY, "task": TASK, "supervision_path": str(state_path), "log": str(case/"train.log"),
                   "launch_record": str(case/"training.launch.json"), "single_launch_guard": str(case/"training.launch.lock"),
                   "monitor_record": str(case/"supervision.monitor.json"), "monitor_snapshot_tool": str(Path(__file__).resolve()),
                   "monitor_snapshot_command": [sys.executable, str(Path(__file__).resolve()), "--monitor", str(case)],
                   "stop_policy_path": str(policy_path), "stop_policy": policy, "parent_launcher_pid": os.getpid(),
                   "source_branch": checked["source_branch"], "source_commit": checked["source_commit"],
                   "source_snapshot": str(source/"source"), "source_manifest": str(case/"source_manifest.json"),
                   "prelaunch_source_hashes": files, "source_hashes": files,
                   "initial_policy": str(frozen_init/"initial_policy.pt"), "initial_policy_sha256": checked["bootstrap"]["initial_policy_sha256"],
                   "bootstrap_output_policy": str(initial), "bootstrap_receipt": str(receipt),
                   "bootstrap_receipt_sha256": checked["bootstrap"]["receipt_sha256"],
                   "accepted_flat_baseline": baseline, "checkpoint": None, "starting_iteration": 0,
                   "initialization_mode": "behavior_cloned_direct_policy", "max_iterations": args.iterations,
                   "save_interval": 500, "num_envs": 1024, "num_steps_per_env": 24,
                   "expected_total_timesteps": args.iterations*1024*24, "expected_final_iteration": args.iterations-1,
                   "network_structure_changed": False, "policy_dimensions": [1029, 866, 73, 12],
                   "manager_supervision_period_s": 60, "automatic_restart_allowed": False,
                   "authorization": previous.get("latest_stair_authorization", "继续完成楼梯任务；网络输入结构改变须先汇报。")}
        immutable(request["single_launch_guard"], request)
        immutable(case/"request.json", request)
        write(state_path, {**request, "status": "launching", "run_dir": None,
                           "automation_status": previous.get("automation_status", "UNKNOWN"),
                           "manager_supervision_enabled": True, "stairs_acceptance_passed": False})
        write(case/"supervision.monitor.json", {"iteration": -1, "checked_at_epoch": time.time(),
                                                "log_checked_offset": 0, "decision": "startup_pending"})
        write(case/"manager_state.json", {"status": "starting", "created_at": now()})
        pointer = dict(previous)
        pointer.update({key: request[key] for key in ("supervision_path", "launch_record", "single_launch_guard",
                       "monitor_record", "monitor_snapshot_tool", "monitor_snapshot_command", "stop_policy_path")})
        pointer.update(active_stair_job=str(case), active_stair_job_kind=KIND, active_stair_learning_case=str(case),
                       active_bootstrap=None, accepted_flat_baseline=baseline, runtime_state_status="stair_direct_policy_ppo_starting",
                       source_branch=checked["source_branch"], source_commit=checked["source_commit"],
                       status="launching", run_dir=None, final_checkpoint=None, checkpoint_sha256=None,
                       latest_checkpoint=None, completed_iterations=0, last_monitor_iteration=-1,
                       launch_record_sha256=None, single_launch_guard_sha256=digest(request["single_launch_guard"]),
                       max_iterations=args.iterations, expected_final_iteration=args.iterations-1,
                       expected_total_timesteps=args.iterations*1024*24, checkpoint=None, starting_iteration=0,
                       walking_acceptance_passed=False, stairs_acceptance_passed=False,
                       new_repair_experiments_authorized=True, manager_supervision_enabled=True,
                       previous_pointer_record=str(case/"previous_pointer.json"),
                       checkpoint_validation_tool=str(ROOT/"tools/validate_main_checkpoint.py"))
        write(POINTER, pointer)
        with (case/"manager.log").open("xb") as stream:
            manager = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--manage", str(case), str(slot_fd)],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(slot_fd,))
        manager_identity = birth(manager.pid)
        immutable(case/"manager_parent_receipt.json", {"manager": manager_identity, "created_at": now()})
        print(json.dumps({"mode": "started", "case": str(case), "manager": manager_identity,
                          "supervision_path": str(state_path), "max_iterations": args.iterations,
                          "save_interval": 500, "checkpoint": None, "initial_policy": request["initial_policy"],
                          "monitor_command": request["monitor_snapshot_command"],
                          "automation_status_preserved": previous.get("automation_status")}, ensure_ascii=False))
    finally:
        if slot_fd is not None:
            os.close(slot_fd)  # The running manager keeps its inherited lock.


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--inspect-bootstrap":
        print(json.dumps(inspect_bootstrap(sys.argv[2], sys.argv[3])))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--manage":
        manage(Path(sys.argv[2]), int(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--initial-policy", "--initial_policy", type=Path)
    source.add_argument("--bootstrap-output", type=Path, help="Bootstrap case or output directory")
    parser.add_argument("--bootstrap-result", type=Path)
    parser.add_argument("--start", action="store_true", help="Explicitly start the bounded PPO job after preflight")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--label", default="direct_stairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", nargs="?", const="current", type=str)
    parser.add_argument("--commit-monitor", action="store_true", help="Commit a read-only snapshot; never sends signals")
    args = parser.parse_args()
    if args.monitor is not None or len(sys.argv) == 1 or args.commit_monitor:
        if args.start or args.initial_policy or args.bootstrap_output:
            parser.error("Monitoring cannot be mixed with a launch request")
        point = snapshot(None if args.monitor in (None, "current") else args.monitor, args.commit_monitor)
        print(json.dumps(point, ensure_ascii=False))
        return
    if not args.initial_policy and not args.bootstrap_output:
        parser.error("Provide --initial_policy or --bootstrap-output; --start is required to launch")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.label):
        parser.error("label must be1..64 alphanumeric, underscore or hyphen characters")
    launch(args)


if __name__ == "__main__":
    main()
