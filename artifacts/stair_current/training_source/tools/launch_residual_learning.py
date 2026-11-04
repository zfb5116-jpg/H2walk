"""Launch one bounded fresh H2 residual-learning run with durable supervision."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    Path(temporary).replace(path)


def birth(pid):
    proc = Path("/proc") / str(pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(fields[19]), "pgid": os.getpgid(pid),
            "argv": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()}


def manage(case):
    request = json.loads((case / "request.json").read_text())
    environment = os.environ.copy()
    environment["CTRLLAB_H2_USD_PATH"] = str(ROOT / "assets/isaaclab/unitree_robots/h2/h2.usd")
    state_path = Path(request["supervision_path"])
    with Path(request["log"]).open("xb") as log:
        process = subprocess.Popen(request["command"], cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        identities = []
        for _ in range(60):
            pids = [process.pid]
            try:
                pids += [int(p) for p in Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()]
            except FileNotFoundError:
                pass
            found = []
            for pid in pids:
                try:
                    identity = birth(pid)
                    if str(ROOT / "tools/train.py") in identity["argv"]:
                        found.append({**identity, "role": "wrapper" if pid == process.pid else "learner"})
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if any(item["role"] == "learner" for item in found):
                identities = found
                break
            if process.poll() is not None:
                identities = found
                break
            time.sleep(.2)
        with (case / "launch.json").open("x") as stream:
            json.dump({**request, "manager": birth(os.getpid()), "processes": identities}, stream,
                      ensure_ascii=False, indent=2)
        state = json.loads(state_path.read_text())
        state.update(processes=identities, wrapper_started_at=now())
        write(state_path, state)
        write(case / "manager_state.json", {"status": "running", "processes": identities})
        code = process.wait()
    state = json.loads(state_path.read_text())
    state.update(wrapper_exit_code=code, wrapper_ended_at=now())
    if code and state.get("status") not in ("failed", "stopped_no_improvement"):
        state.update(status="failed", wrapper_failure=True)
    write(state_path, state)
    write(case / "manager_state.json", {"status": "finished", "exit_code": code,
                                         "ended_at": now(), "processes": identities})


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--manage":
        manage(Path(sys.argv[2]).resolve(strict=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()
    assert 1 <= args.iterations <= 8000
    pointer_path = ROOT / "logs/launches/current_main_training.json"
    previous = json.loads(pointer_path.read_text())
    campaign_path = Path(previous["active_repair_campaign"])
    campaign = json.loads(campaign_path.read_text())
    assert campaign["acceptance"]["minimum_successful_envs"] == 922
    if campaign.get("active_training"):
        active = json.loads(Path(campaign["active_training"]["supervision_path"]).read_text())
        assert active["status"] in ("completed", "failed", "stopped_no_improvement"), active["status"]
    if campaign.get("active_experiment"):
        last_case = Path(campaign["active_experiment"])
        assert json.loads((last_case / "state.json").read_text())["status"] == "finished"
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                   "--format=csv,noheader"], text=True).strip()
    assert not gpu, f"GPU still occupied: {gpu}"
    network_changes = [p for p, h in campaign["initial_network_hashes"].items()
                       if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    assert not network_changes, network_changes
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "main"
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    case = ROOT / "logs/launches" / f"h2_residual_{args.label}_{stamp}"
    case.mkdir(exist_ok=False)
    stop_policy = json.loads(Path(previous["stop_policy_path"]).read_text())
    policy_path, state_path = case / "stop_policy.json", case / "supervision.json"
    monitor_path = case / "supervision.monitor.json"
    write(policy_path, stop_policy)
    write(case / "previous_training_pointer.json", previous)
    shutil.copy2(previous["monitor_snapshot_tool"], case / "monitor_snapshot.py")
    command = ["bash", str(ROOT / "tools/train.sh"), "--task", "Unitree-H2-Velocity-Stairs",
               "--headless", "--device", "cuda:0", "--num_envs", "1024",
               "--max_iterations", str(args.iterations), "--save_interval", "500",
               "--run_name", args.label + "_fresh_save500", "--supervision_state", str(state_path),
               "--stop_policy", str(policy_path), "--expected_git_commit", campaign["source_commit"]]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "source").rglob("*") if p.is_file() and p.suffix != ".pyc"
                     and "__pycache__" not in p.parts}
    request = {"created_at": now(), "case": str(case), "campaign": str(campaign_path),
               "command": command, "supervision_path": str(state_path), "log": str(case / "train.log"),
               "launch_record": str(case / "launch.json"), "single_launch_guard": str(case / "launch.lock"),
               "monitor_record": str(monitor_path), "monitor_snapshot_tool": str(case / "monitor_snapshot.py"),
               "stop_policy_path": str(policy_path), "stop_policy": stop_policy,
               "hypothesis": args.hypothesis, "checkpoint": None, "starting_iteration": 0,
               "max_iterations": args.iterations, "save_interval": 500, "num_envs": 1024,
               "task": "Unitree-H2-Velocity-Stairs", "network_structure_changed": False,
               "prelaunch_source_hashes": source_hashes,
               "acceptance": campaign["acceptance"], "authorization": campaign["authorization"]}
    with (case / "launch.lock").open("x") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
    write(case / "request.json", request)
    write(state_path, {**request, "status": "launching", "run_dir": None, "automation_status": "ACTIVE"})
    write(monitor_path, {"iteration": -1, "checked_at_epoch": time.time(), "log_checked_offset": 0,
                          "automation_status": "ACTIVE", "decision": "startup_pending"})
    pointer = {key: request[key] for key in ("supervision_path", "launch_record", "single_launch_guard",
                                            "monitor_record", "monitor_snapshot_tool", "stop_policy_path")}
    pointer.update(active_repair_campaign=str(campaign_path), active_controller_experiment=None,
                   automation_status="ACTIVE", runtime_state_status="fresh_residual_learning_starting",
                   run_dir=None, source_commit=campaign["source_commit"], max_iterations=args.iterations,
                   expected_final_iteration=args.iterations - 1,
                   expected_total_timesteps=args.iterations * 1024 * 24,
                   previous_pointer_record=str(case / "previous_training_pointer.json"),
                   checkpoint_validation_tool=str(ROOT / "tools/validate_main_checkpoint.py"),
                   latest_authorization=campaign["authorization"], new_repair_experiments_authorized=True,
                   campaign_monitor_record=campaign["monitor_record"], walking_acceptance_passed=False)
    write(pointer_path, pointer)
    campaign.update(active_training={"case": str(case), "supervision_path": str(state_path),
                                     "launch_record": str(case / "launch.json")},
                    latest_controller_experiment=campaign.get("active_experiment"), active_experiment=None,
                    status="fresh_residual_learning", last_updated=now())
    campaign.setdefault("learning_runs", []).append({"case": str(case), "status": "starting",
                                                    "max_iterations": args.iterations, "hypothesis": args.hypothesis})
    write(campaign_path, campaign)
    with (case / "manager.log").open("xb") as log:
        manager = subprocess.Popen(["python3", str(Path(__file__).resolve()), "--manage", str(case)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(case / "manager_state.json", {"status": "starting", "manager": birth(manager.pid)})
    print(json.dumps({"case": str(case), "supervision_path": str(state_path), "manager": birth(manager.pid),
                      "max_iterations": args.iterations, "checkpoint": None, "save_interval": 500}, ensure_ascii=False))


if __name__ == "__main__":
    main()
