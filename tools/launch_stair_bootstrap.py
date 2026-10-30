"""Launch one bounded teacher-data job with immutable birth identities."""
import argparse
from datetime import datetime, timezone
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
POINTER = ROOT/"logs/launches/current_main_training.json"
EXPECTED_TEACHER_SHA = "2b06db2e09061cc660d46578b112ea29d0e601c3e75ac2163f8425aa24553fcb"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = stream.name
    Path(temporary).replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def birth(pid):
    proc = Path("/proc")/str(pid)
    fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=os.getpgid(pid),
                argv=(proc/"cmdline").read_bytes().replace(b"\0", b" ").decode(),
                process_state=fields[0])


def manage(case):
    request = json.loads((case/"request.json").read_text())
    state_path = Path(request["supervision_path"])
    environment = os.environ.copy()
    environment["CTRLLAB_H2_USD_PATH"] = str(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd")
    with Path(request["log"]).open("xb") as stream:
        process = subprocess.Popen(request["command"], cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        identities = []
        for _ in range(100):
            pids = [process.pid]
            try:
                pids += [int(p) for p in Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()]
            except FileNotFoundError:
                pass
            identities = []
            for pid in pids:
                try:
                    identity = birth(pid)
                    if request["entrypoint"] in identity["argv"]:
                        identities.append(dict(identity,
                            role="wrapper" if pid == process.pid else "bootstrap"))
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if any(p["role"] == "bootstrap" for p in identities) or process.poll() is not None:
                break
            time.sleep(.2)
        with (case/"launch.json").open("x") as stream:
            json.dump(dict(request, processes=identities, manager=birth(os.getpid())), stream, indent=2)
        state = json.loads(state_path.read_text())
        state.update(processes=identities, wrapper_started_at=now())
        write(state_path, state)
        write(case/"manager_state.json", dict(status="running", processes=identities))
        code = process.wait()
    state = json.loads(state_path.read_text())
    state.update(wrapper_exit_code=code, wrapper_ended_at=now())
    if code or state.get("status") not in ("completed", "failed"):
        state.update(status="failed", wrapper_failure=True)
    write(state_path, state)
    write(case/"manager_state.json", dict(status="finished", exit_code=code,
                                         ended_at=now(), processes=identities))


def launch(steps):
    previous = json.loads(POINTER.read_text())
    assert not previous.get("stair_experiments_paused_by_user")
    if previous.get("active_stair_job"):
        prior = json.loads(Path(previous["supervision_path"]).read_text())
        assert prior["status"] in ("completed", "failed", "stopped_no_improvement"), prior["status"]
        assert json.loads((Path(previous["supervision_path"]).parent/"manager_state.json").read_text())["status"] == "finished"
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                   "--format=csv,noheader"], text=True).strip()
    assert not gpu, f"GPU occupied: {gpu}"
    teacher = Path(previous["final_checkpoint"]).resolve(strict=True)
    assert sha(teacher) == EXPECTED_TEACHER_SHA
    assert 50 <= steps <= 5000 and steps % 50 == 0
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    assert branch == "codex/h2-stair-ascent-20261030", branch
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case = ROOT/"logs/stair_learning"/f"bootstrap_{stamp}"
    case.mkdir(parents=True, exist_ok=False)
    frozen = case/"snapshot"
    shutil.copytree(ROOT/"source", frozen/"source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protected_tools = ("bootstrap_direct_policy.py", "bootstrap_direct_policy.sh", "h2_bootstrap_contract.py")
    (frozen/"tools").mkdir()
    for name in protected_tools:
        shutil.copy2(ROOT/"tools"/name, frozen/"tools"/name)
    files = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/"source").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    files.update({"tools/"+name: sha(ROOT/"tools"/name) for name in protected_tools})
    output, state_path = case/"output", case/"supervision.json"
    command = ["bash", str(ROOT/"tools/bootstrap_direct_policy.sh"), "--headless", "--device", "cuda:0",
        "--num_envs", "1024", "--steps", str(steps), "--teacher_checkpoint", str(teacher),
        "--source_root", str(frozen/"source"), "--output", str(output), "--state", str(state_path)]
    request = dict(created_at=now(), case=str(case), job_kind="stair_policy_bootstrap", command=command,
        entrypoint=str(ROOT/"tools/bootstrap_direct_policy.py"), supervision_path=str(state_path),
        log=str(case/"bootstrap.log"), launch_record=str(case/"launch.json"),
        single_launch_guard=str(case/"launch.lock"), monitor_record=str(case/"supervision.monitor.json"),
        monitor_snapshot_tool=str(ROOT/"tools/monitor_stair_bootstrap.py"),
        stop_policy_path=previous["stop_policy_path"], source_hashes=files,
        teacher_checkpoint=str(teacher), teacher_checkpoint_sha256=sha(teacher), output=str(output),
        source_root=str(frozen/"source"), num_envs=1024, max_steps=steps, network_structure_changed=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        authorization="继续完成楼梯；奖励可调整，网络与输入布局改变须先汇报。")
    with (case/"launch.lock").open("x") as stream:
        json.dump(request, stream, indent=2, ensure_ascii=False)
    write(case/"request.json", request)
    write(case/"previous_pointer.json", previous)
    write(state_path, dict(request, status="launching", steps=0, formal_stair_acceptance=False))
    write(case/"supervision.monitor.json", dict(checked_at=now(), decision="startup_pending", log_checked_offset=0))
    preserved = {k: previous.get(k) for k in ("final_checkpoint", "checkpoint_sha256", "run_dir",
        "walking_acceptance_passed", "successful_envs", "last_flat_verification", "github_publication_url")}
    pointer = dict(previous)
    pointer.update({key: request[key] for key in ("supervision_path", "launch_record", "single_launch_guard",
        "monitor_record", "monitor_snapshot_tool", "stop_policy_path")})
    pointer.update(active_stair_job=str(case), active_stair_job_kind="stair_policy_bootstrap",
        accepted_flat_baseline=preserved, active_bootstrap=str(case),
        runtime_state_status="stair_bootstrap_starting", stairs_acceptance_passed=False,
        new_repair_experiments_authorized=True, previous_pointer_record=str(case/"previous_pointer.json"))
    write(POINTER, pointer)
    campaign_path = Path(previous["active_stair_campaign"])
    campaign = json.loads(campaign_path.read_text())
    campaign.update(status="stair_bootstrap_starting", active_job=str(case), updated_at=now())
    campaign.setdefault("policy_learning_jobs", []).append(dict(case=str(case), kind="bootstrap", status="starting"))
    write(campaign_path, campaign)
    with (case/"manager.log").open("xb") as stream:
        manager = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--manage", str(case)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(case/"manager_state.json", dict(status="starting", manager=birth(manager.pid)))
    print(json.dumps(dict(case=str(case), manager=birth(manager.pid), steps=steps)))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--manage":
        manage(Path(sys.argv[2]).resolve(strict=True))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--steps", type=int, default=2000)
        launch(parser.parse_args().steps)
