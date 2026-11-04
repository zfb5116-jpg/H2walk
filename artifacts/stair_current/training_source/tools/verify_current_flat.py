"""Freeze current source and launch one1024-env formal flat-walking check.

Uses the retained validated checkpoint and the ORIGINAL20s/.3m/s/90% walking
contract. It never trains, changes a policy, or resumes a stair experiment.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    args = ap.parse_args()
    pointer_path = ROOT / "logs/launches/current_main_training.json"
    pointer = read(pointer_path)
    checkpoint = (args.checkpoint or Path(pointer["final_checkpoint"])).resolve(strict=True)
    validation = read(checkpoint.with_suffix(".validation.json"))
    assert validation["status"] == "passed" and digest(checkpoint) == validation["sha256"]
    for key in ("active_stair_trial", "active_flat_verification"):
        active = pointer.get(key)
        if active:
            assert read(Path(active)/"state.json")["status"] == "finished", f"Active run: {active}"
    assert not subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"
    ], text=True).strip(), "Another GPU application is active; no parallel simulator"
    network = Path("CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks")
    frozen = checkpoint.parent / "training_source/source"
    for name in ("actor_critic_our.py", "lidar_terrain_reconstructor.py", "terrain_reconstructor.py", "__init__.py"):
        assert digest(ROOT/"source"/network/name) == digest(frozen/network/name), "Network source changed"
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    case = ROOT / "logs/verifications" / f"current_flat_{stamp}"
    case.mkdir(parents=True, exist_ok=False)
    source = case / "source_snapshot/source"
    shutil.copytree(ROOT/"source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tool_snapshot = case / "tools_snapshot"
    tool_snapshot.mkdir()
    for name in ("evaluate_main_walk.py", "evaluate_main_walk.sh", "manage_reference_trial.py", "verify_current_flat.py"):
        shutil.copy2(ROOT/"tools"/name, tool_snapshot/name)
    manifest = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    write(case/"source_manifest.json", manifest)
    command = ["bash", str(ROOT/"tools/evaluate_main_walk.sh"), "--headless", "--device", "cuda:0",
        "--num_envs", "1024", "--checkpoint", str(checkpoint), "--source_root", str(source),
        "--terrain", "flat", "--duration", "20", "--speed", "0.3", "--seed", "42", "--video",
        "--output", str(case/"evaluation")]
    request = dict(case=str(case), label="current_source_flat_verification", command=command,
        checkpoint=str(checkpoint), checkpoint_sha256=validation["sha256"],
        controller_source_origin=str(ROOT/"source"), controller_source=str(source),
        checkpoint_training_source=str(frozen), controller_repaired_since_checkpoint_training=True,
        network_structure_changed=False, diagnostic_zero_learned_residual=False,
        diagnostic_only=False, formal_candidate=True, training_updates=0,
        output=str(case/"evaluation"), log=str(case/"evaluation.log"),
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        acceptance=dict(num_envs=1024, minimum_successful_envs=922, duration_s=20, speed_m_s=.3,
            distance_m=[4.8,7.2], minimum_valid_alternating_landings_each_foot=2,
            first_episode_only=True, all_conditions_jointly_required=True),
        source_manifest=str(case/"source_manifest.json"), source_manifest_sha256=digest(case/"source_manifest.json"),
        user_request="暂停楼梯实验；验证当前机器人平地行走；通过则新分支上传原GitHub仓库")
    with (case/"launch.lock").open("x") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
    write(case/"request.json", request)
    write(case/"state.json", dict(status="starting", request=str(case/"request.json")))
    with (case/"manager.log").open("xb") as stream:
        process = subprocess.Popen(["python3", str(ROOT/"tools/manage_reference_trial.py"), str(case)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()
    state = read(case/"state.json")
    state["manager"] = dict(pid=process.pid, start_ticks=int(fields[19]), pgid=os.getpgid(process.pid))
    write(case/"state.json", state)
    pointer.update(active_flat_verification=str(case), runtime_state_status="current_flat_verification_running",
        stair_experiments_paused_by_user=True, automation_status="PAUSED")
    write(pointer_path, pointer)
    print(json.dumps(dict(case=str(case), manager=state["manager"], source_files=len(manifest)), ensure_ascii=False))


if __name__ == "__main__":
    main()
