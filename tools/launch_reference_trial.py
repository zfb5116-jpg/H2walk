"""Launch exactly one isolated 1024-env walking-controller experiment."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    Path(temporary).replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zero_residual", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    cp = args.checkpoint.resolve(strict=True)
    campaign_path = Path(json.loads((ROOT / "logs/launches/current_acceptance_campaign.json").read_text())["campaign_path"])
    campaign = json.loads(campaign_path.read_text())
    training = campaign.get("active_training")
    if training:
        training_state = json.loads(Path(training["supervision_path"]).read_text())
        if training_state["status"] not in ("completed", "failed", "stopped_no_improvement"):
            raise RuntimeError("Learning is active; do not start another simulator")
        manager_state = Path(training["case"]) / "manager_state.json"
        if not manager_state.exists() or json.loads(manager_state.read_text())["status"] != "finished":
            raise RuntimeError("Training wrapper has not confirmed exit")
    active = campaign.get("active_experiment")
    if active:
        state_path = Path(active) / "state.json"
        state = json.loads(state_path.read_text())
        if state["status"] in ("starting", "running"):
            raise RuntimeError("A recorded experiment has not ended; inspect it, never launch a duplicate")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Another GPU compute application is present; no parallel simulator allowed")
    if training:
        campaign["latest_training"] = campaign.pop("active_training")
    changed = [p for p, h in campaign["initial_network_hashes"].items()
               if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if changed and not campaign.get("network_change_reported", False):
        raise RuntimeError("Network structure/source changed without a recorded prior report")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    case = ROOT / "logs/experiments" / f"h2_walk_{args.label}_{stamp}"
    case.mkdir(parents=True, exist_ok=False)
    training_source = cp.parent / "training_source/source"
    source = ROOT / "source" if args.zero_residual else training_source
    if not source.is_dir():
        raise RuntimeError("Formal playback requires the checkpoint's complete preserved training source")
    if not args.zero_residual:
        validation = cp.with_suffix(".validation.json")
        if not validation.exists() or json.loads(validation.read_text()).get("status") != "passed":
            raise RuntimeError("Strict CPU checkpoint validation must pass before formal playback")
        if json.loads(validation.read_text()).get("sha256") != hashlib.sha256(cp.read_bytes()).hexdigest():
            raise RuntimeError("Checkpoint no longer matches its CPU validation receipt")
    shutil.copytree(source, case / "source_snapshot/source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (case / "tools_snapshot").mkdir()
    for name in ("evaluate_main_walk.py", "evaluate_main_walk.sh", "launch_reference_trial.py", "manage_reference_trial.py"):
        shutil.copy2(ROOT / "tools" / name, case / "tools_snapshot" / name)
    output, log = case / "evaluation", case / "evaluation.log"
    command = ["bash", str(ROOT / "tools/evaluate_main_walk.sh"), "--headless", "--device", "cuda:0",
               "--num_envs", "1024", "--checkpoint", str(cp), "--source_root", str(case / "source_snapshot/source"),
               "--terrain", "flat", "--duration", "20", "--speed", "0.3", "--output", str(output)]
    if args.zero_residual:
        command.append("--hold_default_pose")
    if args.video:
        command.append("--video")
    request = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "case": str(case),
               "label": args.label, "campaign": str(campaign_path), "command": command,
               "output": str(output), "log": str(log), "checkpoint": str(cp),
               "checkpoint_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
               "checkpoint_origin_run": str(cp.parent), "controller_source": str(case / "source_snapshot/source"),
               "controller_repaired_since_checkpoint_training": args.zero_residual,
               "controller_source_origin": str(source), "checkpoint_training_source": str(training_source),
               "diagnostic_zero_learned_residual": args.zero_residual,
               "formal_candidate": not args.zero_residual, "acceptance": campaign["acceptance"],
               "training_updates": 0, "network_source_changes": changed,
               "tools_sha256": {n: hashlib.sha256((ROOT / "tools" / n).read_bytes()).hexdigest()
                                  for n in ("evaluate_main_walk.py", "evaluate_main_walk.sh")}}
    with (case / "launch.lock").open("x") as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
    write(case / "request.json", request)
    write(case / "state.json", {"status": "starting", "request": str(case / "request.json")})
    with (case / "manager.log").open("xb") as stream:
        process = subprocess.Popen(["python3", str(ROOT / "tools/manage_reference_trial.py"), str(case)],
                                   stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
    state = {"status": "starting", "manager": {"pid": process.pid, "start_ticks": int(fields[19]),
                                                "pgid": os.getpgid(process.pid)}, "request": str(case / "request.json")}
    write(case / "state.json", state)
    campaign["experiments"].append({"case": str(case), "label": args.label, "status": "running",
                                     "zero_residual": args.zero_residual, "formal_candidate": not args.zero_residual})
    campaign.update(active_experiment=str(case), status="experiment_running")
    write(campaign_path, campaign)
    pointer_path = ROOT / "logs/launches/current_main_training.json"
    pointer = json.loads(pointer_path.read_text())
    pointer.update(active_repair_campaign=str(campaign_path), active_controller_experiment=str(case),
                   runtime_state_status="authorized_controller_experiment_running")
    write(pointer_path, pointer)
    print(json.dumps({"case": str(case), "manager": state["manager"], "zero_residual": args.zero_residual,
                      "formal_candidate": not args.zero_residual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
