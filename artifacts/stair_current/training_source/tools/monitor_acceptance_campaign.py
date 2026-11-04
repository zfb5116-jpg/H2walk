"""Inspect and durably record the active H2 repair case without sending signals."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = stream.name
    Path(temporary).replace(path)


def main():
    pointer_path = ROOT / "logs/launches/current_main_training.json"
    pointer = json.loads(pointer_path.read_text())
    campaign_path = Path(pointer["active_repair_campaign"])
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("active_training"):
        state = json.loads(Path(pointer["supervision_path"]).read_text())
        print(json.dumps({"mode": "training", "state": state.get("status"),
                          "run_dir": state.get("run_dir"), "supervision_path": pointer["supervision_path"],
                          "monitor_snapshot_tool": pointer["monitor_snapshot_tool"],
                          "instruction": "Use the training snapshot tool; review pending evidence before committing."}))
        return
    case = Path(campaign["active_experiment"])
    request = json.loads((case / "request.json").read_text())
    state = json.loads((case / "state.json").read_text())
    launch_path = case / "launch.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {"processes": []}
    process_checks = []
    for identity in launch["processes"]:
        entry = {"recorded_identity": identity}
        try:
            proc = Path("/proc") / str(identity["pid"])
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            entry.update(visible=True, state=fields[0], start_ticks=int(fields[19]),
                         pgid=os.getpgid(identity["pid"]), argv=argv,
                         identity_matches=(int(fields[19]) == identity["start_ticks"]
                                           and os.getpgid(identity["pid"]) == identity["pgid"]
                                           and argv == identity["argv"]))
        except (ProcessLookupError, FileNotFoundError):
            entry.update(visible=False)
        process_checks.append(entry)
    now = datetime.now(timezone.utc)
    log_path = Path(request["log"])
    text = log_path.read_bytes()[-1048576:].decode(errors="replace") if log_path.exists() else ""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    progress = re.findall(r"\[EVAL\] t=([^\n]+)", text)
    result_path = case / "evaluation/result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    fatal_lines = [line for line in text.splitlines() if re.search(
        r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA out of memory|"
        r"illegal memory access|device-side assert|\b(?:nan|inf)\b", line, re.I)]
    monitor = {"checked_at": now.isoformat(), "campaign": str(campaign_path), "case": str(case),
               "state": state, "processes": process_checks, "log": str(log_path),
               "log_bytes": log_path.stat().st_size if log_path.exists() else None,
               "log_age_s": now.timestamp() - log_path.stat().st_mtime if log_path.exists() else None,
               "last_eval_progress": progress[-1] if progress else None,
               "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                 "--format=csv,noheader"], text=True).strip(),
               "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                                           "--format=csv,noheader"], text=True).strip(),
               "fatal_log_candidates": fatal_lines[-12:],
               "network_source_changes": [p for p, h in campaign["initial_network_hashes"].items()
                                           if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h],
               "diagnostic_only": request["diagnostic_zero_learned_residual"],
               "acceptance_passed": False}
    if result is not None:
        monitor["result"] = str(result_path)
        monitor["motion"] = result.get("results")
        monitor["controller"] = result.get("reference_controller_summary")
    ended = (state["status"] == "finished" and not monitor["gpu_processes"]
             and not any(p.get("identity_matches") and p.get("state") != "Z" for p in process_checks))
    if ended:
        valid_result = state.get("exit_code") == 0 and result is not None and result.get("status") == "completed"
        count = result.get("results", {}).get("bilateral_touchdown_screen_count", 0) if result else 0
        formal = bool(request.get("formal_candidate")) and not bool(result.get("diagnostic_only", True)) if result else False
        monitor["conclusion"] = (("completed formal frozen replay; review acceptance evidence"
                                   if formal else "completed diagnostic; walking not yet formally accepted")
                                  if valid_result else "experiment failed; inspect evidence before another launch")
        for entry in campaign["experiments"]:
            if entry["case"] == str(case):
                entry.update(status=("completed_formal_replay" if formal else "completed_diagnostic")
                             if valid_result else "failed",
                             result=str(result_path), successful_envs=count,
                             survived20s_envs=result.get("results", {}).get("survived_20s_count", 0) if result else 0)
        campaign["status"] = "analysis_required"
    else:
        monitor["conclusion"] = "current case still active or completion requires cross-check"
    write(Path(campaign["monitor_record"]), monitor)
    with Path(campaign["monitor_history"]).open("a") as stream:
        stream.write(json.dumps(monitor, ensure_ascii=False) + "\n")
    campaign["last_monitor_check_at"] = monitor["checked_at"]
    write(campaign_path, campaign)
    pointer.update(last_campaign_check_at=monitor["checked_at"],
                   runtime_state_status="repair_analysis_required" if ended else "authorized_controller_experiment_running")
    write(pointer_path, pointer)
    print(json.dumps({key: monitor[key] for key in ("checked_at", "case", "last_eval_progress", "gpu", "conclusion")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
