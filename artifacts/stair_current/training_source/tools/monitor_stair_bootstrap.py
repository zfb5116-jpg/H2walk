"""Read the active bounded bootstrap job and create an uncommitted snapshot."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    current = json.loads((ROOT/"logs/launches/current_main_training.json").read_text())
    state_path = Path(current["supervision_path"])
    state = json.loads(state_path.read_text())
    assert state["job_kind"] == "stair_policy_bootstrap"
    case = state_path.parent
    launch_path = Path(current["launch_record"])
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {"processes": []}
    monitor_path = Path(current["monitor_record"])
    previous = json.loads(monitor_path.read_text())
    log = Path(state["log"])
    size = log.stat().st_size if log.exists() else 0
    offset = min(previous.get("log_checked_offset", 0), size)
    with log.open("rb") as stream:
        stream.seek(offset)
        delta = stream.read(2*1024*1024)
        next_offset = stream.tell()
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", delta.decode(errors="replace"))
    processes = []
    for identity in launch["processes"]:
        proc = Path("/proc")/str(identity["pid"])
        observed = None
        try:
            fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
            observed = dict(start_ticks=int(fields[19]), pgid=os.getpgid(identity["pid"]), state=fields[0],
                            argv=(proc/"cmdline").read_bytes().replace(b"\0", b" ").decode())
        except (FileNotFoundError, ProcessLookupError):
            pass
        matches = bool(observed and observed["start_ticks"] == identity["start_ticks"]
            and observed["pgid"] == identity["pgid"] and launch["entrypoint"] in observed["argv"]
            and observed["state"] not in ("Z", "X"))
        processes.append(dict(identity=identity, observed=observed, matches=matches))
    # Known startup Warp UUID error36 is excluded. Failed initialization retains
    # its actual exception; no process is stopped by this read-only collector.
    known = [line for line in cleaned.splitlines() if "Warp CUDA error:" in line and "cuDeviceGetUuid" in line]
    anomalies = [line for line in cleaned.splitlines() if line not in known and re.search(
        r"Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|CUDA error:|device-side assert|illegal memory access", line)]
    changed = [name for name, expected in state["source_hashes"].items()
               if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected]
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader"], text=True).strip()
    manager = json.loads((case/"manager_state.json").read_text())
    point = dict(checked_at=datetime.now(timezone.utc).isoformat(), checked_at_epoch=time.time(),
        job_kind=state["job_kind"], state_status=state["status"], steps=state.get("steps", 0),
        max_steps=state["max_steps"], log_checked_offset=next_offset, log_size=size,
        log_mtime=log.stat().st_mtime, log_age_s=time.time()-log.stat().st_mtime,
        manager=manager, processes=processes, gpu=gpu, new_anomaly_lines=anomalies, known_startup_warnings=known,
        source_or_tool_changed=changed, initialization_ready=state.get("initialization_ready"),
        result=state.get("initialization_quality"), final_log_excerpt=cleaned[-4500:],
        previous_steps=previous.get("steps"), previous_checked_at_epoch=previous.get("checked_at_epoch"),
        decision="pending_review")
    (case/"supervision.monitor.pending.json").write_text(json.dumps(point, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(point, ensure_ascii=False))


if __name__ == "__main__":
    snapshot()
