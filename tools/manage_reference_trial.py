"""Record the identity and exit of one reference-controller experiment."""

from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    Path(temporary).replace(path)


case = Path(sys.argv[1]).resolve(strict=True)
request = json.loads((case / "request.json").read_text())
env = os.environ.copy()
env["CTRLLAB_H2_USD_PATH"] = "/home/nsgm/python_src/H2walk/assets/isaaclab/unitree_robots/h2/h2.usd"
with Path(request["log"]).open("xb") as stream:
    process = subprocess.Popen(request["command"], cwd="/home/nsgm/python_src/H2walk", env=env,
                               stdout=stream, stderr=subprocess.STDOUT)
    identities = []
    for _ in range(60):
        ids = [process.pid]
        try:
            ids += [int(p) for p in Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()]
        except FileNotFoundError:
            pass
        found = []
        for pid in ids:
            try:
                proc = Path("/proc") / str(pid)
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                if "/home/nsgm/python_src/H2walk/tools/evaluate_main_walk.py" in argv:
                    found.append({"pid": pid, "start_ticks": int(fields[19]), "pgid": os.getpgid(pid),
                                  "role": "wrapper" if pid == process.pid else "evaluator", "argv": argv})
            except (FileNotFoundError, ProcessLookupError):
                pass
        if any(item["role"] == "evaluator" for item in found):
            identities = found
            break
        time.sleep(.2)
    with (case / "launch.json").open("x") as file:
        json.dump({**request, "processes": identities}, file, indent=2)
    state = json.loads((case / "state.json").read_text())
    state.update(status="running", launch=str(case / "launch.json"), started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
    write(case / "state.json", state)
    code = process.wait()
    state = json.loads((case / "state.json").read_text())
    state.update(status="finished", exit_code=code, finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                 result=str(Path(request["output"]) / "result.json"))
    write(case / "state.json", state)
