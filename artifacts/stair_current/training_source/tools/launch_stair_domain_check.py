"""Run exactly one bounded synthetic domain diagnostic after GPU release."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
POINTER=ROOT/"logs/launches/current_main_training.json"
ENTRY=str(ROOT/"tools/validate_stair_domain.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+".domain.tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def birth(pid):
    proc=Path("/proc")/str(pid)
    fields=(proc/"stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=os.getpgid(pid),state=fields[0],
                argv=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode())


def matches(identity):
    try:
        current=birth(identity["pid"])
        return (current["start_ticks"]==identity["start_ticks"] and current["pgid"]==identity["pgid"]
                and current["argv"]==identity["argv"] and ENTRY in current["argv"]
                and current["state"] not in ("Z","X"))
    except (FileNotFoundError,ProcessLookupError):
        return False


def immutable(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def manage(case,slot):
    request=json.loads((case/"request.json").read_text())
    identities=[]
    process=None
    try:
        for name,expected in request["protected_files"].items():
            if digest(name)!=expected:
                raise RuntimeError("Diagnostic input changed after preparation: "+name)
        gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory",
                                     "--format=csv,noheader"],text=True).strip()
        if gpu:
            raise RuntimeError("GPU became occupied before diagnostic startup: "+gpu)
        with (case/"diagnostic.log").open("xb") as log:
            process=subprocess.Popen(request["command"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            for _ in range(75):
                ids=[process.pid]
                try:
                    ids += [int(i) for i in Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()]
                except FileNotFoundError:
                    pass
                identities=[]
                for pid in ids:
                    try:
                        b=birth(pid)
                        if ENTRY in b["argv"]:
                            identities.append(dict(b,role="wrapper" if pid==process.pid else "evaluator"))
                    except (FileNotFoundError,ProcessLookupError):
                        pass
                if any(i["role"]=="evaluator" for i in identities) or process.poll() is not None:
                    break
                time.sleep(.2)
            immutable(case/"launch.json",dict(request,processes=identities,manager=birth(os.getpid())))
            write(case/"manager_state.json",dict(status="running",processes=identities))
            state=json.loads((case/"supervision.json").read_text())
            state.update(status="diagnosing",processes=identities,started_at=now())
            write(case/"supervision.json",state)
            try:
                code=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                for sig,delay in ((signal.SIGINT,10),(signal.SIGTERM,5)):
                    current=json.loads(POINTER.read_text())
                    if current["launch_record"]!=str(case/"launch.json"):
                        raise RuntimeError("Pointer changed; cannot safely stop the old diagnostic")
                    for identity in json.loads((case/"launch.json").read_text())["processes"]:
                        if sig==signal.SIGINT and identity["role"]!="evaluator":
                            continue
                        if matches(identity):
                            os.kill(identity["pid"],sig)
                    try:
                        code=process.wait(timeout=delay)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                else:
                    raise RuntimeError("Matching diagnostic did not exit after bounded INT/TERM")
                raise RuntimeError("Diagnostic exceeded its300s bound")
        result_path=Path(request["output"])/"result.json"
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        for name,expected in request["protected_files"].items():
            if digest(name)!=expected:
                raise RuntimeError("Diagnostic input changed during execution: "+name)
        passed=code==0 and result.get("status")=="passed" and result.get("diagnostic_only") is True
        state=json.loads((case/"supervision.json").read_text())
        state.update(status="completed" if passed else "failed",wrapper_exit_code=code,ended_at=now(),
                     result=str(result_path),diagnostic_passed=passed,formal_acceptance_passed=False)
        write(case/"supervision.json",state)
        write(case/"manager_state.json",dict(status="finished",exit_code=code,ended_at=now(),processes=identities))
    except BaseException as error:
        # Ordinary startup/runtime failures have already exited. A process
        # that remains alive is reported explicitly rather than broadly killed.
        state=json.loads((case/"supervision.json").read_text())
        state.update(status="failed",error=repr(error),ended_at=now(),formal_acceptance_passed=False)
        write(case/"supervision.json",state)
        write(case/"manager_state.json",dict(status="failed_attention" if process and process.poll() is None else "finished",
            error=repr(error),processes=identities,ended_at=now()))
        raise
    finally:
        os.close(slot)


def launch():
    previous=json.loads(POINTER.read_text())
    if json.loads(Path(previous["supervision_path"]).read_text())["status"] not in ("completed","failed","accepted"):
        raise RuntimeError("Current job has not finished")
    if json.loads((Path(previous["active_stair_job"])/"manager_state.json").read_text())["status"]!="finished":
        raise RuntimeError("Current manager has not confirmed finish")
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory",
                                 "--format=csv,noheader"],text=True).strip()
    if gpu:
        raise RuntimeError("No parallel Isaac diagnostic allowed: "+gpu)
    from launch_stair_evaluation import active_simulators
    jobs=active_simulators()
    if jobs:
        raise RuntimeError("Another Isaac/job process is active: "+repr(jobs))
    checkpoint=Path(previous["final_checkpoint"]).resolve(strict=True)
    validation=checkpoint.with_suffix(".validation.json")
    receipt=json.loads(validation.read_text())
    if receipt["status"]!="passed" or receipt["sha256"]!=digest(checkpoint):
        raise RuntimeError("Diagnostic checkpoint has not passed strict CPU validation")
    slot=os.open(ROOT/"logs/stair_learning/domain_check.slot",os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    case=ROOT/"logs/stair_learning"/f"domain_{stamp}_{uuid.uuid4().hex[:8]}"
    case.mkdir(exist_ok=False)
    shutil.copytree(ROOT/"source",case/"source",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    tools=[ROOT/"tools"/name for name in ("validate_stair_domain.py","validate_stair_domain.sh","launch_stair_domain_check.py")]
    protected={str(p):digest(p) for p in (ROOT/"source").rglob("*") if p.is_file()
               and p.suffix!=".pyc" and "__pycache__" not in p.parts}
    protected.update({str(p):digest(p) for p in [*tools,checkpoint,validation]})
    dependencies=json.loads((checkpoint.parent/"run.json").read_text())["external_dependency_sha256"]
    for name,expected in dependencies.items():
        if digest(name)!=expected:
            raise RuntimeError("Original robot/reconstructor dependency changed: "+name)
    protected.update(dependencies)
    request=dict(created_at=now(),case=str(case),job_kind="stair_domain_synthetic_diagnostic",entrypoint=ENTRY,
        output=str(case/"evidence"),log=str(case/"diagnostic.log"),protected_files=protected,
        checkpoint=str(checkpoint),checkpoint_sha256=receipt["sha256"],num_envs=1024,steps=50,
        synthetic_envs=80,diagnostic_only=True,formal_acceptance_passed=False,learning_updates=0,
        supervision_path=str(case/"supervision.json"),launch_record=str(case/"launch.json"),
        single_launch_guard=str(case/"launch.lock"),monitor_record=str(case/"supervision.monitor.json"),
        stop_policy_path=str(case/"stop_policy.json"),
        command=["bash",str(ROOT/"tools/validate_stair_domain.sh"),"--headless","--device","cuda:0",
                 "--checkpoint",str(checkpoint),"--output",str(case/"evidence")])
    immutable(case/"launch.lock",request)
    immutable(case/"request.json",request)
    write(case/"stop_policy.json",dict(max_wall_seconds=300,signal="matched evaluator INT then TERM",
                                      automatic_restart_allowed=False,learning_updates=0))
    write(case/"previous_pointer.json",previous)
    write(case/"supervision.json",dict(request,status="starting"))
    write(case/"manager_state.json",dict(status="starting"))
    write(case/"supervision.monitor.json",dict(status="startup_pending",checked_at=now()))
    pointer=dict(previous)
    pointer.update({key:request[key] for key in ("supervision_path","launch_record","single_launch_guard",
                                               "monitor_record","stop_policy_path")})
    pointer.update(status="starting",phase="domain_integration_diagnostic",active_stair_job=str(case),
        active_stair_job_kind=request["job_kind"],runtime_state_status="domain_diagnostic_starting",
        domain_diagnostic_case=str(case),diagnostic_only=True,stairs_acceptance_passed=False,
        monitor_snapshot_tool=None,monitor_snapshot_command=None)
    write(POINTER,pointer)
    with (case/"manager.log").open("xb") as log:
        manager=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--manage",str(case),"--slot-fd",str(slot)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(slot,))
    os.close(slot)
    print(json.dumps(dict(case=str(case),manager=birth(manager.pid),diagnostic_only=True)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start",action="store_true")
    parser.add_argument("--manage",type=Path)
    parser.add_argument("--slot-fd",type=int)
    a=parser.parse_args()
    if a.manage:
        manage(a.manage.resolve(strict=True),a.slot_fd)
    elif a.start:
        launch()
    else:
        parser.error("Explicit --start required for one diagnostic; it never starts training")
