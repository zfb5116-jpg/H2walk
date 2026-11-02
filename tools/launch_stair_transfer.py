"""Exclusive custody/monitor/once-CPU dispatch for the accepted-flat stair transfer."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

from launch_stair_evaluation import (active_simulators, birth, digest, gpu_snapshot,
                                    immutable, process_check, write)
from launch_stair_learning import parse_training_log, fatal_lines, tail_jsonl, stop_policy

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
POINTER = ROOT/"logs/launches/current_main_training.json"
SLOT = ROOT/"logs/stair_learning/stair_transfer.slot.lock"
ENTRY = ROOT/"tools/train_stair_transfer.py"
TASK = "Unitree-H2-Stair-Ascent-Policy"
PROTOCOL = "accepted_flat_context_stair_clearance_v2"
FORMAT = "h2_guarded_stair_ppo_v2"
KIND = "guarded_stair_transfer_ppo_learning"
EVAL_KIND = "guarded_stair_transfer_frozen_validation"
BUDGET = 1000
TERMINAL = {"completed","failed","stopped_functional_guard","stopped_no_improvement"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, text):
    if not ok: raise RuntimeError(text)


def identity(path):
    s=Path(path).stat()
    return {k:getattr(s,k) for k in ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns")}


def native_environment(cpu=False):
    paths=[str(ROOT.parent/"IsaacLab"/"source"/name) for name in
           ("isaaclab","isaaclab_rl","isaaclab_assets","isaaclab_tasks")]
    paths.append("/home/nsgm/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages")
    result={**os.environ,"PYTHONPATH":os.pathsep.join(paths),"PYTHONDONTWRITEBYTECODE":"1",
            "PYTHONUNBUFFERED":"1","TZ":"Asia/Shanghai","OMP_NUM_THREADS":"2" if cpu else "8",
            "CTRLLAB_H2_USD_PATH":str(ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd")}
    if cpu: result["CUDA_VISIBLE_DEVICES"]=""
    return result


def changes(request):
    changed=[]
    for name, expected in request["frozen_files"].items():
        try:
            if digest(name) != expected: changed.append(name)
        except FileNotFoundError: changed.append(name)
    for name, expected in request.get("retained_stat_identities",{}).items():
        try:
            if identity(name) != expected: changed.append(name)
        except FileNotFoundError: changed.append(name)
    for tree in request.get("retained_trees",[]):
        actual={str(p) for p in Path(tree["root"]).rglob("*") if p.is_file()}
        if actual != set(tree["file_paths"]): changed.append(tree["root"]+": path set changed")
    return changed


def no_sim():
    old=read(POINTER)
    if old.get("supervision_path"):
        state=read(old["supervision_path"])
        require(state["status"] in TERMINAL, "The current producer/evaluator has not ended")
        case=Path(old["supervision_path"]).parent
        require(read(case/"manager_state.json")["status"] == "finished", "Previous manager has not ended")
        launch=read(old["launch_record"])
        for row in launch.get("processes",[]):
            entry=row.get("entrypoint") or next((x for x in row["argv"] if x.endswith(".py") and "/tools/" in x),None)
            require(not process_check(row,entry)["identity_matches"], "A recorded previous birth is still alive")
    require(not gpu_snapshot()["compute_processes"], "GPU compute is still occupied")
    require(not active_simulators(), "Another simulator/manager is active")


def formal_guard(checkpoint):
    checkpoint=Path(checkpoint).resolve(strict=True)
    producer=read(checkpoint.parent/"run.json")
    # Model, its own source and the physical protocol define replay identity.
    # Tool versions, labels and mutable manager state cannot enable a repeat.
    key=hashlib.sha256(json.dumps([digest(checkpoint),producer["source_sha256"],
        "first1024-seed42-90s-six15cm-original922"],sort_keys=True).encode()).hexdigest()
    return SLOT.parent/("stair_formal_"+key+".once.json")


def preflight(plan_path):
    plan_path=Path(plan_path).resolve(strict=True)
    plan=read(plan_path)
    no_sim()
    require(plan["status"] == "reviewed_ready" and plan["trial_protocol"] == PROTOCOL
            and plan["task"] == TASK and plan["max_iterations"] == BUDGET
            and plan["num_envs"] == 1024 and plan["network_dimensions"] == [1034,871,73,12], "Wrong reviewed plan")
    require(subprocess.check_output(["git","status","--porcelain=v1"],cwd=ROOT,text=True).strip() == "", "Commit the reviewed implementation before launch")
    for name, expected in plan["implementation_sha256"].items():
        require(digest(ROOT/name) == expected, "Reviewed implementation changed: "+name)
    for label in ("implementation_review","pre_change_custody_review"):
        path=Path(plan[label])
        require(digest(path) == plan[label+"_sha256"] and read(path)["status"] == "passed", "Missing reviewed admission: "+label)
    require(digest(plan["initial_policy"]) == plan["initial_policy_sha256"] == "8a82278fba63e1e5d8f715c33d4bdfced50291f0fffec73909059d92626cbf9f", "Wrong accepted flat parent")
    require(digest(plan["initial_policy_CPU_receipt"]) == "a6dd8e21690f90362770221879613e373a4baa3bc6a8b47f1853ff43a84ea1d1", "Wrong complete parent CPU receipt")
    formal=read(plan["initial_policy_formal_review"])
    require(digest(plan["initial_policy_formal_review"]) == "0516fa4f5b1a49f75b7a9f2ba4d60c38be18064018a45c7badbb1863552ce881"
            and formal["successful_envs"] == 926 and formal["flat_protocol_passed"] is True, "No accepted direct-flat parent")
    inherited=read(plan["custody_input_manifest"])
    request=dict(frozen_files=inherited["frozen_files"],retained_stat_identities=inherited["retained_stat_identities"],retained_trees=inherited["retained_trees"])
    require(not changes(request), "A retained/frozen input changed")
    return dict(status="passed",at=now(),plan_path=str(plan_path),plan_sha256=digest(plan_path),
                source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                frozen_custody_count=len(request["frozen_files"]),retained_stat_count=len(request["retained_stat_identities"]),
                Torch_imports=0,checkpoint_CPU_loads=0,Sim_starts=0)


def sync(case,status,**fields):
    for path in (POINTER,ROOT/"logs/stairs/campaign.json",ROOT/"logs/stairs_work/current_stair_work_state.json"):
        value=read(path)
        value.update(status=status,current_work_status="stair_transfer_"+status,
                     active_stair_job=str(case),active_simulation=status not in TERMINAL,
                     current_mode="guarded_stair_transfer",stairs_acceptance_passed=False,**fields)
        write(path,value)


def startup(metadata,request):
    if not metadata or metadata.get("status") == "initializing":
        return dict(status="pending",errors=[])
    errors=[]
    def check(ok,message):
        if not ok: errors.append(message)
    check(metadata.get("task") == TASK and metadata.get("trial_protocol") == PROTOCOL, "wrong physical task/profile")
    check(metadata.get("num_envs") == 1024 and metadata.get("max_iterations") == BUDGET
          and metadata.get("num_steps_per_env") == 24, "wrong budget")
    check(metadata.get("observation_shapes") == {"policy":[1024,1034],"critic":[1024,871],"rnd_state":[1024,73]}, "wrong observation ABI")
    transfer=metadata.get("initial_policy_transfer",{})
    check(transfer.get("actor_and_critic_loaded_bitwise") is True and transfer.get("dualnorm_loaded_bitwise") is True
          and transfer.get("parent_RNG_loaded") is False and transfer.get("parent_checkpoint_sha256") == request["initial_policy_sha256"], "incorrect initializer/resume")
    initial=metadata.get("initial_learning_state",{})
    check(initial.get("actor_optimizer_state_entries") == initial.get("critic_optimizer_state_entries")
          == initial.get("rnd_optimizer_state_entries") == 0, "optimizers are not fresh")
    geo=metadata.get("runtime_stair_geometry",{})
    check(geo.get("levels") == 6 and geo.get("riser_min_m") == .03 and geo.get("riser_max_m") == .15
          and geo.get("episode_length_s") == 90 and geo.get("scripted_gait_installed") is False, "incorrect staircase/controller")
    return dict(status="failed" if errors else "passed",errors=errors)


def sample(case,commit=False,reviewed_pending=None):
    case=Path(case).resolve(strict=True)
    request=read(case/"request.json")
    with (case/"monitor.write.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if reviewed_pending:
            reviewed=Path(reviewed_pending).resolve(strict=True)
            require(reviewed.parent == case and read(reviewed)["case"] == str(case), "Wrong reviewed pending")
        old=read(case/"supervision.monitor.json")
        log=case/"job.log"
        size=log.stat().st_size if log.exists() else 0
        offset=old.get("log_checked_offset",0)
        require(size>=offset,"Log was truncated")
        data=b""
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(offset);data=stream.read(4*1024*1024)
        raw=old.get("raw_log_fragment","")+data.decode(errors="replace")
        split=raw.rfind("\n")
        text=old.get("incomplete_iteration_block","")+(raw[:split+1] if split>=0 else "")
        fragment=raw[split+1:] if split>=0 else raw
        parsed=parse_training_log(text,old)
        fatal,known=fatal_lines(text)
        state=read(case/"supervision.json")
        manager=read(case/"manager_state.json")
        active=manager["status"] != "finished"
        run=Path(state["run_dir"]) if state.get("run_dir") else None
        metadata=read(run/"run.json") if run and (run/"run.json").exists() else {}
        contract=startup(metadata,request) if request["job_kind"] == KIND else dict(status="not_training",errors=[])
        files=[]
        if run:
            for path in run.glob("model_*.pt"):
                try: iteration=int(path.stem.split("_")[1])
                except ValueError: continue
                files.append(dict(path=str(path),iteration=iteration,bytes=path.stat().st_size,mtime_epoch=path.stat().st_mtime))
        files.sort(key=lambda row:(row["iteration"],row["mtime_epoch"]))
        rows=read(case/"launch.json").get("processes",[]) if (case/"launch.json").exists() else []
        checked=[process_check(row,request["entrypoint"]) for row in rows]
        changed=changes(request)
        progressed=parsed["iteration"]>old.get("iteration",-1) or size>old.get("log_size",0) and request["job_kind"] == EVAL_KIND
        advanced=time.time() if progressed else old.get("advanced_at_epoch",time.time())
        reasons=[]
        if active:
            if changed: reasons.append("frozen input changed")
            if fatal: reasons.append("new fatal runtime evidence")
            if contract["errors"]: reasons.append("runtime startup contract failed")
            if request["job_kind"] == KIND:
                for due in range(0,max(0,parsed["iteration"]-250)+1,500):
                    if parsed["iteration"]>due+250 and not any(row["iteration"]==due and row["bytes"]>0 for row in files):
                        reasons.append("checkpoint overdue more than250 iterations")
                if parsed["iteration"]>=BUDGET: reasons.append("training budget exceeded")
            if time.time()-advanced>=600 and log.exists() and time.time()-log.stat().st_mtime>=600:
                reasons.append("log and iteration jointly stalled10minutes")
        point=dict(case=str(case),checked_at=now(),checked_at_epoch=time.time(),**parsed,
            state_status=state["status"],manager=manager,log_checked_offset=offset+len(data),log_size=size,
            log_caught_up=offset+len(data)==size,raw_log_fragment=fragment,advanced_at_epoch=advanced,
            log_mtime_epoch=log.stat().st_mtime if log.exists() else None,
            processes=checked,gpu=gpu_snapshot(),changed_frozen_inputs=changed,
            startup_contract=contract,checkpoints=files,fatal_log_candidates=fatal,known_nonfatal=known,
            confirmed_stop_reasons=reasons,reviewed_pending=str(reviewed_pending) if reviewed_pending else None,
            latest_motion=(tail_jsonl(run/"motion_metrics.jsonl") or [None])[-1] if run else None,
            latest_effect=(tail_jsonl(run/"effect_assessments.jsonl") or [None])[-1] if run else None)
        pending=case/("monitor.pending_"+uuid.uuid4().hex[:12]+".json")
        immutable(pending,point)
        if commit:
            write(case/"supervision.monitor.json",point)
            with (case/"supervision.jsonl").open("a") as stream: stream.write(json.dumps(point,ensure_ascii=False)+"\n")
            state=read(case/"supervision.json")
            state["manager_monitor"]=dict(pending=str(pending),iteration=point["iteration"],checked_at=point["checked_at"])
            write(case/"supervision.json",state)
            pointer=read(POINTER)
            if pointer.get("active_stair_job") == str(case):
                pointer.update(last_monitor_pending=str(pending),last_monitor_iteration=point["iteration"],run_dir=str(run) if run else None)
                if metadata.get("source_snapshot"):
                    pointer["source_snapshot"]=metadata["source_snapshot"]
                if files:
                    pointer.update(latest_checkpoint=files[-1]["path"],latest_checkpoint_validation=None)
                write(POINTER,pointer)
        return point,pending


def descendants(pid):
    found,queue={pid},[pid]
    while queue:
        current=queue.pop()
        try: children=Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except FileNotFoundError: continue
        for child in map(int,children):
            if child not in found: found.add(child);queue.append(child)
    return found


def safe_stop(case,reasons):
    sent=[]
    request=read(case/"request.json")
    for sig,grace in ((signal.SIGINT,12),(signal.SIGTERM,8)):
        pointer=read(POINTER)
        require(pointer["supervision_path"] == str(case/"supervision.json"), "Pointer changed before signal")
        rows=read(case/"launch.json")["processes"]
        for row in sorted(rows,key=lambda r:r["role"] != "learner"):
            if sig == signal.SIGINT and row["role"] != "learner": continue
            fresh=process_check(row,request["entrypoint"])
            if fresh["identity_matches"]:
                try: os.kill(row["pid"],sig);sent.append(dict(signal=sig.name,identity=fresh))
                except ProcessLookupError: pass
        deadline=time.monotonic()+grace
        while time.monotonic()<deadline and any(process_check(row,request["entrypoint"])["identity_matches"] for row in rows):
            time.sleep(.25)
        if not any(process_check(row,request["entrypoint"])["identity_matches"] for row in rows): break
    result=dict(at=now(),reasons=reasons,signals=sent,remaining=[process_check(row,request["entrypoint"]) for row in read(case/"launch.json")["processes"]])
    immutable(case/("stop_evidence_"+uuid.uuid4().hex[:8]+".json"),result)
    return result


def manage(case,fd):
    case=Path(case).resolve(strict=True)
    request=read(case/"request.json")
    require(os.fstat(fd).st_ino == SLOT.stat().st_ino,"Wrong inherited slot")
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manager=birth(os.getpid())
    process=None;error=None;code=-1;rows=[]
    try:
        require(not changes(request) and not gpu_snapshot()["compute_processes"],"Frozen inputs/GPU changed before dispatch")
        require(not active_simulators(exclude=[request["parent_launcher_pid"]]),"Another simulator started before dispatch")
        with (case/"job.log").open("xb") as stream:
            process=subprocess.Popen(request["command"],cwd=ROOT,env=native_environment(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+25
            while time.monotonic()<deadline:
                rows=[]
                for pid in descendants(process.pid):
                    try:
                        row=birth(pid)
                        if request["entrypoint"] in row["argv"]:
                            role="learner" if Path(row["argv"][0]).name.startswith("python") else "wrapper"
                            rows.append(dict(**row,role=role,entrypoint=request["entrypoint"]))
                    except (FileNotFoundError,ProcessLookupError): pass
                if any(row["role"] == "learner" for row in rows) or process.poll() is not None: break
                time.sleep(.2)
            immutable(case/"launch.json",dict(manager=manager,processes=rows,request_case=str(case)))
            state=read(case/"supervision.json");state.update(processes=rows,manager=manager);write(case/"supervision.json",state)
            write(case/"manager_state.json",dict(status="running",manager=manager,processes=rows))
            while True:
                point,_=sample(case,commit=True)
                if process.poll() is None and point["confirmed_stop_reasons"]:
                    stop=safe_stop(case,point["confirmed_stop_reasons"])
                    state=read(case/"supervision.json");state.update(status="failed",manager_stop_evidence=stop);write(case/"supervision.json",state)
                if process.poll() is not None: break
                try: process.wait(timeout=60 if point["log_caught_up"] else .1)
                except subprocess.TimeoutExpired: pass
            code=process.wait()
        state=read(case/"supervision.json")
        if request["job_kind"] == EVAL_KIND:
            result=read(case/"evaluation/result.json") if (case/"evaluation/result.json").exists() else {}
            state.update(status="completed" if code == 0 and result.get("status") == "completed" else "failed",evaluation_result=result)
        if code or state.get("status") not in TERMINAL: state["status"]="failed"
        state.update(wrapper_exit_code=code,wrapper_ended_at=now())
        write(case/"supervision.json",state)
    except BaseException as exc:
        error=repr(exc)
        if process is not None and process.poll() is None and (case/"launch.json").exists(): safe_stop(case,[error])
        state=read(case/"supervision.json");state.update(status="failed",manager_error=error);write(case/"supervision.json",state)
    finally:
        state=read(case/"supervision.json")
        write(case/"manager_state.json",dict(status="finished",outcome=state["status"],exit_code=code,error=error,manager=manager,processes=rows))
        point,pending=sample(case,commit=True)
        sync(case,state["status"],run_dir=state.get("run_dir"),terminal_monitor_pending=str(pending))
        ledger=ROOT/"logs/stairs_work/ended_experiments.jsonl"
        # Only this new case can be appended; never rewrite an old ledger row.
        with ledger.open("a+") as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            existing=[json.loads(line) for line in stream if line.strip()]
            if not any(row.get("case") == str(case) for row in existing):
                stream.seek(0,2);stream.write(json.dumps(dict(case=str(case),kind=request["job_kind"],status=state["status"],ended_at=now(),source_commit=request["source_commit"]))+"\n")
        os.close(fd)


def dispatch(plan_path,start,validation_parent=None):
    plan_path=Path(plan_path).resolve(strict=True)
    checked=preflight(plan_path)
    if not start: return dict(mode="preflight_only",**checked)
    plan=read(plan_path)
    SLOT.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(SLOT,os.O_RDWR|os.O_CREAT,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        no_sim()
        kind=EVAL_KIND if validation_parent else KIND
        if validation_parent:
            guard=formal_guard(validation_parent)
            reserved=read(guard)
            receipt=read(validation_parent.with_suffix(".validation.json"))
            require(reserved["checkpoint_sha256"] == digest(validation_parent)
                    and receipt["status"] == "passed" and receipt["scope"] == "natural_final",
                    "Missing pre-CPU formal reservation and complete new final receipt")
            immutable(guard.with_suffix(".dispatched.json"),dict(at=now(),checkpoint=str(validation_parent)))
        else:
            guard_key=hashlib.sha256(json.dumps([kind,plan["initial_policy_sha256"],checked["plan_sha256"]],sort_keys=True).encode()).hexdigest()
            guard=SLOT.parent/("stair_transfer_"+guard_key+".once.json")
            immutable(guard,dict(at=now(),kind=kind,plan_sha256=checked["plan_sha256"],checkpoint=None))
        stamp=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        case=SLOT.parent/("stair_"+("formal_" if validation_parent else "transfer_")+stamp+"_"+uuid.uuid4().hex[:8])
        case.mkdir()
        immutable(case/"preflight.json",checked)
        old=read(POINTER);immutable(case/"previous_pointer.json",old)
        frozen=dict(read(plan["custody_input_manifest"])["frozen_files"])
        frozen.update({str(ROOT/name):sha for name,sha in plan["implementation_sha256"].items()})
        frozen[str(plan_path)]=digest(plan_path)
        for label in ("implementation_review","pre_change_custody_review","custody_input_manifest"):
            frozen[plan[label]]=digest(plan[label])
        frozen[str(guard)]=digest(guard)
        policy=stop_policy();policy["bounded_run_note"]="Independent1000-update clearance/stability repair cap; never extend automatically."
        immutable(case/"stop_policy.json",policy)
        frozen[str(case/"stop_policy.json")]=digest(case/"stop_policy.json")
        entry=str(ROOT/"tools/evaluate_stair_policy.py") if validation_parent else str(ENTRY)
        command=["/home/nsgm/isaac-sim/python.sh",entry,"--headless","--device","cuda:0"]
        if validation_parent:
            command += ["--checkpoint",str(validation_parent),"--output",str(case/"evaluation"),"--num_envs","1024","--seed","42"]
        else:
            command += ["--plan",str(plan_path),"--supervision_state",str(case/"supervision.json"),
                        "--stop_policy",str(case/"stop_policy.json"),"--expected_git_commit",checked["source_commit"]]
        request=dict(case=str(case),job_kind=kind,entrypoint=entry,command=command,task=TASK,trial_protocol=PROTOCOL,
            source_commit=checked["source_commit"],source_branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
            plan_path=str(plan_path),plan_sha256=digest(plan_path),initial_policy=plan["initial_policy"],initial_policy_sha256=plan["initial_policy_sha256"],
            frozen_files=frozen,retained_stat_identities=read(plan["custody_input_manifest"])["retained_stat_identities"],
            retained_trees=read(plan["custody_input_manifest"])["retained_trees"],
            max_iterations=BUDGET,num_envs=1024,num_steps_per_env=24,save_interval=500,expected_total_timesteps=BUDGET*1024*24,
            expected_final_iteration=BUDGET-1,supervision_path=str(case/"supervision.json"),launch_record=str(case/"launch.json"),
            monitor_record=str(case/"supervision.monitor.json"),stop_policy_path=str(case/"stop_policy.json"),
            single_launch_guard=str(guard),parent_launcher_pid=os.getpid(),automatic_restart_allowed=False,
            authorization="用户明确要求测试与修复直到完成上楼；已报告独立1000轮稳定与抬脚训练方案，六级90s原922验收")
        if validation_parent:
            request.update(checkpoint=str(validation_parent),checkpoint_sha256=digest(validation_parent),producer_case=read(validation_parent.parent/"run.json").get("case"))
        immutable(case/"request.json",request)
        write(case/"supervision.json",dict(**request,status="launching",run_dir=None,stairs_acceptance_passed=False))
        write(case/"supervision.monitor.json",dict(iteration=-1,log_checked_offset=0))
        write(case/"manager_state.json",dict(status="starting",created_at=now()))
        pointer=dict(old);pointer.update({k:request[k] for k in ("supervision_path","launch_record","monitor_record","stop_policy_path","single_launch_guard","source_commit","source_branch","plan_path","plan_sha256")})
        pointer.update(active_stair_job=str(case),active_stair_job_kind=kind,status="launching",active_simulation=True,
            run_dir=None,latest_checkpoint=None,latest_checkpoint_validation=None,completed_iterations=0,
            source_snapshot=None,final_checkpoint=None,checkpoint_sha256=None,checkpoint_validation=None,
            latest_checkpoint_sha256=None,latest_checkpoint_iteration=None,latest_checkpoint_size=None,
            max_iterations=BUDGET,expected_final_iteration=BUDGET-1,expected_total_timesteps=BUDGET*1024*24,
            monitor_snapshot_tool=str(SELF),monitor_snapshot_command=[sys.executable,str(SELF),"--monitor",str(case)])
        write(POINTER,pointer)
        sync(case,"launching",current_stair_plan=str(plan_path),current_stair_producer=str(case))
        with (case/"manager.log").open("xb") as stream:
            process=subprocess.Popen([sys.executable,str(SELF),"--manage",str(case),str(fd)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,))
        immutable(case/"manager_parent_receipt.json",dict(manager=birth(process.pid),at=now()))
        return dict(mode="started",case=str(case),manager_pid=process.pid,plan_sha256=digest(plan_path),training_iterations=0 if validation_parent else BUDGET)
    finally: os.close(fd)


def cpu_once(case,scope):
    case=Path(case).resolve(strict=True)
    request=read(case/"request.json")
    state=read(case/"supervision.json")
    run=Path(state["run_dir"]).resolve(strict=True)
    checkpoint=run/("model_0.pt" if scope == "startup" else f"model_{BUDGET-1}.pt")
    require(request["job_kind"] == KIND and checkpoint.exists(),"Missing typed stair packet")
    if scope == "natural_final":
        no_sim()
        require(state["status"] == "completed" and read(case/"manager_state.json")["status"] == "finished", "Final CPU requires natural terminal")
    else:
        require(state["status"] == "training" and read(case/"manager_state.json")["status"] == "running", "Startup CPU requires actual new training")
    require(not changes(request),"Custody changed before CPU")
    if scope == "natural_final":
        immutable(formal_guard(checkpoint),dict(status="reserved_before_any_final_CPU_load",at=now(),
            producer_case=str(case),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            physical_protocol="first1024-seed42-90s-six15cm-original922",target_CPU_loads=1,parent_CPU_loads=1,formal_runs=1))
    claim=case/(scope+"_CPU_review.once.json")
    immutable(claim,dict(status="claimed_before_CPU",at=now(),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                         required_packet_loads=dict(target=1,parent=1),Sim_starts=0))
    result=subprocess.run(["/home/nsgm/isaac-sim/python.sh",str(ROOT/"tools/validate_stair_transfer.py"),"--checkpoint",str(checkpoint),"--plan",request["plan_path"],"--scope",scope],
        cwd=ROOT,env=native_environment(cpu=True),text=True,capture_output=True,timeout=180)
    (case/(scope+"_CPU.stdout")).write_text(result.stdout)
    (case/(scope+"_CPU.stderr")).write_text(result.stderr)
    receipt=checkpoint.with_suffix(".validation.json")
    outcome=dict(status="passed_consumed" if result.returncode == 0 else "failed_consumed_no_retry",at=now(),exit_code=result.returncode,
                 receipt=str(receipt) if receipt.exists() else None,receipt_sha256=digest(receipt) if receipt.exists() else None)
    write(claim,{**read(claim),**outcome})
    require(result.returncode == 0,"New complete stair CPU failed; evidence retained, no automatic reload/retry")
    require(read(receipt)["status"] == "passed", "CPU receipt did not pass")
    return checkpoint,outcome


def main():
    if len(sys.argv)==4 and sys.argv[1]=="--manage":
        manage(Path(sys.argv[2]),int(sys.argv[3]));return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--start",action="store_true")
    parser.add_argument("--monitor",type=Path)
    parser.add_argument("--commit-monitor",action="store_true")
    parser.add_argument("--pending",type=Path)
    parser.add_argument("--startup-cpu",type=Path)
    parser.add_argument("--validate",type=Path)
    args=parser.parse_args()
    if args.monitor:
        point,pending=sample(args.monitor,args.commit_monitor,args.pending);print(json.dumps(dict(point=point,pending=str(pending))));return
    if args.startup_cpu:
        _,outcome=cpu_once(args.startup_cpu,"startup");print(json.dumps(outcome));return
    if args.validate:
        require(args.start,"Final validation and the unique replay must be one integrated --start scope")
        checkpoint,outcome=cpu_once(args.validate,"natural_final")
        print(json.dumps(dispatch(read(args.validate/"request.json")["plan_path"],True,checkpoint)));return
    require(args.plan is not None,"Provide the reviewed stair plan")
    print(json.dumps(dispatch(args.plan,args.start)))


if __name__=="__main__": main()
