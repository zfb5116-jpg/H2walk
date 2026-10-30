"""One bounded direct-policy aggregation job; never starts PPO or declares readiness."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
TASK = "Unitree-H2-Direct-Flat-Initialization"
TEACHER_SHA = "2b06db2e09061cc660d46578b112ea29d0e601c3e75ac2163f8425aa24553fcb"
INITIAL_SHA = "a19a2b498786e8ccba63fa3687d755656d2ecb0fece813184bfd89c8106d12d6"
USD_SHA = "7f3432d01756efaeb6d4337536a3ead5a9d0c2f4130654bd43a353012b6c2dbd"
RECON_SHA = "d260001c9850941c95d1ba83196076f27e118e687e72586d0a2bff84f1674c50"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = Path(str(path)+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_hashes(source):
    return {"source/"+str(path.relative_to(source)):sha(path) for path in sorted(source.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def parse_args():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--initial_policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1024 or args.seed != 42:
        parser.error("This approved collection uses1024 environments and seed42")
    for key in ("source_root", "teacher_checkpoint", "initial_policy", "state"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    return args, AppLauncher


def main():
    args, AppLauncher = parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = dict(status="initializing", task=TASK, physics_controls=0, actor_updates=0, round=-1,
        phase="preparation", initialization_ready=False, ppo_updates=0, rnd_updates=0,
        policy_dimensions=[1029,866,73,12], seed=42, num_envs=1024,
        schedule=dict(admission_controls=150, round_controls=[400]*4, teacher_probabilities=[.75,.5,.25,0.],
                      updates_per_round=800, batch_size=1024, learning_rate=.001, max_gradient_norm=1.),
        source_commit=__import__('subprocess').check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        student_initial_policy=str(args.initial_policy), student_initial_sha256=sha(args.initial_policy),
        teacher_checkpoint=str(args.teacher_checkpoint), teacher_checkpoint_sha256=sha(args.teacher_checkpoint),
        collector_case=str(args.state.parent), started_at_epoch=started)
    app, wrapped, dataset = None, None, None

    def update(**fields):
        metadata.update(fields)
        state = json.loads(args.state.read_text())
        state.update(fields, initialization_ready=False, checked_at_epoch=time.time())
        write(args.state, state); write(args.output/"run.json", metadata)

    try:
        if metadata["teacher_checkpoint_sha256"] != TEACHER_SHA or metadata["student_initial_sha256"] != INITIAL_SHA:
            raise ValueError("Weights differ from the approved accepted-flat teacher and retained BC initializer")
        original_hashes = source_hashes(args.source_root)
        training = args.output/"training_source"
        training.mkdir()
        source = training/"source"
        shutil.copytree(args.source_root, source, ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
        if source_hashes(source) != original_hashes:
            raise RuntimeError("Incomplete production source snapshot")
        write(training/"main_source_sha256.json", original_hashes)
        tool_names = [Path(__file__).name,"collect_closed_loop_initialization.sh","h2_initialization_dataset.py",
                      "h2_initialization_oracle.py","h2_bootstrap_contract.py"]
        (training/"tools").mkdir()
        for name in tool_names:
            shutil.copy2(Path(__file__).parent/name, training/"tools"/name)
        metadata.update(source_root=str(source), source_sha256=original_hashes,
            requested_source_root=str(args.source_root), tool_sha256={name:sha(Path(__file__).parent/name) for name in tool_names})
        # Copied asset modules derive defaults from __file__. Bind the existing
        # audited external bytes before importing the copied task, never assets
        # accidentally inferred underneath output/training_source.
        external = {ROOT/"assets/isaaclab/unitree_robots/h2/h2.usd":USD_SHA,
                    ROOT/"logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt":RECON_SHA}
        for path,expected in external.items():
            if sha(path)!=expected: raise RuntimeError("Initialization external dependency changed: "+str(path))
        os.environ["CTRLLAB_H2_USD_PATH"] = str(next(iter(external)))
        os.environ["CTRLLAB_H2_LIDAR_RECON_CHECKPOINT"] = str(list(external)[1])
        metadata["external_dependency_sha256"] = {str(path):digest for path,digest in external.items()}
        update(phase="launching_physics")
        app = AppLauncher(args).app
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.modules.normalizer import EmpiricalNormalization
        from h2_initialization_dataset import (InitializationDataset, DISK_LIMIT_BYTES, HOST_LIMIT_BYTES,
                                               production_budget, summarize_joint_phase_errors)
        sys.path.insert(0, str(source/"CtrlLab/CtrlLab/tasks/manager_based"))
        import h2_task
        if not Path(h2_task.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Initialization task import did not use the preserved production source")
        from h2_task.networks import H2ActorCriticOur
        from h2_initialization_oracle import FlatInitializationOracle

        def cpu(value, dtype=None):
            value = value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()
            return value.astype(dtype, copy=False) if dtype is not None else value

        def cpu_state(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key:cpu_state(item) for key,item in value.items()}
            if isinstance(value, (list,tuple)):
                return type(value)(cpu_state(item) for item in value)
            return value

        spec = gym.spec(TASK)
        cfg = config(spec.kwargs["env_cfg_entry_point"])
        agent_spec = gym.spec("Unitree-H2-Stair-Ascent-Policy")
        agent = config(agent_spec.kwargs["rsl_rl_cfg_entry_point"])
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = 1024,42,args.device or "cuda:0"
        cfg.io_descriptors_output_dir = str(args.output)
        dump_yaml(str(args.output/"params/env.yaml"),cfg)
        dump_yaml(str(args.output/"params/agent.yaml"),agent)
        asset = Path(cfg.scene.robot.spawn.usd_path).resolve(strict=True)
        recon = Path(cfg.observations.policy.height_scanner.params["checkpoint"]).resolve(strict=True)
        if {str(path):sha(path) for path in (asset,recon)} != metadata["external_dependency_sha256"]:
            raise RuntimeError("Copied task did not bind the audited USD/LiDAR dependencies")
        wrapped = RslRlVecEnvWrapper(gym.make(TASK,cfg=cfg), clip_actions=100.)
        raw = wrapped.unwrapped
        obs, extras = wrapped.reset()
        device = raw.device
        if raw.step_dt != .02 or obs.shape != (1024,1029) or extras["observations"]["critic"].shape != (1024,866):
            raise RuntimeError("Initialization environment control/observation layout drift")
        if extras["observations"]["rnd_state"].shape != (1024,73) or wrapped.num_actions != 12:
            raise RuntimeError("Initialization RND/action dimensions drift")
        term = raw.action_manager.get_term("JointPositionAction")
        robot, ids = raw.scene["robot"], list(term._joint_ids)
        if type(term).__name__ != "DirectLegPositionAction" or term.gait_reference is not None:
            raise RuntimeError("The physical plant must be the original direct-position action, without a reference")
        if any(getattr(act,"support_priority_buffer",None) is not None for act in robot.actuators.values()):
            raise RuntimeError("No native support-priority drive is allowed in the initialization plant")
        metadata.update(observation_shapes={key:list(value.shape) for key,value in extras["observations"].items()},
                        action_joint_names=list(term._joint_names), native_action_ids=ids,
                        normalizer_semantics="Own BC dualnorm frozen/eval throughout collection and fitting",
                        dataset_budget=production_budget(), gpu_soft_limit_bytes=11*2**30)
        pcfg = agent.policy.to_dict(); pcfg.pop("class_name")
        # Model construction/query setup must not consume the plant's future
        # reset random stream. Actor-selection/replay RNGs below are separate.
        cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            student = H2ActorCriticOur(1029,866,12,**pcfg).to(device)
            boot = torch.load(args.initial_policy,map_location=device,weights_only=True)
            if boot.get("format") != "h2_direct_policy_initialization_v1" or boot.get("policy_dimensions") != [1029,866,73,12]:
                raise ValueError("Retained BC package contract differs")
            student.load_state_dict(boot["model_state_dict"],strict=True)
            snorm, cnorm = EmpiricalNormalization([1029]).to(device).eval(), EmpiricalNormalization([866]).to(device).eval()
            snorm.load_state_dict(boot["obs_norm_state_dict"],strict=True)
            cnorm.load_state_dict(boot["privileged_obs_norm_state_dict"],strict=True)
            oracle = FlatInitializationOracle(raw,args.teacher_checkpoint)
        del boot
        normalizer_reference = [cpu_state(norm.state_dict()) for norm in (snorm,cnorm)]
        optimizer = torch.optim.Adam(student.actor.parameters(),lr=.001)
        dataset = InitializationDataset(args.output/"dataset",seed=42)
        actor_rng = np.random.default_rng(72042)
        env_ids = torch.arange(1024,device=device)
        holdout = env_ids % 8 == 0
        teacher_mask = torch.ones(1024,device=device,dtype=torch.bool)
        global_controls, actor_updates, active_stage = 0,0,0
        collection_enabled = False
        probability = 1.
        journal = args.output/"dataset/episodes.jsonl"
        episode_id = torch.zeros(1024,device=device,dtype=torch.long)
        age = torch.zeros(1024,device=device)
        pending = torch.zeros(1024,device=device,dtype=torch.bool)
        first_alive = torch.ones_like(pending)
        first_invalid = torch.zeros_like(pending)
        root_start = robot.data.root_pos_w.clone()
        first_root_start = root_start.clone()
        minimum = root_start.clone()
        endpoint = root_start.clone()
        first_duration = age.clone()
        first_current = torch.ones_like(pending)
        native_names = list(raw.termination_manager.active_terms)
        termination_counts = {name:0 for name in native_names}
        bounds = torch.tensor(raw._stair_domain_bounds_local,device=device)
        below_tolerance = float(raw._stair_below_floor_tolerance)

        def physical_finish(which):
            which = which[pending[which]]
            if not which.numel(): return
            position = robot.data.root_pos_w[which]
            age[which] += raw.step_dt
            minimum[which] = torch.minimum(minimum[which],position)
            local = position-raw.scene.env_origins[which]
            invalid = ((local[:,0]<bounds[0])|(local[:,0]>bounds[1])|(local[:,1]<bounds[2])
                       |(local[:,1]>bounds[3])|(local[:,2]<-below_tolerance)|~torch.isfinite(position).all(-1))
            first_invalid[which] |= invalid & first_current[which]
            fresh = which[first_current[which]]
            endpoint[fresh] = robot.data.root_pos_w[fresh]
            first_duration[fresh] = age[fresh]
            pending[which] = False

        def before_reset(which):
            if not collection_enabled: return
            which = torch.as_tensor(which,device=device,dtype=torch.long)
            physical_finish(which)
            bits = {name:raw.termination_manager.get_term(name)[which].clone() for name in native_names}
            for name, values in bits.items(): termination_counts[name] += int(values.sum())
            payload = dict(stage=active_stage,control_step=global_controls+1,env_ids=cpu(which).tolist(),
                episode_ids=cpu(episode_id[which]).tolist(),duration_s=cpu(age[which]).tolist(),
                start_xyz=cpu(root_start[which]).tolist(),end_xyz=cpu(robot.data.root_pos_w[which]).tolist(),
                minimum_xyz=cpu(minimum[which]).tolist(),
                native_terminated=cpu(raw.reset_terminated[which]).tolist(),
                native_timeout=cpu(raw.reset_time_outs[which]).tolist(),
                termination_bits={name:cpu(value).tolist() for name,value in bits.items()})
            with journal.open("a") as stream: stream.write(json.dumps(payload,separators=(",",":"))+"\n")
            if journal.stat().st_size > 12*2**20:
                raise OSError("Physical episode journal reached its approved12MiB bound")
            first_alive[which] = False

        def after_reset(which):
            which = torch.as_tensor(which,device=device,dtype=torch.long)
            episode_id[which] += 1; age[which] = 0; pending[which] = False
            root_start[which] = robot.data.root_pos_w[which]
            minimum[which] = root_start[which]
            if collection_enabled:
                first_current[which] = False
                choice = actor_rng.random(len(which)) < probability
                teacher_mask[which] = (True if active_stage==0 else
                    torch.as_tensor(choice,device=device) & ~holdout[which])

        raw.add_reset_callback(before_reset,when="before")
        raw.add_reset_callback(after_reset,when="after")
        # Observe returned native command packets only; no extra compute,
        # DelayBuffer push, receiver, motor setter or physical/RNG change.
        trace_dir = args.output/"dataset/native_packets"; trace_dir.mkdir()
        trace_writes = {}; trace_bytes = 0; observed_ids = set()
        oracle_trace_dir = args.output/"dataset/oracle_state"; oracle_trace_dir.mkdir()
        oracle_trace_bytes, oracle_trace_records = 0,[]
        oracle_trace_fields = ("state","valid","invalid_reason_bits","reset_generation","control_step",
            "new_query_mask","reference_time","reference_steps","reference_stance","reference_anchors",
            "reference_q","reference_qdot","reference_failure","reference_load","reference_ik_error","reference_foot_target")
        for name, actuator in robot.actuators.items():
            native = list(range(robot.num_joints)) if isinstance(actuator.joint_indices,slice) else list(actuator.joint_indices)
            if not set(native) <= set(ids): continue
            observed_ids.update(native)
            original_compute = actuator.compute
            def observed(act, command, q, qd, original_compute=original_compute, name=name, native=native):
                nonlocal trace_bytes
                output = original_compute(command,q,qd)
                if collection_enabled:
                    if (not torch.equal(act.positions_delay_buffer.time_lags,act.velocities_delay_buffer.time_lags)
                            or not torch.equal(act.positions_delay_buffer._circular_buffer._num_pushes,
                                               act.velocities_delay_buffer._circular_buffer._num_pushes)):
                        raise RuntimeError("Native q/v transport no longer shares lag/compute-call history")
                    if bool((output.joint_velocities != 0).any() | (output.joint_efforts != 0).any()):
                        raise RuntimeError("Actual initialization plant received nonzero velocity/effort")
                    key = (active_stage,name)
                    count = trace_writes.get(key,0)
                    if count < 8:
                        data = dict(q_received=cpu(output.joint_positions),v_received=cpu(output.joint_velocities),
                            e_received=cpu(output.joint_efforts),q_actual=cpu(q),qd_actual=cpu(qd),
                            stiffness=cpu(act.stiffness),damping=cpu(act.damping),caps=cpu(act.effort_limit),
                            lag=cpu(act.positions_delay_buffer.time_lags),
                            pushes=cpu(act.positions_delay_buffer._circular_buffer._num_pushes),
                            control_step=np.asarray(global_controls+1,dtype=np.int64),
                            native_control_counter=np.asarray(raw.common_step_counter,dtype=np.int64),
                            physics_counter=np.asarray(raw._sim_step_counter,dtype=np.int64),native_ids=np.asarray(native,dtype=np.int32))
                        data["pre_physics_counter_compatible"] = np.asarray(
                            int(raw.common_step_counter)==(int(raw._sim_step_counter)-1)//int(raw.cfg.decimation))
                        path = trace_dir/f"stage{active_stage}_{name}_{count:02d}.npz"
                        np.savez_compressed(path,**data); trace_bytes += path.stat().st_size
                        if trace_bytes > 20*2**20: raise OSError("Native packet trace exceeded20MiB")
                        trace_writes[key] = count+1
                return output
            actuator.compute = types.MethodType(observed,actuator)
        if observed_ids != set(ids):
            raise RuntimeError("Native packet observation did not cover all12 original lower-limb drives")

        actual_published = {}
        original_process = term.process_actions
        def capture_direct(action_term,actions):
            original_process(actions)
            limit = robot.data.soft_joint_pos_limits[:,ids]
            expected = limit.mean(-1)+(limit[...,1]-limit[...,0])*.5*torch.tanh(actions)
            if not torch.allclose(action_term._processed_actions,expected,atol=1e-6,rtol=0):
                raise RuntimeError("Direct publisher differs from the declared full-soft-range decoder")
            actual_published["raw"] = actions.detach().clone()
        term.process_actions = types.MethodType(capture_direct,term)

        def gpu_budget():
            if torch.device(device).type != "cuda": raise RuntimeError("Production collection requires its original CUDA plant")
            allocated,reserved = torch.cuda.memory_allocated(device),torch.cuda.memory_reserved(device)
            if max(allocated,reserved) > 11*2**30:
                raise MemoryError("Initialization GPU allocation/reservation exceeds11GiB; no automatic reduction/retry")
            return dict(allocated_bytes=allocated,reserved_bytes=reserved)

        def quality():
            rows = dataset.heldout.indices()
            errors, phases, valid, clips = [],[],[],[]
            student.eval()
            with torch.no_grad():
                for start in range(0,len(rows),1024):
                    batch = dataset.heldout.get(rows[start:start+1024],["obs","position","stiffness","caps","lower","upper","phase","valid_label","clip"])
                    x = torch.from_numpy(batch["obs"]).to(device)
                    lo,hi = (torch.from_numpy(batch[key]).to(device) for key in ("lower","upper"))
                    predicted = (lo+hi)*.5+(hi-lo)*.5*torch.tanh(student.act_inference(snorm(x)))
                    rad = cpu(predicted)-batch["position"]
                    errors.append(np.stack((rad,np.abs(rad)*batch["stiffness"]/batch["caps"]),axis=1))
                    phases.append(batch["phase"]);valid.append(batch["valid_label"]);clips.append(batch["clip"])
            error,phase,valid,clip = np.concatenate(errors),np.concatenate(phases),np.concatenate(valid),np.concatenate(clips)
            return summarize_joint_phase_errors(error,phase,valid,clip)

        def export(quality_report):
            for normalizer, reference in zip((snorm,cnorm),normalizer_reference):
                if any(not torch.equal(value.detach().cpu(),reference[key]) for key,value in normalizer.state_dict().items()):
                    raise RuntimeError("Own frozen initialization normalization changed")
            opt = dict(format="h2_direct_imitation_optimizer_v1",optimizer_state_dict=cpu_state(optimizer.state_dict()),
                actor_parameter_names=[name for name,_ in student.actor.named_parameters()],updates=actor_updates,
                batch_size=1024,learning_rate=.001,max_gradient_norm=1.,ppo_updates=0,rnd_updates=0,
                dataset_manifest=str(args.output/"dataset/manifest.json"))
            opt_path = args.output/"imitation_optimizer_state.pt"
            torch.save(opt,str(opt_path)+".tmp");Path(str(opt_path)+".tmp").replace(opt_path)
            packet = dict(format="h2_direct_policy_initialization_v1",ppo_completed_iterations=0,
                model_state_dict=cpu_state(student.state_dict()),obs_norm_state_dict=cpu_state(snorm.state_dict()),
                privileged_obs_norm_state_dict=cpu_state(cnorm.state_dict()),action_semantics="full_soft_tanh_position_v1",
                policy_dimensions=[1029,866,73,12],source_root=str(source),source_sha256=original_hashes,
                source_commit=metadata["source_commit"],external_dependency_sha256=metadata["external_dependency_sha256"],
                collector_case=str(args.state.parent),teacher_checkpoint_sha256=TEACHER_SHA,initialization_ready=False,
                metadata=dict(quality=quality_report,physics_controls=global_controls,actor_updates=actor_updates,
                    optimizer_path=str(opt_path),optimizer_sha256=sha(opt_path),
                    critic="Retained BC cold critic, not trained by imitation",independent_validation_required=True))
            path = args.output/"candidate_initial_policy.pt"
            torch.save(packet,str(path)+".tmp");Path(str(path)+".tmp").replace(path)
            return str(path),sha(path)

        def actor_digest():
            digest = hashlib.sha256()
            for name,value in student.actor.state_dict().items():
                digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        stage_summaries=[]
        for stage in range(5):
            active_stage = stage
            probability = (1.,.75,.5,.25,0.)[stage]
            collection_enabled = False
            obs,extras = wrapped.reset()
            age.zero_();pending.zero_();first_alive.fill_(True);first_invalid.zero_();first_current.fill_(True)
            root_start.copy_(robot.data.root_pos_w); minimum.copy_(root_start); endpoint.copy_(root_start);first_duration.zero_()
            first_root_start.copy_(root_start)
            teacher_mask[:] = True if stage==0 else torch.as_tensor(actor_rng.random(1024)<probability,device=device)&~holdout
            counts = {"labels":0,"invalid":0,"clip":np.zeros(12,dtype=np.int64),"teacher_controls":0}
            student.eval(); collection_enabled=True
            collection_actor_sha = actor_digest()
            controls = 150 if stage==0 else 400
            update(status="collecting",phase="admission" if stage==0 else "collecting",round=stage-1,
                   physics_controls=global_controls,actor_updates=actor_updates,gpu=gpu_budget())
            for local_control in range(controls):
                with torch.no_grad():
                    raw_obs, raw_critic = obs.clone(),extras["observations"]["critic"].clone()
                    predicted = student.act_inference(snorm(raw_obs))
                    label = oracle.query(raw_obs)
                    if not torch.equal(raw_obs,obs): raise RuntimeError("Teacher query modified the student's current raw observation")
                    if local_control < 8:
                        trace = {name:cpu(label[name]) for name in oracle_trace_fields}
                        trace["env_id"] = np.arange(1024,dtype=np.int32)
                        if any(value.ndim==0 or value.shape[0]!=1024 for value in trace.values()):
                            raise RuntimeError("Oracle trace must preserve every1024 environment without filtering")
                        buffer = io.BytesIO();np.savez_compressed(buffer,**trace)
                        # Leave bounded space for the common schema/file manifest.
                        if oracle_trace_bytes+buffer.tell()+128*1024>10*2**20:
                            raise OSError("Oracle state trace exceeded its10MiB hard budget")
                        path = oracle_trace_dir/f"stage{stage}_control{local_control+1:02d}.npz"
                        with path.open("xb") as stream: stream.write(buffer.getbuffer())
                        oracle_trace_bytes += path.stat().st_size
                        oracle_trace_records.append(dict(path=path.name,stage=stage,local_control=local_control+1,
                            physics_control=global_controls+1,num_envs=1024,sha256=sha(path),bytes=path.stat().st_size))
                        trace_manifest = dict(maximum_bytes=10*2**20,stored_npz_bytes=oracle_trace_bytes,
                            first_controls_per_stage=8,all_environments_unfiltered=True,
                            array_schema={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in trace.items()},
                            files=oracle_trace_records)
                        write(oracle_trace_dir/"manifest.json",trace_manifest)
                        if sum(item.stat().st_size for item in oracle_trace_dir.iterdir())>10*2**20:
                            raise OSError("Oracle trace plus manifest exceeded10MiB")
                    executed = torch.where(teacher_mask[:,None],label["raw"],predicted).clamp(-100.,100.)
                    if not torch.isfinite(executed).all(): raise RuntimeError("Nonfinite executed direct action")
                    parameters = robot.data.soft_joint_pos_limits[:,ids].clone()
                    batch = dict(obs=cpu(raw_obs,"float32"),critic=cpu(raw_critic,"float32"),
                        root_state=cpu(robot.data.root_state_w,"float32"),
                        position=cpu(label["position"],"float32"),equivalent=cpu(label["equivalent"],"float32"),
                        teacher_q=cpu(label["teacher_q"],"float32"),teacher_v=cpu(label["teacher_v"],"float32"),
                        stiffness=cpu(robot.data.joint_stiffness[:,ids],"float32"),damping=cpu(robot.data.joint_damping[:,ids],"float32"),
                        caps=cpu(robot.data.joint_effort_limits[:,ids],"float32"),lower=cpu(parameters[...,0],"float32"),
                        upper=cpu(parameters[...,1],"float32"),raw_label=cpu(label["raw"],"float32"),
                        raw_teacher=cpu(label["teacher_raw"],"float32"),raw_student=cpu(predicted,"float32"),
                        raw_executed=cpu(executed,"float32"),label_error=cpu(label["affine_request_error"],"float32"),
                        clip=cpu(label["clipped_mask"],"bool"),valid_label=cpu(label["valid"],"bool"),
                        env_id=np.arange(1024,dtype=np.int32),episode_id=cpu(episode_id,"int64"),
                        control_step=np.full(1024,global_controls+1,dtype=np.int64),
                        native_control_counter=np.full(1024,raw.common_step_counter,dtype=np.int64),
                        sample_uid=np.arange(1024,dtype=np.int64)+global_controls*1024,
                        phase=cpu(label["state"],"int16"),actor_source=cpu(teacher_mask,"uint8"))
                    counts["labels"]+=1024;counts["invalid"]+=int((~batch["valid_label"]).sum());counts["clip"]+=batch["clip"].sum(0)
                    counts["teacher_controls"]+=int(teacher_mask.sum())
                    dataset.add(stage,batch)
                    pending.fill_(True);actual_published.clear()
                    next_obs,_,dones,extras = wrapped.step(executed)
                    if "raw" not in actual_published or not torch.equal(actual_published["raw"],executed):
                        raise RuntimeError("Executed ActionManager history differs from the selected published action")
                    physical_finish(env_ids)
                    global_controls+=1;obs=next_obs
                if global_controls%50==0:
                    point=dict(physics_controls=global_controls,actor_updates=actor_updates,round=stage-1,
                        phase="admission" if stage==0 else "collecting",first_episode_alive=int((first_alive&~first_invalid).sum()),
                        heldout_first_episode_alive=int((first_alive&~first_invalid&holdout).sum()),gpu=gpu_budget())
                    print("[H2 INIT COLLECT] "+json.dumps(point),flush=True);update(**point)
            collection_enabled=False
            if actor_digest()!=collection_actor_sha:
                raise RuntimeError("Student weights changed inside a frozen physical collection block")
            dataset.close_stage(stage)
            summary=dict(stage=stage,controls=controls,teacher_probability=probability,
                valid_first_episode_alive=int((first_alive&~first_invalid).sum()),
                heldout_valid_first_episode_alive=int((first_alive&~first_invalid&holdout).sum()),
                first_episode_duration_s=cpu(first_duration).tolist(),
                first_episode_forward_m=cpu(endpoint[:,0]-first_root_start[:,0]).tolist(),
                labels=counts["labels"],oracle_invalid_fraction=counts["invalid"]/counts["labels"],
                per_joint_clip_fraction=(counts["clip"]/counts["labels"]).tolist(),
                collection_actor_sha256=collection_actor_sha,
                actual_teacher_control_fraction=counts["teacher_controls"]/counts["labels"],
                oracle_trace=str(oracle_trace_dir/"manifest.json"),oracle_trace_npz_bytes=oracle_trace_bytes,
                round_boundary_semantics="Still-live episodes censored; reset boundary is not success/timeout")
            stage_summaries.append(summary);write(args.output/f"collection_stage_{stage}.json",summary)
            if stage==0:
                protocol=(summary["oracle_invalid_fraction"]<=.05 and max(summary["per_joint_clip_fraction"])<=.05)
                if summary["valid_first_episode_alive"]<922 or not protocol:
                    manifest=dataset.finalize("rejected_interface")
                    update(status="rejected_interface",phase="finished",physics_controls=global_controls,actor_updates=0,
                        admission=summary,admission_protocol_passed=protocol,dataset_manifest=str(args.output/"dataset/manifest.json"),
                        finished_at_epoch=time.time())
                    write(args.output/"result.json",{**metadata,"initialization_ready":False,"candidate":None,
                          "reason":"Teacher-equivalent direct-position admission failed; no fitting or automatic retry"})
                    return
                continue
            update(status="fitting",phase="fitting",round=stage-1,physics_controls=global_controls,actor_updates=actor_updates)
            fit_counters = (int(raw.common_step_counter),int(raw._sim_step_counter))
            student.train()
            for _ in range(800):
                batch = dataset.training_batch(stage-1)
                tensors={name:torch.from_numpy(value).pin_memory().to(device,non_blocking=True) for name,value in batch.items()}
                with torch.no_grad(): x=snorm(tensors["obs"])
                lo,hi=tensors["lower"],tensors["upper"]
                predicted=(lo+hi)*.5+(hi-lo)*.5*torch.tanh(student.act_inference(x))
                loss=((predicted-tensors["position"])*tensors["stiffness"]/tensors["caps"]).square().mean()
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite direct-imitation loss")
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(student.actor.parameters(),1.)
                if not torch.isfinite(norm): raise RuntimeError("Nonfinite direct-imitation actor gradient")
                optimizer.step();actor_updates+=1
                if actor_updates%100==0:
                    point=dict(phase="fitting",round=stage-1,physics_controls=global_controls,actor_updates=actor_updates,
                               supervised_loss=float(loss),actor_gradient_norm_before_clip=float(norm),gpu=gpu_budget())
                    print("[H2 INIT COLLECT] "+json.dumps(point),flush=True);update(**point)
            student.eval()
            report=quality();write(args.output/f"quality_round_{stage-1}.json",report)
            if fit_counters != (int(raw.common_step_counter),int(raw._sim_step_counter)):
                raise RuntimeError("Physical/reference collection advanced during the fitting-only block")
            candidate,candidate_sha=export(report)
            dataset.progress()
            update(candidate=candidate,candidate_sha256=candidate_sha,latest_quality=report)
        if global_controls!=1750 or actor_updates!=3200: raise RuntimeError("Approved collection/fit budget drift")
        manifest=dataset.finalize("completed")
        if source_hashes(source)!=original_hashes or source_hashes(args.source_root)!=original_hashes:
            raise RuntimeError("Frozen initialization source changed")
        if any(sha(path)!=digest for path,digest in metadata["external_dependency_sha256"].items()):
            raise RuntimeError("Initialization asset or frozen LiDAR changed")
        if sha(args.initial_policy)!=INITIAL_SHA or sha(args.teacher_checkpoint)!=TEACHER_SHA:
            raise RuntimeError("Initialization input weights changed")
        update(status="completed",phase="finished",physics_controls=1750,actor_updates=3200,round=3,
               stage_summaries=stage_summaries,termination_counts=termination_counts,
               dataset_manifest=str(args.output/"dataset/manifest.json"),finished_at_epoch=time.time(),
               candidate=candidate,candidate_sha256=candidate_sha,
               limitation="Regression/collection completion is not autonomous walking or stairs acceptance; separate frozen20s validation required")
        write(args.output/"result.json",metadata)
    except BaseException as error:
        collection_enabled=False
        metadata.update(status="failed",phase="failed",error=type(error).__name__+": "+str(error),
                        finished_at_epoch=time.time(),initialization_ready=False,
                        physics_controls=int(locals().get("global_controls",metadata["physics_controls"])),
                        actor_updates=int(locals().get("actor_updates",metadata["actor_updates"])))
        (args.output/"failure_traceback.txt").write_text(traceback.format_exc())
        if dataset is not None:
            try: dataset.finalize("failed")
            except Exception as secondary: metadata["dataset_finalize_error"]=repr(secondary)
        write(args.output/"result.json",metadata)
        update(**metadata)
        raise
    finally:
        if wrapped is not None: wrapped.close()
        if app is not None: app.close()


if __name__=="__main__":
    main()
