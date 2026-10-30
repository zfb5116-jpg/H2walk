"""Frozen student-only direct-flat initialization validation.

CPU validation certifies saved state integrity only. Physical readiness needs
at least922/1024 same first20s episodes with4.8..7.2m world-X progress and two
original valid alternating landings per foot. This is not stair acceptance.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TASK = "Unitree-H2-Direct-Flat-Initialization"
PACKAGE = Path("CtrlLab/CtrlLab/tasks/manager_based/h2_task")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_hashes(source):
    return {"source/"+str(path.relative_to(source)): sha(path) for path in sorted(source.rglob("*"))
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise RuntimeError(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=True)
    parser.add_argument("--cpu-check-only", action="store_true")
    parser.add_argument("--validation_output", type=Path)
    if "--cpu-check-only" in sys.argv:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cpu")
        AppLauncher = None
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (args.num_envs, args.duration, args.speed, args.seed) != (1024, 20.0, .3, 42):
        parser.error("Initialization validation is fixed at1024 first20s episodes,0.3m/s,seed42")
    if not args.cpu_check_only and args.output is None:
        parser.error("Physical validation requires a unique --output directory")
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.source_root = args.source_root.resolve(strict=True)
    args.validation_output = (args.validation_output.resolve() if args.validation_output else
                              args.checkpoint.with_suffix(".validation.json"))
    if args.output is not None:
        args.output = args.output.resolve()
    args.enable_cameras = not args.cpu_check_only
    return args, AppLauncher


def cpu_imitation_budget(optim, refit_kind=None):
    """Two declared budgets only; the refit also needs external provenance."""
    if refit_kind is None:
        expected = (3200, 1024)
    elif refit_kind == "offline_direct_imitation_refit_v1":
        expected = (6400, 4096)
    else:
        raise RuntimeError("Unrecognized imitation producer protocol")
    require((optim.get("updates"), optim.get("batch_size")) == expected,
            "Incomplete or undeclared bounded imitation update budget")
    return expected


def cpu_refit_provenance(args, saved, optim):
    """Bind a refit to the reviewed immutable plan and completed producer.

    The kind/update numbers in a checkpoint alone never authorize the6400
    branch. The authorized plan, request/guard and exact parent bytes must
    independently agree before any parent state is loaded for comparison.
    """
    refit = saved["metadata"].get("refit_protocol")
    if refit is None:
        require(optim.get("refit_protocol") is None, "Optimizer claims refit without candidate provenance")
        cpu_imitation_budget(optim)
        return None
    kind = "offline_direct_imitation_refit_v1"
    require(refit.get("kind") == kind and optim.get("refit_protocol") == refit,
            "Candidate/optimizer refit provenance mismatch")
    cpu_imitation_budget(optim, kind)
    plan_path = ROOT/"logs/stairs_work/fixed_dataset_calibration_20261031_060313/plan.json"
    plan_sha = "52f0ad5e35bd7e03852163291eac43d5dea52c41dff090eaf037bea36094c1f3"
    require(Path(refit["plan_path"]).resolve(strict=True) == plan_path and
            sha(plan_path) == refit["plan_sha256"] == plan_sha,
            "Refit is not bound to the immutable authorized calibration plan")
    plan = json.loads(plan_path.read_text())
    protocol = plan["protocol"]
    require(protocol["kind"] == kind and
            (protocol["parent_actor_updates"], protocol["additional_actor_updates"],
             protocol["cumulative_actor_updates"], protocol["batch_size"], protocol["physics_controls"]) ==
            (3200, 3200, 6400, 4096, 0), "Authorized refit plan budget changed")
    case = Path(refit["producer_case"]).resolve(strict=True)
    require(case.is_relative_to(ROOT/"logs/stair_learning") and args.checkpoint.parent == case/"output",
            "Refit producer/output identity mismatch")
    request_path = Path(refit["request_path"]).resolve(strict=True)
    require(request_path == case/"request.json" and sha(request_path) == refit["request_sha256"],
            "Refit immutable producer request changed")
    request = json.loads(request_path.read_text())
    guard = json.loads((case/"launch.lock").read_text())
    require(guard["request_sha256"] == refit["request_sha256"], "Refit request/launch guard mismatch")
    require(request["case"] == str(case) and request["output"] == str(case/"output") and
            request["protocol"] == protocol and request["plan_sha256"] == plan_sha and
            Path(request["plan_path"]).resolve(strict=True) == plan_path,
            "Refit request does not declare the authorized protocol")
    require(Path(request["source_root"]).resolve(strict=True) == Path(plan["source_root"]).resolve(strict=True),
            "Refit request uses another source")
    require(Path(refit["parent_collection"]).resolve(strict=True) == Path(plan["parent_collection"]).resolve(strict=True),
            "Refit parent collection changed")
    for name in ("parent_candidate", "parent_optimizer"):
        path = Path(plan[name]).resolve(strict=True)
        expected = plan[name+"_sha256"]
        require(Path(refit[name]).resolve(strict=True) == path and refit[name+"_sha256"] == expected and
                Path(request[name]).resolve(strict=True) == path and request[name+"_sha256"] == expected and
                sha(path) == expected, "Refit parent bytes/provenance changed: "+name)
    manifest = Path(plan["dataset_manifest"]).resolve(strict=True)
    require(Path(refit["frozen_dataset_manifest_path"]).resolve(strict=True) == manifest and
            refit["frozen_dataset_manifest_sha256"] == sha(manifest), "Refit frozen dataset manifest changed")
    for path in (plan_path, Path(plan["parent_candidate"]), Path(plan["parent_optimizer"]), manifest):
        require(request["frozen_files"].get(str(path)) == sha(path), "Request omits a fixed refit input binding")
    if "duplicate_guard" in request:
        require(json.loads(Path(request["duplicate_guard"]).read_text())["request_sha256"] == refit["request_sha256"],
                "Independent refit deduplication guard changed")
    manager = json.loads((case/"manager_state.json").read_text())
    exit_record = json.loads((case/"exit.json").read_text())
    result_path = case/"output/result.json"
    result = json.loads(result_path.read_text())
    require(manager.get("status") == "finished" and manager.get("exit_code") == 0 and
            exit_record.get("wrapper_exit_code") == 0 and not manager.get("error"),
            "Refit producer has not confirmed successful process completion")
    require(result.get("status") == "completed" and result.get("physics_controls") == 0 and
            result.get("simulation_started") is False and result.get("actor_updates") == 3200 and
            result.get("cumulative_actor_updates") == 6400,
            "Refit producer did not complete the fixed offline-only budget")
    require(Path(result["candidate"]).resolve(strict=True) == args.checkpoint and
            result["candidate_sha256"] == sha(args.checkpoint), "Refit result does not bind these candidate bytes")
    require(saved["metadata"].get("actor_updates") == 6400 and
            saved["metadata"].get("additional_actor_updates") == 3200 and
            saved["metadata"].get("cumulative_actor_updates") == 6400 and
            saved["metadata"].get("physics_controls") == 0,
            "Candidate refit metadata budget mismatch")
    for name, expected in (("parent_actor_updates", 3200), ("additional_actor_updates", 3200),
                           ("cumulative_actor_updates", 6400), ("batch_size", 4096), ("physics_controls", 0)):
        require(refit.get(name) == expected, "Refit provenance has an undeclared budget: "+name)
    quality = result.get("quality_gate_passed")
    require(type(quality) is bool and saved["metadata"].get("quality_gate_passed") is quality,
            "Producer/candidate quality-gate record disagrees or is absent")
    require(all(math.isclose(float(group["lr"]), 1e-5, rel_tol=1e-7, abs_tol=1e-12)
                for group in optim["optimizer_state_dict"]["param_groups"]),
            "Refit final Adam learning rate differs from the authorized schedule")
    return dict(kind=kind, plan_path=str(plan_path), plan_sha256=plan_sha,
        request_path=str(request_path), request_sha256=refit["request_sha256"], producer_case=str(case),
        producer_result_path=str(result_path), producer_result_sha256=sha(result_path), producer_completed=True,
        quality_gate_passed=quality, parent_candidate=plan["parent_candidate"],
        parent_candidate_sha256=plan["parent_candidate_sha256"], parent_optimizer=plan["parent_optimizer"],
        parent_optimizer_sha256=plan["parent_optimizer_sha256"],
        dataset_manifest_path=str(manifest), dataset_manifest_sha256=sha(manifest),
        parent_actor_updates=3200, additional_actor_updates=3200, cumulative_actor_updates=6400,
        physics_controls=0, batch_size=4096)


def validate_cpu(args):
    # The CPU path imports no Isaac/SimulationApp and hides accelerators before
    # importing Torch. It never instantiates a PPO optimizer or RND learner.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    receipt = dict(status="failed", cpu_only=True, initialization_ready=False,
        formal_acceptance_passed=False, checkpoint=str(args.checkpoint), sha256=sha(args.checkpoint),
        source_root=str(args.source_root), validator_sha256=sha(Path(__file__)),
        no_simulation=True, training_updates=0)
    tensors = 0

    def finite(value):
        nonlocal tensors
        if isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all()), "Non-finite saved tensor")
            tensors += 1
        elif isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value: finite(item)
        elif isinstance(value, float):
            require(math.isfinite(value), "Non-finite saved scalar")

    require(saved.get("format") == "h2_direct_policy_initialization_v1", "Not an initialization package")
    require(saved.get("ppo_completed_iterations") == 0, "Initialization package must have zero PPO iterations")
    require(saved.get("action_semantics") == "full_soft_tanh_position_v1", "Wrong action semantics")
    require(saved.get("policy_dimensions") == [1029, 866, 73, 12], "Network layout changed")
    require(Path(saved["source_root"]).resolve(strict=True) == args.source_root, "Candidate/source binding mismatch")
    hashes = source_hashes(args.source_root)
    require(saved["source_sha256"] == hashes, "Candidate source fingerprint changed")
    require(json.loads((args.source_root.parent/"main_source_sha256.json").read_text()) == hashes,
            "Producer training_source manifest mismatch")
    require(saved.get("initialization_ready") is False, "A CPU-unvalidated candidate must not claim physical readiness")
    dependencies = saved["external_dependency_sha256"]
    require(bool(dependencies), "Candidate omits external dependencies")
    for path, digest in dependencies.items():
        require(sha(Path(path).resolve(strict=True)) == digest, "External dependency fingerprint changed")
    cfg = yaml.safe_load((args.checkpoint.parent/"params/agent.yaml").read_text())
    policy_cfg = dict(cfg["policy"])
    require(policy_cfg.pop("class_name") == "H2ActorCriticOur", "Network class changed")
    actor_class = load_module(args.source_root/PACKAGE/"networks/actor_critic_our.py", "validation_cpu_actor").H2ActorCriticOur
    actor = actor_class(1029, 866, 12, **policy_cfg).eval()
    actor.load_state_dict(saved["model_state_dict"], strict=True)
    for key, width in (("obs_norm_state_dict", 1029), ("privileged_obs_norm_state_dict", 866)):
        norm = saved[key]
        require(set(norm) == {"_mean", "_var", "_std", "count"}, "Normalizer state layout mismatch")
        require(all(tuple(norm[name].shape) == (1, width) for name in ("_mean", "_var", "_std")),
                "Normalizer dimensions mismatch")
        require(norm["count"].numel() == 1 and int(norm["count"]) > 0 and bool((norm["_std"] >= 0).all()),
                "Unpopulated or invalid saved normalization")
    finite({key:saved[key] for key in ("model_state_dict", "obs_norm_state_dict", "privileged_obs_norm_state_dict")})
    optimizer_path = Path(saved["metadata"]["optimizer_path"])
    optimizer_path = (args.checkpoint.parent/optimizer_path).resolve(strict=True) if not optimizer_path.is_absolute() else optimizer_path.resolve(strict=True)
    require(sha(optimizer_path) == saved["metadata"]["optimizer_sha256"], "Imitation optimizer fingerprint mismatch")
    optim = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    require(optim["format"] == "h2_direct_imitation_optimizer_v1", "Wrong optimizer artifact format")
    provenance = cpu_refit_provenance(args, saved, optim)
    require(optim.get("PPO_updates", optim.get("ppo_updates")) == 0
            and optim.get("RND_updates", optim.get("rnd_updates")) == 0, "Optimizer artifact claims PPO/RND updates")
    parameters = list(actor.actor.named_parameters())
    require(optim["actor_parameter_names"] == [name for name, _ in parameters], "Imitation optimizer actor order mismatch")
    state = optim["optimizer_state_dict"]
    order = [pid for group in state["param_groups"] for pid in group["params"]]
    require(len(order) == len(parameters) and len(set(order)) == len(order), "Optimizer parameter groups are incomplete")
    require(set(state["state"]) == set(order), "Imitation Adam state omits actor parameters")
    for pid, (_, parameter) in zip(order, parameters):
        item = state["state"][pid]
        require({"step", "exp_avg", "exp_avg_sq"}.issubset(item), "Adam moments/step missing")
        require(item["step"].numel() == 1 and int(item["step"]) == optim["updates"], "Adam step count mismatch")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in item:
                require(item[name].shape == parameter.shape, "Adam moment shape mismatch")
    finite(optim)
    if provenance is not None:
        require(len(parameters) == 44, "Refit changed the original44 actor-parameter layout")
        parent = torch.load(provenance["parent_candidate"], map_location="cpu", weights_only=True)
        parent_optim = torch.load(provenance["parent_optimizer"], map_location="cpu", weights_only=True)
        require(parent.get("format") == "h2_direct_policy_initialization_v1" and
                parent.get("ppo_completed_iterations") == 0 and parent.get("policy_dimensions") == [1029,866,73,12] and
                parent.get("action_semantics") == saved["action_semantics"], "Parent initialization package mismatch")
        require(parent["source_sha256"] == saved["source_sha256"] and parent["source_commit"] == saved["source_commit"] and
                parent["external_dependency_sha256"] == dependencies, "Refit changed model source or dependencies")
        require(parent["model_state_dict"].keys() == saved["model_state_dict"].keys(), "Refit changed model tensor layout")
        protected = [name for name in saved["model_state_dict"] if not name.startswith("actor.")]
        require(all(torch.equal(saved["model_state_dict"][name], parent["model_state_dict"][name]) for name in protected),
                "Refit changed the frozen critic/std or another non-actor tensor")
        for key in ("obs_norm_state_dict", "privileged_obs_norm_state_dict"):
            require(saved[key].keys() == parent[key].keys() and
                    all(torch.equal(value, parent[key][name]) for name,value in saved[key].items()),
                    "Refit changed a saved normalization tensor")
        require(parent_optim.get("format") == "h2_direct_imitation_optimizer_v1", "Parent optimizer format mismatch")
        cpu_imitation_budget(parent_optim)
        require(parent_optim["actor_parameter_names"] == optim["actor_parameter_names"], "Parent/refit actor order changed")
        parent_state = parent_optim["optimizer_state_dict"]
        parent_order = [pid for group in parent_state["param_groups"] for pid in group["params"]]
        require(len(parent_order) == len(parameters) and set(parent_state["state"]) == set(parent_order),
                "Parent Adam state is incomplete")
        for pid,(_,parameter) in zip(parent_order,parameters):
            item = parent_state["state"][pid]
            require({"step","exp_avg","exp_avg_sq"}.issubset(item) and item["step"].numel() == 1 and
                    int(item["step"]) == 3200 and item["exp_avg"].shape == item["exp_avg_sq"].shape == parameter.shape,
                    "Parent Adam3200 moments/steps are incomplete")
        require(len(parent_state["param_groups"]) == len(state["param_groups"]), "Refit Adam groups changed")
        for old,new in zip(parent_state["param_groups"],state["param_groups"]):
            require({key:value for key,value in old.items() if key not in ("params","lr")} ==
                    {key:value for key,value in new.items() if key not in ("params","lr")},
                    "Refit changed an undeclared Adam hyperparameter")
        finite({key:parent[key] for key in ("model_state_dict","obs_norm_state_dict","privileged_obs_norm_state_dict")})
        finite(parent_optim)
        provenance.update(non_actor_and_both_normalizers_bitwise_unchanged=True,
            actor_parameters_allowed_to_change=len(parameters), actual_changed_actor_tensors=sum(
                not torch.equal(value,parent["model_state_dict"][name])
                for name,value in saved["model_state_dict"].items() if name.startswith("actor.")),
            real_parent_Adam_step=3200,real_candidate_Adam_step=6400)
    require(not torch.cuda.is_initialized(), "CPU validation initialized CUDA")
    receipt.update(status="passed", finite_tensors=tensors, source_sha256=hashes,
        external_dependency_sha256=dependencies, optimizer_path=str(optimizer_path),
        optimizer_sha256=sha(optimizer_path), actor_optimizer_parameters=len(parameters), imitation_updates=optim["updates"],
        source_commit=saved.get("source_commit"), torch_threads=torch.get_num_threads(), cuda_initialized=False,
        imitation_batch_size=optim["batch_size"], producer_kind=(provenance["kind"] if provenance else "bounded_direct_imitation_v1"),
        refit_provenance=provenance, producer_completed=(provenance["producer_completed"] if provenance else None),
        quality_gate_passed=(provenance["quality_gate_passed"] if provenance else
            saved["metadata"].get("quality",{}).get("regression_quality_passed")),
        interpretation="CPU model/dualnorm/imitation-Adam integrity only; not physical initialization readiness")
    write(args.validation_output, receipt)
    print("[DIRECT_INIT_VALIDATION] CPU "+json.dumps(receipt), flush=True)


def prepare_physical(args):
    receipt = json.loads(args.validation_output.read_text())
    require(receipt.get("status") == "passed" and receipt.get("cpu_only") is True,
            "Run --cpu-check-only successfully before physical validation")
    require(receipt["sha256"] == sha(args.checkpoint) and receipt["source_root"] == str(args.source_root),
            "Candidate CPU receipt binding changed")
    require(receipt["source_sha256"] == source_hashes(args.source_root), "Source changed after CPU validation")
    require(receipt["optimizer_sha256"] == sha(Path(receipt["optimizer_path"])), "Imitation optimizer changed after CPU validation")
    for path, digest in receipt["external_dependency_sha256"].items():
        require(sha(Path(path)) == digest, "External dependency changed after CPU validation")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source_root.parent, args.output/"training_source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source = args.output/"training_source/source"
    require(source_hashes(source) == receipt["source_sha256"], "Validation source copy is incomplete")
    shutil.copy2(args.checkpoint.parent/"params/agent.yaml", args.output/"agent.yaml")
    shutil.copy2(args.validation_output, args.output/"candidate.validation.json")
    (args.output/"tools_snapshot").mkdir()
    for name in ("evaluate_direct_initialization.py", "evaluate_direct_initialization.sh"):
        shutil.copy2(Path(__file__).parent/name, args.output/"tools_snapshot"/name)
    dependencies = receipt["external_dependency_sha256"]
    usd = [path for path in dependencies if Path(path).suffix == ".usd"]
    recon = [path for path in dependencies if "reconstructor" in Path(path).name and Path(path).suffix == ".pt"]
    require(len(usd) == len(recon) == 1, "Need unambiguous recorded USD/LiDAR dependencies")
    os.environ["CTRLLAB_H2_USD_PATH"] = usd[0]
    os.environ["CTRLLAB_H2_LIDAR_RECON_CHECKPOINT"] = recon[0]
    return source, receipt


def evaluate(args, app, source, receipt, metadata):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import yaml
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    sys.path.insert(0, str(source/"CtrlLab/CtrlLab/tasks/manager_based"))
    import h2_task
    from h2_task.direct_leg_action import DirectLegPositionAction, soft_range_targets
    from h2_task.locomotion_progress import FootfallTracker
    from h2_task.networks import H2ActorCriticOur
    require(Path(h2_task.__file__).resolve().is_relative_to(source), "Imported task is not frozen candidate source")
    original_tracker = ROOT/"source"/PACKAGE/"locomotion_progress.py"
    tracker_path = source/PACKAGE/"locomotion_progress.py"
    def class_hash(path):
        node = next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name == "FootfallTracker")
        return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    require(class_hash(tracker_path) == class_hash(original_tracker), "Original independent FootfallTracker changed")
    metadata["footfall_tracker_ast_sha256"] = class_hash(tracker_path)
    spec = gym.spec(TASK); module,name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module), name)()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device or "cuda:0"
    cfg.io_descriptors_output_dir = str(args.output)
    require(cfg.episode_length_s == 20 and not cfg.observations.policy.enable_corruption, "Candidate direct-flat config changed")
    command = cfg.commands.base_velocity
    for ranges in (command.ranges, command.limit_ranges):
        require(ranges.lin_vel_x == (.3,.3) and ranges.lin_vel_y == ranges.ang_vel_z == (0.,0.), "Forward command contract changed")
    cfg.viewer.origin_type, cfg.viewer.asset_name, cfg.viewer.env_index = "asset_root", "robot", 0
    cfg.viewer.eye, cfg.viewer.lookat, cfg.viewer.resolution = (-2.5,3.8,1.8),(0.8,0.,.6),(768,576)
    dump_yaml(str(args.output/"environment.yaml"), cfg)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    agent = yaml.safe_load((args.output/"agent.yaml").read_text())
    env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    video = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions",100.))
        raw, n = env.unwrapped, args.num_envs
        robot, device = raw.scene["robot"], raw.device
        term = raw.action_manager.get_term("JointPositionAction")
        require(type(term) is DirectLegPositionAction and term.gait_reference is None, "Reference/teacher action path present")
        require(not raw.action_manager._locked_action_ids and raw.action_manager.total_action_dim == 12, "Direct actor authority changed")
        qids, held = list(term._joint_ids), list(raw.action_manager.held_joint_ids)
        require(len(held) == 17, "Upper-body default hold changed")
        require(abs(raw.step_dt-.02)<1e-9 and abs(raw.physics_dt-.005)<1e-9, "Original control/physics intervals changed")
        view = robot.root_physx_view
        native = (view.get_dof_stiffnesses().to(device), view.get_dof_dampings().to(device), view.get_dof_max_forces().to(device))
        covered=set(); actuator_certificate={}
        for name,actuator in robot.actuators.items():
            indices=(list(range(robot.num_joints))[actuator.joint_indices] if isinstance(actuator.joint_indices,slice)
                     else [int(i) for i in actuator.joint_indices])
            controls_leg=bool(set(indices).intersection(qids))
            errors=None
            if controls_leg:
                require(actuator.is_implicit_model and set(indices).issubset(qids), "Direct leg native implicit groups changed")
                covered.update(indices)
                errors=[float((actual[:,indices]-expected).abs().max()) for actual,expected in
                        zip(native,(actuator.stiffness,actuator.damping,actuator.effort_limit))]
                require(max(errors)<1e-3, "Actual native leg K/D/cap disagrees with actuator")
            require(getattr(actuator,"support_priority_buffer",None) is None, "Unexpected support force-priority attachment")
            actuator_certificate[name]=dict(class_name=type(actuator).__name__,native_implicit=bool(actuator.is_implicit_model),
                policy_leg_group=controls_leg,readback_errors=errors)
        require(covered==set(qids), "Missing native leg actuator coverage")
        metadata["native_actuator_certificate"]=actuator_certificate
        obs, extras = wrapped.get_observations()
        require(tuple(obs.shape)==(n,1029) and tuple(extras["observations"]["critic"].shape)==(n,866), "Observation layout mismatch")
        require(bool((raw.episode_length_buf==0).all()) and bool((raw.action_manager.action==0).all()), "First episodes do not start at actual reset")
        pcfg = dict(agent["policy"]);pcfg.pop("class_name")
        actor = H2ActorCriticOur(1029,866,12,**pcfg).to(device)
        norm = EmpiricalNormalization([1029],until=1e8).to(device)
        cnorm = EmpiricalNormalization([866],until=1e8).to(device)
        for obj,key in ((actor,"model_state_dict"),(norm,"obs_norm_state_dict"),(cnorm,"privileged_obs_norm_state_dict")):
            obj.load_state_dict(saved[key],strict=True);obj.eval().requires_grad_(False)
        torch.save({key:saved[key] for key in ("obs_norm_state_dict","privileged_obs_norm_state_dict")}, args.output/"saved_observation_normalizers.pt")
        origins = raw.scene.terrain.env_origins.clone(); start = robot.data.root_pos_w.clone()
        sensors = raw._stair_sensors;raw_sensor = raw.scene["contact_forces"]
        alive = torch.ones(n,device=device,dtype=torch.bool); failed=torch.zeros_like(alive);timeouts=torch.zeros_like(alive)
        counts=torch.zeros(n,device=device,dtype=torch.long);epochs=torch.full_like(counts,-1)
        reasons={key:torch.zeros_like(alive) for key in raw.termination_manager.active_terms}
        footfalls=FootfallTracker(n,device);positive=torch.zeros(n,2,device=device,dtype=torch.bool)
        mismatch=torch.zeros(n,2,device=device,dtype=torch.long)
        last_policy=obs.clone();last_critic=extras["observations"]["critic"].clone()
        trajectory=[];events=[]

        def state():
            soles,rot=raw._stair_foot_pose();forces=[]
            for sensor in sensors:
                matrix=sensor.data.force_matrix_w
                require(matrix is not None and tuple(matrix.shape)==(n,1,1,3), "Missing actual ground-filtered foot normal force")
                require(sensor.cfg.filter_prim_paths_expr==["/World/ground/terrain/mesh"], "Ground filter changed")
                forces.append(matrix[:,0,0].clone())
            return dict(root_position=robot.data.root_pos_w.clone(),root_quaternion=robot.data.root_quat_w.clone(),
                root_link_velocity=robot.data.root_link_lin_vel_w.clone(),root_tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)),
                sole_position=soles.clone(),foot_rotation=rot.clone(),filtered_normal_force=torch.stack(forces,1),
                unfiltered_foot_normal_force=raw_sensor.data.net_forces_w[:,raw._stair_unfiltered_ids].clone(),
                joint_position=robot.data.joint_pos.clone(),joint_velocity=robot.data.joint_vel.clone(),
                joint_target=robot.data.joint_pos_target.clone(),raw_action=raw.action_manager.action.clone(),
                control_step=torch.full((n,),int(raw.common_step_counter),device=device,dtype=torch.long))
        endpoint=state()

        def capture(ids):
            ids=ids[alive[ids]&(epochs[ids]!=int(raw.common_step_counter))]
            if not ids.numel():return
            now=state()
            require(all(bool(torch.isfinite(value[ids]).all()) for value in now.values()), "Non-finite first-episode physical state")
            expected=soft_range_targets(now["raw_action"],robot.data.soft_joint_pos_limits[:,qids])
            require(float((now["joint_target"][ids][:,qids]-expected[ids]).abs().max())<2e-6, "Direct position output overridden")
            require(not bool((robot.data.joint_vel_target[ids][:,qids]!=0).any()) and
                    not bool((robot.data.joint_effort_target[ids][:,qids]!=0).any()), "Native direct leg v/E port changed")
            require(float((now["joint_target"][ids][:,held]-robot.data.default_joint_pos[ids][:,held]).abs().max())<2e-6,
                    "Original upper-body default target changed")
            force,unfiltered=now["filtered_normal_force"],now["unfiltered_foot_normal_force"]
            agree=(force-unfiltered).norm(dim=-1)<=1+.02*unfiltered.norm(dim=-1)
            relevant=(force.norm(dim=-1)>5)|(unfiltered.norm(dim=-1)>5)
            mismatch[ids]+=(~agree[ids]&relevant[ids]).long()
            positive[ids]|=(agree&(force[...,2]>5)&(unfiltered[...,2]>5))[ids]
            contact=force.norm(dim=-1)>5
            oldair=footfalls.air_time[ids].clone();oldcounts=footfalls.landings[ids].clone()
            landed=contact[ids]&~footfalls.previous_contact[ids]
            footfalls.update(contact,raw.step_dt,ids)
            for row in landed.nonzero():
                local,side=map(int,row);eid=int(ids[local])
                events.append(dict(env_id=eid,foot=side,time_s=float((counts[eid]+1)*raw.step_dt),
                    air_s=float(oldair[local,side]),valid_alternating=bool(footfalls.landings[eid,side]>oldcounts[local,side]),
                    force_agreement=bool(agree[eid,side])))
            for key in endpoint:endpoint[key][ids]=now[key][ids]
            counts[ids]+=1;epochs[ids]=int(raw.common_step_counter)

        def before_reset(ids):
            ids=torch.as_tensor(ids,device=device,dtype=torch.long);ids=ids[alive[ids]]
            if ids.numel():
                capture(ids);failed[ids]=raw.termination_manager.terminated[ids];timeouts[ids]=raw.termination_manager.time_outs[ids]
                for name,bits in reasons.items():bits[ids]=raw.termination_manager.get_term(name)[ids]
                alive[ids]=False
        raw.add_reset_callback(before_reset,when="before")
        from pxr import UsdGeom
        for eid in range(1,n):UsdGeom.Imageable(raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{eid}")).MakeInvisible()
        video=imageio.get_writer(str(args.output/"env0.mp4"),fps=10,codec="libx264",quality=7)
        metadata.update(status="evaluating",step_dt_s=raw.step_dt,physics_dt_s=raw.physics_dt,
            domain_bounds_local=list(raw.domain_bounds_local),below_floor_tolerance_m=raw._stair_below_floor_tolerance,
            native_term_names=list(reasons),force_semantics="normal-force vectors only; no frictional resultant/motor claim")
        write(args.output/"run.json",metadata)
        view=robot.root_physx_view
        physical=dict(mass=view.get_masses().cpu().numpy(),native_K=view.get_dof_stiffnesses().cpu().numpy(),
            native_D=view.get_dof_dampings().cpu().numpy(),native_cap=view.get_dof_max_forces().cpu().numpy(),
            native_velocity_cap=view.get_dof_max_velocities().cpu().numpy(),armature=view.get_dof_armatures().cpu().numpy(),
            friction=view.get_dof_friction_properties().cpu().numpy(),materials=view.get_material_properties().cpu().numpy(),
            soft_limits=robot.data.soft_joint_pos_limits.cpu().numpy(),joint_names=np.asarray(robot.joint_names),
            initial_root=robot.data.root_state_w.cpu().numpy(),initial_q=robot.data.joint_pos.cpu().numpy(),origins=origins.cpu().numpy())
        for name,actuator in robot.actuators.items():physical[name+"_delay"]=actuator.positions_delay_buffer.time_lags.cpu().numpy()
        np.savez_compressed(args.output/"runtime_physical_parameters.npz",**physical)
        steps=0;error=None;maxsteps=round(args.duration/raw.step_dt)
        try:
            for k in range(maxsteps):
                require(app.is_running(), "Simulator closed early")
                with torch.inference_mode():
                    last_policy[alive]=obs[alive];last_critic[alive]=extras["observations"]["critic"][alive]
                    action=actor.act_inference(norm(obs))
                    require(bool(torch.isfinite(action).all()), "Non-finite frozen action")
                    obs,_,_,extras=wrapped.step(action);capture(alive.nonzero().flatten())
                steps=k+1
                if steps%5==0 or steps==maxsteps or not bool(alive.any()):
                    frame={key:value.cpu().numpy().copy() for key,value in endpoint.items()}
                    frame.update(time_s=np.array(steps*raw.step_dt),first_episode_active=alive.cpu().numpy(),first_episode_time_s=(counts*raw.step_dt).cpu().numpy())
                    trajectory.append(frame)
                    image=Image.fromarray(raw.render());draw=ImageDraw.Draw(image)
                    draw.rectangle((0,0,image.width,42),fill="black")
                    draw.text((8,4),f"Frozen direct student | t={steps*raw.step_dt:.2f}s first active {int(alive.sum())}/{n}",fill="white")
                    draw.text((8,22),"env0 "+("first episode" if bool(alive[0]) else "reset visible; excluded"),fill="white")
                    video.append_data(np.asarray(image))
                if steps%100==0 or not bool(alive.any()):
                    print(f"[DIRECT_INIT_VALIDATION] t={steps*raw.step_dt:.2f}s first_active={int(alive.sum())}/{n}",flush=True)
                if not bool(alive.any()):break
        except BaseException as exc:error=exc
        duration=counts*raw.step_dt;survived=(counts>=maxsteps)&~failed
        dx=endpoint["root_position"][:,0]-start[:,0];progress=(dx>=4.8)&(dx<=7.2)
        bilateral=(footfalls.landings>=2).all(-1);forcevalid=positive.all(-1)&(mismatch.sum(-1)==0)
        success=survived&progress&bilateral&forcevalid
        immutable=True
        for obj,key in ((actor,"model_state_dict"),(norm,"obs_norm_state_dict"),(cnorm,"privileged_obs_norm_state_dict")):
            immutable &= all(torch.equal(value.cpu(),saved[key][name]) for name,value in obj.state_dict().items())
        unchanged=(sha(args.checkpoint)==receipt["sha256"] and source_hashes(source)==receipt["source_sha256"]
            and source_hashes(args.source_root)==receipt["source_sha256"]
            and sha(Path(receipt["optimizer_path"]))==receipt["optimizer_sha256"]
            and all(sha(Path(path))==digest for path,digest in receipt["external_dependency_sha256"].items()))
        if not (immutable and unchanged) and error is None:error=RuntimeError("Frozen candidate/norm/source/dependency changed")
        terminal=dict(**endpoint,start_position=start,origins=origins,duration_s=duration,survived20s=survived,
            terminated=failed,timed_out=timeouts,net_world_x_m=dx,progress=progress,bilateral=bilateral,
            valid_alternating_landings=footfalls.landings,positive_control=positive,force_mismatch_steps=mismatch,
            contact_evidence_valid=forcevalid,success=success,last_prephysics_raw_policy=last_policy,last_prephysics_raw_critic=last_critic,
            **{"termination_"+key:value for key,value in reasons.items()})
        np.savez_compressed(args.output/"first_episode_terminal.npz",**{key:value.cpu().numpy() for key,value in terminal.items()})
        if trajectory:np.savez_compressed(args.output/"trajectories.npz",**{key:np.stack([f[key] for f in trajectory]) for key in trajectory[0]})
        rows=[]
        for i in range(n):
            rows.append(dict(env_id=i,duration_s=float(duration[i]),survived20s=bool(survived[i]),terminated=bool(failed[i]),
                timed_out=bool(timeouts[i]),net_world_x_m=float(dx[i]),progress=bool(progress[i]),bilateral=bool(bilateral[i]),
                left_valid_landings=int(footfalls.landings[i,0]),right_valid_landings=int(footfalls.landings[i,1]),
                contact_evidence_valid=bool(forcevalid[i]),success=bool(success[i]),
                termination_reasons="|".join(key for key,bits in reasons.items() if bool(bits[i]))))
        with (args.output/"episodes.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(args.output/"footfall_events.json",events)
        metadata.update(status="failed" if error else "completed",finished_at_epoch=time.time(),
            simulated_steps=steps,simulated_duration_s=steps*raw.step_dt,required_successful_envs=922,
            successful_envs=int(success.sum()),survived20s_envs=int(survived.sum()),progress_envs=int(progress.sum()),
            bilateral_envs=int(bilateral.sum()),positive_control_envs=int(positive.all(-1).sum()),
            force_mismatch_envs=int((mismatch.sum(-1)>0).sum()),termination_counts={key:int(bits.sum()) for key,bits in reasons.items()},
            policy_and_both_normalizers_unchanged=immutable,frozen_artifacts_unchanged=unchanged,
            initialization_ready=bool(int(success.sum())>=922 and error is None),formal_acceptance_passed=False,
            stairs_acceptance_passed=False,scope="Independent frozen direct-flat initialization only; six-level90s stair acceptance remains separate",
            endpoint_semantics="First terminal state before reset; repeated terminal copies frozen in trajectories",
            last_observation_semantics="Saved last pre-physics actor/critic inputs; not reset-after-death observations")
        if error:metadata["error"]=repr(error)
        write(args.output/"result.json",metadata);write(args.output/"run.json",metadata)
        print("[DIRECT_INIT_VALIDATION] RESULT "+json.dumps(metadata),flush=True)
        if error:raise error
    finally:
        if video is not None:video.close()
        env.close()


def main():
    args,AppLauncher=parse_args()
    if args.cpu_check_only:
        try:validate_cpu(args)
        except BaseException as error:
            write(args.validation_output,dict(status="failed",cpu_only=True,initialization_ready=False,
                formal_acceptance_passed=False,checkpoint=str(args.checkpoint),sha256=sha(args.checkpoint),error=repr(error)))
            raise
        return
    source,receipt=prepare_physical(args)
    metadata=dict(status="initializing",started_at_epoch=time.time(),checkpoint=str(args.checkpoint),
        checkpoint_sha256=receipt["sha256"],source_root=str(args.source_root),imported_source_root=str(source),
        num_envs=1024,duration_s=20.,speed_m_s=.3,seed=42,network_dimensions=[1029,866,73,12],
        training_updates=0,teacher_constructed=False,teacher_interventions=0,initialization_ready=False,
        formal_acceptance_passed=False,stairs_acceptance_passed=False,validator_sha256=sha(Path(__file__)))
    app=None
    try:
        write(args.output/"run.json",metadata)
        app=AppLauncher(args).app
        evaluate(args,app,source,receipt,metadata)
    except BaseException as error:
        metadata.update(status="failed",initialization_ready=False,formal_acceptance_passed=False,error=repr(error))
        write(args.output/"result.json",metadata);write(args.output/"run.json",metadata)
        (args.output/"failure_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:app.close()


if __name__=="__main__":main()
