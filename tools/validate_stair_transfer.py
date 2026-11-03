"""Complete CPU admission for a new typed stair packet; no simulator or solver."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from stair_curriculum_binding import course_config_errors, profile_snapshot_errors
from training_video import video_config_errors
from stair_execution_binding import execution_config_errors, execution_snapshot_errors
from stair_contact_bridge_binding import bridge_config_errors, bridge_snapshot_errors
from stair_initialization_binding import (PROTOCOL, PROTOCOL_FORMATS, INITIALIZATION_PROTOCOLS, CLEARANCE_PROGRESS_PROTOCOLS,
    TEMPORAL_PROTOCOLS, ROLLOUT_CREDIT_PROTOCOLS, HEIGHT_COURSE_PROTOCOLS, EXECUTION_PROTOCOLS, BRIDGE_PROTOCOLS, LEGACY_COURSE_PROTOCOL, ROLLOUT_STEPS_BY_PROTOCOL,
    initialization_config_errors, exploration_snapshot_errors,
    temporal_credit_config_errors, temporal_credit_snapshot_errors, rollout_credit_collection_errors,
    rollout_credit_budget_errors, rollout_credit_seal_errors,
    clearance_progress_config_errors, clearance_progress_snapshot_errors)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_cpu(checkpoint, plan_path, scope):
    import torch
    from guarded_stair_transfer import TASK, BUDGET, parent_admission
    from guarded_flat_ppo import state_digest
    from task_context_normalization import make_task_context_normalizers, normalization_metadata
    checkpoint = Path(checkpoint).resolve(strict=True)
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[1]
    require(scope in ("startup", "natural_final"), "Undeclared CPU scope")
    output = checkpoint.with_suffix(".validation.json")
    require(not output.exists(), "A CPU receipt already exists; do not reload this scope")
    provenance = parent_admission(plan, root)
    protocol = plan["trial_protocol"]
    checkpoint_format = PROTOCOL_FORMATS[protocol]
    if protocol in HEIGHT_COURSE_PROTOCOLS:
        height_course, training_video = plan.get("height_course"), plan.get("training_video")
        require(not course_config_errors(height_course)
                and not video_config_errors(training_video, allow_legacy_view=protocol == LEGACY_COURSE_PROTOCOL),
                "Wrong new course/video declaration")
    rollout_steps = ROLLOUT_STEPS_BY_PROTOCOL[protocol]
    run = json.loads((checkpoint.parent/"run.json").read_text())
    require(run["task"] == TASK and run["trial_protocol"] == protocol and run["max_iterations"] == BUDGET
            and run["plan_sha256"] == sha(plan_path) and run["num_envs"] == 1024
            and run["num_steps_per_env"] == rollout_steps and run["policy_dimensions"] == [1034,871,73,12],
            "CPU admission belongs to a different run")
    if protocol in INITIALIZATION_PROTOCOLS:
        binding = plan["exploration_initialization"]
        request = json.loads((Path(run["case"])/"request.json").read_text())
        require(request.get("exploration_initialization") == run.get("exploration_initialization") == binding
                and request.get("plan_sha256") == run["plan_sha256"]
                and request.get("trial_protocol") == protocol
                and request.get("checkpoint_format") == run.get("checkpoint_format") == checkpoint_format,
                "Fresh exploration plan/request/run binding changed")
        require(not initialization_config_errors(binding, run.get("runtime_exploration_config", {})),
                "Resolved runtime exploration configuration changed")
        require(not exploration_snapshot_errors(binding, run.get("stair_exploration_budget"),
                    initial_learning_state=run.get("initial_learning_state"), at_initialization=True),
                "Actual initial exploration snapshot changed")
    if protocol in TEMPORAL_PROTOCOLS:
        temporal_credit = plan.get("temporal_credit")
        require(not temporal_credit_config_errors(temporal_credit, protocol=protocol)
                and request.get("temporal_credit") == run.get("temporal_credit") == temporal_credit,
                "Temporal credit plan/request/run binding changed")
        for snapshot in (run.get("runtime_temporal_credit_config"), run.get("runtime_temporal_credit"),
                         run.get("initial_learning_state", {}).get("temporal_credit")):
            require(not temporal_credit_snapshot_errors(temporal_credit, snapshot, protocol=protocol),
                    "Resolved or actual initial temporal credit changed")
    if protocol in ROLLOUT_CREDIT_PROTOCOLS:
        require(not rollout_credit_budget_errors(request, protocol=protocol)
                and not rollout_credit_budget_errors(run, protocol=protocol),
                "New rollout source budget differs from plan/request/run")
        collection = plan.get("rollout_collection")
        require(not rollout_credit_collection_errors(collection, protocol=protocol)
                and request.get("rollout_collection") == collection, "New current/fixed collection request changed")
        for snapshot in (run.get("rollout_collection"), run.get("runtime_rollout_collection"),
                         run.get("initial_learning_state", {}).get("rollout_collection"),
                         run.get("guarded_stair_learning", {}).get("rollout_collection")):
            require(not rollout_credit_collection_errors(snapshot, protocol=protocol) and snapshot == collection,
                    "Actual initial48-current/24-fixed collection changed")
    if protocol in CLEARANCE_PROGRESS_PROTOCOLS:
        clearance_progress = plan.get("clearance_progress_potential")
        require(run.get("clearance_progress_potential") == request.get("clearance_progress_potential")
                == clearance_progress and not clearance_progress_config_errors(clearance_progress,
                    run.get("runtime_clearance_progress_config"), protocol=protocol),
                "Clearance progress plan/request/resolved configuration changed")
        for snapshot in (run.get("runtime_clearance_progress"),
                         run.get("initial_learning_state", {}).get("clearance_progress_potential"),
                         run.get("guarded_stair_learning", {}).get("clearance_progress_potential")):
            require(not clearance_progress_snapshot_errors(clearance_progress, snapshot, protocol=protocol,
                                                           expected_updates=0),
                    "Actual clearance progress initialization binding changed")
    source = Path(run["source_snapshot"]).resolve(strict=True)
    source_hashes = {"source/"+str(p.relative_to(source)):sha(p) for p in source.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    require(source_hashes == run["source_sha256"], "Frozen child source changed")
    for path, digest in run["external_dependency_sha256"].items():
        require(sha(path) == digest, "Frozen external dependency changed: "+path)
    require(not torch.cuda.is_initialized(), "CPU admission may not initialize CUDA")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    parent = torch.load(plan["initial_policy"], map_location="cpu", weights_only=True)
    require(saved["format"] == checkpoint_format and saved["trial_protocol"] == protocol and saved["task"] == TASK
            and saved["training_budget"] == BUDGET, "Wrong serialized stair type")
    gs = saved["guarded_ppo_state"]
    if protocol in TEMPORAL_PROTOCOLS:
        require(not temporal_credit_snapshot_errors(temporal_credit, saved.get("temporal_credit"), protocol=protocol)
                and not temporal_credit_snapshot_errors(temporal_credit, gs.get("temporal_credit"), protocol=protocol),
                "Typed packet actual temporal credit binding changed")
        for record in (run, saved, gs):
            require(record.get("inherited_numeric_execution_profile_scope") == "numeric_executor_only"
                    and record.get("ppo_temporal_credit_override") == dict(parameter="gae_lambda", value=0.99,
                        scope="stair_transfer_only", exact_binding="temporal_credit"),
                    "Temporal credit override is not explicitly separated from inherited numeric execution")
    completed = int(gs["completed_updates"])
    require(1 <= completed <= BUDGET and saved["iter"] == completed-1 and gs["uncommitted_rollout"] is False,
            "CPU scope requires an accepted complete update")
    if scope == "startup":
        require(completed == 1 and checkpoint.name == "model_0.pt", "Startup CPU is only M0")
    else:
        require(completed == BUDGET and checkpoint.name == f"model_{BUDGET-1}.pt"
                and run["status"] == "completed" and run["completed_iterations"] == BUDGET
                and run["actual_sampled_environment_steps"] == BUDGET*1024*rollout_steps
                and run["total_timesteps"] == BUDGET*1024*rollout_steps
                and run["discarded_uncommitted_rollout_steps"] == 0, "Not a natural complete stair final")
    if protocol in TEMPORAL_PROTOCOLS:
        with (checkpoint.parent/"motion_metrics.jsonl").open() as stream:
            native_rows = [json.loads(line) for line in stream if line.endswith("\n") and line.strip()]
        native = [row for row in native_rows if row.get("iteration") == saved["iter"]]
        require(len(native) == 1 and not temporal_credit_snapshot_errors(temporal_credit, native[0].get("temporal_credit"),
                                                                        protocol=protocol),
                "Corresponding native iteration actual temporal credit binding changed")
    if protocol in ROLLOUT_CREDIT_PROTOCOLS:
        for record in (saved, gs, native[0]):
            require(not rollout_credit_collection_errors(record.get("rollout_collection"), protocol=protocol)
                    and record["rollout_collection"] == collection,
                    "Typed packet/state/native48-current/24-fixed collection changed")
        bank = gs.get("audit_bank_acquisition")
        require(not rollout_credit_seal_errors(bank, protocol=protocol)
                and run["guarded_stair_learning"].get("fixed_real_bank") == bank,
                "Actual post-step24/RND24 fixed-prefix seal is missing or changed")
        case = Path(run["case"]).resolve(strict=True)
        sealed_path = case/"actual_bank.sealed.json"
        sealed = json.loads(sealed_path.read_text())
        require(sealed.get("case") == str(case) and sealed.get("trial_protocol") == protocol
                and sealed.get("checkpoint_format") == checkpoint_format
                and sealed.get("request_sha256") == sha(case/"request.json")
                and sealed.get("plan_sha256") == sha(plan_path) and sealed.get("fixed_real_bank") == bank
                and not temporal_credit_snapshot_errors(temporal_credit, sealed.get("temporal_credit"), protocol=protocol)
                and not rollout_credit_collection_errors(sealed.get("rollout_collection"), protocol=protocol)
                and type(sealed.get("actual_seal_storage_step")) is int and sealed["actual_seal_storage_step"] == 24
                and type(sealed.get("actual_seal_RND_counter")) is int and sealed["actual_seal_RND_counter"] == 24
                and type(sealed.get("actual_current_rows_at_seal")) is int and sealed["actual_current_rows_at_seal"] == 520,
                "Exclusive actual-bank publication did not observe the true post-step24 seal")
        expected_bindings = dict(case=str(case),request_sha256=sha(case/"request.json"),
                source_commit=run["git_commit"],source_sha256=run["source_sha256"],
                training_tool_sha256=run["training_tool_sha256"],trial_protocol=protocol,
                temporal_credit=temporal_credit,rollout_collection=collection,
                plan_path=str(plan_path),plan_sha256=sha(plan_path),initial_policy_sha256=plan["initial_policy_sha256"])
        if protocol in CLEARANCE_PROGRESS_PROTOCOLS:
            expected_bindings["clearance_progress_potential"] = clearance_progress
        if protocol in HEIGHT_COURSE_PROTOCOLS:
            expected_bindings.update(height_course=height_course, training_video=training_video)
        if protocol in EXECUTION_PROTOCOLS:
            expected_bindings["execution_chain"] = plan["execution_chain"]
        if protocol in BRIDGE_PROTOCOLS:
            expected_bindings["contact_bridge"] = plan["contact_bridge"]
        require(bank["bindings"] == expected_bindings,
                "Actual fixed-prefix source/request/plan bindings changed")
        guard_dir = (checkpoint.parent/"functional_guard").resolve(strict=True)
        for key, hash_key in (("bank_path","bank_sha256"), ("receipt_path","receipt_sha256"),
                              ("acquisition_log_path","acquisition_log_sha256")):
            path = Path(bank[key]).resolve(strict=True)
            require(path.is_relative_to(guard_dir) and sha(path) == bank[hash_key],
                    "Immutable new-run acquisition artifact changed")
        receipt_json = json.loads(Path(bank["receipt_path"]).read_text())
        require(receipt_json == {key:value for key,value in bank.items() if key not in ("receipt_path","receipt_sha256")},
                "Actual fixed-prefix receipt bytes disagree with the typed packet")
    if protocol in CLEARANCE_PROGRESS_PROTOCOLS:
        for record in (saved, gs, native[0]):
            require(not clearance_progress_snapshot_errors(clearance_progress,
                record.get("clearance_progress_potential"), protocol=protocol, expected_updates=completed*rollout_steps),
                "Typed packet/state/native actual clearance reward binding changed")
        require(saved["clearance_progress_potential"] == gs["clearance_progress_potential"]
                == native[0]["clearance_progress_potential"], "Packet and matching native reward states disagree")
        require(not clearance_progress_snapshot_errors(clearance_progress,
            sealed.get("clearance_progress_potential"), protocol=protocol, expected_updates=24),
            "Actual prefix-seal reward state did not observe24 controls")
    tensors = 0
    def finite(value, path=()):
        nonlocal tensors
        if torch.is_tensor(value):
            tensors += 1
            if protocol in BRIDGE_PROTOCOLS and path in (
                    ("contact_bridge_payload","pending_bytes"),("contact_bridge_payload","closed_bytes")):
                expected = (64*1024 if path[-1] == "pending_bytes" else 49152)*3840
                require(value.device.type == "cpu" and value.dtype == torch.uint8
                        and value.is_contiguous() and tuple(value.shape) == (expected,),
                        "Invalid typed bridge byte storage")
                # uint8 cannot contain NaN/Inf. Actual embedded float32 fields
                # are checked below, in <=512-record blocks, by the new schema.
                return
            require(value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Invalid CPU tensor")
        elif isinstance(value, dict):
            for key,child in value.items(): finite(child,path+(key,))
        elif isinstance(value, (tuple,list)):
            for index,child in enumerate(value): finite(child,path+(index,))
        elif isinstance(value, float):
            require(math.isfinite(value), "Nonfinite scalar metadata")
    finite(saved)
    # Instantiate only the preserved independent policy; no task package import.
    model_path = source/"CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py"
    spec = importlib.util.spec_from_file_location("stair_cpu_actor", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.H2ActorCriticOur(1034,871,12,num_joints=29,extra_context_dim=5)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    names = dict(model.named_parameters())
    actor_names = ["actor."+n for n,_ in model.actor.named_parameters()]+["std"]
    critic_names = ["critic."+n for n,_ in model.critic.named_parameters()]
    require(len(actor_names) == 45 and len(critic_names) == 8 and len(names) == 53,
            "Policy parameter schema changed")
    require(saved["optimizer_parameter_names"] == {"actor":actor_names,"critic":critic_names},
            "Split optimizer mapping changed")
    actor_steps = max(0,completed-40)*20
    critic_steps = completed*20
    if protocol in ROLLOUT_CREDIT_PROTOCOLS:
        require(all(type(gs.get(key)) is int for key in ("completed_updates", "actor_optimizer_steps",
                    "critic_optimizer_steps", "rnd_optimizer_steps", "rnd_update_counter"))
                and type(saved["rnd_runtime"].get("update_counter")) is int,
                "New accepted rollout counters require actual integer values")
    require(gs["actor_optimizer_steps"] == actor_steps and gs["critic_optimizer_steps"] == gs["rnd_optimizer_steps"] == critic_steps
            and gs["rnd_update_counter"] == saved["rnd_runtime"]["update_counter"] == completed*rollout_steps,
            "Serialized accepted counters changed")
    def check_optimizer(packet, parameter_names, steps, lr, parameters, foreach=None):
        groups = packet["param_groups"]
        require(len(groups) == 1 and len(groups[0]["params"]) == len(parameter_names), "Wrong Adam groups")
        group = groups[0]
        require(group["lr"] == lr and group["foreach"] is foreach and group["eps"] == 1.e-8
                and tuple(group["betas"]) == (.9,.999) and group["weight_decay"] == 0,
                "Adam numeric settings changed")
        states = packet["state"]
        require((not states) if steps == 0 else set(states) == set(group["params"]), "Incomplete Adam state")
        for key,name in zip(group["params"],parameter_names):
            if steps == 0: continue
            value = states[key]
            require(set(value) == {"step","exp_avg","exp_avg_sq"} and int(value["step"]) == steps,
                    "Adam accepted step differs")
            for moment in ("exp_avg","exp_avg_sq"):
                require(value[moment].shape == parameters[name].shape and value[moment].dtype == torch.float32,
                        "Adam moment schema changed")
    check_optimizer(saved["actor_optimizer_state_dict"],actor_names,actor_steps,1.e-5,names,False)
    check_optimizer(saved["critic_optimizer_state_dict"],critic_names,critic_steps,1.e-3,names)
    rnd_names = saved["rnd_optimizer_parameter_names"]
    require(len(rnd_names) == 4, "RND predictor parameter schema changed")
    rnd_parameters = {name:saved["rnd_state_dict"]["predictor."+name] for name in rnd_names}
    check_optimizer(saved["rnd_optimizer_state_dict"],rnd_names,critic_steps,1.e-3,rnd_parameters)
    require(saved["rnd_runtime"]["initial_weight"] == 0 and gs["rnd_reward_weight"] == 0,
            "Stair RND must stay reward-neutral")
    norms = make_task_context_normalizers("cpu")
    for name,key in (("actor","obs_norm_state_dict"),("critic","privileged_obs_norm_state_dict")):
        norms[name].load_state_dict(saved[key],strict=True)
        require(state_digest(saved[key]) == state_digest(parent[key]), "Frozen own parent norm differs")
    abi = normalization_metadata(norms)
    require(abi == saved["normalization_abi"] and abi["prefix_empirical_counts"] == {"actor":24576000,"critic":24576000},
            "Frozen prefix/fixed context ABI changed")
    if scope == "startup":
        require(all(torch.equal(v,parent["model_state_dict"][key]) for key,v in saved["model_state_dict"].items()
                    if key.startswith("actor.")), "M0 actor mean was updated during critic warmup")
    projection = gs["proposal_projection"]
    require(projection["max_solver_sweeps"] == 2048 and projection["max_active_rows"] == 28
            and projection["jacobian_arena_budget_bytes"] == 1073741824
            and projection["jacobian_arena_capacity_bytes"] == 1052335872
            and projection["jacobian_mean_parameter_elements"] == 782988
            and projection["working_budget_fraction"] == .5 and projection["no_retry"] is True
            and projection["committed_minibatches"] == projection["proposal_calls"] == projection["std_hook_calls"] == actor_steps
            and projection["primal_precision_policy"] == "direct_fp64_primal_envelope_v1"
            and projection["support_polish_policy"] == "bounded_dual_active_set_v1", "Bounded executor metadata changed")
    ex = saved["exploration_state"]["snapshot"]
    require(ex["optimizer_steps"] == actor_steps and len(ex["policy_std"]) == 12
            and all(0<lo<=s<=hi for s,lo,hi in zip(ex["policy_std"],ex["minimum_raw_std"],ex["maximum_raw_std"])),
            "Physical exploration budget changed")
    if protocol in INITIALIZATION_PROTOCOLS:
        require(not exploration_snapshot_errors(binding, ex, initial_learning_state=run["initial_learning_state"]),
                "Typed packet exploration initialization binding changed")
        require(saved["model_state_dict"]["std"].tolist() == ex["policy_std"],
                "Typed packet exploration snapshot disagrees with the serialized policy std")
    if protocol in HEIGHT_COURSE_PROTOCOLS:
        require(request.get("height_course") == run.get("height_course") == height_course
                and request.get("training_video") == run.get("training_video") == training_video,
                "Typed new training profile disagrees with its request/plan")
        initial_state=run.get("actual_initial_course_state", {})
        require(initial_state.get("initial_episode_lengths_all_zero") is True
                and initial_state.get("actual_world_X_bounds_all1024") is True
                and initial_state.get("root_X_bounds_m") == [-.25,.25]
                and initial_state.get("cohort_assignment") == run.get("runtime_height_profile", {}).get("height_course"),
                "Missing actual first training pose/cohort admission")
        for value in (run.get("runtime_height_profile"), run.get("initial_learning_state", {}).get("height_profile"),
                      run.get("guarded_stair_learning", {}).get("height_profile")):
            require(not profile_snapshot_errors(height_course, training_video, value, expected_controls=0),
                    "Actual initial height/video/credit binding changed")
        for value in (saved.get("height_profile"), gs.get("height_profile"), native[0].get("height_profile")):
            require(not profile_snapshot_errors(height_course, training_video, value,
                                                expected_controls=completed*rollout_steps),
                    "Typed actual course/video/credit state differs")
        require(saved["height_profile"] == gs["height_profile"] == native[0]["height_profile"],
                "Packet/state/native new profile states disagree")
        require(not profile_snapshot_errors(height_course, training_video, sealed.get("height_profile"),
                                            expected_controls=24), "Prefix24 new profile state differs")
        require(run["external_dependency_sha256"] == {**run["external_dependency_sha256"],
                **plan["external_dependency_sha256"]}, "External profile dependency receipt differs")
        for name, expected in plan["external_dependency_sha256"].items():
            require(sha(name) == expected, "External profile source/encoder changed")
        require(saved["height_profile"]["training_video"]["encoder_binary_sha256"]
                == plan["external_dependency_sha256"][training_video["encoder_binary"]], "Wrong actual encoder binary")
    if protocol in EXECUTION_PROTOCOLS:
        execution_chain = plan.get("execution_chain")
        require(not execution_config_errors(execution_chain)
                and request.get("execution_chain") == run.get("execution_chain") == execution_chain,
                "Typed execution observer declaration differs from its own source")
        for value in (run.get("runtime_execution_chain"),
                      run.get("initial_learning_state", {}).get("execution_chain"),
                      run.get("guarded_stair_learning", {}).get("execution_chain")):
            require(not execution_snapshot_errors(execution_chain, value, expected_controls=0),
                    "Initial existing-execution binding differs")
        for value in (saved.get("execution_chain"), gs.get("execution_chain"), native[0].get("execution_chain")):
            require(not execution_snapshot_errors(execution_chain, value, expected_controls=completed*rollout_steps),
                    "Typed actual execution-chain window differs")
        require(saved["execution_chain"] == gs["execution_chain"] == native[0]["execution_chain"],
                "Packet/state/native execution-chain snapshots disagree")
        require(not execution_snapshot_errors(execution_chain, sealed.get("execution_chain"), expected_controls=24),
                "Prefix24 existing-execution observer binding differs")
    bridge_cpu = None
    if protocol in BRIDGE_PROTOCOLS:
        from stair_contact_bridge import validate_bridge_payload
        contact_bridge = plan.get("contact_bridge")
        require(not bridge_config_errors(contact_bridge)
                and request.get("contact_bridge") == run.get("contact_bridge") == contact_bridge,
                "New bridge source/plan/request declaration differs")
        for value in (run.get("runtime_contact_bridge"),
                      run.get("initial_learning_state", {}).get("contact_bridge"),
                      run.get("guarded_stair_learning", {}).get("contact_bridge")):
            require(not bridge_snapshot_errors(contact_bridge,value,expected_controls=0),
                    "New bridge initial source state differs")
        for value in (saved.get("contact_bridge"),gs.get("contact_bridge"),native[0].get("contact_bridge")):
            require(not bridge_snapshot_errors(contact_bridge,value,expected_controls=completed*rollout_steps),
                    "New bridge actual packet/native state differs")
        require(saved["contact_bridge"] == gs["contact_bridge"] == native[0]["contact_bridge"],
                "New bridge accepted packet/native snapshots disagree")
        prefix = sealed.get("contact_bridge")
        require(not bridge_snapshot_errors(contact_bridge,prefix,expected_controls=24)
                and prefix["serving_forwards"] == prefix["Adam_commits"] == 0,
                "New bridge changed the original first24 NN/optimizer seal")
        bridge_cpu = validate_bridge_payload(saved["contact_bridge_payload"],contact_bridge,
                                            saved["contact_bridge"],completed=completed)
    geometry = run["runtime_stair_geometry"]
    require(geometry["levels"] == 6 and geometry["riser_min_m"] == (.05 if protocol in HEIGHT_COURSE_PROTOCOLS else .15) and geometry["riser_max_m"] == .15
            and geometry["depth_m"] == .35 and geometry["width_m"] == 2.4 and geometry["approach_m"] == 1.4
            and geometry["episode_length_s"] == 90 and geometry["terrain_rows"] == (3 if protocol in HEIGHT_COURSE_PROTOCOLS else 2)
            and geometry["scripted_gait_installed"] is False and len(geometry["termination_terms"]) == 7,
            "Actual staircase or native terminations changed")
    command = run["world_stair_command"]
    require(command["kind"] == "fixed_world_plusX_base_command_v1"
            and command["world_linear_velocity_m_s"] == [.3,0.,0.]
            and command["prefix_command_units_and_base_frame_preserved"] is True
            and command["frozen_normalization_unchanged"] is True
            and command["joint_or_body_height_controller"] is False,
            "World direction or inherited observation units/frame changed")
    rewards = run["reward_contract"]
    require(run["stair_clearance_target"] == dict(kind="earliest_uncrossed_full_footprint_v1",
            real_riser_count=6, approach_m=1.4, depth_m=.35, rear_margin_m=.008,
            lookahead_m=.20, extra_height_m=.02, distance_scale_m=.05,
            deficit_square_cap=16.0, target_retained_until_whole_sole_crossed=True,
            observation_or_joint_controller=False), "Whole-sole riser target contract changed")
    from selected_row_block import SELECTED_ROW_BLOCK_POLICY, selected_row_block_contract
    require(run["selected_row_evaluation"] == selected_row_block_contract()
            and gs["proposal_projection"]["selected_row_evaluation_policy"] == SELECTED_ROW_BLOCK_POLICY,
            "Original-block selected-row execution contract changed")
    require(run["early_progress_policy"] == plan["early_progress_policy"]
            and run["early_progress_policy_sha256"] == plan["early_progress_policy_sha256"]
            == sha(plan["early_progress_policy"]), "User-authorized early-progress policy binding changed")
    require(len(rewards) == 19 and rewards["failure"] == {"weight":-20.0,"function":"stair_first_qualified_failure"}
            and rewards["world_forward_excess_speed"]["weight"] == -33.333333333333336
            and rewards["outward_tilt_growth"] == {"weight":-20.0/(1.0-math.cos(.75)),"function":"stair_outward_tilt_growth"}
            and rewards["body_angular_rate"] == {"weight":-.025,"function":"stair_body_angular_rate"}
            and rewards["swing_clearance_deficit"] == {"weight":-4.0,"function":"stair_swing_clearance_deficit"},
            "Declared staircase reward/force contract changed")
    with torch.inference_mode():
        zero = torch.zeros(2,1034)
        output_actions = model.act_inference(norms["actor"](zero))
        values = model.evaluate(norms["critic"](torch.zeros(2,871)))
        require(output_actions.shape == (2,12) and values.shape == (2,1)
                and bool(torch.isfinite(output_actions).all()) and bool(torch.isfinite(values).all()),
                "Typed finite policy forward failed")
    require(not torch.cuda.is_initialized(), "CPU check initialized CUDA")
    result = dict(status="passed",scope=scope,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        checkpoint_format=checkpoint_format,trial_protocol=protocol,task=TASK,iteration=saved["iter"],completed_updates=completed,
        finite_tensor_occurrences=tensors,network_dimensions=[1034,871,73,12],model_strict_load=True,
        actor_optimizer_steps=actor_steps,critic_optimizer_steps=critic_steps,rnd_optimizer_steps=critic_steps,
        rnd_rollout_counter=completed*rollout_steps,normalization_abi=abi,source_sha256=source_hashes,
        plan_path=str(plan_path),plan_sha256=sha(plan_path),parent_provenance=provenance,
        checkpoint_packet_loads=1,required_parent_packet_loads=1,parent_load_scope="coupled own-norm and startup actor inheritance",
        NN_forward_batches=2+(bridge_cpu["CPU_forward_batches"] if bridge_cpu is not None else 0),
        policy_NN_forward_batches=2,optimizer_steps_executed=0,QP_calls=0,SVD_calls=0,CUDA_initialized=False,
        complete_policy_CPU_validation=True,physical_or_stair_acceptance=False)
    if protocol in INITIALIZATION_PROTOCOLS:
        result["exploration_initialization"] = binding
        result["exploration_initialization_binding_passed"] = True
    if protocol in TEMPORAL_PROTOCOLS:
        result["temporal_credit"] = temporal_credit
        result["actual_temporal_credit_binding_passed"] = True
        result["native_temporal_credit_iteration"] = saved["iter"]
    if protocol in ROLLOUT_CREDIT_PROTOCOLS:
        result["rollout_collection"] = collection
        result["actual_prefix24_seal_and_full48_packet_binding_passed"] = True
        result["actual_prefix_seal"] = dict(sealed_record=str(sealed_path),sealed_record_sha256=sha(sealed_path),
            storage_step=sealed["actual_seal_storage_step"],rnd_rollout_counter=bank["rnd_rollout_update_counter"],
            current_rows_at_seal=bank["current_rows_at_seal"],full_guard_bank_assembled_at_seal=False,
            completed_controls=bank["completed_controls"],receipt_sha256=bank["receipt_sha256"])
    if protocol in CLEARANCE_PROGRESS_PROTOCOLS:
        result["clearance_progress_potential"] = clearance_progress
        result["actual_clearance_reward_binding_passed"] = True
        result["actual_clearance_reward_state"] = saved["clearance_progress_potential"]
        result["native_clearance_reward_iteration"] = saved["iter"]
    if protocol in HEIGHT_COURSE_PROTOCOLS:
        result.update(height_course=height_course, training_video=training_video,
            actual_height_profile=saved["height_profile"], actual_height_profile_binding_passed=True)
    if protocol in EXECUTION_PROTOCOLS:
        result.update(execution_chain=execution_chain, actual_execution_chain=saved["execution_chain"],
                      actual_execution_chain_binding_passed=True)
    if bridge_cpu is not None:
        result.update(contact_bridge=contact_bridge, actual_contact_bridge=saved["contact_bridge"],
            actual_contact_bridge_binding_passed=True, contact_bridge_CPU=bridge_cpu,
            bridge_CPU_forward_batches=bridge_cpu["CPU_forward_batches"])
    with checkpoint.with_suffix(".validation.json").open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--scope",choices=("startup","natural_final"),required=True)
    a=parser.parse_args()
    print(json.dumps(validate_cpu(a.checkpoint,a.plan,a.scope)))
