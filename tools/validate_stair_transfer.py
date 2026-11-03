"""Complete CPU admission for a new typed stair packet; no simulator or solver."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from stair_initialization_binding import (PROTOCOL, PROTOCOL_FORMATS, INITIALIZATION_PROTOCOLS,
    initialization_config_errors, exploration_snapshot_errors,
    temporal_credit_config_errors, temporal_credit_snapshot_errors)


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
    run = json.loads((checkpoint.parent/"run.json").read_text())
    require(run["task"] == TASK and run["trial_protocol"] == protocol and run["max_iterations"] == BUDGET
            and run["plan_sha256"] == sha(plan_path) and run["num_envs"] == 1024
            and run["num_steps_per_env"] == 24 and run["policy_dimensions"] == [1034,871,73,12],
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
    if protocol == PROTOCOL:
        temporal_credit = plan.get("temporal_credit")
        require(not temporal_credit_config_errors(temporal_credit)
                and request.get("temporal_credit") == run.get("temporal_credit") == temporal_credit,
                "Temporal credit plan/request/run binding changed")
        for snapshot in (run.get("runtime_temporal_credit_config"), run.get("runtime_temporal_credit"),
                         run.get("initial_learning_state", {}).get("temporal_credit")):
            require(not temporal_credit_snapshot_errors(temporal_credit, snapshot),
                    "Resolved or actual initial temporal credit changed")
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
    if protocol == PROTOCOL:
        require(not temporal_credit_snapshot_errors(temporal_credit, saved.get("temporal_credit"))
                and not temporal_credit_snapshot_errors(temporal_credit, gs.get("temporal_credit")),
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
                and run["actual_sampled_environment_steps"] == BUDGET*1024*24
                and run["discarded_uncommitted_rollout_steps"] == 0, "Not a natural complete stair final")
    if protocol == PROTOCOL:
        with (checkpoint.parent/"motion_metrics.jsonl").open() as stream:
            native_rows = [json.loads(line) for line in stream if line.endswith("\n") and line.strip()]
        native = [row for row in native_rows if row.get("iteration") == saved["iter"]]
        require(len(native) == 1 and not temporal_credit_snapshot_errors(temporal_credit, native[0].get("temporal_credit")),
                "Corresponding native iteration actual temporal credit binding changed")
    tensors = 0
    def finite(value):
        nonlocal tensors
        if torch.is_tensor(value):
            tensors += 1
            require(value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Invalid CPU tensor")
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (tuple,list)):
            for child in value: finite(child)
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
    require(gs["actor_optimizer_steps"] == actor_steps and gs["critic_optimizer_steps"] == gs["rnd_optimizer_steps"] == critic_steps
            and gs["rnd_update_counter"] == saved["rnd_runtime"]["update_counter"] == completed*24,
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
    geometry = run["runtime_stair_geometry"]
    require(geometry["levels"] == 6 and geometry["riser_min_m"] == .15 and geometry["riser_max_m"] == .15
            and geometry["depth_m"] == .35 and geometry["width_m"] == 2.4 and geometry["approach_m"] == 1.4
            and geometry["episode_length_s"] == 90 and geometry["terrain_rows"] == 2
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
        rnd_rollout_counter=completed*24,normalization_abi=abi,source_sha256=source_hashes,
        plan_path=str(plan_path),plan_sha256=sha(plan_path),parent_provenance=provenance,
        checkpoint_packet_loads=1,required_parent_packet_loads=1,parent_load_scope="coupled own-norm and startup actor inheritance",
        NN_forward_batches=2,optimizer_steps_executed=0,QP_calls=0,SVD_calls=0,CUDA_initialized=False,
        complete_policy_CPU_validation=True,physical_or_stair_acceptance=False)
    if protocol in INITIALIZATION_PROTOCOLS:
        result["exploration_initialization"] = binding
        result["exploration_initialization_binding_passed"] = True
    if protocol == PROTOCOL:
        result["temporal_credit"] = temporal_credit
        result["actual_temporal_credit_binding_passed"] = True
        result["native_temporal_credit_iteration"] = saved["iter"]
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
