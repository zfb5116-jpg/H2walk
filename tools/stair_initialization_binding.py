"""JSON-only identity, exploration and temporal-credit stair bindings.

These predicates inspect metadata only. They do not construct a policy, load a
checkpoint, or change the inherited exploration arithmetic or optimizer hook.
"""
import math


from stair_execution_binding import PROTOCOL as EXECUTION_PROTOCOL, FORMAT as EXECUTION_FORMAT
from stair_contact_bridge_binding import (PROTOCOL as FIT_BRIDGE_PROTOCOL, FORMAT as FIT_BRIDGE_FORMAT, LEGACY_PROTOCOL as LEGACY_BRIDGE_PROTOCOL,
                                         LEGACY_FORMAT as LEGACY_BRIDGE_FORMAT)

PROTOCOL = "accepted_flat_reward_balanced_anticipatory_support_height_course_v1"
FORMAT = "h2_guarded_stair_ppo_reward_balance_v1"
REWARD_BALANCE_PROTOCOLS = (PROTOCOL,)

LEGACY_COURSE_PROTOCOL = "accepted_flat_fixed_target_height_course_stair_v10"
EXECUTION_PROTOCOLS = (EXECUTION_PROTOCOL, LEGACY_BRIDGE_PROTOCOL, FIT_BRIDGE_PROTOCOL, PROTOCOL)
BRIDGE_PROTOCOLS = (LEGACY_BRIDGE_PROTOCOL, FIT_BRIDGE_PROTOCOL, PROTOCOL)
HEIGHT_COURSE_PROTOCOLS = (LEGACY_COURSE_PROTOCOL, *EXECUTION_PROTOCOLS)
LEGACY_CLEARANCE_PROTOCOL = "accepted_flat_clearance_potential_stair_v9"
CLEARANCE_PROGRESS_PROTOCOLS = (LEGACY_CLEARANCE_PROTOCOL, *HEIGHT_COURSE_PROTOCOLS)
LEGACY_ROLLOUT_PROTOCOL = "accepted_flat_rollout_credit_stair_v8"
ROLLOUT_CREDIT_PROTOCOLS = (LEGACY_ROLLOUT_PROTOCOL, *CLEARANCE_PROGRESS_PROTOCOLS)
LEGACY_PROTOCOL = "accepted_flat_block_consistent_stair_clearance_v5"
LEGACY_INITIALIZATION_PROTOCOL = "accepted_flat_full_rms_exploration_stair_v6"
LEGACY_TEMPORAL_PROTOCOL = "accepted_flat_temporal_credit_stair_v7"
PROTOCOL_FORMATS = {LEGACY_PROTOCOL: "h2_guarded_stair_ppo_v5",
                    LEGACY_INITIALIZATION_PROTOCOL: "h2_guarded_stair_ppo_v6",
                    LEGACY_TEMPORAL_PROTOCOL: "h2_guarded_stair_ppo_v7",
                    LEGACY_ROLLOUT_PROTOCOL: "h2_guarded_stair_ppo_v8",
                    LEGACY_CLEARANCE_PROTOCOL: "h2_guarded_stair_ppo_v9",
                    LEGACY_COURSE_PROTOCOL: "h2_guarded_stair_ppo_v10",
                    EXECUTION_PROTOCOL: EXECUTION_FORMAT, LEGACY_BRIDGE_PROTOCOL: LEGACY_BRIDGE_FORMAT,
                    FIT_BRIDGE_PROTOCOL: FIT_BRIDGE_FORMAT, PROTOCOL: FORMAT}
INITIALIZATION_PROTOCOLS = (LEGACY_INITIALIZATION_PROTOCOL, LEGACY_TEMPORAL_PROTOCOL, *ROLLOUT_CREDIT_PROTOCOLS)
TEMPORAL_CREDIT_BY_PROTOCOL = {
    LEGACY_TEMPORAL_PROTOCOL: {"gamma": 0.99, "gae_lambda": 0.99, "rollout_steps": 24, "control_dt_s": 0.02},
    **{p: {"gamma": 0.99, "gae_lambda": 0.99, "rollout_steps": 48, "control_dt_s": 0.02}
       for p in ROLLOUT_CREDIT_PROTOCOLS},
}
TEMPORAL_PROTOCOLS = tuple(TEMPORAL_CREDIT_BY_PROTOCOL)
ROLLOUT_STEPS_BY_PROTOCOL = {LEGACY_PROTOCOL: 24, LEGACY_INITIALIZATION_PROTOCOL: 24,
                             LEGACY_TEMPORAL_PROTOCOL: 24, **{p: 48 for p in ROLLOUT_CREDIT_PROTOCOLS}}
FIXED_BANK_COLLECTION_CONTROLS = 24
MAXIMUM_ANGLE_STD_RAD = (0.02, 0.015, 0.015, 0.02, 0.01, 0.0075) * 2
_BINDING_KEYS = {"initial_fraction", "minimum_fraction", "maximum_angle_std_rad"}
_RUNNER_KEYS = {"initial_fraction", "minimum_fraction", "maximum_angle_std"}
_TEMPORAL_ADMISSION = dict(enabled=True, algorithm_class="H2GuardedStairPPO",
    algorithm_module="guarded_stair_transfer", stair_trial_protocol=None,
    numeric_execution_profile="explicit_task_context_longer_budget_v1", gamma=0.99, gae_lambda=0.99)
_UNSET = object()
# Products and quotients in the existing float32 snapshot are compared within
# four float32 relative epsilons. Raw initial==maximum and bound checks are exact.
_FLOAT32_RELATIVE_TOLERANCE = 4 * 2.0 ** -23


def reward_balance_terms():
    """Exact19-term configuration; only three existing costs are revised."""
    values = {
        "world_forward": (12.0, "stair_signed_forward", {}),
        "heading_error": (-1.0, "stair_heading_error", {}),
        "platform_hold": (1.5, "stair_qualified_platform_hold", {}),
        "potential": (1.0, "stair_event", {"name":"potential"}),
        "first_valid_foot": (3.0, "stair_event", {"name":"foot"}),
        "bilateral_level": (8.0, "stair_event", {"name":"pair"}),
        "high_foot_transfer": (3.0, "stair_event", {"name":"transfer"}),
        "stable_top": (30.0, "stair_event", {"name":"top"}),
        "failure": (-20.0, "stair_first_qualified_failure", {}),
        "stance_slip": (-0.3, "stair_stance_slip", {}),
        "tilt_margin": (-4.0, "stair_tilt_margin", {"free_angle":0.1, "angle_scale":0.4}),
        "drive_request": (-0.001, "stair_drive_request", {}),
        "target_change": (-0.5, "stair_target_change", {"max_fraction":0.5, "same_generation_only":True}),
        "native_overspeed": (-0.1, "stair_native_overspeed", {}),
        "soft_limit_excess": (-5.0, "stair_soft_limit_excess", {}),
        "world_forward_excess_speed": (-33.333333333333336, "stair_forward_excess_speed", {}),
        "outward_tilt_growth": (-20.0/(1.0-math.cos(0.75)), "stair_outward_tilt_growth", {"margin_angle":0.0}),
        "body_angular_rate": (-0.025, "stair_body_angular_rate", {}),
        "swing_clearance_deficit": (-4.0, "stair_swing_clearance_deficit",
            {"preview":0.35, "full_target_distance":0.2, "terrain_normalized_envelope":True}),
    }
    return {name:dict(weight=weight, function=function, params=params)
            for name,(weight,function,params) in values.items()}


def reward_balance_contract():
    return dict(kind="reward_only_balance_motion_v1", terms=reward_balance_terms(),
        changed_terms=["tilt_margin", "target_change", "swing_clearance_deficit"],
        term_count=19, continuous_payment_dt_s=0.02,
        first_action_history_source="common_step_counter_minus_episode_start_counter_gt1",
        new_reward_terms=0, new_learning_modules=0, extra_NN_forwards=0,
        actor_observation_added=False, physical_or_acceptance_threshold_changed=False)


def reward_balance_clearance_contract():
    return dict(kind="earliest_uncrossed_full_footprint_v1", real_riser_count=6,
        approach_m=1.4, depth_m=0.35, rear_margin_m=0.008,
        lookahead_m=0.35, extra_height_m=0.02, distance_scale_m=None,
        deficit_square_cap=1.0, target_retained_until_whole_sole_crossed=True,
        observation_or_joint_controller=False,
        height_normalization="actual_riser_height_plus_clearance",
        outside_preview_reference="selected_lower_plane", uncrossed_only_cost_mask=True,
        approach_envelope=dict(preview_m=0.35, full_target_distance_m=0.2,
                               interpolation="linear_clipped"))


def _reward_exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_reward_exact(actual[k],v) for k,v in expected.items())
    if isinstance(expected, list):
        return len(actual)==len(expected) and all(_reward_exact(a,b) for a,b in zip(actual,expected))
    return actual == expected


def reward_balance_config_errors(binding, actual_terms=_UNSET):
    errors=[]
    if not _reward_exact(binding, reward_balance_contract()):
        errors.append("reward-only profile differs from its exact19-term contract")
    if actual_terms is not _UNSET and not _reward_exact(actual_terms, reward_balance_terms()):
        errors.append("resolved reward functions, weights or parameters differ")
    return errors


def reward_balance_snapshot_errors(binding, snapshot, *, expected_controls=None):
    errors=reward_balance_config_errors(binding)
    if errors:
        return errors
    keys={"contract", "actual_terms", "controls", "first_action_history_source", "extra_NN_forwards"}
    if not isinstance(snapshot, dict) or set(snapshot)!=keys:
        return ["missing or undeclared actual reward-only snapshot fields"]
    errors.extend(reward_balance_config_errors(snapshot["contract"], snapshot["actual_terms"]))
    if snapshot["contract"]!=binding or snapshot["first_action_history_source"]!=binding["first_action_history_source"]:
        errors.append("actual reward-only declaration/history source disagrees")
    controls=snapshot["controls"]
    if type(controls) is not int or controls < 0 or (expected_controls is not None and controls!=expected_controls):
        errors.append("actual reward-only control count differs")
    if type(snapshot["extra_NN_forwards"]) is not int or snapshot["extra_NN_forwards"]!=0:
        errors.append("reward-only profile may not introduce neural forwards")
    return errors


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive_vector(value):
    return (isinstance(value, list) and len(value) == 12
            and all(_number(item) and item > 0 for item in value))


def temporal_credit_config_snapshot(runner_config, *, control_dt_s):
    """Extract only resolved runner scalars, without constructing a learner."""
    algorithm = runner_config.get("algorithm") if isinstance(runner_config, dict) else None
    if not isinstance(algorithm, dict):
        algorithm = {}
    return dict(gamma=algorithm.get("gamma"), gae_lambda=algorithm.get("lam"),
                rollout_steps=runner_config.get("num_steps_per_env") if isinstance(runner_config, dict) else None,
                control_dt_s=control_dt_s)


def temporal_credit_config_errors(binding, runner_config=_UNSET, *, protocol, control_dt_s=_UNSET):
    """Validate the exact declaration and optionally its resolved runner source.

    The runner keeps its original gamma/lam and num_steps_per_env field names.
    Its environment's resolved control period must be supplied explicitly.
    """
    expected_credit = TEMPORAL_CREDIT_BY_PROTOCOL.get(protocol) if type(protocol) is str else None
    if expected_credit is None:
        return ["unknown temporal credit protocol"]
    errors = []
    if not isinstance(binding, dict) or set(binding) != set(expected_credit):
        return ["temporal_credit requires exactly gamma, gae_lambda, rollout_steps and control_dt_s"]
    for key, expected in expected_credit.items():
        actual = binding[key]
        valid = type(actual) is int if key == "rollout_steps" else _number(actual)
        if not valid or actual != expected:
            errors.append("temporal_credit has incorrect " + key)
    if runner_config is not _UNSET:
        actual = temporal_credit_config_snapshot(runner_config, control_dt_s=control_dt_s)
        errors.extend("resolved runner: " + error for error in temporal_credit_config_errors(actual, protocol=protocol))
        if actual != binding:
            errors.append("resolved runner temporal credit disagrees with the immutable plan")
    return errors


def temporal_credit_snapshot_errors(binding, snapshot, *, protocol):
    """Compare a JSON snapshot with the declaration, retaining strict scalar types."""
    errors = temporal_credit_config_errors(binding, protocol=protocol)
    if errors:
        return errors
    errors.extend("temporal snapshot: " + error for error in temporal_credit_config_errors(snapshot, protocol=protocol))
    if snapshot != binding:
        errors.append("temporal snapshot disagrees with the immutable plan")
    return errors


def temporal_credit_admission_errors(admission):
    """Validate metadata for the one typed constructor discount exception.

    The constructor also checks the actual class object's identity. These JSON
    fields describe that source without importing a policy or its framework.
    """
    if not isinstance(admission, dict) or set(admission) != set(_TEMPORAL_ADMISSION):
        return ["temporal credit constructor admission has undeclared fields"]
    protocol = admission.get("stair_trial_protocol")
    if protocol not in TEMPORAL_PROTOCOLS:
        return ["temporal credit constructor admission has unknown protocol"]
    errors = []
    for key, expected in _TEMPORAL_ADMISSION.items():
        if key == "stair_trial_protocol":
            expected = protocol
        actual = admission[key]
        valid = actual is True if key == "enabled" else (_number(actual) if key in ("gamma", "gae_lambda") else type(actual) is str)
        if not valid or actual != expected:
            errors.append("temporal credit constructor admission has incorrect " + key)
    return errors


def rollout_credit_collection_contract(*, protocol):
    """Declare the new rollout and its separately sealed fixed-bank prefix."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS:
        raise ValueError("unknown rollout credit collection protocol")
    steps = TEMPORAL_CREDIT_BY_PROTOCOL[protocol]["rollout_steps"]
    return dict(training_rollout_steps=steps,
        fixed_bank_collection_controls=FIXED_BANK_COLLECTION_CONTROLS,
        collection_rollout_steps_inclusive=[0, FIXED_BANK_COLLECTION_CONTROLS-1],
        seal_hook="post_inherited_process_env_step", seal_storage_step=FIXED_BANK_COLLECTION_CONTROLS,
        seal_rnd_rollout_update_counter=FIXED_BANK_COLLECTION_CONTROLS, completed_updates_at_seal=0,
        first_bootstrap_pending_at_seal=True, current_rows_per_rollout=1024,
        current_collection_controls=steps,
        current_rows_per_control=[1024//steps + int(k < 1024 % steps) for k in range(steps)],
        total_guard_rows=5120)


def _exact_json_value(actual, expected):
    if type(expected) is list:
        return (type(actual) is list and len(actual) == len(expected)
                and all(_exact_json_value(a, e) for a, e in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def clearance_progress_contract(*, protocol):
    if protocol not in CLEARANCE_PROGRESS_PROTOCOLS:
        raise ValueError("unknown clearance progress protocol")
    return dict(kind="per_foot_real_riser_clearance_potential_v1", enabled=True, gamma=0.99,
        inner_coefficient=1.0, outer_potential_weight=1.0, foot_components=2,
        score_bounds=[0.0, 1.0], added_event_bounds=[-2.0, 1.98],
        extra_height_m=0.02, approach_m=1.4, depth_m=0.35, real_riser_count=6,
        lookahead_m=0.20, rear_margin_m=0.008, positive_tilt_limit_rad=0.5,
        positive_gate="original_force_overspeed_times_unloaded_other70pct_both_forcevalid_tilt_lt0p5_same_riser_both_near",
        negative_gate="none_per_foot_debt_preserved", initial_state="actual_precontrol_full_sole_geometry",
        terminal_state="zero_before_reward_on_termination_timeout_or_ever_invalid",
        old_potential="independently_gated_before_addition", actor_observation_added=False)


def clearance_progress_config_errors(binding, config=_UNSET, *, protocol):
    if protocol not in CLEARANCE_PROGRESS_PROTOCOLS:
        return ["unknown clearance progress protocol"]
    expected = clearance_progress_contract(protocol=protocol)
    if not isinstance(binding, dict) or set(binding) != set(expected):
        return ["clearance_progress_potential requires exactly its declared fields"]
    errors = ["clearance progress has incorrect " + key for key, value in expected.items()
              if not _exact_json_value(binding[key], value)]
    if config is not _UNSET:
        actual = dict(enabled=True, gamma=0.99, outer_potential_weight=1.0,
                      strict_episode_force=True, approach_m=1.4, depth_m=0.35, real_riser_count=6)
        if not isinstance(config, dict) or set(config) != set(actual):
            errors.append("resolved clearance reward config has undeclared fields")
        else:
            errors.extend("resolved clearance config has incorrect " + key for key, value in actual.items()
                          if not _exact_json_value(config[key], value))
    return errors


def clearance_progress_snapshot_errors(binding, snapshot, *, protocol, expected_updates=None):
    errors = clearance_progress_config_errors(binding, protocol=protocol)
    if errors:
        return errors
    if not isinstance(snapshot, dict):
        return ["missing live clearance reward state"]
    config = {key: snapshot.get(key) for key in ("enabled", "gamma", "outer_potential_weight",
              "strict_episode_force", "approach_m", "depth_m", "real_riser_count")}
    errors.extend(clearance_progress_config_errors(binding, config, protocol=protocol))
    if (clearance_progress_config_errors(snapshot.get("contract"), protocol=protocol)
            or snapshot.get("contract") != binding or snapshot.get("configuration_enabled") is not True):
        errors.append("live clearance config/declaration binding disagrees")
    if type(snapshot.get("foot_components")) is not int or snapshot["foot_components"] != 2:
        errors.append("live clearance history is not per foot")
    if not _exact_json_value(snapshot.get("positive_tilt_limit_rad"), 0.5):
        errors.append("live clearance positive posture threshold changed")
    for key in ("update_count", "initialization_count", "initialized_envs"):
        if type(snapshot.get(key)) is not int or snapshot[key] < 0:
            errors.append("invalid live clearance " + key)
    if type(snapshot.get("initialized_envs")) is int and snapshot["initialized_envs"] > 1024:
        errors.append("live clearance initialized environment count exceeds source")
    if expected_updates is not None and snapshot.get("update_count") != expected_updates:
        errors.append("live clearance update count differs from actual control collection")
    if type(snapshot.get("last_control_counter")) is not int or snapshot["last_control_counter"] < -1:
        errors.append("invalid live clearance control epoch")
    # Only new reward metadata uses this FP32 representation envelope. Original
    # physical/numerical gates and their5e-6/2e-7 authorities are untouched.
    ranges = {"score_min": (0.0, 1.0), "score_max": (0.0, 1.0),
              "last_added_event_min": (-2.0, 1.98), "last_added_event_max": (-2.0, 1.98),
              "last_raw_delta_min": (-1.0, 0.99), "last_raw_delta_max": (-1.0, 0.99)}
    for key, (low, high) in ranges.items():
        value = snapshot.get(key)
        epsilon = _FLOAT32_RELATIVE_TOLERANCE * max(abs(low), abs(high))
        if not _number(value) or not low-epsilon <= value <= high+epsilon:
            errors.append("nonfinite or out-of-bound live clearance " + key)
    for low, high in (("score_min", "score_max"), ("last_added_event_min", "last_added_event_max"),
                      ("last_raw_delta_min", "last_raw_delta_max")):
        if _number(snapshot.get(low)) and _number(snapshot.get(high)) and snapshot[low] > snapshot[high]:
            errors.append("unordered live clearance range")
    return errors


def rollout_credit_collection_errors(binding, *, protocol):
    """Check declared 48-current/24-fixed metadata without framework imports."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS:
        return ["unknown rollout credit collection protocol"]
    expected = rollout_credit_collection_contract(protocol=protocol)
    if not isinstance(binding, dict) or set(binding) != set(expected):
        return ["rollout_collection requires exactly its declared fields"]
    return ["rollout_collection has incorrect " + key for key, value in expected.items()
            if not _exact_json_value(binding[key], value)]


def rollout_credit_budget_errors(record, *, protocol):
    """Bind the new source budget; historical parent counts are outside it."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS or not isinstance(record, dict):
        return ["unknown rollout credit budget protocol"]
    expected = dict(trial_protocol=protocol, checkpoint_format=PROTOCOL_FORMATS[protocol],
                    num_envs=1024, max_iterations=1000, num_steps_per_env=48,
                    expected_total_timesteps=49152000)
    return ["rollout credit budget has incorrect " + key for key, value in expected.items()
            if not _exact_json_value(record.get(key), value)]


def rollout_credit_acquisition_contract(*, protocol):
    """Keep the17-field acquisition declaration separate from current quotas."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS:
        raise ValueError("unknown rollout credit acquisition protocol")
    return dict(kind="new_run_first_rollout_actual_preaction_v1", status="planned_new_run_not_acquired",
        fixed_rows=4096, current_rows=1024, phases=4, rows_per_phase=1024,
        proposed_phase_controls=[[0,5],[6,11],[12,17],[18,23]],
        collection_window="First actual24 controls0..23 of the single48-control rollout with actor/std unchanged, all optimizer counters0 and actual RND24. Seal immediately after inherited process_env_step stores transition24, before next act24. The first bootstrap/update waits until48 real stored transitions. No extra NN forward, second rollout or old-data fill.",
        admission="All actual raw1034/native limits/K/caps finite, valid lower<upper/K>0/cap>0, actual forcebit0or1 allowed, existing sampledraw12 finite/unclipped at±100 and native fulltanh finite within softlimits. This is an input/request schema, not safe/walking/teacher valid_label.",
        source_policy="Independently accepted direct flat926 same-width parent; fresh stair run first24controls, no old dataset/heldout/formal inputs.",
        freeze="ROOTsource/tools/plan/oldparent+ancestors/old216dataset frozen from launch. Only declared acquisition output may populate in the preactor window; then seal immutable manifest/hash. No other active source exceptions.",
        shortfall="Preserve actual acquisition evidence and safely stop at specified windowend if exact4096 real stratified valid rows unavailable. No synthetic fill, fallback, retry or extension.",
        no_synthetic_context=True, no_old_dataset_padding=True, no_heldout_or_formal_rows=True,
        phase_step_quotas=[171,171,171,171,170,170], seal_hook="post_inherited_process_env_step")


def rollout_credit_acquisition_errors(binding, *, protocol):
    """Validate only the new profile's original17 acquisition fields."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS:
        return ["unknown rollout credit acquisition protocol"]
    expected = rollout_credit_acquisition_contract(protocol=protocol)
    if not isinstance(binding, dict) or set(binding) != set(expected):
        return ["audit_bank_source requires exactly its17 declared fields"]
    return ["audit_bank_source has incorrect " + key for key, value in expected.items()
            if not _exact_json_value(binding[key], value)]


def rollout_credit_seal_errors(receipt, *, protocol):
    """Validate actual prefix-seal evidence, without reading bank arrays."""
    if protocol not in ROLLOUT_CREDIT_PROTOCOLS or not isinstance(receipt, dict):
        return ["unknown rollout credit seal protocol"]
    contract = rollout_credit_collection_contract(protocol=protocol)
    errors = rollout_credit_collection_errors(
        {key: receipt.get(key) for key in contract}, protocol=protocol)
    errors.extend(temporal_credit_config_errors(receipt.get("temporal_credit"), protocol=protocol))
    actual_prefix_rows = sum(contract["current_rows_per_control"][:FIXED_BANK_COLLECTION_CONTROLS])
    expected = dict(trial_protocol=protocol, status="sealed", rows=4096, completed_controls=24,
        rnd_rollout_update_counter=24, source_actor_optimizer_steps=0, source_critic_optimizer_steps=0,
        source_rnd_optimizer_steps=0, first_bootstrap_pending=True,
        current_controls_at_seal=24, current_rows_at_seal=actual_prefix_rows,
        full_guard_bank_assembled_at_seal=False)
    errors.extend("rollout credit seal has incorrect " + key for key, value in expected.items()
                  if not _exact_json_value(receipt.get(key), value))
    controls = receipt.get("control_steps")
    if (type(controls) is not list or len(controls) != 24
            or any(type(value) is not int for value in controls)
            or controls != list(range(controls[0], controls[0]+24))):
        errors.append("rollout credit seal lacks its first24 contiguous actual controls")
    return errors


def initialization_config_errors(binding, runner_config=_UNSET):
    """Return errors for the exact plan binding and optional resolved runner dict.

    The immutable plan, request and run use maximum_angle_std_rad; the existing
    runner dictionary retains its maximum_angle_std constructor keyword.
    """
    errors = []
    if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        return ["exploration_initialization requires exactly its three declared fields"]
    for key, expected in (("initial_fraction", 1.0), ("minimum_fraction", 0.05)):
        if not _number(binding[key]) or binding[key] != expected:
            errors.append("exploration_initialization has incorrect " + key)
    angles = binding["maximum_angle_std_rad"]
    if not _positive_vector(angles) or angles != list(MAXIMUM_ANGLE_STD_RAD):
        errors.append("exploration_initialization changed the original twelve angular RMS ceilings")
    if runner_config is not _UNSET:
        if not isinstance(runner_config, dict) or set(runner_config) != _RUNNER_KEYS:
            errors.append("resolved runner exploration config requires exactly its three declared fields")
        else:
            actual = {"initial_fraction": runner_config["initial_fraction"],
                      "minimum_fraction": runner_config["minimum_fraction"],
                      "maximum_angle_std_rad": runner_config["maximum_angle_std"]}
            errors.extend("resolved runner: " + error for error in initialization_config_errors(actual))
            if actual != binding:
                errors.append("resolved runner exploration config disagrees with the immutable plan")
    return errors


def exploration_snapshot_errors(binding, snapshot, *, initial_learning_state=None,
                                at_initialization=False):
    """Validate an initial or later JSON snapshot against the fresh-run binding.

    At initialization the actual policy std and learning-state std must exactly
    equal the original maximum raw std. Later snapshots may learn anywhere in
    the unchanged bounds, while retaining that same recorded initial std.
    """
    errors = initialization_config_errors(binding)
    if errors:
        return errors
    if not isinstance(snapshot, dict):
        return ["missing exploration snapshot"]
    for key in ("initial_fraction", "minimum_fraction"):
        if not _number(snapshot.get(key)) or snapshot[key] != binding[key]:
            errors.append("exploration snapshot disagrees with the declared " + key)
    keys = ("policy_std", "minimum_raw_std", "maximum_raw_std", "initial_raw_std",
            "maximum_angle_rms_rad", "current_angle_rms_upper_bound_rad")
    for key in keys:
        if not _positive_vector(snapshot.get(key)):
            errors.append("exploration snapshot requires twelve finite positive values: " + key)
    if errors:
        return errors
    current, minimum, maximum, initial, angles, current_angles = (snapshot[key] for key in keys)
    if initial != maximum:
        errors.append("fresh exploration initial raw std is not exactly its original maximum")
    if not all(lo <= value <= hi for value, lo, hi in zip(current, minimum, maximum)):
        errors.append("trained policy std left the original raw bounds")
    close = lambda actual, expected: math.isclose(
        actual, expected, rel_tol=_FLOAT32_RELATIVE_TOLERANCE, abs_tol=0.0)
    if not all(close(value, expected) for value, expected in zip(angles, binding["maximum_angle_std_rad"])):
        errors.append("exploration snapshot changed the original angular RMS ceilings")
    if not all(close(lo, hi * binding["minimum_fraction"]) for lo, hi in zip(minimum, maximum)):
        errors.append("exploration snapshot changed the original minimum fraction product")
    if not all(close(value, angle * std / hi)
               for value, angle, std, hi in zip(current_angles, angles, current, maximum)):
        errors.append("exploration snapshot angular RMS products disagree with its raw std")
    if at_initialization and current != initial:
        errors.append("actual initial policy std differs from its configured initial std")
    if initial_learning_state is not None or at_initialization:
        initial_policy_std = (initial_learning_state.get("policy_std")
                              if isinstance(initial_learning_state, dict) else None)
        if not _positive_vector(initial_policy_std) or initial_policy_std != initial:
            errors.append("initial learning-state policy std differs from the actual initial snapshot")
    return errors
