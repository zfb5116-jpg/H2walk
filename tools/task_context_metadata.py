"""Shared, standard-library-only metadata admission for the explicit context.

These checks bind declarations and resolved configuration records. They do
not collect an audit bank, inspect tensors, prove a human authorization, or
certify normalization, neural equivalence, or physical walking. Legacy
protocol admission remains in its original callers.
"""

from collections.abc import Mapping
import json
import math
import struct


CONTEXT_PROTOCOL = "explicit_task_context_v1"
CONTEXT_FORMAT = "h2_guarded_flat_ppo_v10"
FORMAT = CONTEXT_FORMAT
CONTEXT_KIND = "controlled_root_link_heading_velocity_world_goal_force_eligibility_v1"
CONTEXT_ORDER = (
    "root_link_heading_vx_m_s",
    "root_link_heading_vy_m_s",
    "world_plusX_heading_x",
    "world_plusX_heading_y",
    "force_credit_eligible",
)
CONTEXT_DIMS = (1034, 871, 73, 12)
CONTEXT_PARENT_KIND = "saved_complete_PPO_v6_context_extension_transfer"
CONTEXT_INITIALIZATION_MODE = "extend_completed_v6_policy_and_frozen_prefix_norm_fresh_optimization"
CONTEXT_PARENT_SHA256 = "fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa"
CONTEXT_REWARD_PROFILE = "overspeed_outward_tilt_growth_clean_force_v1"
CONTEXT_BANK_KIND = "new_run_first_rollout_actual_preaction_v1"
CONTEXT_NORMALIZATION_KIND = "frozen_empirical_prefix_fixed_task_context_v1"
_VELOCITY_CONTEXT_SCALE_FLOAT32 = struct.unpack("<f", struct.pack("<f", 1.0 / 0.3))[0]
_FIXED_CONTEXT_SCALE = (
    _VELOCITY_CONTEXT_SCALE_FLOAT32, _VELOCITY_CONTEXT_SCALE_FLOAT32, 1.0, 1.0, 1.0,
)

_PREFIX_NAMES = {
    "policy": (
        "velocity_commands", "base_ang_vel", "projected_gravity",
        "joint_pos_rel", "joint_vel_rel", "last_action", "height_scanner",
    ),
    "critic": (
        "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos_rel", "joint_vel_rel", "last_action", "joint_effort", "ddq",
        "feet_pos_b", "contact_state", "contact_force", "height_scanner",
    ),
}
_PREFIX_WIDTHS = {
    "policy": (3, 15, 15, 145, 145, 145, 561),
    "critic": (3, 3, 3, 3, 29, 29, 29, 29, 35, 6, 34, 102, 561),
}
_PREFIX_HISTORY = {"policy": (0, 5, 5, 5, 5, 5, 0), "critic": (0,) * 13}
_PHASE_CONTROLS = ((0, 5), (6, 11), (12, 17), (18, 23))
_PHASE_STEP_QUOTAS = (171, 171, 171, 171, 170, 170)


def require(condition, message):
    """Use a local error type so metadata readers have no evaluator dependency."""
    if not condition:
        raise RuntimeError(message)


def _mapping(value, path):
    require(isinstance(value, Mapping), path + " must be an object")
    return value


def _field(value, key, path):
    _mapping(value, path)
    require(key in value, path + "." + key + " is required")
    return value[key]


def _exact(value, expected, path):
    if expected is None:
        require(value is None, path + " must be null")
    elif type(expected) is bool:
        require(value is expected, path + " must be " + str(expected))
    elif type(expected) is int:
        require(type(value) is int and value == expected,
                path + " must be the exact integer " + str(expected))
    elif type(expected) is float:
        require(type(value) in (int, float) and math.isfinite(value) and value == expected,
                path + " differs from the declared numeric value")
    elif isinstance(expected, str):
        require(isinstance(value, str) and value == expected,
                path + " differs from the declared value")
    elif isinstance(expected, (tuple, list)):
        require(isinstance(value, (tuple, list)) and len(value) == len(expected),
                path + " has the wrong ordered width")
        for index, (actual, wanted) in enumerate(zip(value, expected)):
            _exact(actual, wanted, f"{path}[{index}]")
    else:
        raise RuntimeError("Unsupported internal expected metadata type at " + path)


def _expect(record, expected, path):
    for key, value in expected.items():
        _exact(_field(record, key, path), value, path + "." + key)


def _text(value, path):
    require(isinstance(value, str) and bool(value.strip()), path + " must be nonempty text")


def _actual_copy(value, path):
    """Return the supplied metadata as JSON data, adding no invented proof."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(path + " must contain finite JSON metadata only") from error


def validate_task_context_plan(plan):
    """Validate the pinned new-protocol declarations; return its actual context.

    The prior-report record and bank rules are requirements, not evidence that
    a human report or a physical acquisition occurred. Their independent
    provenance and the actual acquisition receipt remain caller obligations.
    """
    plan = _mapping(plan, "plan")
    _expect(plan, {
        "trial_protocol": CONTEXT_PROTOCOL,
        "checkpoint_format": CONTEXT_FORMAT,
        "network_or_input_structure_change": True,
        "network_dimensions": CONTEXT_DIMS,
        "initialization_mode": CONTEXT_INITIALIZATION_MODE,
        "initial_parent_kind": CONTEXT_PARENT_KIND,
        "initial_policy_sha256": CONTEXT_PARENT_SHA256,
        "parent_natural_completion": True,
        "parent_runtime_rollback_completed": False,
        "parent_accepted_updates": 1000,
        "parent_last_complete_iteration": 999,
        "starting_iteration": 0,
        "checkpoint": None,
        "num_steps_per_env": 24,
        "fixed_real_audit_dataset": None,
    }, "plan")
    labels = _mapping(_field(plan, "network_dimension_labels", "plan"), "plan.network_dimension_labels")
    _expect(labels, {"actor": 1034, "critic": 871, "rnd": 73, "action": 12},
            "plan.network_dimension_labels")
    report = _mapping(_field(plan, "prior_user_input_change_report", "plan"),
                      "plan.prior_user_input_change_report")
    _expect(report, {"reported_before_any_source_change": True}, "plan.prior_user_input_change_report")
    _text(_field(report, "report", "plan.prior_user_input_change_report"),
          "plan.prior_user_input_change_report.report")
    context = _mapping(_field(plan, "task_context", "plan"), "plan.task_context")
    _expect(context, {
        "kind": CONTEXT_KIND,
        "raw_context_width": 5,
        "actor_prefix_width": 1029,
        "critic_prefix_width": 866,
        "context_order": CONTEXT_ORDER,
        "history": 1,
        "append_after_old_full_groups": True,
        "no_old_prefix_reorder_or_zero_slot_reuse": True,
        "expected_actor_mean_tensors": 44,
        "expected_actor_Adam_tensors": 45,
        "expected_critic_tensors": 8,
        "expected_mean_elements": 782988,
        "expected_jacobian_row_bytes": 37583424,
        "expected_arena_capacity_bytes": 1052335872,
        "derived_arena_rows": 28,
        "simulator_state_not_hardware_sensor_claim": True,
        "not_full_Markov_claim": True,
    }, "plan.task_context")
    require(context["expected_mean_elements"] * 12 * 4 == context["expected_jacobian_row_bytes"],
            "plan.task_context Jacobian row byte arithmetic is inconsistent")
    require(1073741824 // context["expected_jacobian_row_bytes"] == context["derived_arena_rows"]
            and context["derived_arena_rows"] * context["expected_jacobian_row_bytes"]
            == context["expected_arena_capacity_bytes"],
            "plan.task_context arena arithmetic is inconsistent")
    bank = _mapping(_field(plan, "audit_bank_source", "plan"), "plan.audit_bank_source")
    _expect(bank, {
        "kind": CONTEXT_BANK_KIND,
        "fixed_rows": 4096,
        "current_rows": 1024,
        "phases": 4,
        "rows_per_phase": 1024,
        "proposed_phase_controls": _PHASE_CONTROLS,
        "phase_step_quotas": _PHASE_STEP_QUOTAS,
        "seal_hook": "compute_returns_before_original_bootstrap",
        "no_synthetic_context": True,
        "no_old_dataset_padding": True,
        "no_heldout_or_formal_rows": True,
    }, "plan.audit_bank_source")
    require(sum(bank["phase_step_quotas"]) == bank["rows_per_phase"],
            "plan.audit_bank_source per-control quotas must total one actual phase")
    for key in ("collection_window", "admission", "source_policy", "freeze", "shortfall"):
        _text(_field(bank, key, "plan.audit_bank_source"), "plan.audit_bank_source." + key)
    for key in ("dataset", "source_dataset", "context_padding_dataset"):
        if key in bank:
            require(bank[key] is None, "New context bank cannot use " + key)
    return _actual_copy(context, "plan.task_context")


def validate_task_context_metadata(runtime_flat_geometry, plan=None):
    """Validate B's resolved raw observation ABI; return the actual context.

    In particular external_block_normalization_proven_here=False is correct:
    the environment emits raw SI and cannot certify the entry's separate
    fixed-block normalizer. This function never evaluates an observation.
    """
    runtime = _mapping(runtime_flat_geometry, "runtime_flat_geometry")
    if plan is not None:
        validate_task_context_plan(plan)
    if "trial_protocol" in runtime:
        _exact(runtime["trial_protocol"], CONTEXT_PROTOCOL, "runtime_flat_geometry.trial_protocol")
    if "reward_profile" in runtime:
        _exact(runtime["reward_profile"], CONTEXT_REWARD_PROFILE, "runtime_flat_geometry.reward_profile")
    context = _mapping(_field(runtime, "task_context", "runtime_flat_geometry"),
                       "runtime_flat_geometry.task_context")
    path = "runtime_flat_geometry.task_context"
    _expect(context, {
        "kind": CONTEXT_KIND,
        "raw_context_width": 5,
        "configured_context_width": 5,
        "global_policy_input_changed_by_explicit_context": True,
        "context_order": CONTEXT_ORDER,
        "actor_prefix_width": 1029,
        "critic_prefix_width": 866,
        "actor_shape": (1034,),
        "critic_shape": (871,),
        "rnd_shape": (73,),
        "history": 1,
        "append_after_old_full_groups": True,
        "no_old_prefix_reorder_or_zero_slot_reuse": True,
        "schema_bootstrap_closed": True,
        "missing_real_progress_is_error": True,
        "observation_function_is_pure_read": True,
        "raw_velocity_units": "m/s",
        "raw_goal_units": "unit vector",
        "raw_eligibility_values": (0, 1),
        "environment_context_scale": 1.0,
        "external_block_normalization_proven_here": False,
        "simulator_state_not_hardware_sensor_claim": True,
        "not_full_Markov_claim": True,
        "velocity_point": "robot.data.root_link_lin_vel_w; root rigid-body link origin, not body COM or whole-body COM",
        "velocity_frame": "yaw-only Rz(root_yaw).T times world velocity; XY in m/s",
        "goal_frame": "yaw-only Rz(root_yaw).T times world+X unit vector; XY",
        "eligibility_source": "flat_balance_progress.force_credit_eligible (existing cumulative mismatch sum==0)",
        "eligibility_timing": "stored previous-control qualification at preaction; after true reset use that new episode; never future force",
        "schema_bootstrap": "explicit constructor-only flag and common_step_counter==0; closed after base constructor",
        "context_block_normalization_owner": "learning/evaluation entry, separate from old frozen prefix",
    }, path)
    groups = _mapping(_field(context, "actual_groups", path), path + ".actual_groups")
    for name, shape in (("policy", 1034), ("critic", 871)):
        group_path = path + ".actual_groups." + name
        group = _mapping(_field(groups, name, path + ".actual_groups"), group_path)
        _expect(group, {
            "term_names": _PREFIX_NAMES[name] + ("task_context",),
            "term_shapes": tuple((width,) for width in _PREFIX_WIDTHS[name]) + ((5,),),
            "prefix_width": sum(_PREFIX_WIDTHS[name]),
            "group_shape": (shape,),
            "context_function": "flat_task_context",
            "context_history_length": 1,
            "context_scale": 1.0,
            "context_clip": None,
            "context_noise": None,
            "prefix_history_lengths": _PREFIX_HISTORY[name],
        }, group_path)
        _text(_field(group, "context_function_module", group_path), group_path + ".context_function_module")
    return _actual_copy(context, path)


def validate_normalization_metadata(abi, prefixcounts):
    """Bind the separate frozen-prefix/fixed-context normalization declaration.

    The scales are the exact values represented by the production float32
    buffer, not Python's unrounded 1/.3. Counts describe only the original
    empirical prefixes. Actual state-dict tensors, buffers, mode, and prefix
    transfer still require the caller's independent checks.
    """
    abi = _mapping(abi, "normalization_abi")
    counts = _mapping(prefixcounts, "normalizer_prefix_counts")
    require(set(counts) == {"actor", "critic"},
            "normalizer_prefix_counts must describe only actor and critic prefixes")
    _expect(counts, {"actor": 24576000, "critic": 24576000}, "normalizer_prefix_counts")
    _expect(abi, {
        "kind": CONTEXT_NORMALIZATION_KIND,
        "raw_context_width": 5,
        "context_order": CONTEXT_ORDER,
        "fixed_context_scale": _FIXED_CONTEXT_SCALE,
        "context_statistics_fitted": False,
        "context_empirical_count": None,
        "frozen": True,
    }, "normalization_abi")
    embedded_counts = _mapping(_field(abi, "prefix_empirical_counts", "normalization_abi"),
                               "normalization_abi.prefix_empirical_counts")
    require(set(embedded_counts) == {"actor", "critic"},
            "Embedded normalizer counts must describe only the two prefixes")
    for name in ("actor", "critic"):
        _exact(_field(embedded_counts, name, "normalization_abi.prefix_empirical_counts"),
               counts[name], "normalization_abi.prefix_empirical_counts." + name)
    for name, prefix_width, input_width in (("actor", 1029, 1034), ("critic", 866, 871)):
        block = _mapping(_field(abi, name, "normalization_abi"), "normalization_abi." + name)
        _expect(block, {
            "prefix_width": prefix_width,
            "input_width": input_width,
            "prefix_epsilon": 0.01,
            "prefix_until": 100000000.0,
        }, "normalization_abi." + name)
    for key in ("count", "full_width_empirical_count", "fullwidth_empirical_count"):
        require(key not in abi, "Normalization must not invent a full-width empirical " + key)
    return _actual_copy(abi, "normalization_abi")
