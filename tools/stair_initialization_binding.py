"""JSON-only identity, exploration and temporal-credit stair bindings.

These predicates inspect metadata only. They do not construct a policy, load a
checkpoint, or change the inherited exploration arithmetic or optimizer hook.
"""
import math


PROTOCOL = "accepted_flat_temporal_credit_stair_v7"
FORMAT = "h2_guarded_stair_ppo_v7"
LEGACY_PROTOCOL = "accepted_flat_block_consistent_stair_clearance_v5"
LEGACY_INITIALIZATION_PROTOCOL = "accepted_flat_full_rms_exploration_stair_v6"
PROTOCOL_FORMATS = {LEGACY_PROTOCOL: "h2_guarded_stair_ppo_v5",
                    LEGACY_INITIALIZATION_PROTOCOL: "h2_guarded_stair_ppo_v6", PROTOCOL: FORMAT}
INITIALIZATION_PROTOCOLS = (LEGACY_INITIALIZATION_PROTOCOL, PROTOCOL)
MAXIMUM_ANGLE_STD_RAD = (0.02, 0.015, 0.015, 0.02, 0.01, 0.0075) * 2
_BINDING_KEYS = {"initial_fraction", "minimum_fraction", "maximum_angle_std_rad"}
_RUNNER_KEYS = {"initial_fraction", "minimum_fraction", "maximum_angle_std"}
_TEMPORAL_CREDIT = {"gamma": 0.99, "gae_lambda": 0.99, "rollout_steps": 24, "control_dt_s": 0.02}
_TEMPORAL_ADMISSION = dict(enabled=True, algorithm_class="H2GuardedStairPPO",
    algorithm_module="guarded_stair_transfer", stair_trial_protocol=PROTOCOL,
    numeric_execution_profile="explicit_task_context_longer_budget_v1", gamma=0.99, gae_lambda=0.99)
_UNSET = object()
# Products and quotients in the existing float32 snapshot are compared within
# four float32 relative epsilons. Raw initial==maximum and bound checks are exact.
_FLOAT32_RELATIVE_TOLERANCE = 4 * 2.0 ** -23


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


def temporal_credit_config_errors(binding, runner_config=_UNSET, *, control_dt_s=_UNSET):
    """Validate the exact declaration and optionally its resolved runner source.

    The runner keeps its original gamma/lam and num_steps_per_env field names.
    Its environment's resolved control period must be supplied explicitly.
    """
    errors = []
    if not isinstance(binding, dict) or set(binding) != set(_TEMPORAL_CREDIT):
        return ["temporal_credit requires exactly gamma, gae_lambda, rollout_steps and control_dt_s"]
    for key, expected in _TEMPORAL_CREDIT.items():
        actual = binding[key]
        valid = type(actual) is int if key == "rollout_steps" else _number(actual)
        if not valid or actual != expected:
            errors.append("temporal_credit has incorrect " + key)
    if runner_config is not _UNSET:
        actual = temporal_credit_config_snapshot(runner_config, control_dt_s=control_dt_s)
        errors.extend("resolved runner: " + error for error in temporal_credit_config_errors(actual))
        if actual != binding:
            errors.append("resolved runner temporal credit disagrees with the immutable plan")
    return errors


def temporal_credit_snapshot_errors(binding, snapshot):
    """Compare a JSON snapshot with the declaration, retaining strict scalar types."""
    errors = temporal_credit_config_errors(binding)
    if errors:
        return errors
    errors.extend("temporal snapshot: " + error for error in temporal_credit_config_errors(snapshot))
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
    errors = []
    for key, expected in _TEMPORAL_ADMISSION.items():
        actual = admission[key]
        valid = actual is True if key == "enabled" else (_number(actual) if key in ("gamma", "gae_lambda") else type(actual) is str)
        if not valid or actual != expected:
            errors.append("temporal credit constructor admission has incorrect " + key)
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
