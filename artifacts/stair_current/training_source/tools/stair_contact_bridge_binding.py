"""JSON contract for training-only future contact and local support credit."""
from stair_execution_binding import _same, finite_json

LEGACY_PROTOCOL = "accepted_flat_fixed_target_height_course_anticipatory_support_bridge_v1"
LEGACY_FORMAT = "h2_guarded_stair_ppo_bridge_v1"
PROTOCOL = "accepted_flat_fixed_target_height_course_anticipatory_support_bridge_v2_location_fit"
FORMAT = "h2_guarded_stair_ppo_bridge_v2_location_fit"


def bridge_contract(*, legacy=False):
    result = dict(kind="anticipatory_support_credit_bridge_v2_location_fit", enabled=True,
        input_width=883, output_width=10, layers=[883, 256, 128, 10], parameters=260490,
        independent_actor_parameters=True, critic_input_width=871, received_action_width=12,
        action_feature="tanh_actual_received_pre_transport_delay_action",
        contact_controls=32, recovery_controls=32, pending_controls=64,
        closed_capacity=49152, record_bytes=3840, CPU_cap_bytes=536870912,
        GPU_owned_cap_bytes=134217728, Adam_steps_per_rollout=32, batch_records=512,
        Adam_learning_rate=0.0003, gradient_clip=1.0, private_CPU_seed=20261104,
        warmup_updates=40, calibration_cadence=10, calibration_records=4096,
        lookahead_m=0.60, preparation_m=0.25, retention_floor=0.25,
        potential_weights=[0.25, 0.25, 0.50], gamma=0.99, inner_coefficient=1.0,
        contact_height_tolerance_m=0.02, sole_margin_m=0.008, sole_side_margin_m=0.02,
        recovery_tilt_limit_rad=0.5, final_tilt_limit_rad=0.15,
        final_angular_XY_limit_rad_s=0.75, final_tilt_change_limit_rad=0.01,
        final_controls=4, loaded_controls=4, unloaded_controls=2,
        contact_force_N=5.0, support_weight_fraction=0.7,
        calibration_minimums=dict(joint_labels=2048, joint_positive=128, joint_negative=128,
            contact_positive=128, contact_negative=128, contact_locations=1024),
        calibration_thresholds=dict(Brier_max=0.2, ECE10_max=0.1, AUROC_min=0.65,
            location_median_m_max=0.10, location_p90_m_max=0.25,
            fully_censored_fraction_max=0.75),
        unknown_future_is_censored=True, floor_labels_are_not_stair_feet=True,
        no_new_policy_observation=True, no_joint_command_override=True,
        pre_super_aux_commit=True, physical_rewind=False,
        serving_publication="only_scheduled_calibrated_candidate",
        formal_bridge_enabled=False, original_target500_policy_preserved=True)
    if legacy:
        result.update(kind="anticipatory_support_credit_bridge_v1",layers=[883,128,64,10],
                      parameters=122058,Adam_steps_per_rollout=8)
    return result


def bridge_config_errors(value, *, allow_legacy=False):
    legacy = allow_legacy and type(value) is dict and value.get("kind") == "anticipatory_support_credit_bridge_v1"
    return [] if _same(value, bridge_contract(legacy=legacy)) else ["wrong exact future-contact bridge contract"]


def bridge_snapshot_errors(binding, snapshot, *, expected_controls=None):
    errors = bridge_config_errors(binding,allow_legacy=True)
    if errors or type(snapshot) is not dict:
        return errors or ["missing live contact bridge"]
    if not _same(snapshot.get("contract"), binding) or not finite_json(snapshot):
        errors.append("bridge actual contract or scalar finiteness differs")
    n = snapshot.get("controls")
    if type(n) is not int or n < 0 or expected_controls is not None and n != expected_controls:
        errors.append("bridge actual physical control count differs")
    if snapshot.get("actual_environment_controls") != n or snapshot.get("pending_control") is not None:
        errors.append("bridge has not completed its actual environment control")
    if (snapshot.get("parameters") != binding["parameters"] or snapshot.get("actor_parameters_shared") is not False
            or snapshot.get("CPU_ring_bytes") != (64*1024+49152)*3840
            or snapshot.get("CPU_owned_bytes", 2**63) > binding["CPU_cap_bytes"]
            or snapshot.get("GPU_owned_bytes", 2**63) > binding["GPU_owned_cap_bytes"]):
        errors.append("bridge actual parameter/resource boundary differs")
    for name in ("serving_forwards", "learner_forwards", "calibration_forwards", "backwards",
                 "Adam_attempts", "Adam_commits", "serving_version", "last_reward_model_version",
                 "seal_id", "data_epoch", "anchors", "pending_records", "closed_records"):
        if type(snapshot.get(name)) is not int or snapshot[name] < 0:
            errors.append("invalid bridge counter " + name)
    if n is not None and type(n) is int:
        if snapshot.get("anchors") != n*1024:
            errors.append("bridge anchor denominator differs from actual1024 controls")
        if snapshot.get("serving_forwards", 2**63) > max(0, n-40*48):
            errors.append("bridge extra or premature serving forward")
        cap = ((n+47)//48)*binding["Adam_steps_per_rollout"]
        if (snapshot.get("learner_forwards",2**63)>cap or snapshot.get("Adam_attempts",2**63)>cap
                or snapshot.get("Adam_commits",2**63)>snapshot.get("Adam_attempts",-1)):
            errors.append("bridge auxiliary attempts exceed its actual rollout budget")
    if binding["kind"] == "anticipatory_support_credit_bridge_v2_location_fit":
        if snapshot.get("aux_steps_per_rollout") != 32:
            errors.append("wrong actual32-step auxiliary schedule")
        for name in ("aux_rolled_back_commits","aux_failed_Adam_attempts","skipped_data_rollouts"):
            if type(snapshot.get(name)) is not int or snapshot[name] < 0:
                errors.append("invalid actual auxiliary transaction counter "+name)
    if snapshot.get("original_numeric_arena_unchanged") is not True:
        errors.append("bridge did not preserve the original projection arena")
    return errors
