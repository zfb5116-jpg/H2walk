"""Pure JSON contract for a bounded observer of the existing execution path."""
import math

PROTOCOL = "accepted_flat_execution_chain_observed_height_course_stair_v11"
FORMAT = "h2_guarded_stair_ppo_v11"


def execution_contract():
    return dict(kind="existing_env0_policy_native_execution_chain_terrain_v2", enabled=True,
        env_id=0, completed_updates_inclusive=[480, 500], maximum_controls=1000,
        physics_substeps_per_control=4, maximum_physics_substeps=4000,
        control_dt_s=0.02, physics_dt_s=0.005,
        policy_source="existing_PPO_transition_action_mean_action_sigma_actions",
        native_source="after_original_Articulation_write_data_to_sim_before_physics",
        endpoint_source="original_credit_observer_post_reward_before_reset",
        effort_semantics="existing_implicit_PD_computed_and_clipped_estimates_only",
        true_PhysX_PD_drive_torque_available=False,
        extra_actor_forwards=0, extra_RNG_draws=0, extra_physics_steps=0,
        terrain_input=dict(source="existing_transition_and_already_computed_lidar_caches",
            normalized_actor_slice=[468,1029],height_grid_yx=[17,33],
            capture_phase="preaction_after_real_PPO_act_before_physics",
            observation_producer_epoch="action_control_minus1",
            missing_or_stale="explicit_null_status_no_recompute_no_zero_fill",
            sensor_pose="existing_latest_pose_ring_only_no_sensor_data_access",
            extra_NN_forwards=0,extra_raycasts=0,new_observation_channels=0),
        new_policy_observation=False, original_command_result_returned=True)


def _same(actual, expected):
    if isinstance(expected, dict):
        return (type(actual) is dict and actual.keys() == expected.keys()
                and all(_same(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return type(actual) is list and len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def execution_config_errors(value):
    return [] if _same(value, execution_contract()) else ["wrong exact execution-chain contract"]


def selected_controls_after(controls):
    if type(controls) is not int or controls < 0:
        raise ValueError("actual control count must be a nonnegative integer")
    return min(1000, max(0, controls - 480 * 48))


def execution_snapshot_errors(binding, snapshot, *, expected_controls=None):
    errors = execution_config_errors(binding)
    if errors or type(snapshot) is not dict:
        return errors or ["missing actual execution-chain snapshot"]
    if not _same(snapshot.get("contract"), binding):
        errors.append("execution-chain live contract differs")
    n = snapshot.get("controls_seen")
    if type(n) is not int or n < 0 or expected_controls is not None and n != expected_controls:
        return errors + ["execution-chain control source differs"]
    expected = selected_controls_after(n)
    if (snapshot.get("pending_control") is not None
            or type(snapshot.get("recorded_controls")) is not int or snapshot["recorded_controls"] != expected
            or type(snapshot.get("recorded_physics_substeps")) is not int
            or snapshot["recorded_physics_substeps"] != 4 * expected):
        errors.append("execution-chain complete control/substep count differs")
    available = snapshot.get("terrain_input_records")
    unavailable = snapshot.get("terrain_input_unavailable_records")
    if (type(available) is not int or type(unavailable) is not int
            or available < 0 or unavailable < 0 or available+unavailable != expected):
        errors.append("execution-chain terrain diagnostic record accounting differs")
    if (snapshot.get("original_native_write_bound") is not True
            or snapshot.get("read_only") is not True
            or snapshot.get("true_PhysX_PD_drive_torque_available") is not False):
        errors.append("execution-chain native estimate/read-only boundary differs")
    names = snapshot.get("action_joint_names")
    ids = snapshot.get("action_joint_ids")
    if (type(names) is not list or len(names) != 12 or any(type(x) is not str or not x for x in names)
            or len(set(names)) != 12 or type(ids) is not list or len(ids) != 12
            or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != 12):
        errors.append("execution-chain actual joint binding differs")
    if snapshot.get("original_native_write_source") != dict(module="isaaclab.assets.articulation.articulation",
                                                           qualname="Articulation.write_data_to_sim"):
        errors.append("execution-chain actual native writer identity differs")
    if expected:
        if snapshot.get("first_recorded_control") != 23041 or snapshot.get("last_recorded_control") != 23040 + expected:
            errors.append("execution-chain selected window offsets differ")
    elif snapshot.get("first_recorded_control") is not None or snapshot.get("last_recorded_control") is not None:
        errors.append("execution-chain claimed an unobserved record")
    return errors


def finite_json(value):
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_json(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)
