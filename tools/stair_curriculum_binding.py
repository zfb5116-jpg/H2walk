"""Metadata-only declarations for the human-approved target-cohort course."""
import math
import struct

PROTOCOL = "accepted_flat_fixed_target_height_course_stair_v10"
FORMAT = "h2_guarded_stair_ppo_v10"


def early_policy_contract():
    return dict(kind="fixed_target15_saved500_progress_v1",
        status="active_for_subsequent_training_decisions", original_source_cap=1000,
        continue_upper_cap=1000, decision_checkpoint_iteration=500,
        required_native_window_iterations=[299, 399, 499], required_complete_episodes_per_window=5000,
        fixed_target_env_ids_inclusive=[0, 255], fixed_target_riser_height_m=0.15,
        curriculum_promoted_envs_in_target_window=False,
        incomplete_nonfinite_action="needs_attention_no_signal_no_extension")


def _same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_json(a,b) for a,b in zip(actual,expected))
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_same_json(actual[k],v) for k,v in expected.items())
    return actual == expected


def early_policy_errors(value):
    if not isinstance(value, dict):
        return ["missing target-only user decision policy"]
    expected = early_policy_contract()
    if set(value) != set(expected) | {"human_course_choice_file", "human_course_choice_sha256", "original_formal_protocol_preserved"}:
        return ["target-only early policy has undeclared fields"]
    errors = ["target-only early policy changed " + key for key, item in expected.items()
              if not _same_json(value.get(key), item)]
    if not isinstance(value.get("human_course_choice_file"), str) or not isinstance(value.get("human_course_choice_sha256"), str):
        errors.append("target-only policy lacks direct human choice binding")
    if value.get("original_formal_protocol_preserved") is not True:
        errors.append("original formal contract was not preserved")
    return errors


def course_contract():
    return dict(kind="fixed_target_real_height_course_v1", heights_m=[0.05, 0.10, 0.15],
        num_envs=1024, target_ids_inclusive=[0, 255], curriculum_ids_inclusive=[256, 1023],
        target_row=2, initial_curriculum_row=0, promotion_window=5, promotion_required=4,
        certificate_minimum_s=20.0, certificate_levels=2, certificate_tilt_lt_rad=0.5,
        certificate_positive_root_rise=True, certificate_transfers=True,
        terminal_force_and_physical_recheck=True, initial_partial_excluded=True,
        promotion_only_at_true_reset=True, maximum_promotions_per_env=2,
        original_root_x_m=[-0.25, 0.25], target_window_episodes=5000,
        target_window_iterations=[299, 399, 499], target_cohort_only=True,
        incomplete_window_action="needs_attention_no_signal_no_extension",
        formal_heights_m=[0.15, 0.15], formal_root_x_m=[-0.25, 0.25],
        formal_first_envs=1024, formal_duration_s=90.0, formal_required=922)


def course_config_errors(binding):
    expected = course_contract()
    if type(binding) is not dict or set(binding) != set(expected):
        return ["height course requires its exact declared fields"]
    def exact(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(b, list):
            return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    return ["height course changed " + k for k in expected if not exact(binding[k], expected[k])]


def course_snapshot_errors(binding, value, *, expected_controls=None):
    errors = course_config_errors(binding)
    if not isinstance(value, dict):
        return errors + ["missing actual course snapshot"]
    if not _same_json(value.get("contract"), binding) or value.get("cohort_assignment_before_observations") is not True:
        errors.append("actual initial cohort/config binding disagrees")
    for name in ("controls", "target_completed_episodes", "curriculum_completed_episodes", "promotions"):
        if type(value.get(name)) is not int or value[name] < 0:
            errors.append("invalid course counter " + name)
    if expected_controls is not None and value.get("controls") != expected_controls:
        errors.append("course control count differs from actual rollout")
    counts = value.get("actual_row_counts")
    if (type(counts) is not list or len(counts) != 3
            or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != 1024
            or counts[2] < 256 or value.get("fixed_target_all_row2") is not True):
        errors.append("actual cohort rows disagree")
    if not _same_json(value.get("initial_row_counts"), [768, 0, 256]):
        errors.append("actual initial row distribution changed")
    if type(value.get("promotions")) is int and value["promotions"] > 1536:
        errors.append("course exceeded two promotions per curriculum environment")
    return errors


def profile_snapshot_errors(binding, video_binding, value, *, expected_controls=None):
    from training_video import video_config_errors
    if type(value) is not dict or set(value) != {"height_course", "training_video", "credit_coverage"}:
        return ["missing exact live course/video/credit profile"]
    errors = course_snapshot_errors(binding, value["height_course"], expected_controls=expected_controls)
    errors.extend(video_config_errors(video_binding, allow_legacy_view=True))
    video = value["training_video"]
    if not isinstance(video, dict) or not _same_json(video.get("contract"), video_binding):
        return errors + ["live video config differs"]
    for k in ("attempts", "accepted_frames", "manual_render_calls"):
        if type(video.get(k)) is not int or video[k] < 0:
            errors.append("invalid video counter " + k)
    if (video.get("attempts", 501) > 500 or video.get("accepted_frames") != video.get("attempts")
            or video.get("manual_render_calls") != video.get("attempts", -5) + 4
            or video.get("has_rtx_sensors") is not False):
        errors.append("video exceeded its admitted path/frame bounds")
    observed = video.get("observed_controls")
    if (type(observed) is not list or len(observed) != 3
            or any(type(n) is not int or not 0 <= n <= cap for n, cap in zip(observed, (500, 1000, 1000)))):
        errors.append("invalid video control windows")
    elif sum(n // 5 for n in observed) != video.get("attempts"):
        errors.append("video capture cadence differs")
    probe = value["credit_coverage"]
    n = probe.get("controls") if isinstance(probe, dict) else None
    if type(n) is not int or n < 0 or expected_controls is not None and n != expected_controls:
        return errors + ["credit coverage missed actual controls"]
    if (probe.get("terminal_before_reset") is not True or probe.get("new_actor_observation") is not False
            or type(probe.get("extra_NN_calls")) is not int or probe["extra_NN_calls"] != 0):
        errors.append("credit observation boundary changed")
    counts = probe.get("counts")
    if (type(counts) is not list or len(counts) != 6
            or any(type(row) is not list or len(row) != 12
                   or any(type(x) is not int or x < 0 for x in row) for row in counts)):
        errors.append("invalid credit intersection counts")
    elif (any(counts[i][0] != 0 for i in (0, 1)) or counts[2][0] != 512*n
            or sum(counts[i][0] for i in (3, 4, 5)) != 1536*n):
        errors.append("credit target/course denominator changed")
    for k, width in (("actual_reward_integrals", 19),
                     ("raw_positive_negative_paid_positive_negative_delta_integrals", 4)):
        matrix = probe.get(k)
        if (type(matrix) is not list or len(matrix) != 6
                or any(type(row) is not list or len(row) != width
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in row) for row in matrix)):
            errors.append("invalid actual credit integrals " + k)
    fixed = probe.get("fixed_input_height_counts")
    current = probe.get("current_input_height_counts")
    for name, rows in (("fixed", fixed), ("current", current)):
        if type(rows) is not list or len(rows) != 3 or any(type(x) is not int or x < 0 for x in rows):
            errors.append("invalid selected-input actual heights " + name)
    if n >= 24 and fixed != [3072, 0, 1024]:
        errors.append("fixed input heights disagree with actual first24 all-env strata")
    if type(current) is list and len(current) == 3:
        steps = n % 48 or (48 if n else 0)
        expected_rows = sum((22 if k < 16 else 21) for k in range(steps))
        if sum(current) != expected_rows:
            errors.append("current input height accounting lost selected rows")
    return errors


def target_window_errors(native, index):
    """Never manufacture zeros from incomplete target-only evidence."""
    errors = []
    if native.get("iteration") != index or native.get("completed_iterations") != index + 1:
        errors.append("incorrect target native epoch")
    course = native.get("height_course", {})
    errors.extend(course_snapshot_errors(course_contract(), course, expected_controls=(index+1)*48))
    target = native.get("target15_progress", {})
    target_f32 = struct.unpack("f", struct.pack("f", .15))[0]
    if (not _same_json(target.get("cohort_ids_inclusive"), [0, 255])
            or not _same_json(target.get("authored_riser_height_m"), 0.15)
            or not _same_json(target.get("actual_riser_height_m"), target_f32)
            or not _same_json(target.get("actual_target_riser_min_max_m"), [target_f32, target_f32])):
        errors.append("wrong fixed target cohort")
    m = target.get("motion", {})
    for k in ("episode_count", "window_episode_count", "completed_episode_count",
              "physical_valid_episode_count", "invalid_physical_episode_count"):
        if type(m.get(k)) is not int or m[k] < 0:
            errors.append("invalid target counter " + k)
    if (m.get("episode_count") != 5000 or m.get("window_episode_count") != 5000
            or m.get("completed_episode_count", -1) < 5000
            or m.get("completed_episode_count") != course.get("target_completed_episodes")):
        errors.append("incomplete target5000 window")
    if (m.get("physical_valid_episode_count", -1) + m.get("invalid_physical_episode_count", -1)) != 5000:
        errors.append("target failure denominator changed")
    if m.get("initial_partial_episodes_excluded") is not True:
        errors.append("target initial-partial semantics changed")
    for k, cap in (("mean_verified_feet", 12), ("mean_completed_levels", 6),
                   ("mean_transferred_levels", 6), ("first_level_bilateral_fraction", 1),
                   ("all_levels_bilateral_fraction", 1), ("stair_success_fraction", 1)):
        v = m.get(k)
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= cap:
            errors.append("invalid target metric " + k)
    return errors
