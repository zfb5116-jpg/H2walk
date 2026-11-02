"""Five physical context channels for direct stair transfer; legacy prefix unchanged."""
import torch
from .flat_task_context import (heading_task_context, TASK_CONTEXT_KIND, TASK_CONTEXT_DIM,
                                TASK_CONTEXT_ORDER, TASK_CONTEXT_TERM, _PREFIX_LAYOUT, _PREFIX_HISTORY)


def stair_task_context(env):
    """Pure-read observation of current physical state and stored qualification."""
    raw = env.unwrapped
    robot = raw.scene["robot"].data
    progress = getattr(raw, "stair_progress", None)
    if progress is None:
        # ObservationManager probes dimensions inside the base constructor,
        # before StairAscentProgress exists. This flag is closed immediately
        # when that constructor returns; it is never a real-rollout fallback.
        if (getattr(raw, "_stair_task_context_schema_bootstrap", False) is not True
                or getattr(raw, "common_step_counter", None) != 0):
            raise RuntimeError("Real direct-stair observations require episode qualification state")
        eligible = torch.ones(robot.root_quat_w.shape[0], device=robot.root_quat_w.device,
                              dtype=torch.bool)
    else:
        eligible = progress.force_credit_eligible
    return heading_task_context(robot.root_quat_w, robot.root_link_lin_vel_w, eligible)


def append_stair_task_context(observations):
    """Append independent terms after the already-configured original groups."""
    from isaaclab.managers import ObservationTermCfg

    for name, layout in _PREFIX_LAYOUT.items():
        group = getattr(observations, name)
        before = tuple(group.__dict__.items())
        terms = tuple(key for key, value in before if isinstance(value, ObservationTermCfg))
        if terms != tuple(key for key, _ in layout):
            raise RuntimeError(f"Original direct-stair {name} term order changed")
        if hasattr(group, TASK_CONTEXT_TERM) or group.history_length is not None:
            raise RuntimeError("Context cannot replace a term or override original histories")
        if not group.concatenate_terms or group.concatenate_dim != -1:
            raise RuntimeError("Context requires the original flat concatenated groups")
        setattr(group, TASK_CONTEXT_TERM, ObservationTermCfg(
            func=stair_task_context, history_length=1, flatten_history_dim=True,
            scale=1.0, noise=None, clip=None,
        ))
        # Appending is the only mutation: retain every old config object,
        # including LiDAR params, noise settings, joint order, and histories.
        if any(group.__dict__.get(key) is not value for key, value in before):
            raise RuntimeError("Appending task context changed an original observation setting")


def stair_task_context_runtime_contract(env):
    """Read resolved ObservationManager configuration without evaluating a term."""
    if type(env.cfg.stair_task_context_dim) is not int or env.cfg.stair_task_context_dim != TASK_CONTEXT_DIM:
        raise RuntimeError("Actual stair context declaration must be exactly five values")
    manager = env.observation_manager
    actual = {}
    for name, layout in _PREFIX_LAYOUT.items():
        names = list(manager.active_terms[name])
        dimensions = [list(map(int, shape)) for shape in manager.group_obs_term_dim[name]]
        expected_names = [key for key, _ in layout] + [TASK_CONTEXT_TERM]
        expected_dimensions = [[width] for _, width in layout] + [[TASK_CONTEXT_DIM]]
        if names != expected_names or dimensions != expected_dimensions:
            raise RuntimeError(f"Actual {name} context/prefix ABI differs from the declared layout")
        if not manager.group_obs_concatenate[name]:
            raise RuntimeError("Actual context group is not concatenated")
        # The installed SDK exposes resolved names/dimensions publicly and
        # stores their matching term configs in this parallel list.
        configs = manager._group_obs_term_cfgs[name]
        if len(configs) != len(names):
            raise RuntimeError("ObservationManager resolved-term list is inconsistent")
        if tuple(term.history_length for term in configs[:-1]) != _PREFIX_HISTORY[name]:
            raise RuntimeError("Actual original observation histories changed")
        term = configs[-1]
        actual_scale = float(term.scale) if term.scale is not None else None
        if (term.func is not stair_task_context or term.history_length != 1
                or not term.flatten_history_dim or actual_scale != 1.0
                or term.noise is not None or term.clip is not None or term.modifiers is not None):
            raise RuntimeError("Actual raw context preprocessing differs from SI identity")
        actual[name] = {
            "term_names": names,
            "term_shapes": dimensions,
            "prefix_width": sum(width for _, width in layout),
            "group_shape": list(map(int, manager.group_obs_dim[name])),
            "context_function": term.func.__name__,
            "context_function_module": term.func.__module__,
            "context_history_length": int(term.history_length),
            "context_scale": actual_scale,
            "context_clip": None,
            "context_noise": None,
            "prefix_history_lengths": [int(term.history_length) for term in configs[:-1]],
        }
    if tuple(manager.group_obs_dim["rnd_state"]) != (73,):
        raise RuntimeError("The independent RND73 observation changed")
    if getattr(env, "stair_progress", None) is None:
        raise RuntimeError("Runtime context contract requires the actual episode state")
    if getattr(env, "_stair_task_context_schema_bootstrap", False):
        raise RuntimeError("Schema bootstrap must be closed before runtime admission")
    return {
        "kind": TASK_CONTEXT_KIND,
        "raw_context_width": TASK_CONTEXT_DIM,
        "configured_context_width": env.cfg.stair_task_context_dim,
        "global_policy_input_changed_by_explicit_context": True,
        "context_order": list(TASK_CONTEXT_ORDER),
        "actor_prefix_width": actual["policy"]["prefix_width"],
        "critic_prefix_width": actual["critic"]["prefix_width"],
        "actor_shape": actual["policy"]["group_shape"],
        "critic_shape": actual["critic"]["group_shape"],
        "rnd_shape": list(map(int, manager.group_obs_dim["rnd_state"])),
        "actual_groups": actual,
        "velocity_point": "robot.data.root_link_lin_vel_w; root rigid-body link origin, not body COM or whole-body COM",
        "velocity_frame": "yaw-only Rz(root_yaw).T times world velocity; XY in m/s",
        "goal_frame": "yaw-only Rz(root_yaw).T times world+X unit vector; XY",
        "eligibility_source": "stair_progress.force_credit_eligible (existing cumulative mismatch sum==0)",
        "eligibility_timing": "stored previous-control qualification at preaction; after true reset use that new episode; never future force",
        "history": 1,
        "append_after_old_full_groups": True,
        "no_old_prefix_reorder_or_zero_slot_reuse": True,
        "schema_bootstrap": "explicit constructor-only flag and common_step_counter==0; closed after base constructor",
        "schema_bootstrap_closed": True,
        "missing_real_progress_is_error": True,
        "observation_function_is_pure_read": True,
        "raw_velocity_units": "m/s",
        "raw_goal_units": "unit vector",
        "raw_eligibility_values": [0, 1],
        "environment_context_scale": 1.0,
        "context_block_normalization_owner": "learning/evaluation entry, separate from old frozen prefix",
        "external_block_normalization_proven_here": False,
        "simulator_state_not_hardware_sensor_claim": True,
        "not_full_Markov_claim": True,
    }
