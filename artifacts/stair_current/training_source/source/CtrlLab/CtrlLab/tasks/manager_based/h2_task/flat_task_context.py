"""Explicit raw task context for the direct flat policy only.

The observation reads the present root-link state and the already-latched
episode qualification. It does not update rewards, contacts, or any clock.
The separate learning entry owns normalization of this five-value block.
"""

import torch


TASK_CONTEXT_KIND = "controlled_root_link_heading_velocity_world_goal_force_eligibility_v1"
TASK_CONTEXT_DIM = 5
TASK_CONTEXT_TERM = "task_context"
TASK_CONTEXT_ORDER = (
    "root_link_heading_vx_m_s",
    "root_link_heading_vy_m_s",
    "world_plusX_heading_x",
    "world_plusX_heading_y",
    "force_credit_eligible",
)

# Existing term names and flattened dimensions, before the appended block.
_PREFIX_LAYOUT = {
    "policy": (
        ("velocity_commands", 3), ("base_ang_vel", 15),
        ("projected_gravity", 15), ("joint_pos_rel", 145),
        ("joint_vel_rel", 145), ("last_action", 145),
        ("height_scanner", 561),
    ),
    "critic": (
        ("base_lin_vel", 3), ("base_ang_vel", 3),
        ("projected_gravity", 3), ("velocity_commands", 3),
        ("joint_pos_rel", 29), ("joint_vel_rel", 29),
        ("last_action", 29), ("joint_effort", 29), ("ddq", 35),
        ("feet_pos_b", 6), ("contact_state", 34),
        ("contact_force", 102), ("height_scanner", 561),
    ),
}
_PREFIX_HISTORY = {
    "policy": (0, 5, 5, 5, 5, 5, 0),
    "critic": (0,) * 13,
}


def heading_task_context(root_quaternion, root_link_velocity_world, force_eligible):
    """Return SI velocity, unit goal direction, and a 0/1 eligibility value.

    Quaternions are scalar-first, as in the existing articulation data. Only
    yaw rotates the horizontal velocity; roll, pitch, and vertical velocity
    do not change which point or horizontal plane the task controls.
    """
    if root_quaternion.ndim != 2 or root_quaternion.shape[1] != 4:
        raise ValueError("Task context requires N x4 scalar-first root quaternions")
    n = root_quaternion.shape[0]
    if tuple(root_link_velocity_world.shape) != (n, 3):
        raise ValueError("Task context requires N x3 root-link world velocity")
    if tuple(force_eligible.shape) != (n,) or force_eligible.dtype != torch.bool:
        raise ValueError("Task context requires N boolean episode qualifications")
    if (root_quaternion.device != root_link_velocity_world.device
            or force_eligible.device != root_link_velocity_world.device
            or root_quaternion.dtype != root_link_velocity_world.dtype):
        raise ValueError("Task context inputs must share the physical state device/dtype")
    if not bool(torch.isfinite(root_quaternion).all()
                & torch.isfinite(root_link_velocity_world).all()):
        raise RuntimeError("Non-finite direct-flat task context state")
    w, x, y, z = root_quaternion.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    vx, vy = root_link_velocity_world[:, 0], root_link_velocity_world[:, 1]
    return torch.stack((cosine * vx + sine * vy,
                        -sine * vx + cosine * vy,
                        cosine, -sine,
                        force_eligible.to(dtype=root_link_velocity_world.dtype)), dim=-1)


def flat_task_context(env):
    """Pure-read observation of current physical state and stored qualification."""
    raw = env.unwrapped
    robot = raw.scene["robot"].data
    progress = getattr(raw, "flat_balance_progress", None)
    if progress is None:
        # ObservationManager probes dimensions inside the base constructor,
        # before FlatBalanceProgress exists. This flag is closed immediately
        # when that constructor returns; it is never a real-rollout fallback.
        if (getattr(raw, "_flat_task_context_schema_bootstrap", False) is not True
                or getattr(raw, "common_step_counter", None) != 0):
            raise RuntimeError("Real direct-flat observations require episode qualification state")
        eligible = torch.ones(robot.root_quat_w.shape[0], device=robot.root_quat_w.device,
                              dtype=torch.bool)
    else:
        eligible = progress.force_credit_eligible
    return heading_task_context(robot.root_quat_w, robot.root_link_lin_vel_w, eligible)


def append_flat_task_context(observations):
    """Append independent terms after the already-configured original groups."""
    from isaaclab.managers import ObservationTermCfg

    for name, layout in _PREFIX_LAYOUT.items():
        group = getattr(observations, name)
        before = tuple(group.__dict__.items())
        terms = tuple(key for key, value in before if isinstance(value, ObservationTermCfg))
        if terms != tuple(key for key, _ in layout):
            raise RuntimeError(f"Original direct-flat {name} term order changed")
        if hasattr(group, TASK_CONTEXT_TERM) or group.history_length is not None:
            raise RuntimeError("Context cannot replace a term or override original histories")
        if not group.concatenate_terms or group.concatenate_dim != -1:
            raise RuntimeError("Context requires the original flat concatenated groups")
        setattr(group, TASK_CONTEXT_TERM, ObservationTermCfg(
            func=flat_task_context, history_length=1, flatten_history_dim=True,
            scale=1.0, noise=None, clip=None,
        ))
        # Appending is the only mutation: retain every old config object,
        # including LiDAR params, noise settings, joint order, and histories.
        if any(group.__dict__.get(key) is not value for key, value in before):
            raise RuntimeError("Appending task context changed an original observation setting")


def task_context_runtime_contract(env):
    """Read resolved ObservationManager configuration without evaluating a term."""
    if type(env.cfg.flat_task_context_dim) is not int or env.cfg.flat_task_context_dim != TASK_CONTEXT_DIM:
        raise RuntimeError("Actual flat context declaration must be exactly five values")
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
        if (term.func is not flat_task_context or term.history_length != 1
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
    if getattr(env, "flat_balance_progress", None) is None:
        raise RuntimeError("Runtime context contract requires the actual episode state")
    if getattr(env, "_flat_task_context_schema_bootstrap", False):
        raise RuntimeError("Schema bootstrap must be closed before runtime admission")
    return {
        "kind": TASK_CONTEXT_KIND,
        "raw_context_width": TASK_CONTEXT_DIM,
        "configured_context_width": env.cfg.flat_task_context_dim,
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
        "eligibility_source": "flat_balance_progress.force_credit_eligible (existing cumulative mismatch sum==0)",
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
