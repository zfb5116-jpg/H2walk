"""Reward-mode wiring and resume contracts, using config containers and spies."""

import ast
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from test_walking_profiles import TASK, _config, _load_profile_module, _apply, _resolved_profile
from test_walk_entry import module as entry_module
from test_walk_protocol import module as protocol_module, protocol, checkpoint, runner


def test_new_walk_profile_selects_alternation_and_held_target():
    module = _load_profile_module()
    cfg = _resolved_profile(module)
    assert cfg.walking_reward_mode == "alternating_v1"
    assert cfg.walking_reward_version == "h2_walk_rewards_v1"
    assert cfg.rewards.feet_stair_clearance.params["target_mode"] == "nearest_latched"
    assert cfg.rewards.feet_swing_height.params["swing_mode"] == "alternating_v1"
    assert cfg.rewards.feet_swing_height.weight == .75
    for name in ("feet_air_time", "feet_contact_time"):
        assert getattr(cfg.rewards, name).weight == 0.
    for name in ("walk_signed_progress", "walk_alternating_steps", "walk_support_quality"):
        assert getattr(cfg.rewards, name).weight > 0.
    module.validate_walking_profile(cfg)


def test_fresh_legacy_mode_retains_original_formulas_and_weights():
    module = _load_profile_module()
    cfg = _config()
    cfg.rewards.feet_stair_clearance.params["target_mode"] = "nearest"
    before = deepcopy(cfg.rewards.__dict__)
    _apply(module, cfg, "shallow", reward_mode="legacy", cross_time_mode="legacy")
    assert cfg.walking_reward_mode == "legacy"
    for name in ("feet_swing_height", "feet_air_time", "feet_contact_time", "feet_stair_clearance"):
        assert getattr(cfg.rewards, name).__dict__ == before[name].__dict__
    for name in ("walk_signed_progress", "walk_alternating_steps", "walk_support_quality"):
        assert getattr(cfg.rewards, name) is None


@pytest.mark.parametrize("mutate", [
    lambda cfg: setattr(cfg, "walking_reward_mode", "legacy"),
    lambda cfg: setattr(cfg, "walking_reward_version", "unknown"),
    lambda cfg: cfg.walking_reward_parameters.update(min_step_distance=.1),
    lambda cfg: setattr(cfg.rewards.feet_air_time, "weight", .45),
    lambda cfg: setattr(cfg.rewards.walk_signed_progress, "weight", 0.),
    lambda cfg: cfg.rewards.feet_stair_clearance.params.update(target_mode="nearest"),
])
def test_reward_mode_cannot_be_relabelled_after_profile_resolution(mutate):
    module = _load_profile_module()
    cfg = _resolved_profile(module)
    mutate(cfg)
    with pytest.raises(ValueError, match="reward|target"):
        module.validate_walking_profile(cfg)


@pytest.mark.parametrize("mode", [None, [], True, "alternating", "v2"])
def test_unknown_reward_mode_fails_before_mutation(mode):
    module = _load_profile_module()
    cfg = _config()
    with pytest.raises(ValueError, match="reward_mode"):
        _apply(module, cfg, "bootstrap", reward_mode=mode)
    assert cfg.scene.terrain.terrain_generator is None


def test_entry_baseline_choice_is_applied_at_config_construction():
    entry = entry_module()
    args = entry.parse_args(["--reward_mode", "legacy", "--cross_time_mode", "legacy"])
    calls = []

    def loader(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(name=name, kwargs=kwargs)

    gym = SimpleNamespace(spec=lambda task: SimpleNamespace(kwargs={
        "env_cfg_entry_point": "fake:Env", "rsl_rl_cfg_entry_point": "fake:Runner",
    }))
    entry.registry_configs(gym, args.task, loader, env_kwargs={
        "walking_reward_mode": args.reward_mode, "walking_cross_time_mode": args.cross_time_mode,
    })
    assert calls == [("fake:Env", {"walking_reward_mode": "legacy", "walking_cross_time_mode": "legacy"}),
                     ("fake:Runner", {})]


def test_walk_class_wires_declared_reward_functions_only_for_walk():
    from test_walking_env_config import _load_configs, _new_cfg

    namespace = _load_configs()
    cfg = _new_cfg(namespace, "RobotWalkBootstrapEnvCfg")
    for name in ("walk_signed_progress", "walk_alternating_steps", "terrain_walk_swing",
                 "terrain_walk_support", "terrain_walk_landing", "terrain_walk_failure"):
        assert getattr(cfg.rewards, name).func is getattr(namespace["h2_mdp"], name)
    assert cfg.walking_reward_mode == "terrain_v2"
    assert cfg.rewards.walk_support_quality.weight == 0.
    legacy = namespace["RobotWalkBootstrapEnvCfg"]()
    legacy.walking_reward_mode = "legacy"
    legacy.__post_init__()
    for name in ("walk_signed_progress", "walk_alternating_steps", "walk_support_quality"):
        assert getattr(legacy.rewards, name) is None
    assert legacy.rewards.feet_swing_height.weight == .75
    assert "swing_mode" not in legacy.rewards.feet_swing_height.params


def _signature_cfg():
    return SimpleNamespace(
        observations={}, actions={}, rewards={"progress": {"weight": 1., "params": {}}},
        walking_reward_mode="alternating_v1", walking_reward_version="h2_walk_rewards_v1",
        walking_reward_parameters={"min_step_distance": .03},
        contact_diagnostic_threshold=1., stage_support_height_tolerance=.015,
        sim=SimpleNamespace(dt=.005), decimation=4,
        scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace(joint_pos={}))),
    )


@pytest.mark.parametrize("change", ["weight", "params", "mode", "version", "state", "force_threshold", "support_tolerance"])
def test_signature_changes_with_reward_semantics(change):
    entry, proto = entry_module(), protocol_module()
    cfg = _signature_cfg()
    agent = SimpleNamespace(policy={}, clip_actions=100.)
    old = entry.resolved_configuration_signature(cfg, agent, proto.configuration_fingerprint)
    if change == "weight":
        cfg.rewards["progress"]["weight"] = 2.
    elif change == "params":
        cfg.rewards["progress"]["params"]["scale"] = .5
    elif change == "mode":
        cfg.walking_reward_mode = "legacy"
    elif change == "version":
        cfg.walking_reward_version = "unknown"
    elif change == "state":
        cfg.walking_reward_parameters["min_step_distance"] = .04
    elif change == "force_threshold":
        cfg.contact_diagnostic_threshold = 2.
    else:
        cfg.stage_support_height_tolerance = .01
    assert entry.resolved_configuration_signature(cfg, agent, proto.configuration_fingerprint) != old


@pytest.mark.parametrize("mutation", ["old_id", "missing_mode", "other_mode", "other_version"])
def test_old_or_changed_reward_protocol_rejects_before_state_application(mutation):
    mod = protocol_module()
    saved = checkpoint(mod)
    marker = saved["infos"]["h2_walk_protocol"]
    if mutation == "old_id":
        marker["id"] = "h2_walk_fixed_scales_v1"
    elif mutation == "missing_mode":
        marker.pop("reward_mode")
    elif mutation == "other_mode":
        marker["reward_mode"] = "legacy"
    else:
        marker["reward_version"] = "unknown"
    value = runner()
    with pytest.raises(ValueError, match="protocol|reward"):
        mod.restore_checkpoint(value, saved, protocol(mod))
    assert value.calls == []


@pytest.mark.parametrize("stage", ["bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20"])
def test_terrain_mode_has_one_explicit_budget_for_every_stage(stage):
    module = _load_profile_module()
    cfg = _resolved_profile(module, stage, reward_mode="terrain_v2")
    active = {name: term.weight for name, term in vars(cfg.rewards).items()
              if term is not None and term.weight != 0.}
    assert active == module.TERRAIN_REWARD_WEIGHTS
    assert len(active) == 22
    assert cfg.walking_reward_version == "h2_walk_rewards_v2"
    assert cfg.walking_reward_parameters == module.TERRAIN_REWARD_PARAMETERS
    assert cfg.rewards.feet_air_time_variance.weight == 0.
    assert cfg.rewards.feet_stair_clearance.weight == cfg.rewards.feet_swing_height.weight == 0.
    module.validate_walking_profile(cfg)


@pytest.mark.parametrize("mutation", ["weight", "parameter", "function", "disabled", "extra", "mode", "version", "state"])
def test_terrain_budget_rejects_post_resolution_semantic_changes(mutation):
    module = _load_profile_module()
    cfg = _resolved_profile(module, reward_mode="terrain_v2")
    if mutation == "weight":
        cfg.rewards.terrain_walk_swing.weight = 1.
    elif mutation == "parameter":
        cfg.rewards.feet_slide.params["new_threshold"] = 1.
    elif mutation == "function":
        cfg.rewards.feet_slide.func = max
    elif mutation == "disabled":
        cfg.rewards.feet_swing_height.weight = .1
    elif mutation == "extra":
        cfg.rewards.extra = SimpleNamespace(func=abs, weight=0., params={})
    elif mutation == "mode":
        cfg.walking_reward_mode = "alternating_v1"
    elif mutation == "version":
        cfg.walking_reward_version = "h2_walk_rewards_v1"
    else:
        cfg.walking_reward_parameters["semantic_version"] = "changed"
    with pytest.raises(ValueError, match="reward|version"):
        module.validate_walking_profile(cfg)


def test_terrain_cli_default_and_explicit_previous_modes_are_available():
    entry = entry_module()
    assert entry.parse_args([]).reward_mode == "terrain_v2"
    for mode in ("legacy", "alternating_v1", "terrain_v2"):
        assert entry.parse_args(["--reward_mode", mode]).reward_mode == mode


def test_all_reward_versions_are_mode_specific_and_cross_mode_restore_rejects():
    mod = protocol_module()
    for mode, version in (("legacy", "h2_walk_rewards_v1"),
                          ("alternating_v1", "h2_walk_rewards_v1"),
                          ("terrain_v2", "h2_walk_rewards_v2")):
        marker = protocol(mod, reward_mode=mode)
        assert marker["reward_version"] == version
        mod.validate_protocol(marker)
    saved = checkpoint(mod)
    value = runner()
    with pytest.raises(ValueError, match="reward|protocol"):
        mod.restore_checkpoint(value, saved, protocol(mod, reward_mode="terrain_v2"))
    assert value.calls == []
    saved["infos"]["h2_walk_protocol"] = protocol(mod, reward_mode="terrain_v2")
    assert mod.validate_checkpoint(saved, protocol(mod, "stairs_20", reward_mode="terrain_v2")) == 10


def test_terrain_budget_closes_over_every_real_rewards_cfg_declaration():
    """Execute the real term/config statements with backend-free containers."""
    from test_walking_env_config import _load_configs

    class Functions:
        def __getattr__(self, name):
            return abs

    path = TASK / "config/rewards_cfg.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    scope = {
        "math": math, "mdp": Functions(), "h2_mdp": Functions(),
        "RewTerm": lambda **kwargs: SimpleNamespace(**{"params": {}, **kwargs}),
        "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        "configclass": lambda cls: cls,
    }
    for name in ("H2_ARM_JOINT_NAMES", "H2_BASE_BODY_NAMES", "H2_FALL_CONTACT_BODY_NAMES",
                 "H2_FOOT_BODY_NAMES", "H2_LEG_JOINT_NAMES", "H2_UNDESIRED_CONTACT_BODY_NAMES",
                 "H2_WAIST_JOINT_NAMES"):
        scope[name] = (name,)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), scope)
    declarations = {name: deepcopy(term) for name, term in vars(scope["RewardsCfg"]).items()
                    if not name.startswith("_")}
    cfg = _resolved_profile(_load_profile_module(), reward_mode="terrain_v2")
    cfg.rewards = SimpleNamespace(**declarations)
    cfg.walking_reward_mode = "terrain_v2"
    _load_configs()["_configure_walking_reward_terms"](cfg)

    source = TASK / "robot_env_cfg.py"
    tracking = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_configure_walking_tracking_rewards")
    tracking_scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tracking], type_ignores=[])), str(source), "exec"), tracking_scope)
    cfg.walking_tracking_mode, cfg.walking_tracking_relative_weight = "command_relative_mixed", .75
    tracking_scope[tracking.name](cfg)
    original_tracking = deepcopy(cfg.rewards.track_lin_vel_xy.params)
    module = _load_profile_module()
    _apply(module, cfg, "shallow", reward_mode="terrain_v2")
    active = {name: term.weight for name, term in vars(cfg.rewards).items()
              if term is not None and term.weight != 0.}
    assert active == module.TERRAIN_REWARD_WEIGHTS
    assert cfg.rewards.track_lin_vel_xy.params == original_tracking
    for name in ("track_forward_progress", "base_linear_velocity", "base_height", "action_rate2",
                 "joint_leg_deviation", "feet_air_vel", "feet_vel", "feet_acc", "feet_clearance",
                 "feet_swing_height", "feet_stair_clearance", "support_force_z_axis", "stand_still"):
        assert getattr(cfg.rewards, name).weight == 0.
    module.validate_walking_profile(cfg)
