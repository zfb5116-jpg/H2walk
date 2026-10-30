"""Reward-mode wiring and resume contracts, using config containers and spies."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_walking_profiles import _config, _load_profile_module, _apply, _resolved_profile
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
    for name in ("walk_signed_progress", "walk_alternating_steps", "walk_support_quality"):
        assert getattr(cfg.rewards, name).func is getattr(namespace["h2_mdp"], name)
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
