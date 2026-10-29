"""Protocol-control tests with synthetic dictionaries and spies, never weights."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"


def module():
    spec = importlib.util.spec_from_file_location(
        "_walk_protocol_test", TOOLS / "walk_protocol.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def protocol(mod, stage="bootstrap", mode="bounded"):
    return mod.make_protocol(
        stage,
        mode,
        0.5,
        {"git_commit": None, "source_sha256": "a" * 64},
        configuration_sha256="b" * 64,
    )


def checkpoint(mod):
    return {
        "model_state_dict": {"schema": "synthetic"},
        "optimizer_state_dict": {"schema": "synthetic"},
        "iter": 9,
        "infos": {
            "h2_walk_protocol": protocol(mod),
            "h2_walk_progress": {
                "completed_iterations": 10,
                "total_timesteps": 2400,
                "total_time": 12.5,
            },
        },
    }


def runner():
    calls = []

    def model_load(state, strict=True):
        assert strict is True
        calls.append(("model", state))
        # nn.Module returns _IncompatibleKeys, not RSL ActorCritic's bool.
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    opt = SimpleNamespace(
        load_state_dict=lambda state: calls.append(("optimizer", state)),
        param_groups=[{"lr": 0.0003}],
    )
    return SimpleNamespace(
        alg=SimpleNamespace(
            policy=SimpleNamespace(load_state_dict=model_load),
            optimizer=opt,
            rnd=None,
            learning_rate=0.001,
        ),
        empirical_normalization=False,
        current_learning_iteration=0,
        tot_timesteps=0,
        tot_time=0.0,
        calls=calls,
    )


def test_cross_stage_resume_restores_once_and_advances_iteration():
    mod = module()
    value = runner()
    saved = checkpoint(mod)
    mod.restore_checkpoint(
        value, saved, protocol(mod, "stairs_08"), expected_completed_iterations=10
    )
    assert [kind for kind, _ in value.calls] == ["model", "optimizer"]
    assert value.current_learning_iteration == 10
    assert value.tot_timesteps == 2400
    assert value.tot_time == 12.5
    assert value.alg.learning_rate == 0.0003


@pytest.mark.parametrize(
    "key",
    [
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
        "rnd_state_dict",
        "rnd_optimizer_state_dict",
    ],
)
def test_normalizer_or_rnd_is_rejected_before_any_state_application(key):
    mod = module()
    value = runner()
    saved = checkpoint(mod)
    saved[key] = {}
    with pytest.raises(ValueError, match="fresh run"):
        mod.restore_checkpoint(value, saved, protocol(mod))
    assert value.calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "normalization",
        "shape",
        "mode",
        "bound",
        "iteration",
        "progress",
        "source",
        "expected",
    ],
)
def test_incompatible_or_incomplete_protocol_is_rejected_before_mutation(mutation):
    mod = module()
    saved = checkpoint(mod)
    marker = saved["infos"]["h2_walk_protocol"]
    expected = None
    if mutation == "unknown":
        saved["infos"] = {}
    elif mutation == "normalization":
        marker["empirical_normalization"] = True
    elif mutation == "shape":
        marker["actor_observations"] = 468
    elif mutation == "mode":
        marker["cross_time_mode"] = "legacy"
    elif mutation == "bound":
        marker["max_time_difference"] = 0.4
    elif mutation == "iteration":
        saved["iter"] = True
    elif mutation == "progress":
        saved["infos"]["h2_walk_progress"]["completed_iterations"] = 9
    elif mutation == "source":
        marker["source"]["source_sha256"] = "unknown"
    elif mutation == "expected":
        expected = 11
    value = runner()
    with pytest.raises(ValueError):
        mod.restore_checkpoint(
            value, saved, protocol(mod), expected_completed_iterations=expected
        )
    assert value.calls == []


def test_no_optimizer_restore_is_explicit():
    mod = module()
    value = runner()
    mod.restore_checkpoint(value, checkpoint(mod), protocol(mod), load_optimizer=False)
    assert [kind for kind, _ in value.calls] == ["model"]


def test_same_width_changed_scale_or_order_is_rejected_before_state_application():
    mod = module()
    for changed in (
        {"angular": {"scale": 0.3, "width": 3}, "gravity": {"scale": 1.0, "width": 3}},
        {"gravity": {"scale": 1.0, "width": 3}, "angular": {"scale": 0.2, "width": 3}},
    ):
        original = {
            "angular": {"scale": 0.2, "width": 3},
            "gravity": {"scale": 1.0, "width": 3},
        }
        saved = checkpoint(mod)
        saved["infos"]["h2_walk_protocol"]["configuration_sha256"] = (
            mod.configuration_fingerprint(original)
        )
        expected = protocol(mod)
        expected["configuration_sha256"] = mod.configuration_fingerprint(changed)
        value = runner()
        with pytest.raises(ValueError, match="protocol differs"):
            mod.restore_checkpoint(value, saved, expected)
        assert value.calls == []


def test_configuration_fingerprint_has_stable_callable_and_slice_support():
    mod = module()

    class Config:
        def to_dict(self):
            return {"func": abs, "ids": slice(None), "names": ["left", "right"]}

    assert mod.configuration_fingerprint(Config()) == mod.configuration_fingerprint(
        Config()
    )
    assert mod.configuration_fingerprint(
        {"scale": 0.25}
    ) != mod.configuration_fingerprint({"scale": 0.11875})
    with pytest.raises(ValueError, match="Unsupported"):
        mod.configuration_fingerprint(object())
    with pytest.raises(ValueError, match="Unsupported"):
        mod.configuration_fingerprint({"scale": float("nan")})


def test_every_base_save_receives_producer_marker_and_actual_counters():
    mod = module()
    calls = []

    class Base:
        def __init__(self, env, cfg, **kwargs):
            self.empirical_normalization = False
            self.alg = SimpleNamespace(rnd=None)
            self.current_learning_iteration = 0
            self.tot_timesteps = 24
            self.tot_time = 1.0

        def save(self, path, infos=None):
            calls.append((path, deepcopy(infos)))

    cfg = {
        "empirical_normalization": False,
        "algorithm": {"class_name": "PPO", "rnd_cfg": None},
        "policy": {"class_name": "H2ActorCriticOur"},
    }
    guarded = mod.guarded_runner_class(Base, protocol(mod))
    value = guarded(None, deepcopy(cfg))
    caller_infos = {"note": "retained", "h2_walk_protocol": {"wrong": True}}
    value.save("synthetic_path_0", infos=caller_infos)
    value.current_learning_iteration = 7
    value.tot_timesteps = 192
    value.save("synthetic_path_7")
    assert calls[0][1]["note"] == "retained"
    assert calls[0][1]["h2_walk_protocol"] == protocol(mod)
    assert calls[0][1]["h2_walk_progress"]["completed_iterations"] == 1
    assert calls[1][1]["h2_walk_progress"]["completed_iterations"] == 8
    assert calls[1][1]["h2_walk_progress"]["total_timesteps"] == 192
    assert caller_infos["h2_walk_protocol"] == {"wrong": True}
    bad = deepcopy(cfg)
    bad["empirical_normalization"] = True
    with pytest.raises(ValueError):
        guarded(None, bad)
