"""Import and CLI checks never import Isaac, construct a runner or start main."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "_walk_entry_test", ROOT / "tools/train_walk.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_import_and_help_do_not_import_backend(capsys):
    before = set(sys.modules)
    value = module()
    with pytest.raises(SystemExit) as error:
        value.main(["--help"])
    assert error.value.code == 0
    assert "Unitree-H2-Velocity-Walk-Bootstrap" in capsys.readouterr().out
    assert not any(
        name.startswith(("isaaclab", "rsl_rl", "gymnasium"))
        for name in set(sys.modules) - before
    )


def test_all_six_training_ids_and_default_are_supported():
    value = module()
    parser = value.build_parser()
    assert parser.parse_args([]).task == "Unitree-H2-Velocity-Walk-Bootstrap"
    for suffix in (
        "Bootstrap",
        "Shallow",
        "Stairs08",
        "Stairs12",
        "Stairs16",
        "Stairs20",
    ):
        task = "Unitree-H2-Velocity-Walk-" + suffix
        assert parser.parse_args(["--task", task]).task == task
    with pytest.raises(SystemExit):
        parser.parse_args(["--task", "Unitree-H2-Velocity-Stairs"])


@pytest.mark.parametrize(
    "args",
    [
        ["--num_envs", "0"],
        ["--max_iterations", "0"],
        ["--save_interval", "-1"],
        ["--run_name", "../escape"],
        ["--expected_completed_iterations", "10"],
    ],
)
def test_invalid_cli_rejected_without_backend(args):
    value = module()
    with pytest.raises(SystemExit):
        value.parse_args(args)


def test_registry_resolves_both_configs():
    value = module()
    entries = []

    def load(entry):
        entries.append(entry)
        return SimpleNamespace(entry=entry)

    registry = SimpleNamespace(
        spec=lambda task: SimpleNamespace(
            kwargs={
                "env_cfg_entry_point": "fake:Env",
                "rsl_rl_cfg_entry_point": "fake:Runner",
            }
        )
    )
    env, agent = value.registry_configs(registry, "explicit_task", load)
    assert entries == ["fake:Env", "fake:Runner"]
    assert env.entry == "fake:Env" and agent.entry == "fake:Runner"


def test_signature_includes_actual_wrapper_action_clipping():
    value = module()
    proto_spec = importlib.util.spec_from_file_location(
        "_entry_protocol", ROOT / "tools/walk_protocol.py"
    )
    proto = importlib.util.module_from_spec(proto_spec)
    proto_spec.loader.exec_module(proto)
    env = SimpleNamespace(
        observations={"scale": 0.2},
        actions={"scale": 0.25},
        rewards={"progress": {"weight": 1.0}},
        walking_reward_mode="alternating_v1",
        walking_reward_version="h2_walk_rewards_v1",
        walking_reward_parameters={},
        contact_diagnostic_threshold=1.0,
        stage_support_height_tolerance=.015,
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
        scene=SimpleNamespace(
            robot=SimpleNamespace(init_state=SimpleNamespace(joint_pos={"knee": 0.3}))
        ),
    )
    agent = SimpleNamespace(
        policy={"class_name": "H2ActorCriticOur"}, clip_actions=100.0
    )
    original = value.resolved_configuration_signature(
        env, agent, proto.configuration_fingerprint
    )
    agent.clip_actions = 1.0
    changed = value.resolved_configuration_signature(
        env, agent, proto.configuration_fingerprint
    )
    assert original != changed
