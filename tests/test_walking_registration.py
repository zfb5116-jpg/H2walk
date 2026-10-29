"""Check opt-in Gym and runner declarations without importing Isaac Lab."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as Box


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def registrations():
    tree = ast.parse((TASK / "__init__.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    records = []
    scope = {"gym": Box(register=lambda **kwargs: records.append(kwargs)),
             "agents": Box(__name__="h2_task.agents"), "__name__": "h2_task"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "registration", "exec"), scope)
    return {record["id"]: record for record in records}


def test_original_tasks_keep_their_baseline_runner():
    records = registrations()
    for suffix in ("", "-Play"):
        assert records["Unitree-H2-Velocity-Stairs" + suffix]["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":H2RoughPPORunnerCfg")


def test_each_explicit_stage_has_matching_train_and_play_tasks():
    records = registrations()
    for suffix in ("Bootstrap", "Shallow", "Stairs08", "Stairs12", "Stairs16", "Stairs20"):
        for play in (False, True):
            name = "Unitree-H2-Velocity-Walk-" + suffix + ("-Play" if play else "")
            record = records[name]
            assert record["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":H2WalkPPORunnerCfg")
            expected = f":RobotWalk{suffix}{'Play' if play else ''}EnvCfg"
            assert record["kwargs"]["env_cfg_entry_point"].endswith(expected)


def test_contact_diagnostic_task_uses_baseline_learning_protocol():
    records = registrations()
    for suffix in ("", "-Play"):
        record = records["Unitree-H2-Velocity-Stairs-Diagnostic" + suffix]
        assert record["kwargs"]["rsl_rl_cfg_entry_point"].endswith(":H2RoughPPORunnerCfg")


def test_new_runner_uses_fixed_observation_scales_without_mutating_old_runner():
    tree = ast.parse((TASK / "agents/rsl_rl_ppo_cfg.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "H2WalkPPORunnerCfg")
    node.decorator_list = []

    class Baseline:
        algorithm = Box(rnd_cfg=Box(weight=25.))
        empirical_normalization = True
        policy = Box(class_name="H2ActorCriticOur")

    scope = {"H2RoughPPORunnerCfg": Baseline, "deepcopy": deepcopy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "walk_runner", "exec"), scope)
    runner = scope["H2WalkPPORunnerCfg"]()
    runner.__post_init__()
    assert runner.empirical_normalization is False
    assert runner.algorithm.rnd_cfg is None
    assert runner.policy.class_name == "H2ActorCriticOur"
    assert Baseline.empirical_normalization is True
    assert Baseline.algorithm.rnd_cfg.weight == 25.
