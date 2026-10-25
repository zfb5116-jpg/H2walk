"""Register the real policy function against isolated RSL module doubles.

No real RSL runner, environment, model or optimizer is imported or constructed.
The runner double deliberately starts with a pre-existing globals namespace,
matching the v2.3.3 resolver's eval(class_name) lookup location.
"""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


POLICY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py"
)


def _load_registration(policy_class):
    tree = ast.parse(POLICY_SOURCE.read_text(encoding="utf-8"), filename=str(POLICY_SOURCE))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "register_rsl_rl_policy")
    namespace = {"H2ActorCriticOur": policy_class}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(POLICY_SOURCE), "exec"), namespace)
    return namespace[function.name]


@pytest.fixture
def rsl_modules(monkeypatch):
    package = ModuleType("rsl_rl")
    package.__path__ = []
    modules = ModuleType("rsl_rl.modules")
    runners = ModuleType("rsl_rl.runners")
    runners.__path__ = []
    runner = ModuleType("rsl_rl.runners.on_policy_runner")
    package.modules = modules
    package.runners = runners
    runners.on_policy_runner = runner
    for module in (package, modules, runners, runner):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class ExistingActorCritic:
        pass

    class CustomPolicy:
        pass

    modules.ActorCritic = ExistingActorCritic
    runner.ActorCritic = ExistingActorCritic
    return SimpleNamespace(modules=modules, runner=runner, policy=CustomPolicy,
                           existing=ExistingActorCritic, register=_load_registration(CustomPolicy))


def test_policy_is_resolved_from_both_modules_and_preexisting_runner_globals(rsl_modules):
    with pytest.raises(NameError):
        eval("H2ActorCriticOur", vars(rsl_modules.runner))
    rsl_modules.register()
    assert eval("H2ActorCriticOur", vars(rsl_modules.modules)) is rsl_modules.policy
    assert eval("H2ActorCriticOur", vars(rsl_modules.runner)) is rsl_modules.policy
    assert rsl_modules.modules.ActorCritic is rsl_modules.runner.ActorCritic is rsl_modules.existing


def test_repeated_registration_is_idempotent(rsl_modules):
    for _ in range(3):
        rsl_modules.register()
        assert rsl_modules.modules.H2ActorCriticOur is rsl_modules.policy
        assert rsl_modules.runner.H2ActorCriticOur is rsl_modules.policy


@pytest.mark.parametrize("missing_name", ["rsl_rl", "rsl_rl.modules"])
def test_missing_rsl_modules_retains_installation_error_and_original_cause(monkeypatch, missing_name):
    original_import = builtins.__import__
    missing = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)

    def failed_import(name, *args, **kwargs):
        if name == "rsl_rl.modules":
            raise missing
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failed_import)
    with pytest.raises(ImportError, match="Install Isaac Lab/RSL-RL") as error:
        _load_registration(type("CustomPolicy", (), {}))()
    assert error.value.__cause__ is missing


def test_internal_modules_import_error_is_not_swallowed(monkeypatch):
    original_import = builtins.__import__
    failure = ImportError("A dependency inside rsl_rl.modules could not be imported")

    def failed_import(name, *args, **kwargs):
        if name == "rsl_rl.modules":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failed_import)
    with pytest.raises(ImportError, match="Install Isaac Lab/RSL-RL") as error:
        _load_registration(type("CustomPolicy", (), {}))()
    assert error.value.__cause__ is failure


@pytest.mark.parametrize("internal", [False, True])
def test_runner_import_failure_propagates_without_partial_registration(rsl_modules, monkeypatch, internal):
    original_import = builtins.__import__
    failure = (ImportError("A dependency inside the runner failed") if internal else
               ModuleNotFoundError("No module named 'rsl_rl.runners.on_policy_runner'",
                                   name="rsl_rl.runners.on_policy_runner"))

    def failed_import(name, *args, **kwargs):
        if name == "rsl_rl.runners.on_policy_runner":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failed_import)
    with pytest.raises(ImportError) as error:
        rsl_modules.register()
    assert error.value is failure
    assert not hasattr(rsl_modules.modules, "H2ActorCriticOur")
    assert not hasattr(rsl_modules.runner, "H2ActorCriticOur")


def test_modules_are_imported_before_runner_and_no_policy_is_constructed(rsl_modules, monkeypatch):
    original_import = builtins.__import__
    import_order = []

    def observed_import(name, *args, **kwargs):
        if name in {"rsl_rl.modules", "rsl_rl.runners.on_policy_runner"}:
            import_order.append(name)
        return original_import(name, *args, **kwargs)

    class MustNotConstruct:
        def __init__(self):
            raise AssertionError("Registration must not construct the policy")

    monkeypatch.setattr(builtins, "__import__", observed_import)
    _load_registration(MustNotConstruct)()
    assert import_order == ["rsl_rl.modules", "rsl_rl.runners.on_policy_runner"]
    assert rsl_modules.modules.H2ActorCriticOur is rsl_modules.runner.H2ActorCriticOur is MustNotConstruct
