"""CPU checks at the production contact reward/termination call sites.

Only those function ASTs are compiled to avoid importing Isaac Lab. The shared
tensor helper, when present, is loaded from the production file unchanged.
No simulator, checkpoint, policy, or optimizer is constructed.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def _load_term(filename, function_name):
    source = TASK / "mdp" / filename
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    helper_path = TASK / "contact_events.py"
    if helper_path.exists():
        spec = importlib.util.spec_from_file_location("h2_contact_events_cpu_tests", helper_path)
        helper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper_module)
        namespace["stumble_from_force_history"] = helper_module.stumble_from_force_history
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[function_name]


@pytest.fixture(scope="module", params=[("rewards.py", "feet_stumble_norm"), ("terminations.py", "illegal_feet_stumble")])
def term(request):
    return _load_term(*request.param)


def _call(term, forces, body_ids=slice(None), **kwargs):
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"contact_forces": sensor}))
    cfg = SimpleNamespace(name="contact_forces", body_ids=body_ids)
    return term(env, cfg, **kwargs)


def test_previous_support_does_not_hide_later_riser_collision(term):
    forces = torch.tensor([[[[60.0, 0.0, 0.0]], [[0.0, 0.0, 100.0]], [[0.0, 0.0, 0.0]]]])
    assert _call(term, forces).tolist() == [True]


def test_collision_is_separate_from_healthily_supported_or_other_environment(term):
    forces = torch.zeros(4, 3, 2, 3)
    forces[0, 0, 0] = torch.tensor([60.0, 0.0, 20.0])  # Same-time support: no lateral dominance.
    forces[1, 0, 0] = torch.tensor([60.0, 0.0, 0.0])
    forces[1, 1, 0] = torch.tensor([0.0, 0.0, 100.0])
    forces[2, 0, 0] = torch.tensor([0.0, 0.0, 100.0])  # Other foot cannot mask this foot.
    forces[2, 0, 1] = torch.tensor([0.0, 60.0, 0.0])
    result = _call(term, forces)
    assert result.tolist() == [False, True, True, False]
    assert result.dtype == torch.bool and result.shape == (4,) and result.device == forces.device


@pytest.mark.parametrize("body_ids", [[2, 0], slice(0, 3, 2), [2]])
def test_noncontiguous_body_selection_and_order(term, body_ids):
    forces = torch.zeros(2, 3, 4, 3)
    forces[0, 0, 2, 1] = 60.0
    forces[1, 0, 1, 0] = 60.0  # Unselected colliding body.
    assert _call(term, forces, body_ids).tolist() == [True, False]
    assert _call(term, forces, list(reversed(range(4)))).tolist() == [True, True]


@pytest.mark.parametrize("force_ratio", [5.0, 10.0])
@pytest.mark.parametrize("vertical_sign", [-1.0, 1.0])
def test_threshold_is_strict_and_uses_vertical_magnitude(term, force_ratio, vertical_sign):
    forces = torch.zeros(3, 1, 1, 3, dtype=torch.float64)
    forces[:, 0, 0, 0] = torch.tensor([force_ratio - 0.01, force_ratio, force_ratio + 0.01])
    forces[:, 0, 0, 2] = vertical_sign
    assert _call(term, forces, force_ratio=force_ratio).tolist() == [False, False, True]


def test_original_reward_and_termination_multipliers_are_retained():
    forces = torch.tensor([[[[6.0, 0.0, 1.0]]]])
    reward = _load_term("rewards.py", "feet_stumble_norm")
    termination = _load_term("terminations.py", "illegal_feet_stumble")
    assert _call(reward, forces).tolist() == [True]
    assert _call(termination, forces).tolist() == [False]


@pytest.mark.parametrize("body_ids", [[], slice(0, 0)])
@pytest.mark.parametrize("mode", ["per_sample", "independent_max"])
def test_empty_foot_selection_is_false(term, body_ids, mode):
    assert _call(term, torch.zeros(2, 3, 4, 3), body_ids, temporal_mode=mode).tolist() == [False, False]


def _sample_oracle(forces, ratio):
    return [
        any(math.hypot(*force[:2]) > ratio * abs(force[2]) for sample in episode for force in sample)
        for episode in forces.tolist()
    ]


@pytest.mark.parametrize("history_size,feet", [(1, 1), (3, 2), (6, 4)])
@pytest.mark.parametrize("ratio", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_production_terms_match_time_paired_reference(term, history_size, feet, ratio, dtype):
    generator = torch.Generator().manual_seed(42)
    forces = torch.randint(-20, 21, (7, history_size, feet, 3), generator=generator).to(dtype)
    original = forces.clone()
    expected = _sample_oracle(forces, ratio)
    assert _call(term, forces, force_ratio=ratio).tolist() == expected
    torch.testing.assert_close(forces, original, rtol=0, atol=0)
    # Both modes are order invariant, but only the new one pairs each force in time.
    assert _call(term, forces.flip(1), force_ratio=ratio).tolist() == expected


@pytest.mark.parametrize("history_size", [1, 3, 6])
@pytest.mark.parametrize("ratio", [5.0, 10.0])
def test_baseline_exactly_reproduces_independent_historical_maxima(term, history_size, ratio):
    forces = torch.randn(8, history_size, 3, 3, generator=torch.Generator().manual_seed(7))
    horizontal = forces[..., :2].norm(dim=-1).max(dim=1).values
    vertical = forces[..., 2].abs().max(dim=1).values
    legacy = (horizontal > ratio * vertical).any(dim=-1)
    actual = _call(term, forces, temporal_mode="independent_max", force_ratio=ratio)
    torch.testing.assert_close(actual, legacy)
    paired = _call(term, forces, force_ratio=ratio)
    assert (paired | ~legacy).all()  # A true legacy event is also a time-paired event.
    if history_size == 1:
        torch.testing.assert_close(paired, actual)


def test_baseline_retains_original_missed_event_and_no_state_leaks(term):
    forces = torch.tensor([[[[60.0, 0.0, 0.0]], [[0.0, 0.0, 100.0]]]])
    assert _call(term, forces, temporal_mode="independent_max").tolist() == [False]
    assert _call(term, forces).tolist() == [True]
    assert _call(term, torch.zeros_like(forces)).tolist() == [False]


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_force_ratio_is_rejected(term, ratio):
    with pytest.raises(ValueError, match="force_ratio"):
        _call(term, torch.zeros(1, 1, 1, 3), force_ratio=ratio)


def test_unknown_temporal_mode_is_rejected(term):
    with pytest.raises(ValueError, match="temporal_mode"):
        _call(term, torch.zeros(1, 1, 1, 3), temporal_mode="max")


def test_sensor_history_requires_a_time_sample(term):
    with pytest.raises(ValueError, match="force_history"):
        _call(term, torch.zeros(1, 0, 2, 3))


def test_production_functions_import_the_actual_shared_helper():
    for name in ("rewards.py", "terminations.py"):
        tree = ast.parse((TASK / "mdp" / name).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.level == 2 and node.module == "contact_events"
            and any(alias.name == "stumble_from_force_history" and alias.asname is None for alias in node.names)
            for node in tree.body
        )
