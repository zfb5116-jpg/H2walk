"""CPU-only tests; no simulator, policy, checkpoint or optimizer is created."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_recomputed_pure_predicates_distinguish_cooccurrence_from_last_report(module):
    calls = []

    def pure(env, threshold):
        calls.append(threshold)
        return torch.tensor([True, False])

    pure.__module__, pure.__name__ = "isaaclab.envs.mdp.terminations", "illegal_contact"

    class Stateful:
        def __call__(self, env):
            raise AssertionError("Unknown stateful predicates must never run twice")

    terms = {"ground": SimpleNamespace(func=pure, params={"threshold": 1.}),
             "other_contact": SimpleNamespace(func=pure, params={"threshold": 2.}),
             "custom": SimpleNamespace(func=Stateful(), params={})}
    manager = SimpleNamespace(active_terms=list(terms), get_term_cfg=terms.__getitem__,
                              get_term=lambda name: torch.tensor([name == "other_contact", False]))
    result = module.capture_terminal_predicates(SimpleNamespace(num_envs=2, termination_manager=manager), torch.tensor([0]))
    assert calls == [1., 2.]
    assert result["reported_term_flags"]["ground"].tolist() == [False]
    assert all(flags.tolist() == [True] for flags in result["recomputed_predicate_flags"].values())
    assert result["predicates_not_recomputed"] == ("custom",)


@pytest.mark.parametrize("output", [torch.ones(2), torch.ones(2, 1, dtype=torch.bool), None])
def test_invalid_recomputed_predicate_output_is_an_error(module, output):
    def pure(env):
        return output

    pure.__module__, pure.__name__ = "isaaclab.envs.mdp.terminations", "time_out"
    manager = SimpleNamespace(active_terms=["term"], get_term_cfg=lambda name: SimpleNamespace(func=pure, params={}),
                              get_term=lambda name: torch.zeros(2, dtype=torch.bool))
    with pytest.raises(RuntimeError, match="invalid tensor"):
        module.capture_terminal_predicates(SimpleNamespace(num_envs=2, termination_manager=manager), torch.tensor([0]))


PATH = Path(__file__).resolve().parents[1] / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/contact_diagnostics.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("h2_contact_diagnostics_cpu", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs(n=1, t=3, m=2, dtype=torch.float32):
    return (torch.zeros(n, t, 1, 3, dtype=dtype),
            torch.zeros(n, t, 1, m, 3, dtype=dtype),
            torch.zeros(n, t, 1, m, 3, dtype=dtype))


def test_all_exclusive_window_categories(module):
    net, ground, robot = inputs(n=6)
    net[:5, 0, 0, 2] = 1.01
    ground[0, 0, 0, 0, 2] = 1.01
    robot[1, 0, 0, 0, 2] = 1.01
    ground[2, 0, 0, 0, 2] = 1.01
    robot[2, 0, 0, 1, 0] = 2
    ground[4, 0, 0, 0, 2] = 1.01
    ground[4, 1, 0, 0, 2] = torch.nan
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [1, 2, 3, 4, -1, 0]
    assert result["net_trigger"].tolist() == [True, True, True, True, True, False]
    assert result["valid"].tolist() == [True, True, True, True, False, True]


def test_prior_ground_support_cannot_explain_later_robot_collision(module):
    net, ground, robot = inputs()
    net[0, 0, 0, 2] = 5
    robot[0, 0, 0, 0, 2] = 5
    ground[0, 1, 0, 0, 2] = 100
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [2]
    assert result["ground_detected"].tolist() == [True]
    assert result["ground_at_trigger"].tolist() == [False]


def test_distinct_time_sources_are_not_reported_simultaneous(module):
    net, ground, robot = inputs()
    net[0, :2, 0, 2] = 5
    ground[0, 0, 0, 0, 2] = 5
    robot[0, 1, 0, 0, 2] = 5
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code_samples"].tolist() == [[1, 2, 0]]
    assert result["source_code"].tolist() == [3]
    assert result["both_in_history"].tolist() == [True]
    assert result["both_same_sample"].tolist() == [False]


def test_unattributed_sample_remains_visible_beside_known_ground(module):
    net, ground, robot = inputs()
    net[0, :2, 0, 2] = 5
    ground[0, 0, 0, 0, 2] = 5
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [4]
    assert result["ground_at_trigger"].tolist() == [True]
    assert result["unattributed_at_trigger"].tolist() == [True]


def test_net_cancellation_does_not_erase_observed_pair_forces(module):
    net, ground, robot = inputs()
    ground[0, 0, 0, 0, 2] = 5
    robot[0, 0, 0, 0, 2] = -5
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [0]  # No original net-force trigger.
    assert result["ground_detected"].tolist() == [True]
    assert result["self_detected"].tolist() == [True]


def test_opposite_pairs_are_detected_before_vector_sum(module):
    net, ground, robot = inputs()
    net[0, 0, 0, 0] = 2
    ground[0, 0, 0, :, 2] = torch.tensor([5., -5.])
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [1]
    assert result["ground_peak_pair_n"].tolist() == [5]


@pytest.mark.parametrize("missing", ["ground", "robot", "both", "zero_filters"])
def test_missing_channel_is_invalid_not_assumed_no_contact(module, missing):
    net, ground, robot = inputs()
    net[0, 0, 0, 2] = 5
    if missing in ("ground", "both"):
        ground = None
    if missing in ("robot", "both"):
        robot = None
    if missing == "zero_filters":
        ground = ground[:, :, :, :0]
    result = module.classify_pelvis_contact(net, ground, robot)
    assert result["source_code"].tolist() == [-1]
    assert result["valid"].tolist() == [False]
    assert result["net_trigger"].tolist() == [True]


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_force_marks_only_affected_environment(module, index, value):
    forces = list(inputs(n=2))
    forces[index][0, 0].fill_(value)
    result = module.classify_pelvis_contact(*forces)
    assert result["source_code"].tolist() == [-1, 0]
    assert result["valid"].tolist() == [False, True]


@pytest.mark.parametrize("threshold", [0., -1., float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(module, threshold):
    with pytest.raises(ValueError, match="threshold"):
        module.classify_pelvis_contact(*inputs(), threshold=threshold)


def test_strict_threshold_and_empty_reset_batch(module):
    net, ground, robot = inputs(n=2, t=1)
    net[:, 0, 0, 2] = torch.tensor([1., 1.01])
    ground[:, 0, 0, 0, 2] = torch.tensor([1., 1.01])
    assert module.classify_pelvis_contact(net, ground, robot)["source_code"].tolist() == [0, 1]
    assert module.classify_pelvis_contact(*inputs(n=0))["source_code"].shape == (0,)


@pytest.mark.parametrize("bad", ["time", "bodies", "xyz", "integer"])
def test_incompatible_layout_is_rejected(module, bad):
    net, ground, robot = inputs()
    if bad == "time":
        ground = ground[:, :2]
    elif bad == "bodies":
        ground = ground.expand(1, 3, 2, 2, 3)
    elif bad == "xyz":
        net = net[..., :2]
    else:
        net = net.to(torch.int64)
    with pytest.raises(ValueError):
        module.classify_pelvis_contact(net, ground, robot)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stateless_order_invariant_window_and_unmodified_inputs(module, dtype):
    forces = list(inputs(n=2, dtype=dtype))
    forces[0][0, 0, 0, 2] = 5
    forces[1][0, 0, 0, 1, 2] = 5
    original = [f.clone() for f in forces]
    first = module.classify_pelvis_contact(*forces)
    reversed_time = module.classify_pelvis_contact(*(f.flip(1) for f in forces))
    torch.testing.assert_close(first["source_code"], reversed_time["source_code"])
    for f, old in zip(forces, original):
        torch.testing.assert_close(f, old, rtol=0, atol=0)


def state_inputs():
    return {"pelvis_position_w": torch.tensor([[0., 0., 1.], [0., 0., .4]]),
            "projected_gravity_b": torch.tensor([[0., 0., -1.], [1., 0., 0.]]),
            "joint_position_error": torch.tensor([[.1, -.2], [.3, -.4]]),
            "joint_velocity": torch.tensor([[1., -2.], [3., -4.]]),
            "applied_torque": torch.tensor([[10., -20.], [30., -40.]]),
            "feet_force_w": torch.tensor([[[0., 0., 10.], [0., 0., 0.]], [[0., 0., 10.], [0., 0., 20.]]]),
            "feet_velocity_w": torch.tensor([[[.1, 0., 9.], [9., 0., 9.]], [[.2, 0., 9.], [.3, 0., 9.]]]),
            "raw_action": torch.tensor([[1., -2.], [3., -4.]]),
            "processed_action": torch.tensor([[.1, -.2], [.3, -.4]])}


def test_terminal_metrics_keep_per_environment_pre_reset_values(module):
    state = state_inputs()
    result = module.terminal_state_metrics(**state, ground_height=torch.tensor([.1, .2]))
    torch.testing.assert_close(result["pelvis_ground_clearance"], torch.tensor([.9, .2]))
    torch.testing.assert_close(result["pelvis_tilt_rad"], torch.tensor([0., torch.pi / 2]))
    assert result["feet_positive_net_z_count"].tolist() == [1, 2]
    torch.testing.assert_close(result["feet_contact_proxy_xy_speed_max"], torch.tensor([.1, .3]))
    torch.testing.assert_close(result["joint_position_error_max"], torch.tensor([.2, .4]))
    torch.testing.assert_close(result["joint_velocity_abs_max"], torch.tensor([2., 4.]))
    torch.testing.assert_close(result["joint_torque_abs_max"], torch.tensor([20., 40.]))
    assert result["state_valid"].tolist() == [True, True]
    state["pelvis_position_w"].zero_()
    torch.testing.assert_close(result["pelvis_height_w"], torch.tensor([1., .4]))


def test_no_ground_data_does_not_fabricate_world_zero_floor(module):
    result = module.terminal_state_metrics(**state_inputs())
    assert result["ground_height_valid"].tolist() == [False, False]
    assert torch.isnan(result["pelvis_ground_clearance"]).all()
    assert result["state_valid"].tolist() == [True, True]


def test_nonfinite_state_or_zero_gravity_is_explicitly_invalid(module):
    state = state_inputs()
    state["joint_velocity"][0, 0] = torch.nan
    state["projected_gravity_b"][0] = 0
    result = module.terminal_state_metrics(**state, ground_height=torch.tensor([torch.nan, .2]))
    assert result["state_valid"].tolist() == [False, True]
    assert result["ground_height_valid"].tolist() == [False, True]
    assert torch.isnan(result["joint_velocity_abs_max"][0])
    assert torch.isnan(result["pelvis_tilt_rad"][0])
    assert torch.isfinite(result["joint_velocity_abs_max"][1])


def test_state_shape_mismatch_is_rejected(module):
    state = state_inputs()
    state["feet_velocity_w"] = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError):
        module.terminal_state_metrics(**state)
