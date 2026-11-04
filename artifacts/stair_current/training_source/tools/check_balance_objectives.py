"""CPU counterexamples for the balance objectives and physical landing receipts."""

import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def inverse(q, v):
    xyz = -q[..., 1:]
    twice = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * twice + torch.cross(xyz, twice, dim=-1)


def yaw(q):
    angle = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                        1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    out = torch.zeros_like(q)
    out[:, 0], out[:, 3] = (angle / 2).cos(), (angle / 2).sin()
    return out


class Term:
    def __init__(self, *_):
        pass


def main():
    # Only the simulator-independent objective arithmetic is run on CPU.
    tree = ast.parse((TASK / "mdp/walking_rewards.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    scope = dict(torch=torch, ManagerTermBase=Term, SceneEntityCfg=object,
                 quat_apply_inverse=inverse, yaw_quat=yaw)
    exec(compile(ast.fix_missing_locations(tree), "walking_objectives", "exec"), scope)
    spec = importlib.util.spec_from_file_location("physical_progress", TASK / "locomotion_progress.py")
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)

    data = NS(root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
              root_lin_vel_w=torch.tensor([[0.3, 0., 0.]]),
              projected_gravity_b=torch.tensor([[0., 0., -1.]]),
              body_pos_w=torch.tensor([[[0., 0.067, 0.], [0., -0.067, 0.]]]),
              joint_effort_limits=torch.tensor([[19., 66., 360.]]),
              computed_torque=torch.tensor([[20., 70., 100.]]))
    sensor_data = NS(current_air_time=torch.zeros(1, 2), current_contact_time=torch.ones(1, 2),
                     last_air_time=torch.zeros(1, 2))
    first_contact = torch.ones(1, 2, dtype=torch.bool)
    sensor = NS(data=sensor_data, compute_first_contact=lambda dt: first_contact.clone())
    env = NS(scene={"robot": NS(data=data), "contact_forces": sensor}, num_envs=1, device="cpu", step_dt=0.02,
             command_manager=NS(get_command=lambda name: torch.tensor([[0.3, 0., 0.]])))
    cfg = NS(name="robot", body_ids=[0, 1], joint_ids=[0, 1, 2])
    sensor_cfg = NS(name="contact_forces", body_ids=[0, 1])
    good = scope["stable_forward_tracking"](env, "velocity")
    data.root_lin_vel_w[:] = torch.tensor([[0., 0., -1.]])
    data.root_quat_w[:] = torch.tensor([[math.cos(.3), 0., math.sin(.3), 0.]])
    data.projected_gravity_b[:] = torch.tensor([[math.sin(.6), 0., -math.cos(.6)]])
    assert good.item() > .99 and scope["stable_forward_tracking"](env, "velocity").item() == 0
    data.root_lin_vel_w[:] = torch.tensor([[.3, 0., 0.]])
    assert 0 < scope["stable_forward_tracking"](env, "velocity").item() < good.item()
    data.root_quat_w[:] = torch.tensor([[1., 0., 0., 0.]])
    data.projected_gravity_b[:] = torch.tensor([[0., 0., -1.]])
    assert scope["feet_lateral_separation"](env, cfg).item() == 0
    data.body_pos_w[0, 0] = torch.tensor([.8, -.01, 0.])
    assert scope["feet_lateral_separation"](env, cfg).item() > 0
    data.body_pos_w[0, 0, 1] = -.1
    assert scope["feet_lateral_separation"](env, cfg).item() > 1
    assert scope["joint_torque_saturation"](env, cfg).item() > 0

    term = scope["feet_support_transition"](None, env)
    assert term(env, "velocity", sensor_cfg).item() == 0  # initial landing
    sensor_data.current_contact_time[:] = torch.tensor([[0., .2]])
    sensor_data.current_air_time[:] = torch.tensor([[.2, 0.]])
    first_contact[:] = False
    assert term(env, "velocity", sensor_cfg).item() > 0
    sensor_data.current_air_time[0, 0] = .8
    assert term(env, "velocity", sensor_cfg).item() == 0  # one-legged stall
    sensor_data.current_air_time[:] = 0
    sensor_data.current_contact_time[:] = 1
    sensor_data.last_air_time[:] = torch.tensor([[.2, 0.]])
    first_contact[:] = torch.tensor([[True, False]])
    assert term(env, "velocity", sensor_cfg).item() > 0
    assert term(env, "velocity", sensor_cfg).item() == 0  # repeated same foot
    term.reset(torch.tensor([0]))
    assert term(env, "velocity", sensor_cfg).item() == 0

    tracker = progress.FootfallTracker(1, "cpu")
    both, left_air, right_air = (torch.tensor([pair]) for pair in
                                 ([True, True], [False, True], [True, False]))
    tracker.update(both, .02)
    assert tracker.landings.sum() == 0
    for _ in range(5):
        tracker.update(torch.zeros_like(both), .02)
    tracker.update(both, .02)
    assert tracker.landings.sum() == 0  # synchronized hop
    for _ in range(5):
        tracker.update(left_air, .02)
    tracker.update(both, .02)
    assert tracker.landings.tolist() == [[1., 0.]]
    for _ in range(5):
        tracker.update(left_air, .02)
    tracker.update(both, .02)
    assert tracker.landings.tolist() == [[1., 0.]]
    for _ in range(5):
        tracker.update(right_air, .02)
    tracker.update(both, .02)
    assert tracker.landings.tolist() == [[1., 1.]]
    tracker.reset(torch.tensor([0]))
    tracker.update(both, .02)
    assert tracker.landings.sum() == 0
    print(json.dumps({"status": "passed", "counterexamples": ["falling vertical velocity", "tilted progress",
        "forward-staggered narrow feet", "crossed feet", "ankle saturation", "initial landing", "one-leg stall",
        "same-foot repeats", "synchronized hopping", "reset state", "genuine alternating contacts"]}))


if __name__ == "__main__":
    main()
