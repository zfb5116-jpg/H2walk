"""CPU readiness checks for USD IK, LIPM transitions and signed net progress."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("h2_leg_geometry", TASK / "leg_kinematics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kin = module.H2LegKinematics("cpu", torch.float64)
    q = kin.home[None].clone()
    p, r, jac = kin.forward(q, jacobian=True)
    constants = json.loads((TASK / "assets/h2_leg_chain.json").read_text())
    expected = torch.tensor([constants["legs"][s]["home_sole_center_in_pelvis_m"]
                             for s in ("left", "right")], dtype=torch.float64)
    assert (p[0] - expected).abs().max() < 1e-7
    finite_difference_error = 0.0
    for i in range(6):
        plus, minus = q.clone(), q.clone()
        plus[..., i] += 1e-6
        minus[..., i] -= 1e-6
        difference = (kin.forward(plus)[0] - kin.forward(minus)[0]) / 2e-6
        finite_difference_error = max(finite_difference_error, float((difference - jac[..., :3, i]).abs().max()))
    assert finite_difference_error < 1e-7
    # Double-support wrench allocation must act inside each physical sole.
    # The combined midpoint is not a valid contact point for either foot.
    supported_force = torch.zeros(1, 2, 3, dtype=q.dtype)
    supported_force[..., 2] = 75.5875 * 9.81 * .5
    support_torque = kin.support_torque(q, supported_force, p)
    assert support_torque[..., 4].abs().max() < 19.0
    gravity = kin.support_torque(q, torch.zeros(1, 2, 3, dtype=q.dtype), p)

    def potential(angles):
        frames = kin.forward(angles, frames=True)[-1]
        centers = (frames[..., :3, :3] @ kin.com_local[None, ..., None]).squeeze(-1) + frames[..., :3, 3]
        return (centers[..., 2] * kin.mass * 9.81).sum(-1)

    gravity_error = 0.0
    for i in range(6):
        plus, minus = q.clone(), q.clone()
        plus[..., i] += 1e-6
        minus[..., i] -= 1e-6
        derivative = (potential(plus) - potential(minus)) / 2e-6
        gravity_error = max(gravity_error, float((derivative - gravity[..., i]).abs().max()))
    assert gravity_error < 1e-6
    angle = torch.tensor(.10, dtype=q.dtype)
    root_rotation = torch.tensor([[angle.cos(), 0., angle.sin()], [0., 1., 0.],
                                   [-angle.sin(), 0., angle.cos()]], dtype=q.dtype)
    world_target = p.clone()
    world_target[..., 0] += .08
    world_target[..., 2] += .03
    body_target = (root_rotation.T @ world_target[..., None]).squeeze(-1)
    body_rotation = root_rotation.T @ r
    rotated_q, _ = kin.solve(body_target, body_rotation, q, iterations=30, damping=.001)
    actual_p, _ = kin.forward(rotated_q)
    world_foot_error = ((root_rotation @ actual_p[..., None]).squeeze(-1) - world_target).norm(dim=-1)
    assert world_foot_error.max() < .003
    # Independent sole-space poses cover lift, forward placement and lateral COM sway.
    targets, rotations = [], []
    for x, y, z_left, z_right in [(0., 0., 0., 0.), (.09, .034, 0., .06),
                                  (-.09, -.034, .06, 0.), (.18, 0., 0., 0.)]:
        value = expected.clone()
        value[:, 2] += .03
        value[:, 0] -= x
        value[:, 1] -= y
        value[0, 2] += z_left
        value[1, 2] += z_right
        targets.append(value)
        rotations.append(r[0])
    targets, rotations = torch.stack(targets), torch.stack(rotations)
    solved, error = kin.solve(targets, rotations, q.expand(4, -1, -1), iterations=30, damping=.001)
    assert error.max() < .003, error
    functions = ast.parse((TASK / "walking_reference.py").read_text())
    functions.body = [node for node in functions.body if isinstance(node, ast.FunctionDef)]
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(functions), "lipm_primitives", "exec"), scope)
    omega = torch.tensor([[9.81 / .98]], dtype=torch.float64).sqrt()
    b, duration = .105, torch.tensor([[.6]], dtype=torch.float64)
    v0 = omega * b * torch.tanh(omega * duration / 2)
    end_y, end_v = scope["lipm"](torch.zeros_like(omega), v0, torch.full_like(omega, b), omega, duration)
    assert end_y.abs().max() < 1e-12 and (end_v + v0).abs().max() < 1e-12
    prep_duration = torch.ones_like(omega)
    cp0, cp1 = scope["two_cops"](torch.zeros_like(omega), torch.zeros_like(omega),
                                    torch.zeros_like(omega), v0, omega, prep_duration)
    yp, vp = scope["lipm"](torch.zeros_like(omega), torch.zeros_like(omega), cp0, omega, prep_duration / 2)
    yp, vp = scope["lipm"](yp, vp, cp1, omega, prep_duration / 2)
    assert yp.abs().max() < 1e-12 and (vp - v0).abs().max() < 1e-12
    assert max(float(cp0.abs().max()), float(cp1.abs().max())) < .13
    step_length = .18
    forward_end_velocity = omega * (step_length / 2) / torch.tanh(omega * duration / 2)
    cp0_x, cp1_x = scope["two_cops"](torch.zeros_like(omega), torch.zeros_like(omega),
                                       torch.full_like(omega, step_length / 2), forward_end_velocity,
                                       omega, duration)
    assert max(float(cp0_x.abs().max()), float(cp1_x.abs().max())) < .06
    xx, vx = scope["lipm"](torch.zeros_like(omega), torch.zeros_like(omega), cp0_x, omega, duration / 2)
    xx, vx = scope["lipm"](xx, vx, cp1_x, omega, duration / 2)
    assert float(xx) > .08 and float(vx) > .3
    endpoint = torch.tensor([[.09, 0.]], dtype=q.dtype)
    landing_velocity = torch.cat((forward_end_velocity, -v0), -1)
    next_velocity = torch.cat((forward_end_velocity, v0), -1)
    center, margin = scope["landing_center"](endpoint, landing_velocity, next_velocity,
                                              omega, duration, r[:, 1, :2, :2])
    capture = endpoint + landing_velocity / omega
    next_cop = endpoint + (landing_velocity * torch.cosh(omega * duration) - next_velocity) / (
        omega * torch.sinh(omega * duration))
    rotation_inverse = r[:, 1, :2, :2].transpose(-1, -2)
    capture_relative = (rotation_inverse @ (capture - center)[..., None]).squeeze(-1)
    cop_relative = (rotation_inverse @ (next_cop - center)[..., None]).squeeze(-1)
    assert (capture_relative.abs() <= torch.tensor([.105, .030], dtype=q.dtype)).all()
    assert (cop_relative.abs() <= torch.tensor([.06, .025], dtype=q.dtype)).all()
    assert float(margin) > .01
    tree = ast.parse((TASK / "mdp/walking_rewards.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "net_forward_progress")
    reward_scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "signed_progress", "exec"), reward_scope)
    robot = NS(data=NS(root_pos_w=torch.zeros(1, 3)))
    tracker = NS(desired_xy=torch.tensor([[.3, 0.]]), pre_xy=torch.zeros(1, 2))
    env = NS(scene={"robot": robot}, locomotion_progress=tracker, step_dt=.02)
    robot.data.root_pos_w[:, 0] = .006
    forward = reward_scope["net_forward_progress"](env, "base_velocity")
    tracker.pre_xy[:, 0] = .006
    robot.data.root_pos_w[:, 0] = 0
    backward = reward_scope["net_forward_progress"](env, "base_velocity")
    assert float(forward + backward) <= 0.0
    tracker.pre_xy[:] = 0
    stationary = reward_scope["net_forward_progress"](env, "base_velocity")
    assert stationary.item() < 0
    robot.data.root_pos_w[:, 0] = .012
    overspeed = reward_scope["net_forward_progress"](env, "base_velocity")
    assert float(overspeed) < float(forward)
    result = {"status": "passed", "Jacobian_max_error": finite_difference_error,
              "gravity_energy_derivative_error": gravity_error, "IK_position_error_max_m": float(error.max()),
              "IK_knee_min_rad": float(solved[..., 3].min()),
              "double_support_ankle_roll_FF_max_Nm": float(support_torque[..., 4].abs().max()),
              "tilted_root_world_foot_IK_error_max_m": float(world_foot_error.max()),
              "feasible_forward_start_CoP_m": [float(cp0_x), float(cp1_x)],
              "landing_feasible_region_halfwidth_min_m": float(margin),
              "landing_capture_relative_xy_m": capture_relative[0].tolist(),
              "landing_next_CoP_relative_xy_m": cop_relative[0].tolist(),
              "LIPM_periodic_boundary_error": float(end_y.abs().max()),
              "preparation_CoP_m": [float(cp0), float(cp1)],
              "zero_net_rocking_reward_sum": float(forward + backward),
              "standing_progress_reward": float(stationary), "simulation_acceptance": False}
    result["double_command_speed_reward"] = float(overspeed)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
