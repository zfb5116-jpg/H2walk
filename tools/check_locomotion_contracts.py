"""CPU checks for joint identity, complete-episode curriculum and policy dimensions."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    checks = {}
    mapping = module_at("h2_action_contract_test", TASK / "action_contract.py")
    motion = module_at("h2_progress_test", TASK / "locomotion_progress.py")
    result = json.loads((ROOT / "logs/diagnostics/current_instability_diagnosis.json").read_text())
    runtime = json.loads((Path(result["directory"]) / "baseline_flat_1024/result.json").read_text())["runtime_joint_action_mapping"]
    names = runtime["action_joint_names"]
    leg_names = names[:12]
    # Use a permutation unrelated to articulation ids; channel values identify names.
    permutation = torch.randperm(29, generator=torch.Generator().manual_seed(17)).tolist()
    values = torch.tensor(permutation, dtype=torch.float32).repeat(3, 1)
    selected = mapping.actions_in_joint_order(values, [names[i] for i in permutation], leg_names)
    assert torch.equal(selected, torch.arange(12).float().repeat(3, 1))
    padded = mapping.actions_in_joint_order(selected, leg_names, names, strict=False)
    assert torch.equal(padded[:, :12], selected) and (padded[:, 12:] == 0).all()
    try:
        mapping.actions_in_joint_order(selected, leg_names, [names[12]])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing required action channel was silently accepted")
    checks["joint_names_determine_action_identity"] = True
    checks["uncontrolled_observed_joints_are_zero_padded"] = True

    count, dt = 1024, 0.02
    ids = torch.arange(count)
    terminated = torch.zeros(count, dtype=torch.bool)
    quat = torch.zeros((count, 4)); quat[:, 0] = 1
    command = torch.zeros((count, 3)); command[:, 0] = 0.3
    tracker = motion.LocomotionProgress(count, "cpu")
    position = torch.zeros((count, 2))

    def episode(walking):
        for step in range(1000):
            tracker.begin_step(position, quat, command, dt)
            if walking:
                position[:, 0] += 0.3 * dt
            # Valid alternating stance/flight, with >80 ms between landings.
            contact = torch.tensor([step % 50 < 25, step % 50 >= 25]).repeat(count, 1)
            tracker.finish_step(position, contact)
        tracker.record_episodes(ids, terminated)

    # Discard the initial potentially random-length rollout even if it walked.
    episode(True)
    assert tracker.summary()["episode_count"] == 0
    tracker.reset(ids)
    episode(False)
    standing = tracker.summary()
    assert standing["survival_fraction"] == 1 and standing["mean_progress_ratio"] == 0
    assert not motion.curriculum_ready(standing)
    checks["standing_cannot_advance_curriculum"] = True
    checks["random_length_bootstrap_excluded"] = True
    tracker.advance_generation(); tracker.reset(ids)
    episode(True)
    walking = tracker.summary()
    assert motion.curriculum_ready(walking), walking
    checks["sustained_progress_and_bilateral_landings_allow_expansion"] = True

    # Execute the actual curriculum function with a fake environment; no simulator
    # imports are needed. An episode receipt can authorize only one expansion.
    syntax = ast.parse((TASK / "mdp/curriculums.py").read_text())
    function = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "lin_vel_cmd_levels")
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    function.args.defaults = [ast.Constant("tracking_lin_vel_x")]
    program = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {"torch": torch, "curriculum_ready": motion.curriculum_ready}
    exec(compile(program, "curriculum_under_test", "exec"), scope)
    ranges = SimpleNamespace(lin_vel_x=[-0.1, 0.3], lin_vel_y=[0.0, 0.0])
    limits = SimpleNamespace(lin_vel_x=[-1.0, 1.2], lin_vel_y=[0.0, 0.0])
    term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges, limit_ranges=limits))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term),
                          common_step_counter=1000, max_episode_length=1000,
                          locomotion_progress=tracker, device="cpu")
    scope["lin_vel_cmd_levels"](env, ids)
    assert abs(ranges.lin_vel_x[1] - 0.4) < 1e-6 and ranges.lin_vel_y == [0.0, 0.0]
    env.common_step_counter = 2000
    scope["lin_vel_cmd_levels"](env, ids)
    assert abs(ranges.lin_vel_x[1] - 0.4) < 1e-6
    # All still-active easier episodes are excluded after an expansion.
    tracker.record_episodes(ids, terminated)
    assert tracker.summary()["episode_count"] == 0
    checks["old_command_generation_cannot_trigger_repeated_expansion"] = True

    # Reset teleport is outside a transition and must not count as travel.
    tracker.reset(ids)
    position[:, 0] += 100.0
    tracker.finish_step(position, torch.ones((count, 2), dtype=torch.bool))
    assert (tracker.projected_distance == 0).all()
    checks["reset_teleport_not_counted_as_progress"] = True
    # RSL's wrapper resets before the first simulation step. At that point
    # reset_terminated does not exist on ManagerBasedRLEnv yet.
    robot_syntax = ast.parse((TASK / "robot_env.py").read_text())
    robot_class = next(node for node in robot_syntax.body if isinstance(node, ast.ClassDef))
    reset_function = next(node for node in robot_class.body
                          if isinstance(node, ast.FunctionDef) and node.name == "_reset_idx")
    for argument in reset_function.args.args:
        argument.annotation = None
    reset_class = ast.ClassDef(name="ResetUnderTest", bases=[ast.Name(id="FakeBase", ctx=ast.Load())],
                              keywords=[], body=[reset_function], decorator_list=[])
    class FakeBase:
        def _reset_idx(self, env_ids):
            self.parent_reset_called = True
    reset_scope = {"FakeBase": FakeBase}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reset_class], type_ignores=[])),
                 "initial_reset_under_test", "exec"), reset_scope)
    initial_env = reset_scope["ResetUnderTest"]()
    initial_env.locomotion_progress = motion.LocomotionProgress(count, "cpu")
    initial_env.termination_manager = SimpleNamespace(terminated=terminated)
    initial_env._finish_progress_step = lambda _: None
    initial_env.update_feet_state = lambda: None
    initial_env.feet_pos_b = torch.zeros((count, 2, 3))
    initial_env.prev_feet_pos_b = torch.zeros_like(initial_env.feet_pos_b)
    initial_env.feet_vel_b = torch.zeros_like(initial_env.feet_pos_b)
    initial_env._reset_idx(ids)
    assert initial_env.parent_reset_called and not hasattr(initial_env, "reset_terminated")
    assert initial_env.locomotion_progress.summary()["episode_count"] == 0
    checks["initial_wrapper_reset_needs_no_previous_step_termination_field"] = True
    policy = module_at("h2_policy_contract_test", TASK / "networks/actor_critic_our.py")
    net = policy.H2ActorCriticOur(1029, 866, 12, num_joints=29)
    obs = torch.zeros((8, 1029))
    actions = net.act(obs)
    assert actions.shape == (8, 12) and net.action_std.shape == (8, 12)
    assert net.entropy.shape == (8,) and net.get_actions_log_prob(actions).shape == (8,)
    checks["policy_probability_distribution_contains_only_12_executed_actions"] = True
    reward_syntax = ast.parse((TASK / "mdp/rewards.py").read_text())
    reward_function = next(node for node in reward_syntax.body
                           if isinstance(node, ast.FunctionDef) and node.name == "tracking_lin_vel_x")
    for argument in reward_function.args.args:
        argument.annotation = None
    reward_function.returns = None
    reward_scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reward_function], type_ignores=[])),
                 "forward_reward_under_test", "exec"), reward_scope)
    cmd = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [0.3, 0, 0], [0., 0, 0]])
    velocity = torch.tensor([[0., 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [-0.3, 0, 0], [0., 0, 0]])
    reward_env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=velocity))},
                                 command_manager=SimpleNamespace(get_command=lambda _: cmd))
    rewards = reward_scope["tracking_lin_vel_x"](
        reward_env, "base_velocity", command_threshold=0.05, require_forward_progress=True)
    assert torch.equal(rewards, torch.tensor([0., 1., 1., 0., 0.])), rewards
    checks["stair_tracking_reward_requires_commanded_forward_or_backward_progress"] = True
    return {"status": "passed", "checks": checks, "walking_acceptance": "requires Isaac Sim",
            "standing_receipts": standing, "walking_receipts": walking}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
