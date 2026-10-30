"""CPU checks for reward hooks around the production RobotEnv transition."""

from types import SimpleNamespace as Box

import pytest
import torch

from test_command_progress_integration import env_type


class RewardRecorder:
    def __init__(self, num_envs, device):
        self.calls = []
        self.start = None

    def begin_step(self, env):
        self.calls.append("begin")
        self.start = env.asset.data.root_pos_w.clone()
        self.command = env.command_manager.get_command("base_velocity").clone()

    def reset(self, ids):
        self.calls.append(("reset", torch.as_tensor(ids).tolist()))

    def abort_step(self):
        self.calls.append("abort")


def make_env(env_type, mode="alternating_v1"):
    env_type.__init__.__globals__["WalkingRewardState"] = RewardRecorder
    return env_type(Box(feet_body_names=["left", "right"], walking_reward_mode=mode))


def test_only_opt_in_mode_constructs_reward_state(env_type):
    env = make_env(env_type)
    assert isinstance(env.walking_reward_state, RewardRecorder)
    baseline = make_env(env_type, "legacy")
    assert baseline.walking_reward_state is None


def test_begin_captures_executed_command_and_pre_physics_position(env_type):
    env = make_env(env_type)
    env.next_command = -env.command.clone()
    env.step(torch.zeros(2, 1))
    state = env.walking_reward_state
    assert state.calls == ["begin"]
    torch.testing.assert_close(state.start, torch.zeros(2, 3))
    torch.testing.assert_close(state.command[:, 0], torch.full((2,), .2))
    assert (env.command[:, 0] < 0).all()


def test_partial_and_manual_reset_clear_only_requested_reward_rows(env_type):
    env = make_env(env_type)
    env.next_reset_ids = [0]
    env.step(torch.zeros(2, 1))
    assert env.walking_reward_state.calls == ["begin", ("reset", [0])]
    assert env.asset.data.root_pos_w[0, 0] == 100.
    env._reset_idx([1])
    assert env.walking_reward_state.calls[-1] == ("reset", [1])


@pytest.mark.parametrize("field", ["raise_before_reset", "raise_after_reset"])
def test_parent_failure_aborts_reward_state_and_propagates(env_type, field):
    env = make_env(env_type)
    setattr(env, field, True)
    with pytest.raises(RuntimeError, match="synthetic"):
        env.step(torch.zeros(2, 1))
    assert env.walking_reward_state.calls == ["begin", "abort"]


def test_constructor_time_reset_tolerates_absent_reward_state(env_type):
    env = make_env(env_type)
    del env.walking_reward_state
    env._reset_idx([0])
