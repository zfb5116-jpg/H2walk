"""CPU-only policy shape and autograd checks; no Isaac Lab or optimizer needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


NETWORK_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py"
)
SPEC = importlib.util.spec_from_file_location("h2_actor_critic_under_test", NETWORK_PATH)
assert SPEC is not None and SPEC.loader is not None
NETWORK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NETWORK)


def observation_width(actions: int, history_length: int = 5, height_scan_dim: int = 561) -> int:
    return 3 + history_length * (6 + 3 * actions) + height_scan_dim


class H2ActorCriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def make_policy(self, actions=29, critic_width=137, **kwargs):
        width = observation_width(
            actions, kwargs.get("history_length", 5), kwargs.get("height_scan_dim", 561)
        )
        return NETWORK.H2ActorCriticOur(
            width,
            critic_width,
            actions,
            actor_hidden_dims=[16],
            critic_hidden_dims=[16],
            **kwargs,
        )

    def test_29_action_h2_uses_1029_observations_and_independent_critic(self):
        policy = self.make_policy()
        self.assertEqual(policy.actor.obs_dim, 1029)
        self.assertEqual(policy.actor.scan_start, 468)
        observations = torch.randn(2, 1029)
        self.assertEqual(tuple(policy.act_inference(observations).shape), (2, 29))
        self.assertEqual(tuple(policy.act(observations).shape), (2, 29))
        self.assertEqual(tuple(policy.get_actions_log_prob(policy.action_mean).shape), (2,))
        self.assertEqual(tuple(policy.evaluate(torch.randn(2, 137)).shape), (2, 1))

    def test_explicit_28_action_layout(self):
        policy = self.make_policy(actions=28)
        self.assertEqual(policy.actor.obs_dim, 1014)
        self.assertEqual(policy.actor.scan_start, 453)
        self.assertEqual(tuple(policy.act_inference(torch.randn(2, 1014)).shape), (2, 28))

    def test_actor_remains_torchscript_exportable(self):
        policy = self.make_policy()
        observations = torch.randn(2, 1029)
        scripted_actor = torch.jit.script(policy.actor)
        torch.testing.assert_close(scripted_actor(observations), policy.act_inference(observations))

    def test_custom_history_and_height_scan_keep_term_boundaries(self):
        policy = self.make_policy(history_length=2, height_scan_dim=9)
        width = observation_width(29, 2, 9)
        observations = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        expected_current = torch.cat(
            (
                observations[:, :3],
                observations[:, 6:9],
                observations[:, 12:15],
                observations[:, 44:73],
                observations[:, 102:131],
                observations[:, 160:189],
            ),
            dim=-1,
        )
        torch.testing.assert_close(policy.actor._current_obs(observations), expected_current)
        self.assertEqual(policy.actor.scan_start, 189)
        self.assertEqual(tuple(policy.act_inference(observations).shape), (1, 29))

    def test_constructor_rejects_both_short_and_long_actor_observations(self):
        for width in (1014, 1028, 1030):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "exactly 1029"):
                NETWORK.H2ActorCriticOur(width, 137, 29)

    def test_runtime_actor_and_critic_width_errors_are_explicit(self):
        policy = self.make_policy()
        for observations in (torch.zeros(2, 1030), torch.zeros(1029)):
            with self.subTest(shape=observations.shape), self.assertRaisesRegex(ValueError, "Actor"):
                policy.act_inference(observations)
        for observations in (torch.zeros(2, 1029), torch.zeros(2, 136), torch.zeros(137)):
            with self.subTest(shape=observations.shape), self.assertRaisesRegex(ValueError, "Critic"):
                policy.evaluate(observations)

    def test_all_three_history_encoders_receive_gradients_by_default(self):
        policy = self.make_policy()
        policy.act_inference(torch.randn(3, 1029)).square().mean().backward()
        for name in ("velocity_encoder", "contact_encoder", "predict_encoder"):
            with self.subTest(encoder=name):
                gradients = [parameter.grad for parameter in getattr(policy.actor, name).parameters()]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)

    def test_detachment_remains_available_for_auxiliary_training_callers(self):
        policy = self.make_policy(detach_estimates=True)
        policy.act_inference(torch.randn(3, 1029)).square().mean().backward()
        for name in ("velocity_encoder", "contact_encoder", "predict_encoder"):
            with self.subTest(encoder=name):
                self.assertTrue(all(parameter.grad is None for parameter in getattr(policy.actor, name).parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in policy.actor.terrain_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
