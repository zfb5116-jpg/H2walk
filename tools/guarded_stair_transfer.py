"""Same-width accepted-flat transfer and the unchanged bounded numeric executor."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from guarded_flat_ppo import (H2GuardedPPO, LONGER_BUDGET_CONTEXT_TRIAL, require, sha, state_digest)
from task_context_normalization import normalization_metadata

PROTOCOL = "accepted_flat_full_footprint_stair_clearance_v4"
FORMAT = "h2_guarded_stair_ppo_v4"
TASK = "Unitree-H2-Stair-Ascent-Policy"
PARENT_SHA = "8a82278fba63e1e5d8f715c33d4bdfced50291f0fffec73909059d92626cbf9f"
PARENT_CPU_SHA = "a6dd8e21690f90362770221879613e373a4baa3bc6a8b47f1853ff43a84ea1d1"
FORMAL_SHA = "0516fa4f5b1a49f75b7a9f2ba4d60c38be18064018a45c7badbb1863552ce881"
BUDGET = 1000


class H2GuardedStairPPO(H2GuardedPPO):
    """Only task/serialization metadata differs; all numerical methods inherit."""
    def __init__(self, *args, trial_protocol=PROTOCOL, **kwargs):
        require(trial_protocol == PROTOCOL, "Undeclared stair transfer profile")
        # This selects the existing 5D, reserved, resource bounded, strict FP64
        # executor. It does not select a task, reward, terrain or training cap.
        super().__init__(*args, trial_protocol=LONGER_BUDGET_CONTEXT_TRIAL, **kwargs)
        self.checkpoint_format = FORMAT

    def learning_state(self):
        result = super().learning_state()
        result.update(trial_protocol=PROTOCOL, task=TASK, training_budget=BUDGET,
                      inherited_numeric_execution_profile=LONGER_BUDGET_CONTEXT_TRIAL)
        result["proposal_projection"]["protocol"] = PROTOCOL
        return result

    def packet(self, iteration, infos=None):
        result = super().packet(iteration, infos)
        result.update(trial_protocol=PROTOCOL, task=TASK, training_budget=BUDGET,
                      inherited_numeric_execution_profile=LONGER_BUDGET_CONTEXT_TRIAL)
        return result


def parent_admission(plan, root):
    """Receipt/custody checks only; never loads a parent checkpoint."""
    root = Path(root).resolve(strict=True)
    require(plan["trial_protocol"] == PROTOCOL and plan["task"] == TASK
            and plan["max_iterations"] == BUDGET and plan["num_envs"] == 1024
            and plan["network_dimensions"] == [1034, 871, 73, 12], "Wrong stair plan")
    parent = Path(plan["initial_policy"]).resolve(strict=True)
    canonical = root/"logs/rsl_rl/h2_main_policy_manager_based/2026-11-02_08-58-39_guarded_flat_balance_d880d7b7/model_2568_guard_restored.pt"
    require(parent == canonical and plan["initial_policy_sha256"] == PARENT_SHA
            and sha(parent) == PARENT_SHA, "Accepted flat parent changed")
    cpu_path = parent.with_suffix(".validation.json")
    formal_path = root/"logs/stair_learning/flat_balance_20261102_175824_d7863f27/formal_result_review.json"
    require(sha(cpu_path) == PARENT_CPU_SHA and sha(formal_path) == FORMAL_SHA,
            "Required independent parent proofs changed")
    cpu = json.loads(cpu_path.read_text())
    formal = json.loads(formal_path.read_text())
    require(cpu.get("status") == "passed", "Parent complete CPU receipt is not passed")
    # Its ROOT review explicitly reconciles the historical generic false flag.
    require(formal.get("joint_successful_envs", formal.get("successful_envs", 0)) == 926
            or formal.get("flat_joint_successful_envs") == 926
            or formal.get("joint_success_count") == 926,
            "Parent independent first1024 flat conjunction must be exactly926")
    require(formal.get("flat_protocol_passed") is True and formal.get("scope_closed") is True
            and formal.get("all1024_original_conjunction_independently_verified") is True,
            "Parent physical gate review is not closed")
    return dict(kind=PROTOCOL, parent_checkpoint=str(parent), parent_checkpoint_sha256=PARENT_SHA,
                parent_CPU_receipt=str(cpu_path), parent_CPU_receipt_sha256=PARENT_CPU_SHA,
                parent_formal_review=str(formal_path), parent_formal_review_sha256=FORMAL_SHA,
                parent_source_commit="946742f45fc377718c9bb7710058e7457e5d2bb3",
                parent_last_complete_iteration=2568, parent_accepted_updates=2569,
                parent_natural_completion=False, parent_runtime_rollback_completed=True,
                parent_uncommitted_rollout=True, parent_formal_successful_envs=926,
                parent_flat_acceptance=True, parent_stair_acceptance=False,
                parent_network_dimensions=[1034, 871, 73, 12], network_dimensions=[1034, 871, 73, 12],
                same_width_context_transfer=True, model_and_dualnorm_only=True,
                fresh_actor_optimizer=True, fresh_critic_optimizer=True, fresh_RND=True,
                fresh_RND_optimizer=True, parent_RNG_loaded=False, new_starting_iteration=0)


def apply_stair_transfer(algorithm, normalizers, saved, provenance):
    """Use the single initialization load; copy model/norms without resume state."""
    require(saved["format"] == "h2_guarded_flat_ppo_v14" and saved["iter"] == 2568,
            "Wrong accepted-flat serialized type")
    state = saved["guarded_ppo_state"]
    require(state["completed_updates"] == 2569 and state["actor_optimizer_steps"] == 50580
            and state["critic_optimizer_steps"] == state["rnd_optimizer_steps"] == 51380
            and state["rnd_update_counter"] == 61680 and state["uncommitted_rollout"] is True,
            "Accepted-flat serialized counters changed")
    require(algorithm.completed_updates == algorithm.actor_optimizer_steps == algorithm.critic_optimizer_steps
            == algorithm.rnd_optimizer_steps == algorithm.rnd.update_counter == 0
            and not algorithm.actor_optimizer.state and not algorithm.critic_optimizer.state
            and not algorithm.rnd_optimizer.state, "Stair optimization must start fresh")
    rnd_before = state_digest(algorithm.rnd.state_dict())
    def rng():
        return state_digest(dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                                 cuda=torch.cuda.get_rng_state_all() if str(algorithm.device).startswith("cuda") else None))
    rng_before = rng()
    algorithm.policy.load_state_dict(saved["model_state_dict"], strict=True)
    for key, value in algorithm.policy.state_dict().items():
        require(torch.equal(value.detach().cpu(), saved["model_state_dict"][key].detach().cpu()),
                "Transferred model differs: "+key)
    for name, key in (("actor", "obs_norm_state_dict"), ("critic", "privileged_obs_norm_state_dict")):
        normalizers[name].load_state_dict(saved[key], strict=True)
        normalizers[name].eval()
        require(state_digest(normalizers[name].state_dict()) == state_digest(saved[key]), "Own normalization differs")
    abi = normalization_metadata(normalizers)
    require(abi == saved["normalization_abi"] and abi["prefix_empirical_counts"] == {"actor":24576000,"critic":24576000},
            "Same-width own normalization ABI changed")
    require(rnd_before == state_digest(algorithm.rnd.state_dict()) and rng_before == rng(),
            "Transfer changed fresh RND or imported parent RNG")
    result = copy.deepcopy(provenance)
    result.update(actor_and_critic_loaded_bitwise=True, dualnorm_loaded_bitwise=True,
                  normalization_abi=abi, fresh_RND_digest=rnd_before,
                  loading_did_not_change_fresh_RNG=True)
    algorithm.initialization_transfer = result
    return result
