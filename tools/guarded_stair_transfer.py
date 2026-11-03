"""Same-width accepted-flat transfer with guarded original-block row execution."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from guarded_flat_ppo import (H2GuardedPPO, LONGER_BUDGET_CONTEXT_TRIAL, require, sha, state_digest)
from task_context_normalization import normalization_metadata
from selected_row_block import SELECTED_ROW_BLOCK_POLICY
from stair_initialization_binding import (FORMAT, PROTOCOL, PROTOCOL_FORMATS, INITIALIZATION_PROTOCOLS, CLEARANCE_PROGRESS_PROTOCOLS,
                                          TEMPORAL_PROTOCOLS, ROLLOUT_CREDIT_PROTOCOLS, FIXED_BANK_COLLECTION_CONTROLS,
                                          initialization_config_errors, temporal_credit_config_errors,
                                          temporal_credit_snapshot_errors, rollout_credit_collection_contract,
                                          rollout_credit_collection_errors, rollout_credit_budget_errors,
                                          rollout_credit_acquisition_errors, clearance_progress_config_errors,
                                          clearance_progress_snapshot_errors)

TASK = "Unitree-H2-Stair-Ascent-Policy"
PARENT_SHA = "8a82278fba63e1e5d8f715c33d4bdfced50291f0fffec73909059d92626cbf9f"
PARENT_CPU_SHA = "a6dd8e21690f90362770221879613e373a4baa3bc6a8b47f1853ff43a84ea1d1"
FORMAL_SHA = "0516fa4f5b1a49f75b7a9f2ba4d60c38be18064018a45c7badbb1863552ce881"
BUDGET = 1000


class H2GuardedStairPPO(H2GuardedPPO):
    """Keep original optimizer/constraints and select original-block row calls."""
    def __init__(self, *args, trial_protocol=PROTOCOL, **kwargs):
        require(trial_protocol in PROTOCOL_FORMATS, "Undeclared stair transfer profile")
        self.stair_trial_protocol = trial_protocol
        # This selects the existing 5D, reserved, resource bounded, strict FP64
        # executor. It does not select a task, reward, terrain or training cap.
        super().__init__(*args, trial_protocol=LONGER_BUDGET_CONTEXT_TRIAL,
                         stair_temporal_credit=trial_protocol in TEMPORAL_PROTOCOLS, **kwargs)
        self.checkpoint_format = PROTOCOL_FORMATS[trial_protocol]
        self.selected_row_evaluation_policy = SELECTED_ROW_BLOCK_POLICY

    def bind_temporal_credit(self, runner, environment, binding):
        """Bind live runner/environment sources once, before the first rollout."""
        require(self.stair_trial_protocol in TEMPORAL_PROTOCOLS and runner.alg is self
                and not hasattr(self, "_temporal_credit_binding"), "Wrong or repeated temporal credit binding")
        require(not temporal_credit_config_errors(binding, runner.cfg, protocol=self.stair_trial_protocol,
                                                  control_dt_s=environment.step_dt),
                "Actual runner temporal configuration differs from its plan")
        self._temporal_credit_binding = copy.deepcopy(binding)
        self._temporal_credit_runner = runner
        self._temporal_credit_environment = environment
        return self.temporal_credit_snapshot()

    def temporal_credit_snapshot(self):
        """Read actual algorithm attributes and live rollout/environment scalars."""
        if self.stair_trial_protocol not in TEMPORAL_PROTOCOLS:
            return None
        require(hasattr(self, "_temporal_credit_binding"), "Actual temporal credit is not bound")
        steps = self._temporal_credit_runner.num_steps_per_env
        require(type(steps) is int and type(self.storage.num_transitions_per_env) is int
                and self.storage.num_transitions_per_env == steps,
                "Actual rollout storage and runner lengths disagree")
        result = dict(gamma=self.gamma, gae_lambda=self.lam, rollout_steps=steps,
                      control_dt_s=self._temporal_credit_environment.step_dt)
        require(not temporal_credit_snapshot_errors(self._temporal_credit_binding, result,
                                                    protocol=self.stair_trial_protocol),
                "Actual temporal credit differs from the immutable plan")
        return result

    def rollout_collection_snapshot(self):
        """Bind declared prefix/current lengths to the already live storage."""
        if self.stair_trial_protocol not in ROLLOUT_CREDIT_PROTOCOLS:
            return None
        temporal = self.temporal_credit_snapshot()
        result = rollout_credit_collection_contract(protocol=self.stair_trial_protocol)
        require(temporal["rollout_steps"] == result["training_rollout_steps"],
                "Live rollout differs from its current-bank declaration")
        if self.guard is not None:
            require(self.guard.current_rollout_steps == result["current_collection_controls"]
                    and self.guard.fixed_collection_controls == result["fixed_bank_collection_controls"],
                    "Actual guard collection differs from the live rollout")
        return result

    def bind_clearance_progress(self, environment, binding):
        require(self.stair_trial_protocol in CLEARANCE_PROGRESS_PROTOCOLS
                and environment is self._temporal_credit_environment
                and not hasattr(self, "_clearance_progress_binding")
                and not clearance_progress_config_errors(binding, protocol=self.stair_trial_protocol),
                "Wrong or repeated clearance progress binding")
        self._clearance_progress_binding = copy.deepcopy(binding)
        return self.clearance_progress_snapshot()

    def clearance_progress_snapshot(self):
        if self.stair_trial_protocol not in CLEARANCE_PROGRESS_PROTOCOLS:
            return None
        require(hasattr(self, "_clearance_progress_binding"), "Actual clearance progress is not bound")
        environment = self._temporal_credit_environment
        result = dict(environment.stair_progress.clearance_progress_snapshot() or {},
            contract=copy.deepcopy(self._clearance_progress_binding),
            configuration_enabled=environment.cfg.stair_clearance_progress_potential,
            outer_potential_weight=float(environment.reward_manager.get_term_cfg("potential").weight))
        require(not clearance_progress_snapshot_errors(self._clearance_progress_binding, result,
                                                       protocol=self.stair_trial_protocol),
                "Actual clearance reward state differs from its immutable contract")
        return result

    def bind_height_profile(self, environment, observer, video, binding, video_binding):
        from stair_curriculum_binding import course_config_errors
        from training_video import video_config_errors
        require(self.stair_trial_protocol == PROTOCOL and not hasattr(self, "_height_profile")
                and environment is self._temporal_credit_environment
                and environment.cfg.stair_height_course == binding
                and not course_config_errors(binding) and not video_config_errors(video_binding),
                "Wrong or repeated height/video profile binding")
        self._height_profile = (environment, observer, video, copy.deepcopy(binding), copy.deepcopy(video_binding))
        return self.height_profile_snapshot()

    def height_profile_snapshot(self):
        if self.stair_trial_protocol != PROTOCOL:
            return None
        from stair_curriculum_binding import profile_snapshot_errors
        require(hasattr(self, "_height_profile"), "Actual height/video profile is not bound")
        environment, observer, video, binding, video_binding = self._height_profile
        result = dict(height_course=environment.stair_height_course.snapshot(),
                      training_video=video.snapshot(), credit_coverage=observer.snapshot())
        require(not profile_snapshot_errors(binding, video_binding, result),
                "Actual height/video/credit state disagrees with the admitted profile")
        return result

    def process_env_step(self, rewards, dones, infos):
        result = super().process_env_step(rewards, dones, infos)
        if self.stair_trial_protocol in ROLLOUT_CREDIT_PROTOCOLS:
            self.temporal_credit_snapshot()
            require(self.guard is not None, "Rollout credit step has no actual-input guard")
            if self.completed_updates == 0 and self.guard.bank_receipt["status"] == "collecting_first_rollout":
                step = self.storage.step
                if type(step) is not int or not 1 <= step <= FIXED_BANK_COLLECTION_CONTROLS:
                    self.guard._acquisition_reject("Fixed bank missed its post-step24 seal boundary")
                if step == FIXED_BANK_COLLECTION_CONTROLS:
                    self.guard.seal_first_rollout(step, self.completed_updates,
                        seal_hook="post_inherited_process_env_step")
            require(self.guard.bank_receipt["status"] == "sealed"
                    or (self.completed_updates == 0 and self.storage.step < FIXED_BANK_COLLECTION_CONTROLS),
                    "Fixed bank must seal before the next control24 act")
        return result

    def compute_returns(self, last_critic_obs):
        if self.stair_trial_protocol in ROLLOUT_CREDIT_PROTOCOLS:
            temporal = self.temporal_credit_snapshot()
            require(self.storage.step == temporal["rollout_steps"]
                    and self.guard is not None and self.guard.bank_receipt["status"] == "sealed"
                    and len(self.guard.current) == temporal["rollout_steps"],
                    "The single48-control rollout must finish with its prefix bank already sealed")
        return super().compute_returns(last_critic_obs)

    def learning_state(self):
        result = super().learning_state()
        result.update(trial_protocol=self.stair_trial_protocol, task=TASK, training_budget=BUDGET,
                      inherited_numeric_execution_profile=LONGER_BUDGET_CONTEXT_TRIAL)
        result["proposal_projection"]["protocol"] = self.stair_trial_protocol
        result["proposal_projection"]["selected_row_evaluation_policy"] = SELECTED_ROW_BLOCK_POLICY
        if self.stair_trial_protocol in TEMPORAL_PROTOCOLS:
            result["temporal_credit"] = self.temporal_credit_snapshot()
            result["inherited_numeric_execution_profile_scope"] = "numeric_executor_only"
            result["ppo_temporal_credit_override"] = dict(parameter="gae_lambda", value=self.lam,
                scope="stair_transfer_only", exact_binding="temporal_credit")
        if self.stair_trial_protocol in ROLLOUT_CREDIT_PROTOCOLS:
            result["rollout_collection"] = self.rollout_collection_snapshot()
        if self.stair_trial_protocol in CLEARANCE_PROGRESS_PROTOCOLS:
            result["clearance_progress_potential"] = self.clearance_progress_snapshot()
        if self.stair_trial_protocol == PROTOCOL:
            result["height_profile"] = self.height_profile_snapshot()
        return result

    def packet(self, iteration, infos=None):
        result = super().packet(iteration, infos)
        result.update(trial_protocol=self.stair_trial_protocol, task=TASK, training_budget=BUDGET,
                      inherited_numeric_execution_profile=LONGER_BUDGET_CONTEXT_TRIAL)
        if self.stair_trial_protocol in TEMPORAL_PROTOCOLS:
            result["temporal_credit"] = self.temporal_credit_snapshot()
            result["inherited_numeric_execution_profile_scope"] = "numeric_executor_only"
            result["ppo_temporal_credit_override"] = dict(parameter="gae_lambda", value=self.lam,
                scope="stair_transfer_only", exact_binding="temporal_credit")
        if self.stair_trial_protocol in ROLLOUT_CREDIT_PROTOCOLS:
            result["rollout_collection"] = self.rollout_collection_snapshot()
        if self.stair_trial_protocol in CLEARANCE_PROGRESS_PROTOCOLS:
            result["clearance_progress_potential"] = self.clearance_progress_snapshot()
        if self.stair_trial_protocol == PROTOCOL:
            result["height_profile"] = self.height_profile_snapshot()
        return result


def parent_admission(plan, root):
    """Receipt/custody checks only; never loads a parent checkpoint."""
    root = Path(root).resolve(strict=True)
    require(plan["trial_protocol"] in PROTOCOL_FORMATS and plan["task"] == TASK
            and plan["max_iterations"] == BUDGET and plan["num_envs"] == 1024
            and plan["network_dimensions"] == [1034, 871, 73, 12], "Wrong stair plan")
    if plan["trial_protocol"] in INITIALIZATION_PROTOCOLS:
        require(plan.get("checkpoint_format") == PROTOCOL_FORMATS[plan["trial_protocol"]]
                and not initialization_config_errors(plan.get("exploration_initialization")),
                "Wrong fresh stair exploration initialization plan")
    if plan["trial_protocol"] in TEMPORAL_PROTOCOLS:
        require(not temporal_credit_config_errors(plan.get("temporal_credit"), protocol=plan["trial_protocol"]),
                "Wrong stair temporal credit plan")
    if plan["trial_protocol"] in ROLLOUT_CREDIT_PROTOCOLS:
        require(not rollout_credit_budget_errors(plan, protocol=plan["trial_protocol"])
                and not rollout_credit_collection_errors(plan.get("rollout_collection"), protocol=plan["trial_protocol"])
                and not rollout_credit_acquisition_errors(plan.get("audit_bank_source"), protocol=plan["trial_protocol"]),
                "Wrong48-current/24-fixed stair collection plan")
    if plan["trial_protocol"] in CLEARANCE_PROGRESS_PROTOCOLS:
        require(not clearance_progress_config_errors(plan.get("clearance_progress_potential"),
                                                     protocol=plan["trial_protocol"]),
                "Wrong new clearance progress reward plan")
    if plan["trial_protocol"] == PROTOCOL:
        from stair_curriculum_binding import course_config_errors, early_policy_errors
        from training_video import video_config_errors
        require(not course_config_errors(plan.get("height_course"))
                and not video_config_errors(plan.get("training_video"))
                and not early_policy_errors(json.loads(Path(plan["early_progress_policy"]).read_text())),
                "Missing human-approved fixed-target height/video plan")
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
    return dict(kind=plan["trial_protocol"], parent_checkpoint=str(parent), parent_checkpoint_sha256=PARENT_SHA,
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
