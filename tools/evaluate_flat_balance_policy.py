"""Strict split-PPO checkpoint integrity and independent direct flat walking.

The physical procedure is the preserved direct-flat first-episode evaluator:
1024 episodes,20s,world+X4.8..7.2m,eachfoot>=2 original valid alternating
landings,and same-foot ground/raw normal-force evidence. CPU completeness is
not physical walking. This tool never restores optimizers into a live plant.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TASK='Unitree-H2-Flat-Balance-Policy'
PACKAGE=Path('CtrlLab/CtrlLab/tasks/manager_based/h2_task')
LEGACY_INITIAL_SHA='ce822132385de7b9c70315087bbcc4da1739d0da8460cf656c39e14aa1424688'
TRANSFER_INITIAL_SHA='01e1584898c5313ae996ae3c3c57ce23d4cc06ba51d7ab8decbae01c59e88a68'
NEW_PROTOCOL='overspeed_credit_action_margin_v1'
PROPOSAL_PROTOCOL='simultaneous_output_proposal_v1'
PROPOSAL_INITIAL_SHA='319ff350be8c8e6359be978313f484262e2e1f9ddef03d675b7bf0a6ac7968c5'
RESERVED_PROTOCOL='reserved_simultaneous_output_proposal_v1'
RESERVED_INITIAL_SHA='ffcec4bbe62f63f77fcdfdac53d322fe19ef355dda40b7a2b1ee394015f81ed0'
RESOURCE_PROTOCOL='resource_bounded_reserved_output_proposal_v1'
RESOURCE_INITIAL_SHA='e96a9fd29cc0d7457db742133345c0224f2dbcdde522692b52eba88ebdd6977d'
PRECISION_PROTOCOL='strict_primal_resource_output_proposal_v1'
PRECISION_INITIAL_SHA='4cecd1fc24d618502402c1bd3ed5bcb0fdc7f44a0981a3bfe5855895715176b2'
VELOCITY_COST_PROTOCOL='command_scaled_velocity_cost_v1'
VELOCITY_COST_INITIAL_SHA='fe7a82e0dd9a00a6ae4551ab8b254c840d5c50b1c907cd8f2798c717456135fa'
OUTWARD_TILT_PROTOCOL='outward_tilt_growth_v1'
CLEAN_FORCE_PROTOCOL='episode_clean_force_eligibility_v1'
from task_context_metadata import (CONTEXT_PROTOCOL, CONTEXT_FORMAT, CONTEXT_DIMS,
    ACTIVE_CONTEXT_PROTOCOL, ACTIVE_CONTEXT_FORMAT, CONTEXT_PROTOCOLS, CONTEXT_FORMATS,
    validate_task_context_plan, validate_task_context_metadata, validate_normalization_metadata)
OUTWARD_REWARD_PROTOCOLS=(OUTWARD_TILT_PROTOCOL,CLEAN_FORCE_PROTOCOL,*CONTEXT_PROTOCOLS)
NATURAL_PARENT_PROTOCOLS=(VELOCITY_COST_PROTOCOL,OUTWARD_TILT_PROTOCOL,CLEAN_FORCE_PROTOCOL,CONTEXT_PROTOCOL)
NEW_REWARD_PROFILE='overspeed_positive_credit_v1'
OUTWARD_REWARD_PROFILE='overspeed_outward_tilt_growth_v1'
CLEAN_FORCE_REWARD_PROFILE='overspeed_outward_tilt_growth_clean_force_v1'
TRANSFER_CPU_RECEIPT_SHA='6300bf22b993ba92bbe5fdfedc7c9278712eb4891e38ac7d8db1b22e7ccbd4d4'
TRANSFER_RUN_SHA='ffc94e26c39ebf399055bc997a6577fb3e3a481ce08581117583139ba3837e54'
PROTECTED_RELATIVE_FILES=(
    'action_contract.py','direct_leg_action.py','config/observations_cfg.py','mdp/observations.py',
    'manager/action_manager.py','manager/actuators.py','manager/leg_action.py','assets/robot.py',
    'direct_initialization.py','stair_ascent_cfg.py','networks/__init__.py','networks/actor_critic_our.py',
    'networks/terrain_reconstructor.py','networks/lidar_terrain_reconstructor.py')
spec=importlib.util.spec_from_file_location('h2_preserved_direct_physical',Path(__file__).with_name('evaluate_direct_initialization.py'))
physical=importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical)
physical.TASK=TASK
physical.__doc__=__doc__
require,sha,write,source_hashes=physical.require,physical.sha,physical.write,physical.source_hashes


def validate_outward_reward_plan(plan):
    """Pure metadata admission for the reported fixed outward-growth cost."""
    cost=plan.get('outward_tilt_growth_cost',{})
    interval=math.cos(.35)-math.cos(.75)
    require(plan.get('trial_protocol') in OUTWARD_REWARD_PROTOCOLS
            and cost.get('kind')=='existing_body_gravity_gyro_outward_potential_rate_v1'
            and cost.get('margin_rad')==.35 and cost.get('original_native_termination_rad')==.75
            and cost.get('failure_charge')==20.
            and math.isclose(cost.get('weight',0.),-20./interval,rel_tol=0.,abs_tol=1e-10)
            and math.isclose(cost.get('potential_interval',0.),interval,rel_tol=0.,abs_tol=1e-12)
            and cost.get('continuous_monotone_margin_to_termination_cost')==-20.
            and cost.get('reward_dt_multiplications')==1
            and cost.get('existing_XY_weight_restored')==-.5
            and cost.get('new_policy_input_or_structure') is False
            and cost.get('old_tilt_margin_cost_unchanged') is True
            and cost.get('old_angular_rate_cost_unchanged') is True,
            'Reviewed outward-growth reward plan differs from its fixed original-margin/failure calibration')
    return cost


def validate_outward_reward_metadata(runtime,plan=None):
    """Check recorded actual RewardManager getters, without a model or Sim."""
    if plan is not None:
        validate_outward_reward_plan(plan)
    cost=runtime.get('outward_tilt_growth_cost',{})
    speed=runtime.get('velocity_cost_scaling',{})
    interval=math.cos(.35)-math.cos(.75)
    expected_profile=(CLEAN_FORCE_REWARD_PROFILE if runtime.get('trial_protocol') in (CLEAN_FORCE_PROTOCOL,*CONTEXT_PROTOCOLS)
                      else OUTWARD_REWARD_PROFILE)
    require(runtime.get('reward_profile')==expected_profile
            and cost.get('kind')=='existing_body_gravity_gyro_outward_potential_rate_v1'
            and cost.get('actual_function_name')=='flat_outward_tilt_growth'
            and cost.get('actual_failure_weight')==-20. and cost.get('failure_charge')==20.
            and cost.get('margin_rad')==.35 and cost.get('original_native_termination_rad')==.75
            and math.isclose(cost.get('actual_weight',0.),-20./interval,rel_tol=0.,abs_tol=1e-10)
            and math.isclose(cost.get('margin_potential',0.),1.-math.cos(.35),rel_tol=0.,abs_tol=1e-12)
            and math.isclose(cost.get('termination_potential',0.),1.-math.cos(.75),rel_tol=0.,abs_tol=1e-12)
            and math.isclose(cost.get('potential_interval',0.),interval,rel_tol=0.,abs_tol=1e-12)
            and math.isclose(cost.get('continuous_monotone_margin_to_termination_cost',0.),-20.,rel_tol=0.,abs_tol=1e-10)
            and cost.get('actual_tilt_margin_weight')==-1.
            and cost.get('actual_body_angular_rate_weight')==-.025
            and cost.get('actual_world_xy_velocity_error_weight')==-.5
            and cost.get('command_m_s')==.3
            and all(cost.get(key) is True for key in ('uses_existing_body_gravity_and_gyro',
                'same_reward_endpoint','no_new_policy_input','discrete_total_not_exact_failure_charge'))
            and type(cost.get('reward_dt_multiplications')) is int
            and cost.get('reward_dt_multiplications')==1,
            'Actual outward-growth RewardManager function/weight/frame/margin/dt profile differs')
    require(speed.get('kind')=='restored_existing_world_xy_error_v1'
            and speed.get('actual_world_xy_velocity_error_weight')==-.5
            and speed.get('actual_signed_forward_weight')==10.
            and speed.get('command_m_s')==.3 and speed.get('maximum_positive_forward_rate')==3.
            and math.isclose(speed.get('one_command_speed_error_cost_per_s',0.),-.045,rel_tol=0.,abs_tol=1e-12)
            and speed.get('function_unchanged') is True and speed.get('no_new_reward_information') is True,
            'Outward-growth trial must restore the original unchanged-kernel XY cost')
    return cost


def episode_force_policy_expectations(context=False):
    """Fixed goal-alignment metadata; no model, sensor query, or reward execution."""
    result=dict(kind='whole_episode_clean_force_positive_credit_and_first_union_failure_v1',
        agreement_absolute_tolerance_N=1.,agreement_relative_raw_norm_tolerance=.02,
        relevant_norm_threshold_N=5.,positive_control_Fz_threshold_N=5.,
        original_force_predicate_unchanged=True,
        positive_progress_and_landing_require_no_prior_or_current_relevant_mismatch=True,
        qualification_from_existing_cumulative_mismatch=True,
        physical_ever_invalid_and_survival_unchanged=True,native_termination_terms_unchanged=7,
        actual_peak_always_consumed=True,negative_debt_physical_valid_only=True,negative_debt_unweighted=True,
        first_of_union_force_failure_or_native_failure_once=True,failure_charge=20.,
        no_double_charge_later_native_failure=True,pure_timeout_exempt=True,
        timeout_with_first_force_failure_charged_once=True,
        initial_no_contact_or_missing_positive_control_not_failure=True,
        per_counter_update_idempotent=True,reward_reads_non_destructive=True,
        record_before_reset=True,reset_only_real_env_ids=True,
        positive_credit_never_reactivated_after_force_failure=True,
        all_other_reward_weights_and_kernels_unchanged_relative_v8=True,
        no_new_term_sensor_controller_or_policy_input=not context,no_fixed_foot_width_target=True)
    if context:
        result.update(no_new_reward_term_sensor_or_controller=True,global_policy_input_changed_by_explicit_context=True)
    return result


def validate_episode_force_reward_plan(plan):
    """Admit the reported clean-force lifecycle without changing physical gates."""
    validate_outward_reward_plan(plan)
    policy=plan.get('episode_force_credit_policy',{})
    require(isinstance(policy,dict),'Episode force-credit plan metadata must be a mapping')
    context=plan.get('trial_protocol') in CONTEXT_PROTOCOLS
    if context:
        validate_task_context_plan(plan)
    require(plan.get('trial_protocol') in (CLEAN_FORCE_PROTOCOL,*CONTEXT_PROTOCOLS)
            and plan.get('network_or_input_structure_change') is (context and plan.get('trial_protocol')!=ACTIVE_CONTEXT_PROTOCOL),
            'Episode force-credit plan must declare its unchanged input layout and exact protocol')
    for key,expected in episode_force_policy_expectations(context).items():
        actual=policy.get(key)
        require(actual is expected if isinstance(expected,bool) else actual==expected,
                'Episode force-credit plan differs: '+key)
    return policy


def validate_episode_force_reward_metadata(runtime,plan=None):
    """Validate configuration getters; actual event execution needs its own evidence."""
    if plan is not None:
        validate_episode_force_reward_plan(plan)
    validate_outward_reward_metadata(runtime,plan)
    context=runtime.get('trial_protocol') in CONTEXT_PROTOCOLS
    if context:
        validate_task_context_metadata(runtime,plan)
    policy=runtime.get('episode_force_credit_policy',{})
    require(isinstance(policy,dict),'Episode force-credit runtime metadata must be a mapping')
    require(runtime.get('trial_protocol') in (CLEAN_FORCE_PROTOCOL,*CONTEXT_PROTOCOLS)
            and runtime.get('reward_profile')==CLEAN_FORCE_REWARD_PROFILE,
            'Episode force-credit runtime profile differs')
    for key,expected in episode_force_policy_expectations(context).items():
        actual=policy.get(key)
        require(actual is expected if isinstance(expected,bool) else actual==expected,
                'Episode force-credit runtime differs: '+key)
    require(policy.get('actual_failure_weight')==-20.
            and policy.get('actual_failure_function')=='flat_failure',
            'Actual first-union failure function/weight differs')
    return policy


def reviewed_parent(protocol):
    if protocol==ACTIVE_CONTEXT_PROTOCOL:
        from v11_parent_custody import reviewed_parent as active_parent
        return active_parent()
    if protocol==CONTEXT_PROTOCOL:
        parent=reviewed_parent(CLEAN_FORCE_PROTOCOL)
        return dict(parent,kind='saved_complete_PPO_v6_context_extension_transfer',
                    mode='extend_completed_v6_policy_and_frozen_prefix_norm_fresh_optimization')
    if protocol in NATURAL_PARENT_PROTOCOLS:
        return dict(sha=VELOCITY_COST_INITIAL_SHA,
            receipt='e7016cd9f64cf81e7eb9562bbde371caf44c0d63b8e6bc308ae882bf2d77ee4b',
            run='797aecdb0c137d57f396fd7705940921b0ea9624c7b75323f8bcde58caf9e33c',
            format='h2_guarded_flat_ppo_v6',iteration=999,updates=1000,status='completed',
            kind='saved_complete_PPO_v6_policy_transfer',
            mode='transfer_completed_v6_policy_and_dualnorm_fresh_optimization',restored=False)
    if protocol==PRECISION_PROTOCOL:
        return dict(sha=PRECISION_INITIAL_SHA,
            receipt='ce8563d9b7d715928d0d4cc2c058ac65553d3674e259a273bc2569722cd10075',
            run='3717e86bd5c007f47837714b92fa4f78cdd8e3f30c9637e3c184224f8e281f52',
            format='h2_guarded_flat_ppo_v5',iteration=182,updates=183,status='stopped_functional_guard',
            kind='saved_restored_PPO_v5_policy_transfer',
            mode='transfer_restored_v5_policy_and_dualnorm_fresh_optimization',restored=True)
    if protocol==RESOURCE_PROTOCOL:
        return dict(sha=RESOURCE_INITIAL_SHA,
            receipt='2c488d6e4a67f530e392486d5fd7b42c9c2d1f14bba509698ea3e8b169a14192',
            run='413c9af79a8e033652eed65daba296af0dd5baada81b51971c10011a1aa04a1f',
            format='h2_guarded_flat_ppo_v4',iteration=612,updates=613,status='stopped_functional_guard',
            kind='saved_restored_PPO_v4_policy_transfer',
            mode='transfer_restored_v4_policy_and_dualnorm_fresh_optimization',restored=True)
    if protocol==RESERVED_PROTOCOL:
        return dict(sha=RESERVED_INITIAL_SHA,
            receipt='338743a01a8590aea06acc58baf2192e4d5026d6875d606859a942ee57802e8e',
            run='b52da5dd5ef483d5ac1d465df88f2be45dec30a638f0fe8fe45e9b6df924f315',
            format='h2_guarded_flat_ppo_v3',iteration=104,updates=105,status='stopped_functional_guard',
            kind='saved_restored_PPO_v3_policy_transfer',
            mode='transfer_restored_v3_policy_and_dualnorm_fresh_optimization',restored=True)
    if protocol==NEW_PROTOCOL:
        return dict(sha=TRANSFER_INITIAL_SHA,receipt=TRANSFER_CPU_RECEIPT_SHA,run=TRANSFER_RUN_SHA,
            format='h2_guarded_flat_ppo_v1',iteration=164,updates=165,status='failed',
            kind='saved_pre_guard_PPO_v1_policy_transfer',
            mode='transfer_pre_guard_policy_and_dualnorm_fresh_optimization',restored=False)
    require(protocol==PROPOSAL_PROTOCOL,'Unrecognized reviewed parent protocol')
    return dict(sha=PROPOSAL_INITIAL_SHA,
        receipt='df70999cd1b10a78dabe4d04ce900a6487c5500a0153f7aafc2108e85a6572d4',
        run='431216e00a2c2b408480255fe85a93af819d6685f48b84cd513803e9d8bac3c8',
        format='h2_guarded_flat_ppo_v2',iteration=100,updates=101,status='stopped_functional_guard',
        kind='saved_restored_PPO_v2_policy_transfer',
        mode='transfer_restored_policy_and_dualnorm_fresh_optimization',restored=True)


def declared_checkpoint_profile(fmt, algorithm, state, run, plan, transfer=None, packet_protocol=None):
    """Metadata protocol discrimination; never authorizes arbitrary init SHAs."""
    if fmt=='h2_guarded_flat_ppo_v1':
        require(algorithm.get('trial_protocol','guarded_flat_balance_v1')=='guarded_flat_balance_v1'
                and state.get('trial_protocol','guarded_flat_balance_v1')=='guarded_flat_balance_v1'
                and plan.get('trial_protocol','guarded_flat_balance_v1')=='guarded_flat_balance_v1',
                'A new protocol must not masquerade as legacyv1')
        require(float(algorithm.get('actor_action_budget_penalty_weight',0.))==0.,
                'Legacyv1 cannot claim the new action-margin optimization')
        require(state.get('action_margin',{}).get('weight',0.)==0.
                and packet_protocol in (None,'guarded_flat_balance_v1')
                and run.get('runtime_flat_geometry',{}).get('reward_profile') in (None,'legacy_supported_peak_credit_v1'),
                'Legacyv1 may not conceal a new reward/margin profile')
        require(plan['initial_policy_sha256']==LEGACY_INITIAL_SHA,'Legacyv1 initializer changed')
        require(transfer is None,'Legacyv1 must not claim a typedPPO transfer')
        return dict(trial_protocol='guarded_flat_balance_v1',initial_sha256=LEGACY_INITIAL_SHA,
                    checkpoint_format=fmt,typed_policy_transfer=False)
    active=fmt==ACTIVE_CONTEXT_FORMAT
    context=fmt in CONTEXT_FORMATS
    clean_force=context or fmt=='h2_guarded_flat_ppo_v9'
    outward_tilt=clean_force or fmt=='h2_guarded_flat_ppo_v8'
    velocity_cost=fmt=='h2_guarded_flat_ppo_v7'
    natural_parent=(outward_tilt or velocity_cost) and not active
    precision=outward_tilt or velocity_cost or fmt=='h2_guarded_flat_ppo_v6'
    resource=precision or fmt=='h2_guarded_flat_ppo_v5'
    reserved=resource or fmt=='h2_guarded_flat_ppo_v4'
    projected=reserved or fmt=='h2_guarded_flat_ppo_v3'
    require(projected or fmt=='h2_guarded_flat_ppo_v2','Unrecognized PPO checkpoint format')
    protocol=ACTIVE_CONTEXT_PROTOCOL if active else CONTEXT_PROTOCOL if context else CLEAN_FORCE_PROTOCOL if clean_force else OUTWARD_TILT_PROTOCOL if outward_tilt else VELOCITY_COST_PROTOCOL if velocity_cost else PRECISION_PROTOCOL if precision else RESOURCE_PROTOCOL if resource else RESERVED_PROTOCOL if reserved else PROPOSAL_PROTOCOL if projected else NEW_PROTOCOL
    parent=reviewed_parent(protocol)
    require(packet_protocol==algorithm.get('trial_protocol')==state.get('trial_protocol')
            ==plan.get('trial_protocol')==protocol,'Declared typed profiles disagree')
    require(algorithm.get('actor_action_budget_penalty_weight')==(0. if projected else .01)
            and algorithm.get('actor_action_budget_penalty_epsilon')==1e-6
            and algorithm['gamma']==.99 and algorithm['lam']==.95,
            'Declared actor objective/gamma/GAE contract differs')
    margin=state['action_margin']
    if projected:
        require(margin['weight']==0.,'Projected proposal may not silently retain the top1 soft objective')
        projection=state['proposal_projection']
        require(projection['kind']==algorithm.get('actor_proposal_projection')==('simultaneous_actual_bank_reserved_linear_v1' if reserved else 'simultaneous_actual_bank_linear_v1')
                and projection['interior_fraction']==algorithm.get('actor_proposal_interior_fraction')==.5
                and projection['max_active_rows']==(28 if resource else 8)
                and algorithm.get('actor_proposal_max_active_rows')==(None if resource else 8)
                and projection['max_solver_sweeps']==algorithm.get('actor_proposal_max_solver_sweeps')==2048
                and all(projection[k] is True for k in ('whole_update_reference','linear_full_bank','hard_gate_unchanged','no_retry')),
                'Saved simultaneous proposal contract differs')
        if reserved:
            require(projection['working_budget_fraction']==algorithm.get('actor_proposal_working_budget_fraction')==.5
                    and projection['whole_update_fixed_working_bound'] is True
                    and projection['solver_tolerance_reserved_inward'] is True,
                    'Saved fixed whole-update working reserve differs')
        if resource:
            require(algorithm.get('actor_proposal_jacobian_arena_budget_bytes')==1073741824
                    and projection.get('jacobian_arena_budget_bytes')==1073741824
                    and projection.get('configured_max_active_rows','missing') is None
                    and projection.get('jacobian_mean_parameter_elements')==(782988 if context else 780428)
                    and projection.get('jacobian_bytes_per_row')==(37583424 if context else 37460544)
                    and projection.get('jacobian_arena_capacity_bytes')==(1052335872 if context else 1048895232)
                    and projection.get('resource_kind')=='single_cpu_float32_jacobian_arena_v1',
                    'Saved compute coverage is not the fixed CPU-byte-derived arena protocol')
        if precision:
            require(algorithm.get('actor_proposal_primal_precision_policy')=='direct_fp64_primal_envelope_v1'
                    and projection.get('primal_precision_policy')=='direct_fp64_primal_envelope_v1'
                    and (active or (projection.get('max_polish_per_QP')==1
                    and projection.get('max_support_pivots')==0
                    and projection.get('max_polish_per_projection')==28))
                    and projection.get('solver_optimality_and_direct_primal_separate') is True
                    and projection.get('primal_envelope_is_machine_derived') is True
                    and projection.get('final_same_direction_cast_JVP_required') is True,
                    'Saved inner feasibility precision differs from the reviewed strict primal protocol')
        else:
            require(algorithm.get('actor_proposal_primal_precision_policy') is None
                    and projection.get('primal_precision_policy') is None,
                    'A strict primal solve cannot masquerade as a legacy proposal profile')
        if active:
            from v11_parent_custody import validate_support_projection_metadata
            validate_support_projection_metadata(algorithm,projection)
    else:
        require(margin['kind']=='worst_actual_input_log_barrier' and margin['weight']==.01
                and margin['epsilon']==1e-6 and margin['whole_update_reference'] is True
                and margin['hard_gate_unchanged'] is True,'Saved action-margin contract differs')
    require(plan['initial_policy_sha256']==parent['sha']
            and plan['initialization_mode']==parent['mode']
            and run['initialization_mode']==plan['initialization_mode'],'Typed initializer is not the exact reviewed transfer')
    require(isinstance(transfer,dict) and transfer==run['initial_policy_transfer'],'Packet/run transfer provenance disagrees')
    require(transfer['kind']==parent['kind']
            and transfer['mode']==plan['initialization_mode'] and transfer['parent_checkpoint_sha256']==parent['sha']
            and transfer['parent_accepted_updates']==parent['updates'] and transfer['parent_last_complete_iteration']==parent['iteration']
            and transfer['parent_uncommitted_rollout'] is (not natural_parent)
            and transfer.get('parent_natural_completion',False) is natural_parent
            and transfer['parent_runtime_rollback_completed'] is parent['restored'],'Parent transfer identity/history differs')
    require(all(transfer[k] is True for k in (
            ('model_prefix_and_dualnorm_prefix_only' if context and not active else 'model_and_dualnorm_only'),'fresh_actor_optimizer',
            'fresh_critic_optimizer','fresh_RND','fresh_RND_optimizer',
            ('actor_and_critic_prefix_loaded_bitwise' if context and not active else 'actor_and_critic_loaded_bitwise'),
            ('dualnorm_prefix_loaded_bitwise' if context and not active else 'dualnorm_loaded_bitwise'),'loading_did_not_change_fresh_RNG'))
            and transfer['parent_RNG_loaded'] is False and transfer['new_starting_iteration']==0
            and transfer['new_checkpoint'] is None
            and transfer['normalizer_prefix_counts' if context else 'norm_counts']=={'actor':24576000,'critic':24576000},
            'Typed policy transfer must not resume parent optimization/RND/RNG')
    if context:
        validate_task_context_plan(plan)
        validate_normalization_metadata(state.get('normalization_abi'),state.get('normalizer_prefix_counts'))
        if active:
            require(transfer.get('same_width_context_transfer') is True
                    and transfer.get('network_or_input_structure_changed') is False
                    and transfer.get('raw_context_semantics_unchanged') is True
                    and transfer.get('parent_network_dimensions')==transfer.get('network_dimensions')==[1034,871,73,12],
                    'v11 cannot claim a new extension, changed input semantics or hidden optimization resume')
        else:
            require(transfer.get('new_context_columns_zero') is True, 'Context extension columns were not initialized to zero')
    runtime=run['runtime_flat_geometry']
    require(runtime['trial_protocol']==protocol and runtime['reward_profile']==(CLEAN_FORCE_REWARD_PROFILE if clean_force else OUTWARD_REWARD_PROFILE if outward_tilt else NEW_REWARD_PROFILE),
            'Actual environment reward profile differs from the declared optimization')
    speed=runtime['positive_credit_speed_weight']
    require(speed['command_m_s']==speed['full_at_or_below_m_s']==.3
            and speed['zero_at_or_above_m_s']==.6 and speed['actual_peak_consumed_unweighted'] is True
            and speed['negative_debt_unweighted'] is True and speed['physical_or_action_speed_limit_changed'] is False,
            'Actual positive-credit weighting changed the declared reward/physical semantics')
    if velocity_cost:
        cost=runtime.get('velocity_cost_scaling',{})
        require(cost.get('kind')=='command_scaled_existing_world_xy_error_v1'
                and cost.get('actual_world_xy_velocity_error_weight')==-(10./.3)
                and cost.get('actual_signed_forward_weight')==10.
                and cost.get('command_m_s')==.3 and cost.get('maximum_positive_forward_rate')==3.
                and abs(cost.get('one_command_speed_error_cost_per_s',0.)+3.)<1e-12
                and cost.get('function_unchanged') is True and cost.get('no_new_reward_information') is True,
                'Actual existing-XY-cost calibration differs from its reviewed command/rate scale')
    if clean_force:
        validate_episode_force_reward_metadata(runtime,plan)
    elif outward_tilt:
        validate_outward_reward_metadata(runtime,plan)
    return dict(trial_protocol=protocol,initial_sha256=parent['sha'],
                checkpoint_format=fmt,typed_policy_transfer=True,action_margin=margin,
                proposal_projection=state.get('proposal_projection') if projected else None)


def verify_transfer_files(plan, transfer, child_source):
    """Verify old-source provenance and only the declared protected shared files.

    A new reward implementation must have its own full source manifest. It is
    not required to be byte-identical to all old reward/task files.
    """
    if plan['trial_protocol']==ACTIVE_CONTEXT_PROTOCOL:
        return verify_active_transfer_files(plan,transfer,child_source)
    expected=reviewed_parent(plan['trial_protocol'])
    require(plan['initial_policy_CPU_receipt_sha256']==expected['receipt']
            and plan['initial_parent_run_sha256']==expected['run'],
            'New protocol must use the reviewed immutable parentCPU/run receipts')
    bindings=(('parent_checkpoint','initial_policy'),('parent_CPU_receipt','initial_policy_CPU_receipt'),
              ('parent_run','initial_parent_run'),('parent_source_root','initial_parent_source_root'))
    for key,plan_key in bindings:
        require(Path(transfer[key]).resolve(strict=True)==Path(plan[plan_key]).resolve(strict=True),
                'Typed transfer file identity changed: '+key)
        if key!='parent_source_root':
            require(transfer[key+'_sha256']==plan[plan_key+'_sha256']==sha(Path(plan[plan_key])),
                    'Typed transfer bytes changed: '+key)
    parent_receipt=json.loads(Path(plan['initial_policy_CPU_receipt']).read_text())
    parent_run=json.loads(Path(plan['initial_parent_run']).read_text())
    parent_source=Path(plan['initial_parent_source_root']).resolve(strict=True)
    parent_hashes=source_hashes(parent_source)
    require(parent_receipt['status']=='passed' and parent_receipt['cpu_only'] is True
            and parent_receipt['sha256']==expected['sha']
            and parent_receipt['checkpoint_format']==expected['format']
            and parent_receipt['checkpoint_iteration']==expected['iteration'] and parent_receipt['committed_updates']==expected['updates'],
            'ParentCPU receipt is not the exact reviewed pre-update snapshot')
    require(Path(parent_receipt['checkpoint']).resolve(strict=True)==Path(plan['initial_policy']).resolve(strict=True)
            and parent_receipt['source_root']==str(parent_source)
            and parent_receipt['producer_run_sha256']==plan['initial_parent_run_sha256']
            and parent_receipt['source_sha256']==parent_hashes==transfer['parent_source_sha256'],
            'ParentCPU/source/run custody differs')
    require(parent_run['git_commit']==transfer['parent_source_commit']
            and parent_run['completed_iterations']==expected['updates'] and parent_run['status']==expected['status'],
            'Parent source/budget/terminal status differs from its exact reviewed protocol')
    if plan['trial_protocol'] in NATURAL_PARENT_PROTOCOLS:
        require(plan.get('parent_natural_completion') is True
                and plan.get('parent_runtime_rollback_completed') is False
                and parent_run.get('actual_sampled_environment_steps')==24576000
                and parent_run.get('discarded_uncommitted_rollout_steps')==0
                and parent_run.get('guarded_ppo_state',{}).get('uncommitted_rollout') is False
                and transfer.get('parent_rollback_evidence') is None,
                'Natural complete parent must not be relabeled a guard-restored or discarded-rollout snapshot')
    if expected['restored']:
        rollback=transfer['parent_rollback_evidence']
        rollback_path=Path(parent_run['functional_guard_evidence']).resolve(strict=True)
        require(Path(rollback['path']).resolve(strict=True)==rollback_path
                and sha(rollback_path)==rollback['sha256'],'Exact parent rollback evidence changed')
        proof=json.loads(rollback_path.read_text())
        require(proof.get('complete_update_restored') is True
                and proof['rollback_state_digest_before']==proof['rollback_state_digest_after']==rollback['persistent_state_digest']
                and proof['restored_checkpoint_sha256']==expected['sha']
                and proof['completed_updates']==expected['updates'],'Parent actual coherent restore proof differs')
    require(transfer['parent_external_dependency_sha256']==parent_receipt['external_dependency_sha256']
            ==parent_run['external_dependency_sha256'],'Parent external asset/reconstruction provenance differs')
    require(isinstance(transfer['fresh_RND_digest'],str) and len(transfer['fresh_RND_digest'])==64,
            'Typed transfer lacks the freshRND state digest')
    protected=plan['protected_network_input_native_source_sha256']
    if plan['trial_protocol']==CONTEXT_PROTOCOL:
        expected_deltas={'source/'+str(PACKAGE/'networks/actor_critic_our.py'),
                         'source/'+str(PACKAGE/'direct_initialization.py'),'tools/functional_proposal.py'}
        require(set(plan.get('reported_source_delta_sha256',{}))==expected_deltas,
                'Context source admission may alter only the two reported source adapters and projector input shape')
        require(sha(Path(child_source).parent/'functional_proposal.py')==plan['reported_source_delta_sha256']['tools/functional_proposal.py'],
                'Actual producer projector source differs from the reviewed input adapter')
    expected_keys={'source/'+str(PACKAGE/name) for name in PROTECTED_RELATIVE_FILES}
    require(set(protected)==expected_keys and transfer['protected_network_input_native_source_sha256']==protected,
            'Protected network/input/native file declaration changed')
    for key,digest in protected.items():
        require(key.startswith('source/') and '..' not in Path(key).parts,'Invalid protected source key')
        relative=key.removeprefix('source/')
        require(parent_receipt['source_sha256'][key]==sha(parent_source/relative)==digest,
                'Protected parent network/input/native source changed: '+key)
        child_expected=(plan['reported_source_delta_sha256'].get(key,digest)
                        if plan['trial_protocol']==CONTEXT_PROTOCOL else digest)
        require(sha(Path(child_source)/relative)==child_expected,
                'Protected child source is not the exact reviewed input adapter: '+key)
    return dict(parent_checkpoint=str(Path(plan['initial_policy'])),parent_checkpoint_sha256=expected['sha'],parent_CPU_receipt=str(Path(plan['initial_policy_CPU_receipt'])),
        parent_CPU_receipt_sha256=plan['initial_policy_CPU_receipt_sha256'],
        parent_run=str(Path(plan['initial_parent_run'])),parent_run_sha256=plan['initial_parent_run_sha256'],
        parent_source_root=str(parent_source),parent_source_sha256=parent_hashes,
        parent_rollback_evidence=transfer.get('parent_rollback_evidence'),
        parent_external_dependency_sha256=transfer['parent_external_dependency_sha256'],
        parent_accepted_updates=expected['updates'],parent_last_complete_iteration=expected['iteration'],protected_file_count=14,
        model_and_dualnorm_only=plan['trial_protocol']!=CONTEXT_PROTOCOL,
        model_prefix_and_dualnorm_prefix_only=plan['trial_protocol']==CONTEXT_PROTOCOL,
        parent_optimization_and_RND_and_RNG_not_resumed=True)


def verify_active_transfer_files(plan, transfer, child_source):
    from v11_parent_custody import verify_parent_files, PARENT_KIND, INITIALIZATION_MODE
    actual=verify_parent_files(plan,ROOT)
    require(transfer['kind']==PARENT_KIND and transfer['mode']==INITIALIZATION_MODE,
            'v11 parent transfer kind/mode differs')
    for name,key in (('parent_checkpoint','initial_policy'),('parent_CPU_receipt','initial_policy_CPU_receipt'),
                     ('parent_run','initial_parent_run'),('parent_source_root','initial_parent_source_root')):
        require(Path(transfer[name]).resolve(strict=True)==Path(plan[key]).resolve(strict=True),
                'v11 transfer path differs: '+name)
        if name!='parent_source_root':
            require(transfer[name+'_sha256']==plan[key+'_sha256'], 'v11 transfer hash differs: '+name)
    require(transfer['parent_source_sha256']==actual['parent_source_sha256']
            and transfer['parent_rollback_evidence']==actual['parent_rollback_evidence']
            and transfer['parent_persistent_state_digest']==actual['parent_persistent_state_digest'],
            'v11 actual parent source/whole-state proof differs')
    protected=plan['protected_network_input_native_source_sha256']
    expected_keys={'source/'+str(PACKAGE/name) for name in PROTECTED_RELATIVE_FILES}
    require(set(protected)==expected_keys and transfer['protected_network_input_native_source_sha256']==protected,
            'v11 protected14 declaration differs')
    for key,digest in protected.items():
        require(actual['parent_source_sha256'][key]==digest                and sha(Path(child_source)/key.removeprefix('source/'))==digest,
                'v11 may not silently re-extend or modify the existing network/native inputs: '+key)
    parent_receipt=json.loads(Path(plan['initial_policy_CPU_receipt']).read_text())
    require(transfer['parent_external_dependency_sha256']==parent_receipt['external_dependency_sha256'],
            'v11 external dependency provenance differs')
    require(sha(Path(child_source).parent/'task_context_normalization.py')            ==parent_receipt['producer_training_tool_sha256']['task_context_normalization.py'],
            'v11 must retain the exact parent block-normalization implementation')
    return {key:transfer[key] for key in ('parent_checkpoint','parent_checkpoint_sha256','parent_CPU_receipt',
        'parent_CPU_receipt_sha256','parent_run','parent_run_sha256','parent_source_root','parent_source_sha256',
        'parent_rollback_evidence','parent_external_dependency_sha256','parent_accepted_updates','parent_last_complete_iteration',
        'parent_persistent_state_digest','parent_formal_review','parent_formal_review_sha256','parent_formal_result',
        'parent_formal_result_sha256')} | dict(protected_file_count=14,model_and_dualnorm_only=True,
        same_width_context_transfer=True,parent_optimization_and_RND_and_RNG_not_resumed=True)


def transfer_provenance_unchanged(receipt):
    transfer=receipt.get('typed_initialization_transfer')
    if transfer and transfer.get('same_width_context_transfer'):
        if not all(sha(Path(transfer[k]))==transfer[k+'_sha256'] for k in ('parent_formal_review','parent_formal_result')):
            return False
    return transfer is None or (sha(Path(transfer['parent_checkpoint']))==transfer.get('parent_checkpoint_sha256',TRANSFER_INITIAL_SHA)
        and sha(Path(transfer['parent_CPU_receipt']))==transfer['parent_CPU_receipt_sha256']
        and sha(Path(transfer['parent_run']))==transfer['parent_run_sha256']
        and (transfer.get('parent_rollback_evidence') is None or
             sha(Path(transfer['parent_rollback_evidence']['path']))==transfer['parent_rollback_evidence']['sha256'])
        and source_hashes(Path(transfer['parent_source_root']))==transfer['parent_source_sha256'])


def context_bank_provenance_unchanged(receipt):
    bank=receipt.get("fixed_context_bank_custody")
    if bank is None:
        return receipt.get("checkpoint_format") not in CONTEXT_FORMATS
    return all(Path(bank[key]).is_file() and sha(Path(bank[key]))==bank[sha_key]
               for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),
                                   ("seal_path","seal_sha256"),("acquisition_log_path","acquisition_log_sha256")))


def check_adam(torch, state, names, parameters, expected_step, expected_lr, label):
    """Check the actual parameter order and all Adam moments, including warm0."""
    require(isinstance(state,dict) and set(('state','param_groups')).issubset(state),label+' Adam packet missing')
    groups=state['param_groups']
    order=[pid for group in groups for pid in group['params']]
    require(len(order)==len(parameters)==len(names) and len(set(order))==len(order),label+' Adam parameter coverage differs')
    require(all(math.isclose(float(group['lr']),expected_lr,rel_tol=1e-7,abs_tol=1e-12) for group in groups),label+' Adam LR differs')
    for group in groups:
        require(tuple(group['betas'])==(.9,.999) and math.isclose(float(group['eps']),1e-8)
                and float(group['weight_decay'])==0. and not group.get('amsgrad',False),label+' Adam hyperparameters differ')
    if expected_step==0:
        require(not state['state'],label+' Adam has moments during declared zero-update warm state')
    else:
        require(set(state['state'])==set(order),label+' Adam moment coverage is incomplete')
        for pid,parameter in zip(order,parameters):
            item=state['state'][pid]
            require({'step','exp_avg','exp_avg_sq'}.issubset(item),label+' Adam moments missing')
            require(isinstance(item['step'],torch.Tensor) and item['step'].numel()==1
                    and float(item['step'])==expected_step,label+' Adam step disagrees with actual update counters')
            require(item['exp_avg'].shape==item['exp_avg_sq'].shape==parameter.shape,label+' Adam moment shape differs')
    return dict(parameter_count=len(names),steps=expected_step,learning_rate=expected_lr,names=names)


def verify_context_bank_custody(run, plan):
    """Verify the new bank's once-sealed files and request binding, without replay."""
    bank=run["guarded_flat_learning"]["fixed_real_bank"]
    require(bank.get("status")=="sealed" and bank.get("rows")==4096,
            "Context PPO must seal the real first-rollout bank before update0")
    directory=Path(run["source_snapshot"]).resolve(strict=True).parent.parent/"functional_guard"
    for key,sha_key in (("bank_path","bank_sha256"),("receipt_path","receipt_sha256"),
                        ("acquisition_log_path","acquisition_log_sha256")):
        path=Path(bank[key]).resolve(strict=True)
        require(path.is_relative_to(directory) and sha(path)==bank[sha_key],"New bank custody differs: "+key)
    actual=json.loads(Path(bank["receipt_path"]).read_text())
    require(all(actual.get(k)==v for k,v in bank.items() if k not in ("receipt_path","receipt_sha256")),
            "Actual bank receipt and producer binding differ")
    bindings=actual["bindings"]
    case=Path(bindings["case"]).resolve(strict=True)
    request=case/"request.json"
    seal=json.loads((case/"actual_bank.sealed.json").read_text())
    require(seal["case"]==str(case) and seal["fixed_real_bank"]==bank
            and seal["request_sha256"]==sha(request)==bindings["request_sha256"]
            and seal["plan_sha256"]==bindings["plan_sha256"]==run["guarded_flat_learning"]["plan_sha256"]
            and bindings["initial_policy_sha256"]==plan["initial_policy_sha256"]
            and bindings["source_commit"]==run["git_commit"]
            and bindings["training_tool_sha256"]==run["training_tool_sha256"],
            "Actual bank was not bound to this request/source/plan before update0")
    return dict(bank_path=bank["bank_path"],bank_sha256=bank["bank_sha256"],
                receipt_path=bank["receipt_path"],receipt_sha256=bank["receipt_sha256"],
                acquisition_log_path=bank["acquisition_log_path"],acquisition_log_sha256=bank["acquisition_log_sha256"],
                seal_path=str(case/"actual_bank.sealed.json"),seal_sha256=sha(case/"actual_bank.sealed.json"),
                raw_width=1034,rows=4096,finite_bank_not_physical_acceptance=True)


def validate_cpu(args, *, restored_state_inspector=None):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    import yaml
    from rsl_rl.modules.rnd import RandomNetworkDistillation
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    required={'format','iter','model_state_dict','actor_optimizer_state_dict','critic_optimizer_state_dict',
        'optimizer_parameter_names','obs_norm_state_dict','privileged_obs_norm_state_dict',
        'rnd_state_dict','rnd_optimizer_state_dict','rnd_optimizer_parameter_names','rnd_runtime',
        'guarded_ppo_state','exploration_state','rng_state'}
    require(isinstance(saved,dict) and required.issubset(saved),'Not a complete guarded split-PPO checkpoint')
    require(saved['format'] in ('h2_guarded_flat_ppo_v1','h2_guarded_flat_ppo_v2','h2_guarded_flat_ppo_v3','h2_guarded_flat_ppo_v4','h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'),'Unrecognized PPO checkpoint format')
    require('optimizer_state_dict' not in saved,'A shared optimizer must not masquerade as the split optimizers')
    active=saved['format']==ACTIVE_CONTEXT_FORMAT
    context=saved['format'] in CONTEXT_FORMATS
    iteration=int(saved['iter'])
    require(0<=iteration<1000,'Checkpoint exceeds the bounded flat experiment')
    run_dir=args.checkpoint.parent
    run_path=run_dir/'run.json'
    run=json.loads(run_path.read_text())
    require(run['task']==TASK and run['num_envs']==1024 and run['num_steps_per_env']==24,
            'Checkpoint was produced by another task/rollout')
    require(Path(run['source_snapshot']).resolve(strict=True)==args.source_root,'PPO source snapshot binding differs')
    require(run.get('checkpoint') is None and run.get('starting_iteration')==0,'This trial must start fresh PPO at0')
    hashes=source_hashes(args.source_root)
    require(json.loads((args.source_root.parent/'main_source_sha256.json').read_text())==hashes,
            'PPO training source changed')
    training_tools=run['training_tool_sha256']
    require({'train.py','training_supervision.py','guarded_flat_ppo.py','stair_exploration.py'}.issubset(training_tools),
            'Guarded optimizer producer tool fingerprints are incomplete')
    if context:
        require({'task_context_normalization.py','task_context_metadata.py'}.issubset(training_tools),
                'Producer omitted the explicit normalization/metadata tool dependencies')
        if active:
            require('v11_parent_custody.py' in training_tools, 'v11 omitted its strict restored-parent admission dependency')
            for name in ('task_context_metadata.py','v11_parent_custody.py'):
                require(training_tools[name]==sha(Path(__file__).with_name(name)),
                        'Actual v11 CPU-imported helper differs from its producer snapshot: '+name)
    if saved['format'] in ('h2_guarded_flat_ppo_v9',*CONTEXT_FORMATS):
        require({'evaluate_flat_balance_policy.py','evaluate_direct_initialization.py'}.issubset(training_tools),
                'Episode force-credit admission source dependencies are not frozen in the producer snapshot')
    for name,digest in training_tools.items():
        require(sha(args.source_root.parent/name)==digest,'Recorded producer tool changed: '+name)
    if saved['format'] in ('h2_guarded_flat_ppo_v3','h2_guarded_flat_ppo_v4','h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'):
        require('functional_proposal.py' in training_tools,'Producer omitted simultaneous projection implementation')
    agent=yaml.safe_load((run_dir/'params/agent.yaml').read_text())
    algo=agent['algorithm']
    protocol=run['guarded_flat_learning']
    plan_path=Path(protocol['plan_path']).resolve(strict=True)
    require(sha(plan_path)==protocol['plan_sha256'],'Authorized flat plan no longer matches producer receipt')
    plan=json.loads(plan_path.read_text())
    profile=declared_checkpoint_profile(saved['format'],algo,saved['guarded_ppo_state'],run,plan,
        saved.get('initialization_transfer'),saved.get('trial_protocol'))
    require(algo['class_name']=='H2GuardedPPO' and algo['schedule']=='fixed'
            and algo['num_learning_epochs']==5 and algo['num_mini_batches']==4,'Saved algorithm contract differs')
    for key,value in (('actor_learning_rate',1e-5),('critic_learning_rate',.001),
                      ('critic_only_iterations',40),('actor_max_grad_norm',1.),('critic_max_grad_norm',1.)):
        require(float(algo[key])==value,'Saved split optimizer contract differs: '+key)
    require(algo['rnd_cfg']['weight']==0. and algo['entropy_coef']==1e-4,'Novelty/entropy setting differs')
    policy_cfg=dict(agent['policy'])
    require(policy_cfg.pop('class_name')=='H2ActorCriticOur','Original H2 network class changed')
    cls=physical.load_module(args.source_root/PACKAGE/'networks/actor_critic_our.py','flat_cpu_original_network').H2ActorCriticOur
    require(policy_cfg.get('extra_context_dim',0)==(5 if context else 0), 'Network context declaration differs')
    model=cls(1034 if context else 1029,871 if context else 866,12,**policy_cfg).eval()
    model.load_state_dict(saved['model_state_dict'],strict=True)
    require(tuple(model.std.shape)==(12,) and bool((model.std>0).all()),'Original exploration std missing or invalid')
    actor_named=[('actor.'+name,p) for name,p in model.actor.named_parameters()]+[('std',model.std)]
    critic_named=[('critic.'+name,p) for name,p in model.critic.named_parameters()]
    require(len(actor_named)==45 and len(critic_named)==8,'Split45 actor+std/8critic layout changed')
    names=saved['optimizer_parameter_names']
    require(names=={'actor':[n for n,_ in actor_named],'critic':[n for n,_ in critic_named]},'Actual optimizer name/order differs')
    gs=saved['guarded_ppo_state']
    # Counters reflect only committed whole updates. A veto restores all three
    # optimizers; the iteration filename alone is not used as an Adam counter.
    count=int(gs['completed_updates'])
    require(0<=count<=1000 and count<=iteration+1,'Invalid committed PPO update count')
    expected=dict(actor=max(0,count-40)*20,critic=count*20,rnd=count*20)
    if profile['typed_policy_transfer']:
        if profile['proposal_projection'] is not None:
            require(gs['proposal_projection']['committed_minibatches']==expected['actor']
                    and gs['proposal_projection']['proposal_calls']==expected['actor']
                    and gs['proposal_projection']['std_hook_calls']==expected['actor']
                    and gs['action_margin']['committed_minibatches']==0,
                    'Proposal-projection committed-minibatch count differs from actual actorAdam')
        else:
            require(gs['action_margin']['committed_minibatches']==expected['actor'],
                    'Action-margin committed-minibatch counter differs from actorAdam')
        require(plan.get('status')=='reviewed_ready','New profile plan has not completed its prelaunch review')
    if saved['format'] in ('h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'):
        projection=gs['proposal_projection']
        width=sum(p.numel() for _,p in actor_named[:-1]);row_bytes=12*width*4
        require(projection['jacobian_mean_parameter_elements']==width==(782988 if context else 780428)
                and projection['jacobian_bytes_per_row']==row_bytes
                and projection['max_active_rows']==min(5120,1073741824//row_bytes),
                'Actual actor parameter layout differs from byte-derived compute coverage')
        if expected['actor']:
            summary=projection['latest_summary'];used=summary['jacobian_used_rows']
            require(summary['resource_kind']==projection['resource_kind']
                    and summary['jacobian_arena_budget_bytes']==1073741824
                    and summary['jacobian_bytes_per_row']==row_bytes
                    and summary['jacobian_derived_max_active_rows']==28
                    and summary['jacobian_arena_capacity_bytes']==(1052335872 if context else 1048895232)
                    and 0<=used<=28 and summary['jacobian_used_bytes']==used*row_bytes
                    and summary['signed_midpoint_bounds_verified_coordinates']==61440
                    and summary['nonlinear_hard_check_passed'] is True,
                    'Actual last committed actor update lacks its full-bank resource and nonlinear certificate')
            if saved['format'] in ('h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'):
                precision=summary.get('primal_precision_summary',{})
                require(summary.get('primal_precision_policy')=='direct_fp64_primal_envelope_v1'
                        and precision.get('direct_primal_certified') is True
                        and (active or (precision.get('polish_call_limit_per_subproblem')==1
                        and precision.get('polish_call_limit_per_projection')==28
                        and precision.get('support_pivot_limit')==0))
                        and precision.get('final_same_direction_row_JVP_checked') is True
                        and precision==projection.get('primal_precision_summary'),
                        'Actual last committed actor update lacks its direct-primal/native-direction certificate')
                if active:
                    from v11_parent_custody import validate_support_projection_metadata
                    validate_support_projection_metadata(algo,projection,summary)
    require({name:int(gs[name+'_optimizer_steps']) for name in expected}==expected,
            'Committed split/RND update budget differs')
    require(gs['critic_only_iterations']==40 and gs['actor_updates_accepted']==max(0,count-40)
            and gs['critic_warmup_complete']==(count>=40),'Critic warm/accepted actor counters disagree')
    require(gs['actor_learning_rate']==1e-5 and gs['critic_learning_rate']==.001
            and gs['normalizers_frozen'] is True and
            gs['normalizer_prefix_counts' if context else 'normalizer_counts']=={'actor':24576000,'critic':24576000},'Frozen normalization/fixed LR runtime contract differs')
    require(type(gs['uncommitted_rollout']) is bool,'Uncommitted rollout state is not explicit')
    sampled_rollouts=count+int(gs['uncommitted_rollout'])
    require(int(gs['rnd_update_counter'])==sampled_rollouts*24 and
            int(saved['rnd_runtime']['update_counter'])==int(gs['rnd_update_counter']) and
            saved['rnd_runtime']['initial_weight']==saved['rnd_runtime']['weight']==0.,
            'RND rollout runtime disagrees with committed updates and optional uncommitted rollout')
    if run.get('status')=='completed':
        require(count==iteration+1,'Natural completion has an incomplete committed update')
        if run.get('completed_iterations')==1000:
            require(iteration==999 and expected==dict(actor=19200,critic=20000,rnd=20000)
                    and run.get('total_timesteps')==24576000,'Final bounded budget is incomplete')
    optimizers={}
    for name,parameters,lr in (('actor',actor_named,1e-5),('critic',critic_named,.001)):
        optimizers[name]=check_adam(torch,saved[name+'_optimizer_state_dict'],[n for n,_ in parameters],
            [p for _,p in parameters],expected[name],lr,name)
    if saved['format'] in ('h2_guarded_flat_ppo_v3','h2_guarded_flat_ppo_v4','h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'):
        require(all(g.get('foreach') is False and g.get('fused') in (None,False)
                    for g in saved['actor_optimizer_state_dict']['param_groups']),
                'Prospective Adam must declare its single-tensor execution path')
    rcfg=dict(algo['rnd_cfg']);rlr=float(rcfg.pop('learning_rate'))
    rcfg.pop('class_name',None)
    rcfg['num_states']=73
    rcfg['device']='cpu'
    rnd=RandomNetworkDistillation(**rcfg)
    rnd.load_state_dict(saved['rnd_state_dict'],strict=True)
    rnd_named=list(rnd.predictor.named_parameters())
    require(saved['rnd_optimizer_parameter_names']==[name for name,_ in rnd_named],
            'RND predictor optimizer name/order differs')
    optimizers['rnd']=check_adam(torch,saved['rnd_optimizer_state_dict'],[n for n,_ in rnd_named],
        [p for _,p in rnd_named],expected['rnd'],rlr,'RND predictor')
    initial=Path(run['initial_policy']).resolve(strict=True)
    initial_sha=profile['initial_sha256']
    require(sha(initial)==run['initial_policy_sha256']==initial_sha,'Reviewed initial policy changed')
    require(gs['initial_policy_sha256']==initial_sha,'Saved guarded state initial-policy binding differs')
    initial_saved=torch.load(initial,map_location='cpu',weights_only=True)
    transfer_receipt=None
    warm_actor_transfer_checked=False
    if profile['typed_policy_transfer']:
        transfer_receipt=verify_transfer_files(plan,saved['initialization_transfer'],args.source_root)
        parent_expected=reviewed_parent(profile['trial_protocol'])
        require(initial_saved['format']==parent_expected['format'] and initial_saved['iter']==parent_expected['iteration']
                and initial_saved['guarded_ppo_state']['completed_updates']==parent_expected['updates']
                and initial_saved['guarded_ppo_state']['uncommitted_rollout'] is (profile['trial_protocol'] not in NATURAL_PARENT_PROTOCOLS),
                'Actual parent packet differs from the exact reviewed accepted/restored budget')
        start=run['initial_learning_state']
        require(start['checkpoint_loaded'] is False and run['checkpoint'] is None
                and start['iteration']==0 and start['imitation_optimizer_loaded'] is False
                and all(start[k]==0 for k in ('actor_optimizer_state_entries','critic_optimizer_state_entries','rnd_optimizer_state_entries')),
                'New typed transfer imported parent optimization or iteration')
        initial_gs=start['guarded_ppo_state']
        require(all(initial_gs[k]==0 for k in ('completed_updates','actor_optimizer_steps',
            'critic_optimizer_steps','rnd_optimizer_steps','rnd_update_counter')),'New RND/optimizer counters were not fresh')
        require(run['std_reset_to_half_budget'] is True and
                start['policy_std']==run['stair_exploration_budget']['initial_raw_std'],
                'New trial did not explicitly reset std to the same half exploration budget')
        require(run['external_dependency_sha256']==saved['initialization_transfer']['parent_external_dependency_sha256'],
                'New typed transfer changed native USD or frozen LiDAR dependencies')
        if saved['format'] in ('h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') and count<=40:
            for name,_ in actor_named[:-1]:
                actual,parent_value=saved['model_state_dict'][name],initial_saved['model_state_dict'][name]
                if context and not active and name in ('actor.fusion.net.0.weight','actor.actor.net.0.weight'):
                    require(actual.shape==(parent_value.shape[0],parent_value.shape[1]+5)
                            and torch.equal(actual[:,:-5],parent_value) and not bool(actual[:,-5:].any()),
                            'Warm actor context columns/prefix differ: '+name)
                else:
                    require(torch.equal(actual,parent_value),'Actual warm actor differs from parent: '+name)
            require(torch.equal(saved['model_state_dict']['std'],torch.tensor(start['policy_std'],dtype=model.std.dtype)),
                    'Warm std differs from the explicitly reset half budget')
            warm_actor_transfer_checked=True
    for key,width in (('obs_norm_state_dict',1029),('privileged_obs_norm_state_dict',866)):
        norm=saved[key]
        if context:
            require(set(norm)=={'prefix._mean','prefix._var','prefix._std','prefix.count','fixed_context_scale'},
                    'Explicit context normalization state layout differs')
            prefix={name.removeprefix('prefix.'):value for name,value in norm.items() if name.startswith('prefix.')}
            require(torch.equal(norm['fixed_context_scale'],torch.tensor([1/.3,1/.3,1.,1.,1.],dtype=norm['fixed_context_scale'].dtype)),
                    'Fixed context scale changed')
        else:
            prefix=norm
            require(set(prefix)=={'_mean','_var','_std','count'},'Normalizer state layout differs')
        require(all(tuple(prefix[n].shape)==(1,width) for n in ('_mean','_var','_std')),'Normalizer prefix shape differs')
        require(prefix['count'].numel()==1 and int(prefix['count'])==24576000
                and bool((prefix['_std']>=0).all()),'Frozen inherited normalization prefix is invalid')
        parent_norm=({name.removeprefix('prefix.'):value for name,value in initial_saved[key].items() if name.startswith('prefix.')}
                     if active else initial_saved[key])
        require(prefix.keys()==parent_norm.keys() and
                all(torch.equal(value,parent_norm[name]) for name,value in prefix.items()),
                'Frozen own normalizer prefix changed during the new PPO')
        if active:
            require(norm.keys()==initial_saved[key].keys()
                    and all(torch.equal(value,initial_saved[key][name]) for name,value in norm.items()),
                    'v11 full own block normalizer, including fixed scales, changed')
    context_bank=None
    if context:
        norm_module=physical.load_module(args.source_root.parent/'task_context_normalization.py','flat_cpu_own_context_normalization')
        actual_norms=norm_module.make_task_context_normalizers('cpu')
        for label,key in (('actor','obs_norm_state_dict'),('critic','privileged_obs_norm_state_dict')):
            actual_norms[label].load_state_dict(saved[key],strict=True)
            actual_norms[label].eval()
        actual_abi=norm_module.normalization_metadata(actual_norms)
        validate_normalization_metadata(actual_abi,gs['normalizer_prefix_counts'])
        require(actual_abi==gs['normalization_abi']==saved.get('normalization_abi')
                ==run['guarded_flat_learning']['normalization_abi'],
                'Actual saved block normalization ABI differs from its declaration')
        context_bank=verify_context_bank_custody(run,plan)
    dependencies=run['external_dependency_sha256']
    require(dependencies and all(sha(Path(path).resolve(strict=True))==digest for path,digest in dependencies.items()),
            'Recorded external USD/LiDAR dependency changed')
    ex=saved['exploration_state'];std_snapshot=ex['snapshot']
    require(int(ex['optimizer_steps'])==expected['actor'] and int(std_snapshot['optimizer_steps'])==expected['actor'],
            'Exploration projection hook counter disagrees with actual actor Adam')
    require(torch.equal(torch.tensor(std_snapshot['policy_std'],dtype=model.std.dtype),model.std.detach()),
            'Saved exploration snapshot and actual policy std differ')
    maximum=torch.tensor(std_snapshot['maximum_raw_std'],dtype=model.std.dtype)
    minimum=torch.tensor(std_snapshot['minimum_raw_std'],dtype=model.std.dtype)
    angle=torch.tensor(std_snapshot['maximum_angle_rms_rad'],dtype=model.std.dtype)
    require(maximum.shape==minimum.shape==angle.shape==(12,) and bool((maximum>0).all())
            and torch.allclose(minimum,maximum*.05) and bool((model.std>=minimum).all())
            and bool((model.std<=maximum).all())
            and torch.allclose(angle,torch.tensor([.02,.015,.015,.02,.01,.0075]*2,dtype=model.std.dtype)),
            'Policy std left the original per-joint target-RMS projection budget')
    require(plan['task']==TASK and plan['max_iterations']==1000 and plan['num_envs']==1024
            and plan['initial_policy_sha256']==initial_sha,'Producer plan is not this bounded flat protocol')
    tensors=0
    def finite(value):
        nonlocal tensors
        if isinstance(value,torch.Tensor):
            require(bool(torch.isfinite(value).all()),'Non-finite checkpoint tensor')
            tensors+=1
        elif isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,(tuple,list)):
            for item in value:finite(item)
        elif isinstance(value,float):require(math.isfinite(value),'Non-finite checkpoint scalar')
    finite(saved)
    receipt=dict(status='passed',cpu_only=True,no_simulation=True,training_updates=0,
        flat_balance_ready=False,initialization_ready=False,formal_acceptance_passed=False,stairs_acceptance_passed=False,
        checkpoint=str(args.checkpoint),sha256=sha(args.checkpoint),checkpoint_format=saved['format'],
        checkpoint_iteration=iteration,committed_updates=count,optimizers=optimizers,finite_tensors=tensors,
        trial_protocol=profile['trial_protocol'],typed_initialization_transfer=transfer_receipt,
        actor_action_margin=({key:gs['action_margin'][key] for key in
            ('kind','weight','epsilon','whole_update_reference','hard_gate_unchanged','committed_minibatches')}
            if profile['typed_policy_transfer'] else None),
        actor_proposal_projection=({key:gs['proposal_projection'][key] for key in
            ('kind','interior_fraction','max_active_rows','max_solver_sweeps','whole_update_reference',
             'linear_full_bank','hard_gate_unchanged','no_retry','committed_minibatches')}
            if profile.get('proposal_projection') is not None else None),
        actor_whole_update_working_reserve=({key:gs['proposal_projection'][key] for key in
            ('working_budget_fraction','whole_update_fixed_working_bound','solver_tolerance_reserved_inward')}
            if saved['format'] in ('h2_guarded_flat_ppo_v4','h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') else None),
        actor_compute_resource_coverage=({key:gs['proposal_projection'][key] for key in
            ('resource_kind','configured_max_active_rows','max_active_rows','jacobian_arena_budget_bytes',
             'jacobian_mean_parameter_elements','jacobian_bytes_per_row','jacobian_arena_capacity_bytes')}
            if saved['format'] in ('h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') else None),
        actor_primal_precision=({key:gs['proposal_projection'][key] for key in
            ('primal_precision_policy','max_polish_per_QP','max_support_pivots','max_polish_per_projection',
             'solver_optimality_and_direct_primal_separate','primal_envelope_is_machine_derived',
             'final_same_direction_cast_JVP_required','primal_precision_summary')}
            if saved['format'] in ('h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') else None),
        actor_support_polish=(gs['proposal_projection'].get('support_polish_policy') if active else None),
        world_xy_velocity_cost_scaling=(run['runtime_flat_geometry'].get('velocity_cost_scaling')
            if saved['format'] in ('h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') else None),
        outward_tilt_growth_cost=(run['runtime_flat_geometry'].get('outward_tilt_growth_cost')
            if saved['format'] in ('h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11') else None),
        episode_force_credit_policy=(run['runtime_flat_geometry'].get('episode_force_credit_policy')
            if saved['format'] in ('h2_guarded_flat_ppo_v9',*CONTEXT_FORMATS) else None),
        actual_warm_actor44_and_reset_std_checked=warm_actor_transfer_checked,
        uncommitted_rollout=gs['uncommitted_rollout'],rnd_rollout_update_counter=int(gs['rnd_update_counter']),
        source_root=str(args.source_root),source_sha256=hashes,producer_run=str(run_path),producer_run_sha256=sha(run_path),
        producer_training_tool_sha256=training_tools,
        external_dependency_sha256=dependencies,normalizers_frozen_and_parent_bitwise_equal=(not context or active),
        normalizer_prefix_frozen_and_parent_bitwise_equal=context,
        same_width_context_transfer_verified=active,
        task_context=(run['runtime_flat_geometry'].get('task_context') if context else None),
        normalization_abi=(gs.get('normalization_abi') if context else None),
        normalizer_prefix_counts=(gs.get('normalizer_prefix_counts') if context else None),
        network_dimensions=CONTEXT_DIMS if context else [1029,866,73,12],
        fixed_context_bank_custody=context_bank,
        initial_policy=str(initial),initial_policy_sha256=initial_sha,
        plan_path=str(plan_path),plan_sha256=protocol['plan_sha256'],
        optimizer_path=str(args.checkpoint),optimizer_sha256=sha(args.checkpoint),
        optimizer_path_semantics='All actual split PPO/RND optimizer tensors are inside this same complete checkpoint.',
        validator_sha256=sha(Path(__file__)),physical_validator_sha256=sha(Path(physical.__file__)),
        rnd_semantics='Weight0 means no intrinsic reward; actual fresh predictor Adam and normalization continue and are checked.',
        integrity_is_not_walking_acceptance=True)
    if restored_state_inspector is not None:
        receipt['rollback_equivalence']=restored_state_inspector(saved)
        require(receipt['rollback_equivalence'].get('status')=='passed','Restored-state inspector did not pass')
    write(args.validation_output,receipt)
    print('[FLAT_BALANCE_CPU] '+json.dumps({key:receipt[key] for key in (
        'status','cpu_only','checkpoint','sha256','checkpoint_iteration','committed_updates',
        'finite_tensors','flat_balance_ready','source_root')}),flush=True)
    return receipt


def prepare_physical(args):
    receipt=json.loads(args.validation_output.read_text())
    require(receipt.get('status')=='passed' and receipt.get('cpu_only') is True,'Run complete split-PPO CPU validation first')
    require(receipt['checkpoint_format'] in ('h2_guarded_flat_ppo_v1','h2_guarded_flat_ppo_v2','h2_guarded_flat_ppo_v3','h2_guarded_flat_ppo_v4','h2_guarded_flat_ppo_v5','h2_guarded_flat_ppo_v6','h2_guarded_flat_ppo_v7','h2_guarded_flat_ppo_v8','h2_guarded_flat_ppo_v9','h2_guarded_flat_ppo_v10','h2_guarded_flat_ppo_v11'),'Wrong CPU receipt format')
    require(receipt['sha256']==sha(args.checkpoint) and receipt['source_root']==str(args.source_root)
            and receipt['source_sha256']==source_hashes(args.source_root),'Checkpoint/source changed after CPU validation')
    require(receipt['validator_sha256']==sha(Path(__file__)) and
            receipt['physical_validator_sha256']==sha(Path(physical.__file__)),'Validator changed after CPU receipt')
    require(sha(Path(receipt['producer_run']))==receipt['producer_run_sha256'],'Producer run evidence changed')
    require(sha(Path(receipt['plan_path']))==receipt['plan_sha256'],'Producer plan changed after CPU validation')
    require(transfer_provenance_unchanged(receipt),'Typed parent provenance changed after CPU validation')
    require(context_bank_provenance_unchanged(receipt),'Sealed real context bank evidence changed after CPU validation')
    for name,digest in receipt['producer_training_tool_sha256'].items():
        require(sha(args.source_root.parent/name)==digest,'Producer tool changed after CPU validation: '+name)
    for path,digest in receipt['external_dependency_sha256'].items():
        require(sha(Path(path))==digest,'External dependency changed after CPU receipt')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(args.source_root.parent,args.output/'training_source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=args.output/'training_source/source'
    require(source_hashes(source)==receipt['source_sha256'],'Frozen source copy is incomplete')
    shutil.copy2(args.checkpoint.parent/'params/agent.yaml',args.output/'agent.yaml')
    shutil.copy2(args.validation_output,args.output/'checkpoint.validation.json')
    shutil.copy2(Path(receipt['producer_run']),args.output/'checkpoint_run.json')
    (args.output/'tools_snapshot').mkdir()
    for name in ('evaluate_flat_balance_policy.py','evaluate_flat_balance_policy.sh',
                 'evaluate_direct_initialization.py','evaluate_direct_initialization.sh',
                 *(['task_context_metadata.py','task_context_normalization.py'] if receipt['checkpoint_format'] in CONTEXT_FORMATS else []),
                 *(['v11_parent_custody.py'] if receipt['checkpoint_format']==ACTIVE_CONTEXT_FORMAT else [])):
        shutil.copy2(Path(__file__).parent/name,args.output/'tools_snapshot'/name)
    deps=receipt['external_dependency_sha256']
    usd=[p for p in deps if Path(p).suffix=='.usd']
    lidar=[p for p in deps if 'reconstructor' in Path(p).name and Path(p).suffix=='.pt']
    require(len(usd)==len(lidar)==1,'Ambiguous original USD/LiDAR dependencies')
    os.environ['CTRLLAB_H2_USD_PATH']=usd[0]
    os.environ['CTRLLAB_H2_LIDAR_RECON_CHECKPOINT']=lidar[0]
    return source,receipt


def main():
    args,AppLauncher=physical.parse_args()
    if args.cpu_check_only:
        try:validate_cpu(args)
        except BaseException as error:
            write(args.validation_output,dict(status='failed',cpu_only=True,no_simulation=True,
                flat_balance_ready=False,checkpoint=str(args.checkpoint),sha256=sha(args.checkpoint),error=repr(error)))
            raise
        return
    source,receipt=prepare_physical(args)
    metadata=dict(status='initializing',task=TASK,started_at_epoch=time.time(),checkpoint=str(args.checkpoint),
        checkpoint_sha256=receipt['sha256'],source_root=str(args.source_root),imported_source_root=str(source),
        num_envs=1024,duration_s=20.,speed_m_s=.3,seed=42,network_dimensions=receipt.get('network_dimensions',[1029,866,73,12]),
        training_updates=0,teacher_constructed=False,teacher_interventions=0,initialization_ready=False,
        flat_balance_ready=False,formal_acceptance_passed=False,stairs_acceptance_passed=False,
        validator_sha256=sha(Path(__file__)),checkpoint_format=receipt['checkpoint_format'],
        trial_protocol=receipt.get('trial_protocol','guarded_flat_balance_v1'),
        optimizer_loaded_for_control=False,physical_procedure='Original first-episode physical loop/gates; versioned input/model/normalizer construction for explicit context only')
    app=None
    try:
        write(args.output/'run.json',metadata)
        app=AppLauncher(args).app
        physical.evaluate(args,app,source,receipt,metadata)
        result=json.loads((args.output/'result.json').read_text())
        provenance_unchanged=(sha(Path(__file__))==receipt['validator_sha256']
            and sha(Path(physical.__file__))==receipt['physical_validator_sha256']
            and sha(Path(receipt['producer_run']))==receipt['producer_run_sha256']
            and sha(Path(receipt['plan_path']))==receipt['plan_sha256']
            and transfer_provenance_unchanged(receipt)
            and context_bank_provenance_unchanged(receipt)
            and all(sha(args.source_root.parent/name)==digest
                    for name,digest in receipt['producer_training_tool_sha256'].items()))
        result.update(flat_balance_ready=bool(result['initialization_ready']),
            scope='Independent frozen autonomous direct flat PPO20s; not six-level90s stair acceptance.',
            cpu_optimizer_integrity_only=True,checkpoint_format=receipt['checkpoint_format'],
            actual_split_optimizer_steps={k:v['steps'] for k,v in receipt['optimizers'].items()},
            full_producer_and_evaluator_provenance_unchanged=provenance_unchanged)
        metadata.update(result)
        require(provenance_unchanged,'Producer/evaluator provenance changed during frozen physical validation')
        write(args.output/'result.json',result)
        write(args.output/'run.json',result)
        print('[FLAT_BALANCE_VALIDATION] RESULT '+json.dumps(result),flush=True)
    except BaseException as error:
        metadata.update(status='failed',flat_balance_ready=False,initialization_ready=False,error=repr(error))
        write(args.output/'result.json',metadata)
        write(args.output/'run.json',metadata)
        (args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':main()
