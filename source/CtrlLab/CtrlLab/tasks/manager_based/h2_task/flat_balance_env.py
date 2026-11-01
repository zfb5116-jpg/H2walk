"""Physical progress and rewards for autonomous direct flat walking.

No reference trajectory, teacher action, pose preload, or extra actuator port
is installed. Episode evidence is finalized before the real scene reset.
"""
from collections import deque
from math import cos
from pathlib import Path

import numpy as np
import torch

from .direct_initialization import DirectInitializationEnv
from .flat_task_context import task_context_runtime_contract
from .locomotion_progress import FootfallTracker
from .stair_ascent_evidence import StairEpisodeEvidence
from .stair_ascent_progress import signed_forward_ledger
from .stair_ascent_safety import physical_validity

TRIAL_PROTOCOL='explicit_task_context_full_outward_growth_v1'
POSITIVE_CREDIT_PROFILE='overspeed_full_outward_tilt_growth_clean_force_v1'
EPISODE_FORCE_CREDIT_POLICY={
    'kind':'whole_episode_clean_force_positive_credit_and_first_union_failure_v1',
    'agreement_absolute_tolerance_N':1.0,
    'agreement_relative_raw_norm_tolerance':.02,
    'relevant_norm_threshold_N':5.0,'positive_control_Fz_threshold_N':5.0,
    'original_force_predicate_unchanged':True,
    'positive_progress_and_landing_require_no_prior_or_current_relevant_mismatch':True,
    'qualification_from_existing_cumulative_mismatch':True,
    'physical_ever_invalid_and_survival_unchanged':True,
    'native_termination_terms_unchanged':7,'actual_peak_always_consumed':True,
    'negative_debt_physical_valid_only':True,'negative_debt_unweighted':True,
    'first_of_union_force_failure_or_native_failure_once':True,'failure_charge':20.0,
    'no_double_charge_later_native_failure':True,'pure_timeout_exempt':True,
    'timeout_with_first_force_failure_charged_once':True,
    'initial_no_contact_or_missing_positive_control_not_failure':True,
    'per_counter_update_idempotent':True,'reward_reads_non_destructive':True,
    'record_before_reset':True,'reset_only_real_env_ids':True,
    'positive_credit_never_reactivated_after_force_failure':True,
    'all_other_reward_weights_and_kernels_unchanged_relative_v8':True,
    'no_new_term_sensor_controller_or_policy_input':False,
    'no_new_reward_term_sensor_or_controller':True,
    'global_policy_input_changed_by_explicit_context':True,'no_fixed_foot_width_target':True,
    'scope':'Bounded fixedworld+.3directflat; independent firstepisode gate unchanged, not automatic stairs reuse.',
}


def positive_forward_speed_weight(world_vx, command_speed=.3):
    """Smooth reward weighting only; no native/action speed limit is changed."""
    if command_speed != .3:
        raise ValueError('This declared flat reward profile requires its0.3m/s command')
    u=((world_vx-command_speed)/command_speed).clamp(0.,1.)
    # Algebraically1-3u^2+2u^3; factored to avoid cancellation nearu=1.
    return ((1.-u).square()*(1.+2.*u)).clamp(0.,1.)


class FlatBalanceProgress:
    """One physical endpoint per epoch; invalid/partial episodes remain explicit."""
    def __init__(self, n, device, masses, bounds, floor_tolerance, directory=None,
                 *,trial_protocol=TRIAL_PROTOCOL,reward_profile=POSITIVE_CREDIT_PROFILE):
        if trial_protocol!=TRIAL_PROTOCOL or reward_profile!=POSITIVE_CREDIT_PROFILE:
            raise ValueError('Unrecognized declared flat reward/optimization profile')
        self.n, self.device = n, device
        self.trial_protocol,self.reward_profile=trial_protocol,reward_profile
        self.bounds, self.floor_tolerance = tuple(bounds), float(floor_tolerance)
        self.weight = masses * 9.81
        self.footfalls = FootfallTracker(n, device)
        self.recent = deque(maxlen=5000)
        self.completed_episode_count = 0
        self.pending, self.last_counter = False, -1
        self.needs_start = torch.ones(n, dtype=torch.bool, device=device)
        self.partial = torch.zeros_like(self.needs_start)
        self.ever_invalid = torch.zeros_like(self.needs_start)
        self.ever_outside = torch.zeros_like(self.needs_start)
        self.ever_below = torch.zeros_like(self.needs_start)
        for name in ('duration', 'paid_m', 'unweighted_paid_m', 'backward_m', 'actual_backward_m', 'event_distance',
                     'slip_integral', 'velocity_error_integral', 'signed_step', 'landing_step',
                     'slip_square', 'peak_x', 'failure_charge_step',
                     'positive_forward_paid_after_force_failure_m',
                     'backward_debt_after_force_failure_m'):
            setattr(self, name, torch.zeros(n, device=device))
        self.positive_speed_weight=torch.ones(n,device=device)
        for name in ('origins', 'start', 'end', 'minimum', 'pre_root'):
            setattr(self, name, torch.zeros(n, 3, device=device))
        self.pre_soles = torch.zeros(n, 2, 3, device=device)
        self.end_quaternion = torch.zeros(n, 4, device=device)
        self.ground_end = torch.zeros(n, 2, 3, device=device)
        self.raw_end = torch.zeros_like(self.ground_end)
        self.positive = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.mismatch = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.failure_charge_count = torch.zeros(n, dtype=torch.long, device=device)
        self.landing_bonus_events_after_force_failure = torch.zeros_like(self.failure_charge_count)
        self.first_force_failure_control_counter = torch.full_like(self.failure_charge_count,-1)
        self.first_failure_charge_control_counter = torch.full_like(self.failure_charge_count,-1)
        self.first_failure_force_cause = torch.zeros(n, dtype=torch.bool, device=device)
        self.first_failure_native_cause = torch.zeros_like(self.first_failure_force_cause)
        self.start_counter = torch.zeros(n, dtype=torch.long, device=device)
        self.end_counter = torch.zeros_like(self.start_counter)
        self.evidence = StairEpisodeEvidence(directory, {
            'task': 'autonomous direct flat balance', 'domain_bounds_local_m': list(bounds),
            'below_floor_tolerance_m': floor_tolerance, 'original_FootfallTracker_unchanged': True,
            'force_semantics': 'world contact-normal vectors only; not frictional GRF or measured motor torque',
            'invalid_episodes_kept_as_failures': True, 'initial_partial_excluded_from_rolling_metrics': True,
            'positive_forward_speed_cap_m_s': .3, 'forward_upright_gate_rad': .5,
            'forward_supported_weight_fraction': .7, 'landing_reward_min_new_progress_m': .04,
            'trial_protocol':trial_protocol,'reward_profile':reward_profile,
            'positive_speed_weight':'u=clip((world_vx-.3)/.3,0,1); g=1-3u^2+2u^3',
            'unweighted_actual_peak_always_consumed':True,'negative_distance_debt_unweighted':True,
            'episode_force_credit_policy':dict(EPISODE_FORCE_CREDIT_POLICY),
        })

    @property
    def force_credit_eligible(self):
        return self.mismatch.sum(-1)==0

    def begin(self, root, soles, origins, episode_lengths, counter):
        ids = self.needs_start.nonzero().flatten()
        if ids.numel():
            self.start[ids] = root[ids]
            self.end[ids] = root[ids]
            self.minimum[ids] = root[ids]
            self.origins[ids] = origins[ids]
            self.peak_x[ids] = root[ids, 0]
            self.partial[ids] = episode_lengths[ids] != 0
            self.start_counter[ids] = int(counter)
            self.needs_start[ids] = False
        self.pre_root.copy_(root)
        self.pre_soles.copy_(soles)
        self.signed_step.zero_()
        self.landing_step.zero_()
        self.failure_charge_step.zero_()
        self.pending = True

    def update(self, counter, root, quaternion, velocity, tilt, soles, ground, raw_force, terminated, dt):
        if not self.pending or self.last_counter == int(counter):
            return
        for value in (root, quaternion, velocity, tilt, soles, ground, raw_force):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Non-finite physical flat-balance episode state')
        self.pending, self.last_counter = False, int(counter)
        outside, below, physical = physical_validity(root, self.origins, self.bounds, self.floor_tolerance)
        self.ever_outside |= outside
        self.ever_below |= below
        self.ever_invalid |= ~physical
        usable = ~self.ever_invalid
        self.duration += dt
        self.end.copy_(root)
        self.end_quaternion.copy_(quaternion)
        self.minimum[:] = torch.minimum(self.minimum, root)
        self.end_counter.fill_(int(counter))
        self.ground_end.copy_(ground)
        self.raw_end.copy_(raw_force)
        agreement = (ground-raw_force).norm(dim=-1) <= 1. + .02*raw_force.norm(dim=-1)
        relevant = (ground.norm(dim=-1)>5.) | (raw_force.norm(dim=-1)>5.)
        self.mismatch += ((~agreement)&relevant).long()
        force_clean = self.force_credit_eligible
        first_force = ~force_clean & (self.first_force_failure_control_counter<0)
        self.first_force_failure_control_counter[first_force] = int(counter)
        charge = (~force_clean | terminated) & (self.failure_charge_count==0)
        self.failure_charge_step.copy_(charge.float())
        self.failure_charge_count += charge.long()
        self.first_failure_charge_control_counter[charge] = int(counter)
        self.first_failure_force_cause |= charge & ~force_clean
        self.first_failure_native_cause |= charge & terminated
        self.positive |= agreement & (ground[..., 2]>5.) & (raw_force[..., 2]>5.)
        contact = ground.norm(dim=-1)>5.
        old_landings = self.footfalls.landings.clone()
        self.footfalls.update(contact, dt)
        new_landing = (self.footfalls.landings>old_landings).any(-1)
        valid_fz = torch.where(agreement & (ground[..., 2]>0.), ground[..., 2], 0.)
        supported = valid_fz.sum(-1) >= .7*self.weight
        positive_allowed = supported & (tilt<=.5) & ~terminated & force_clean
        peak, paid, negative = signed_forward_ledger(self.pre_root[:, 0], self.peak_x, root[:, 0],
            positive_allowed, usable, torch.zeros_like(usable), dt, maximum_credit_speed=.3)
        # The actual peak is consumed before weighting. An uncredited fast
        # excursion can never be saved for a later slow transition or retrace.
        self.peak_x.copy_(peak)
        self.unweighted_paid_m += paid
        self.positive_speed_weight.copy_(positive_forward_speed_weight(velocity[:,0]))
        paid=paid*self.positive_speed_weight
        self.paid_m += paid
        self.backward_m -= negative
        self.actual_backward_m += (root[:, 0]-self.pre_root[:, 0]).clamp_max(0).neg()
        self.signed_step.copy_(paid+negative)
        self.event_distance[:] = (self.event_distance+paid).clamp_max(.04)
        event = (new_landing & usable & positive_allowed & agreement.all(-1)
                 & (self.event_distance>=.04-1e-6) & (paid>0.))
        self.landing_step[:] = event.float()
        self.event_distance[event] = 0.
        self.positive_forward_paid_after_force_failure_m += torch.where(~force_clean,paid,0.)
        self.landing_bonus_events_after_force_failure += (event & ~force_clean).long()
        self.backward_debt_after_force_failure_m -= torch.where(~force_clean,negative,0.)
        speeds = (soles[..., :2]-self.pre_soles[..., :2]).norm(dim=-1)/dt
        loaded = contact & agreement
        self.slip_square[:] = (speeds.square()*loaded).sum(-1)/loaded.sum(-1).clamp_min(1)
        self.slip_integral += torch.sqrt(self.slip_square)*dt
        target = torch.tensor([.3, 0.], device=root.device, dtype=root.dtype)
        self.velocity_error_integral += (velocity[:, :2]-target).norm(dim=-1)*dt

    def record(self, ids, terminated, timed_out, term_bits):
        ids = ids[self.duration[ids]>0]
        if not ids.numel():
            return
        physical_valid = ~self.ever_invalid[ids]
        force_valid = self.positive[ids].all(-1) & (self.mismatch[ids].sum(-1)==0)
        survived = (self.duration[ids]>=20.-1e-5) & ~terminated[ids] & physical_valid
        dx = self.end[ids, 0]-self.start[ids, 0]
        ratio = dx/(.3*self.duration[ids]).clamp_min(1e-6)
        progress = (dx>=4.8)&(dx<=7.2)&physical_valid
        bilateral = (self.footfalls.landings[ids]>=2).all(-1)&physical_valid
        success = survived&progress&bilateral&force_valid
        complete = ~self.partial[ids]
        self.evidence.append(dict(env_id=ids, start_control_counter=self.start_counter[ids],
            end_control_counter=self.end_counter[ids], duration_s=self.duration[ids],
            initial_partial=self.partial[ids], ever_invalid=self.ever_invalid[ids],
            ever_out_of_tile=self.ever_outside[ids], ever_below_floor=self.ever_below[ids],
            true_timeout=timed_out[ids]&~terminated[ids], terminated=terminated[ids],
            start_root_world=self.start[ids], end_root_world=self.end[ids],
            end_root_quaternion_wxyz=self.end_quaternion[ids], origins=self.origins[ids],
            minimum_world_xyz=self.minimum[ids], minimum_local_xyz=self.minimum[ids]-self.origins[ids],
            net_world_x_m=dx, forward_paid_m=self.paid_m[ids], backward_debt_m=self.backward_m[ids],
            unweighted_eligible_forward_paid_m=self.unweighted_paid_m[ids],
            positive_speed_weight_at_terminal=self.positive_speed_weight[ids],
            signed_forward_credit_m=torch.where(physical_valid,self.paid_m[ids]-self.backward_m[ids],0.),
            actual_backward_distance_m=self.actual_backward_m[ids], positive_control=self.positive[ids],
            force_mismatch_steps=self.mismatch[ids], ground_normal_force=self.ground_end[ids],
            force_credit_eligible=self.force_credit_eligible[ids],
            failure_charge_count=self.failure_charge_count[ids],
            first_force_failure_control_counter=self.first_force_failure_control_counter[ids],
            first_failure_charge_control_counter=self.first_failure_charge_control_counter[ids],
            first_failure_force_cause=self.first_failure_force_cause[ids],
            first_failure_native_cause=self.first_failure_native_cause[ids],
            failure_charge_this_control=self.failure_charge_step[ids],
            positive_forward_paid_after_force_failure_m=self.positive_forward_paid_after_force_failure_m[ids],
            landing_bonus_events_after_force_failure=self.landing_bonus_events_after_force_failure[ids],
            backward_debt_after_force_failure_m=self.backward_debt_after_force_failure_m[ids],
            raw_foot_normal_force=self.raw_end[ids], valid_alternating_landings=self.footfalls.landings[ids],
            survived20s=survived, progress=progress, bilateral=bilateral, success=success,
            **{'termination_'+name:bits[ids] for name,bits in term_bits.items()}))
        if complete.any():
            age = self.duration[ids]
            values = torch.stack((torch.where(physical_valid,age,0.), age, survived.float(),
                terminated[ids].float(), torch.where(physical_valid,ratio,0.).clamp(-2,2),
                bilateral.float(), (survived&progress).float(), success.float(),
                self.velocity_error_integral[ids]/age.clamp_min(1e-6),
                self.slip_integral[ids]/age.clamp_min(1e-6), self.ever_invalid[ids].float(),
                force_valid.float(), dx, self.footfalls.landings[ids,0], self.footfalls.landings[ids,1]),-1)
            rows=values[complete].cpu().numpy()
            self.recent.extend(rows)
            self.completed_episode_count += len(rows)

    def reset(self, ids):
        self.needs_start[ids] = True
        for name in ('partial','ever_invalid','ever_outside','ever_below','duration','paid_m','unweighted_paid_m','backward_m',
                     'actual_backward_m','event_distance','slip_integral','velocity_error_integral',
                     'signed_step','landing_step','slip_square','positive','mismatch',
                     'failure_charge_step','failure_charge_count',
                     'positive_forward_paid_after_force_failure_m',
                     'landing_bonus_events_after_force_failure','backward_debt_after_force_failure_m',
                     'first_failure_force_cause','first_failure_native_cause'):
            getattr(self,name)[ids] = 0
        self.first_force_failure_control_counter[ids]=-1
        self.first_failure_charge_control_counter[ids]=-1
        self.positive_speed_weight[ids]=1.
        self.footfalls.reset(ids)

    def summary(self):
        if not self.recent:
            # A full zero-valued canonical mapping must override MotionProbe
            # until a complete nonpartial episode exists. window0 denotes no
            # evidence, never a successful zero-error or surviving episode.
            empty={name:0. for name in (
                'mean_episode_s','median_episode_s','raw_mean_episode_s','survival20s_fraction',
                'survival_horizon_fraction','early_termination_fraction','command_progress_ratio',
                'bilateral_landings_fraction','survival_and_progress_fraction','locomotion_success_fraction',
                'mean_xy_velocity_error_m_s','mean_stance_slip_m_s','physical_invalid_fraction',
                'contact_evidence_valid_fraction','mean_net_world_x_m','mean_raw_net_world_x_m',
                'mean_left_valid_landings','mean_right_valid_landings')}
            empty.update(window_episode_count=0,completed_episode_count=self.completed_episode_count,moving_episode_count=0)
            return empty
        a=np.asarray(self.recent)
        physical=(a[:,10]==0)
        return dict(window_episode_count=len(a),completed_episode_count=self.completed_episode_count,
            mean_episode_s=float(a[:,0].mean()),median_episode_s=float(np.median(a[:,0])),
            # Column0 is full-episode eligible credit (invalid=>0). Column1
            # remains the raw simulated lifetime and must never replace it.
            raw_mean_episode_s=float(a[:,1].mean()),survival20s_fraction=float(a[:,2].mean()),
            survival_horizon_fraction=float(a[:,2].mean()),early_termination_fraction=float(a[:,3].mean()),
            command_progress_ratio=float(a[:,4].mean()),bilateral_landings_fraction=float(a[:,5].mean()),
            survival_and_progress_fraction=float(a[:,6].mean()),locomotion_success_fraction=float(a[:,7].mean()),
            mean_xy_velocity_error_m_s=float(a[:,8].mean()),mean_stance_slip_m_s=float(a[:,9].mean()),
            physical_invalid_fraction=float(a[:,10].mean()),contact_evidence_valid_fraction=float(a[:,11].mean()),
            mean_net_world_x_m=float(np.where(physical,a[:,12],0.).mean()),
            mean_raw_net_world_x_m=float(a[:,12].mean()),
            mean_left_valid_landings=float(np.where(physical,a[:,13],0.).mean()),
            mean_right_valid_landings=float(np.where(physical,a[:,14],0.).mean()),moving_episode_count=len(a))


class FlatBalanceEnv(DirectInitializationEnv):
    """Direct policy physics, physical reward bookkeeping and pre-reset evidence."""
    def __init__(self, cfg, render_mode=None, **kwargs):
        self._flat_task_context_schema_bootstrap = True
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._flat_task_context_schema_bootstrap = False
        directory = getattr(cfg,'io_descriptors_output_dir',None)
        directory = str(Path(directory)/'flat_episode_evidence') if directory else None
        self.flat_balance_progress = FlatBalanceProgress(self.num_envs,self.device,self.masses,
            self.domain_bounds_local,self._stair_below_floor_tolerance,directory,
            trial_protocol=cfg.flat_trial_protocol,reward_profile=cfg.flat_reward_profile)
        generator=self.scene.terrain.cfg.terrain_generator
        command=self.command_manager.get_term('base_velocity').cfg
        held=[int(i) for i in self.action_manager.held_joint_ids]
        expected_terms={'time_out','bad_orientation','base_contact','fall_body_contact',
                        'contact_out_of_time','stair_below_floor','stair_out_of_tile'}
        if (tuple(map(float,generator.size))!=(16.,6.) or generator.num_rows!=1 or generator.num_cols!=1
                or len(held)!=17 or set(self.termination_manager.active_terms)!=expected_terms):
            raise RuntimeError('Actual flat geometry, original upper hold, or seven native/domain terminations changed')
        upper_error=float((self.asset.data.joint_pos_target[:,held]
                            -self.asset.data.default_joint_pos[:,held]).abs().max())
        velocity_error_weight=float(self.reward_manager.get_term_cfg('world_xy_velocity_error').weight)
        signed_forward_weight=float(self.reward_manager.get_term_cfg('signed_forward').weight)
        command_speed=float(command.ranges.lin_vel_x[0])
        growth_cfg=self.reward_manager.get_term_cfg('outward_tilt_growth')
        failure_cfg=self.reward_manager.get_term_cfg('failure')
        failure_weight=float(failure_cfg.weight)
        growth_margin=float(growth_cfg.params['margin_angle'])
        growth_limit=float(self.termination_manager.get_term_cfg('bad_orientation').params['limit_angle'])
        growth_interval=cos(growth_margin)-cos(growth_limit)
        self.runtime_contract={
            'trial_protocol':cfg.flat_trial_protocol,'reward_profile':cfg.flat_reward_profile,
            'task_context':task_context_runtime_contract(self),
            'episode_force_credit_policy':dict(EPISODE_FORCE_CREDIT_POLICY,
                actual_failure_weight=failure_weight,actual_failure_function=failure_cfg.func.__name__),
            'positive_credit_speed_weight':{'command_m_s':.3,'full_at_or_below_m_s':.3,
                'zero_at_or_above_m_s':.6,'formula':'1-3u^2+2u^3; u=clip((world_vx-.3)/.3,0,1)',
                'actual_peak_consumed_unweighted':True,'negative_debt_unweighted':True,
                'physical_or_action_speed_limit_changed':False},
            'velocity_cost_scaling':{
                'kind':'restored_existing_world_xy_error_v1',
                'actual_world_xy_velocity_error_weight':velocity_error_weight,
                'actual_signed_forward_weight':signed_forward_weight,
                'command_m_s':command_speed,
                'maximum_positive_forward_rate':signed_forward_weight*command_speed,
                'one_command_speed_error_cost_per_s':velocity_error_weight*command_speed**2,
                'function_unchanged':True,'no_new_reward_information':True},
            'outward_tilt_growth_cost':{
                'kind':'existing_body_gravity_gyro_outward_potential_rate_v1',
                'actual_function_name':growth_cfg.func.__name__,
                'actual_weight':float(growth_cfg.weight),
                'actual_failure_weight':failure_weight,
                'failure_charge':abs(failure_weight),
                'margin_rad':growth_margin,'original_native_termination_rad':growth_limit,
                'margin_potential':1.-cos(growth_margin),
                'termination_potential':1.-cos(growth_limit),
                'potential_interval':growth_interval,
                'continuous_monotone_margin_to_termination_cost':float(growth_cfg.weight)*growth_interval,
                'actual_tilt_margin_weight':float(self.reward_manager.get_term_cfg('tilt_margin').weight),
                'actual_body_angular_rate_weight':float(self.reward_manager.get_term_cfg('body_angular_rate').weight),
                'actual_world_xy_velocity_error_weight':velocity_error_weight,
                'command_m_s':command_speed,
                'uses_existing_body_gravity_and_gyro':True,'same_reward_endpoint':True,
                'no_new_policy_input':True,
                'no_new_policy_input_scope':'Only this auxiliary reward still uses existing body gravity/gyro; the explicit task context changes the global policy input.',
                'reward_dt_multiplications':1,
                'discrete_total_not_exact_failure_charge':True,
                'activation_profile':'full_upright_to_native_limit_outward_growth_v1',
                'actual_reward_term_names':list(self.reward_manager.active_terms),
                'actual_reward_weights':{name:float(self.reward_manager.get_term_cfg(name).weight)
                                        for name in self.reward_manager.active_terms},
                'actual_state_tilt_function_name':self.reward_manager.get_term_cfg('tilt_margin').func.__name__,
                'state_tilt_margin_source_constant_rad':.35,
                'state_tilt_margin_source_scope':'Unchanged kernel source constant; AST-bound, not a runtimecfg parameter getter.',
                'upright_potential_zero_is_geometry':True,
                'only_existing_outward_activation_and_calibration_changed':True},
            'generator_size_m':list(map(float,generator.size)),
            'terrain_rows':int(generator.num_rows),'terrain_columns':int(generator.num_cols),
            'terrain_subtypes':{name:type(term).__name__ for name,term in generator.sub_terrains.items()},
            'domain_bounds_local_m':list(self.domain_bounds_local),
            'below_floor_tolerance_m':self._stair_below_floor_tolerance,
            'termination_terms':list(self.termination_manager.active_terms),
            'command_ranges':{name:list(getattr(command.ranges,name))
                              for name in ('lin_vel_x','lin_vel_y','ang_vel_z','heading')},
            'command_limit_ranges':{name:list(getattr(command.limit_ranges,name))
                                    for name in ('lin_vel_x','lin_vel_y','ang_vel_z','heading')},
            'command_heading_enabled':bool(command.heading_command),
            'command_standing_fraction':float(command.rel_standing_envs),
            'scripted_reference_installed':self.action_manager.get_term('JointPositionAction').gait_reference is not None,
            'action_dimension':int(self.action_manager.total_action_dim),
            'held_upper_joint_ids':held,'held_upper_joint_names':[self.asset.joint_names[i] for i in held],
            'held_upper_count':len(held),'default_upper_target_observable':True,
            'initial_upper_target_max_error_rad':upper_error,
            'upper_target_semantics':'Observed native target minus original default; before first control publication only. Frozen evaluator also checks every first-episode control endpoint.',
            'ground_filter_paths':[list(sensor.cfg.filter_prim_paths_expr) for sensor in self._stair_sensors],
            'episode_evidence_directory':directory,
            'observation_shapes':{name:list(self.observation_manager.group_obs_dim[name])
                                  for name in ('policy','critic','rnd_state')},
            'control_step_s':self.step_dt,'physics_step_s':self.physics_dt,
        }

    def _update_flat_balance(self):
        progress=getattr(self,'flat_balance_progress',None)
        if progress is None or not progress.pending:
            return
        sole,_=self._stair_foot_pose()
        forces=[]
        for sensor in self._stair_sensors:
            matrix=sensor.data.force_matrix_w
            if matrix is None or tuple(matrix.shape)!=(self.num_envs,1,1,3):
                raise RuntimeError('Flat balance requires original ground-filtered foot force sensors')
            forces.append(matrix[:,0,0])
        robot=self.asset.data
        progress.update(self.common_step_counter,robot.root_pos_w,robot.root_quat_w,
            robot.root_link_lin_vel_w,torch.acos((-robot.projected_gravity_b[:,2]).clamp(-1,1)),sole,
            torch.stack(forces,1),self.scene['contact_forces'].data.net_forces_w[:,self._stair_unfiltered_ids],
            self.termination_manager.terminated,self.step_dt)

    def step(self,action):
        soles,_=self._stair_foot_pose()
        self.flat_balance_progress.begin(self.asset.data.root_pos_w,soles,
            self.scene.terrain.env_origins,self.episode_length_buf,self.common_step_counter)
        result=super().step(action)
        self.flat_balance_progress.evidence.maybe_flush(self.common_step_counter)
        return result

    def _reset_idx(self,env_ids):
        if hasattr(self,'flat_balance_progress'):
            self._update_flat_balance()
            self.flat_balance_progress.record(env_ids,self.termination_manager.terminated,
                self.termination_manager.time_outs,
                {name:self.termination_manager.get_term(name) for name in self.termination_manager.active_terms})
        super()._reset_idx(env_ids)
        if hasattr(self,'flat_balance_progress'):
            self.flat_balance_progress.reset(env_ids)

    def close(self):
        try:
            if hasattr(self,'flat_balance_progress'):
                self.flat_balance_progress.evidence.close()
        finally:
            super().close()


def flat_signed_progress(env):
    env._update_flat_balance()
    return env.flat_balance_progress.signed_step/env.step_dt


def flat_valid_landing(env):
    env._update_flat_balance()
    return env.flat_balance_progress.landing_step/env.step_dt


def flat_support_slip(env):
    env._update_flat_balance()
    return env.flat_balance_progress.slip_square


def flat_failure(env):
    # The original weight applies once to the first force/native failure union.
    env._update_flat_balance()
    return env.flat_balance_progress.failure_charge_step/env.step_dt


def flat_velocity_error(env):
    v=env.asset.data.root_link_lin_vel_w
    return (v[:,0]-.3).square()+v[:,1].square()


def flat_heading_error(env):
    q=env.asset.data.root_quat_w
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
    return 1-torch.cos(yaw)


def flat_tilt_margin(env):
    angle=torch.acos((-env.asset.data.projected_gravity_b[:,2]).clamp(-1,1))
    return (angle-.35).clamp_min(0).square()


def flat_body_angular_rate(env):
    return env.asset.data.root_ang_vel_w[:,:2].square().sum(-1)


def flat_outward_tilt_growth(env, margin_angle=.35):
    """Positive root-tilt-potential rate beyond the existing angular margin."""
    gravity=env.asset.data.projected_gravity_b
    angular=env.asset.data.root_ang_vel_b
    outward=torch.relu(gravity[:,0]*angular[:,1]-gravity[:,1]*angular[:,0])
    return outward * (-gravity[:,2] < cos(margin_angle))


def flat_drive_request(env):
    ids=env.action_manager.get_term('JointPositionAction')._joint_ids
    data=env.asset.data
    return (data.computed_torque[:,ids]/data.joint_effort_limits[:,ids].clamp_min(1e-6)).clamp(-5,5).square().mean(-1)


def flat_target_change(env):
    return (env.action_manager.action.tanh()-env.action_manager.prev_action.tanh()).square().mean(-1)


def flat_native_overspeed(env):
    ids=env.action_manager.get_term('JointPositionAction')._joint_ids
    data=env.asset.data
    return (data.joint_vel[:,ids].abs()/data.joint_vel_limits[:,ids].clamp_min(1e-6)-1).clamp_min(0).square().sum(-1)


def flat_soft_excess(env):
    ids=env.action_manager.get_term('JointPositionAction')._joint_ids
    data=env.asset.data
    q,limits=data.joint_pos[:,ids],data.soft_joint_pos_limits[:,ids]
    return ((limits[...,0]-q).clamp_min(0)+(q-limits[...,1]).clamp_min(0)).sum(-1)
