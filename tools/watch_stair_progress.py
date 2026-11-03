"""User-authorized, case-local500 checkpoint decision. No policy/array loading."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import argparse
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--authorization-sha256", required=True)
args = parser.parse_args()
CASE = args.case.resolve(strict=True)
HERE = CASE/"user_progress_gate_500"
AUTH_SHA = args.authorization_sha256
_authority = json.loads((HERE/"human_authorization.json").read_text())
SOURCE = _authority["source_commit"]
PLAN_SHA = _authority["plan_sha256"]
WINDOWS = (299, 399, 499)
MAX_SIGNAL_ITERATION = 525
sys.path.insert(0, str(ROOT/'tools'))
import launch_stair_transfer as L
from stair_curriculum_binding import target_window_errors, early_policy_errors
from stair_initialization_binding import HEIGHT_COURSE_PROTOCOLS
from stair_execution_binding import PROTOCOL, execution_snapshot_errors
# Reject a wrong/closed producer before creating any operator artifact.
if (not CASE.is_relative_to(ROOT/'logs/stair_learning')
        or L.digest(HERE/'human_authorization.json') != AUTH_SHA
        or L.read(CASE/'request.json').get('trial_protocol') not in HEIGHT_COURSE_PROTOCOLS):
    raise RuntimeError('Wrong current-course producer/authorization; no operator artifacts created')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite progress metadata')
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def validate_window(native, index):
    finite(native)
    require(native['iteration'] == index and native['completed_iterations'] == index+1,
            'Wrong exact progress window iteration')
    schema = native['effect_metric_schema']
    require(schema['profile'] == 'stairs' and schema['source'] == 'stair_progress'
            and schema['complete_episode_evidence'] is True
            and schema['evidence_status'] == 'complete_episode_window', 'Wrong progress provider')
    target_errors = target_window_errors(native, index)
    require(not target_errors, 'Target15 window rejected: '+ '; '.join(target_errors))
    request=L.read(CASE/'request.json')
    if request['trial_protocol'] == PROTOCOL:
        require(not execution_snapshot_errors(request.get('execution_chain'), native.get('execution_chain'),
                    expected_controls=(index+1)*48), 'Wrong actual execution-chain window source')
    m = native['target15_progress']['motion']
    for key in ['window_episode_count','episode_count','completed_episode_count',
                'physical_valid_episode_count','invalid_physical_episode_count']:
        require(type(m[key]) is int and m[key] >= 0, 'Invalid episode counter: '+key)
    require(m['window_episode_count'] == m['episode_count'] == 5000
            and m['completed_episode_count'] >= 5000, 'Need a complete5000-episode window')
    require(m['initial_partial_episodes_excluded'] is True
            and m['metrics_are_training_evidence_not_formal_acceptance'] is True,
            'Wrong initial-partial/denominator semantics')
    require(m['physical_valid_episode_count']+m['invalid_physical_episode_count'] == 5000,
            'Failure/invalid denominator changed')
    for key, upper in [('mean_verified_feet',12), ('mean_completed_levels',6),
                       ('mean_transferred_levels',6), ('first_level_bilateral_fraction',1),
                       ('all_levels_bilateral_fraction',1), ('stair_success_fraction',1)]:
        value = m[key]
        require(isinstance(value, (int,float)) and not isinstance(value,bool)
                and math.isfinite(value) and 0 <= value <= upper, 'Missing/invalid stair metric: '+key)
    require(native['exploration']['optimizer_steps'] == (index+1-40)*20,
            'Window actor counter differs from this new run')
    if m['mean_verified_feet'] == 0:
        require(all(m[key] == 0 for key in ['mean_completed_levels','mean_transferred_levels',
                'first_level_bilateral_fraction','all_levels_bilateral_fraction','stair_success_fraction']),
                'Zero verified feet contradict other accepted stair progress')
    return m


def decide(native_windows):
    require(len(native_windows) == 3, 'Need exactly the three declared windows')
    metrics = [validate_window(native,index) for native,index in zip(native_windows,WINDOWS)]
    counts = [m['completed_episode_count'] for m in metrics]
    require(counts[0] < counts[1] < counts[2], 'Episode evidence did not advance')
    zero = all(m['mean_verified_feet'] == 0 for m in metrics)
    return dict(action='stop_user_zero_first_riser_progress' if zero else 'continue_original1000',
                all_three_zero_qualified_verified_feet=zero,
                window_iterations=list(WINDOWS), completed_episode_counts=counts,
                mean_verified_feet=[m['mean_verified_feet'] for m in metrics],
                first_level_bilateral_fraction=[m['first_level_bilateral_fraction'] for m in metrics],
                complete_window_episodes=5000, failures_invalid_kept_in_denominator=True,
                not_independent_formal_trials_or_convergence_proof=True)


def state(status, **fields):
    L.write(HERE/'state.json', dict(status=status,at=L.now(),case=str(CASE),**fields))


def bindings(auth):
    require(L.digest(HERE/'human_authorization.json') == AUTH_SHA, 'Authorization changed')
    require(L.read(L.POINTER)['active_stair_job'] == str(CASE), 'Current pointer changed')
    req = L.read(CASE/'request.json')
    require(req['trial_protocol'] in HEIGHT_COURSE_PROTOCOLS
            and not early_policy_errors(L.read(req['early_progress_policy']))
            and req['early_progress_policy_sha256'] == L.digest(req['early_progress_policy']),
            'Wrong human-approved target15 policy')
    policy=L.read(req['early_progress_policy'])
    require(L.digest(policy['human_course_choice_file']) == policy['human_course_choice_sha256']
            and L.read(policy['human_course_choice_file'])['answer'] == '采用该课程方案',
            'Direct human course choice changed')
    require(req['case'] == str(CASE) and req['source_commit'] == SOURCE
            and req['plan_sha256'] == PLAN_SHA and req['max_iterations'] == 1000
            and req['expected_final_iteration'] == 999, 'Wrong producer/source/budget')
    for filename,key in [('request.json','request_sha256'),('launch.json','launch_sha256'),
                         ('stop_policy.json','stop_policy_sha256')]:
        require(L.digest(CASE/filename) == auth[key], 'Immutable producer metadata changed: '+filename)
    require(str(Path(L.read(CASE/'supervision.json')['run_dir']).resolve()) == auth['run_dir'],
            'Wrong actual run directory')
    admitted = L.read(HERE/'admission_review.json')
    require(admitted['status'] == 'passed_ready' and admitted['operator_sha256'] == L.digest(__file__),
            'Operator differs from its reviewed bytes')
    require(L.digest(L.SELF) == req['frozen_files'][str(L.SELF)], 'Frozen signal helper changed')
    return req


def prior_stop_priority():
    point = L.read(CASE/'supervision.monitor.json')
    supervision = L.read(CASE/'supervision.json')
    manager = L.read(CASE/'manager_state.json')
    reasons = []
    if supervision['status'] != 'training' or manager['status'] != 'running':
        reasons.append('producer_or_manager_already_ended_or_stopping')
    for key in ['fatal_log_candidates','confirmed_stop_reasons','changed_frozen_inputs']:
        if point[key]:
            reasons.append(key)
    if point['startup_contract']['status'] != 'passed' or point['startup_contract']['errors']:
        reasons.append('startup_contract')
    if supervision.get('manager_stop_evidence') or supervision.get('error') or supervision.get('traceback'):
        reasons.append('existing_manager_or_runtime_stop_evidence')
    guarded = supervision.get('guarded_ppo_state',{})
    if guarded.get('last_guard_rejected') or guarded.get('last_guard',{}).get('rejected'):
        reasons.append('existing_functional_guard_rejection')
    if reasons:
        L.immutable(HERE/'prior_stop_priority.json',dict(at=L.now(),reasons=reasons,
            committed_monitor=point,actual_supervision_status=supervision['status'],
            actual_manager_status=manager['status'],manager_stop_evidence=supervision.get('manager_stop_evidence'),
            original_cause_kept=True,user_stop_signal_sent=False))
        raise RuntimeError('Original stop/fatal/frozen evidence takes priority: '+','.join(reasons))


def capture_window(point, origin):
    require(point['case'] == str(CASE) and point['startup_contract']['status'] == 'passed',
            'Window not bound to the actual current producer')
    native = point.get('latest_motion')
    if not native or native['iteration'] not in WINDOWS:
        return
    index = native['iteration']
    window_errors=target_window_errors(native,index)
    if window_errors:
        L.immutable(HERE/'windows'/f'window_{index:04d}.incomplete.json',dict(at=L.now(),
            case=str(CASE),source_commit=SOURCE,plan_sha256=PLAN_SHA,native=native,
            errors=window_errors,no_zero_substitution=True,no_signal=True))
        raise RuntimeError('Target window incomplete/invalid; no signal and no budget extension')
    validate_window(native,index)
    path = HERE/'windows'/f'window_{index:04d}.json'
    if path.exists():
        require(L.read(path)['native'] == native, 'A previously captured window changed')
        return
    L.immutable(path,dict(at=L.now(),case=str(CASE),source_commit=SOURCE,plan_sha256=PLAN_SHA,
                         origin=origin,monitor_iteration=point['iteration'],native=native))


def admitted_point(point):
    require(point['case'] == str(CASE) and point['state_status'] == 'training'
            and point['manager']['status'] == 'running' and point['log_caught_up'],
            'Producer is not healthy and active at the decision boundary')
    require(500 <= point['iteration'] <= MAX_SIGNAL_ITERATION, '500 decision is missing or stale')
    require(point['fatal_log_candidates'] == point['confirmed_stop_reasons']
            == point['changed_frozen_inputs'] == [] and point['startup_contract']['status'] == 'passed',
            'Pre-existing fatal/guard/frozen evidence takes priority')
    require(len(point['processes']) == 2 and all(x['identity_matches'] for x in point['processes']),
            'Both exact recorded births must match before stopping')
    learner = next(x['recorded_identity'] for x in point['processes']
                   if x['recorded_identity']['role'] == 'learner')
    gpu = point['gpu']['compute_processes']
    require(len(gpu.splitlines()) == 1 and gpu.split(',')[0].strip() == str(learner['pid']),
            'GPU owner differs from the exact learner birth')
    scalars = point['scalars']
    require(sum(k.startswith('Episode_Reward/') for k in scalars) == 19
            and sum(k.startswith('Episode_Termination/') for k in scalars) == 7,
            'Incomplete actual training footer')
    finite(scalars)


def checkpoint_bytes(checkpoint, point, pending):
    marker = CASE/'midpoint_500_checkpoint_bytes.once.json'
    review = CASE/'midpoint_500_checkpoint_byte_review.json'
    if marker.exists() and not review.exists():
        deadline = time.monotonic()+60
        while time.monotonic() < deadline and not review.exists():
            time.sleep(1)
    if review.exists():
        result = L.read(review)
        require(result['status'] == 'passed_bytes_only' and result['scope_closed'] is True
                and result['checkpoint'] == str(checkpoint)
                and L.identity(checkpoint) == result['identity'], 'Existing midpoint custody differs')
        L.immutable(HERE/'checkpoint_byte_receipt_reuse.json',dict(at=L.now(),review=str(review),
                    review_sha256=L.digest(review),freshstat_stable=True,new_byte_opens=0))
        return result
    L.immutable(marker,dict(status='claimed_before_byte_read',at=L.now(),owner=str(HERE)))
    before = L.identity(checkpoint);h = hashlib.sha256()
    with checkpoint.open('rb') as inp, (HERE/'checkpoint_500.bytes.cached.pt').open('xb') as out:
        opened = {k:getattr(os.fstat(inp.fileno()),k) for k in before}
        for block in iter(lambda:inp.read(1048576),b''):
            out.write(block);h.update(block)
        out.flush();os.fsync(out.fileno())
        after_fd = {k:getattr(os.fstat(inp.fileno()),k) for k in before}
    after = L.identity(checkpoint)
    require(before == opened == after_fd == after and before['st_size'] > 0,
            'New midpoint bytes were not stable; do not reopen or signal')
    result = dict(status='passed_bytes_only',scope_closed=True,at=L.now(),checkpoint=str(checkpoint),
        checkpoint_sha256=h.hexdigest(),identity=before,opened_fstat=opened,after_fstat=after_fd,
        after_path_identity=after,ROOT_checkpoint_byte_opens=1,CPU_loads=0,NPZ_decodes=0,
        optimizer_RNG_serialization_not_inspected=True,complete_policy_CPU_validation=False,
        validation_receipt_exists=False,metadata_pending=str(pending),
        metadata_pending_sha256=L.digest(pending),complete_footer_at_byte_admission=point['iteration'],
        physical_acceptance=False,byte_capture_owner=str(HERE))
    L.immutable(review,result)
    L.write(marker,dict(status='passed_consumed',at=L.now(),owner=str(HERE),review=str(review),
                        review_sha256=L.digest(review)))
    return result


def append_ended_once(auth, classification, byte_review):
    ledger = ROOT/'logs/stairs/ended_experiments.jsonl'
    prefix_path=Path(auth['canonical_prefix_cache'])
    prefix = prefix_path.read_bytes()
    prefix_count=auth['canonical_prefix_lines']
    require(hashlib.sha256(prefix).hexdigest() == auth['canonical_prefix_sha256']
            and len(prefix.splitlines()) == prefix_count == 23, 'Old23-row ledger prefix changed')
    with ledger.open('a+b') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0);before=stream.read()
        require(before.startswith(prefix), 'Canonical protected prefix changed')
        rows=[json.loads(x) for x in before.splitlines() if x.strip()]
        own=[x for x in rows if x.get('case') == str(CASE)]
        require(len(own) <= 1, 'Duplicate existing producer ledger rows')
        if not own:
            require(len(rows) == prefix_count, 'Unexpected new ledger case before this sole producer')
            row=dict(case=str(CASE),kind='guarded_stair_transfer_ppo_learning',
                status='stopped_user_no_first_riser_progress',ended_at=L.now(),source_commit=SOURCE,
                original_source_budget=1000,selected_checkpoint=byte_review['checkpoint'],
                checkpoint_sha256=byte_review['checkpoint_sha256'],physical_acceptance_passed=False,
                not_an_independent_formal_failure=True,classification=str(HERE/'terminal_classification.json'))
            stream.seek(0,2);stream.write((json.dumps(row,ensure_ascii=False)+'\n').encode())
            stream.flush();os.fsync(stream.fileno())
        stream.seek(0);after=stream.read()
    L.immutable(HERE/'ended_ledger_review.json',dict(status='passed',old23_prefix_unchanged=True,
        prior_rows=len(rows),current_case_exactly_once=True,existing_row_reused=bool(own),
        after_sha256=hashlib.sha256(after).hexdigest(),shadow_ledger_not_modified=True))


def stop_and_classify(auth, decision, point, pending):
    req=bindings(auth);admitted_point(point);prior_stop_priority()
    L.immutable(HERE/'stop_claim.once.json',dict(status='claimed_before_checkpoint_bytes_or_signals',
        at=L.now(),case=str(CASE),decision=decision,pending=str(pending),
        request_sha256=auth['request_sha256'],operator_sha256=L.digest(__file__)))
    checkpoint=Path(auth['run_dir'])/'model_500.pt'
    require(not checkpoint.with_suffix('.validation.json').exists(), 'Unexpected midpoint CPU receipt')
    byte_review=checkpoint_bytes(checkpoint,point,pending)
    prior_stop_priority()
    fixed,fp=L.sample(CASE,commit=True,reviewed_pending=pending);admitted_point(fixed)
    journal=(CASE/'supervision.jsonl').read_text().splitlines()
    require(fixed == L.read(fp) == json.loads(journal[-1])
            and Path(fixed['reviewed_pending']).resolve() == pending.resolve(), 'Exact monitor commit differs')
    bindings(auth)
    require(not (HERE/'cancel.once.json').exists(), 'User cancelled before signals')
    L.immutable(HERE/'before_signal_manifest.json',dict(at=L.now(),point=fixed,decision=decision,
        fixed_pending=str(fp),supervision_journal_line_1based=len(journal),supervision_journal_index0=len(journal)-1,wholepoint_equal=True,
        selected_checkpoint_byte_review=byte_review,original_source_budget1000_unchanged=True,
        stop_policy_SHA_unchanged=True,serialized_optimizer_RNG_not_inspected=True))
    state('stopping_authorized_zero_first_riser_progress',iteration=fixed['iteration'])
    prior_stop_priority()
    result=L.safe_stop(CASE,['user-authorized500checkpoint: three complete fixed target15 windows have zero qualified verified stair feet'])
    L.immutable(HERE/'stop_result.json',result)
    require(any(x['signal']=='SIGINT' and x['identity']['recorded_identity']['role']=='learner'
                for x in result['signals']), 'No exact learner INT was sent; cannot classify an authorized stop')
    deadline=time.monotonic()+240
    while time.monotonic() < deadline:
        require(L.read(L.POINTER)['active_stair_job']==str(CASE), 'Pointer changed during terminal audit')
        manager=L.read(CASE/'manager_state.json');launch=L.read(CASE/'launch.json')
        ended=all(not L.process_check(x,req['entrypoint'])['identity_matches'] for x in launch['processes'])
        manager_ended=not L.process_check(manager['manager'],str(L.SELF))['identity_matches']
        if manager['status']=='finished' and ended and manager_ended and not L.gpu_snapshot()['compute_processes']:
            try:
                L.no_sim()
            except RuntimeError:
                pass
            else:
                break
        state('waiting_for_actual_manager_births_GPU_terminal');time.sleep(1)
    else:
        raise RuntimeError('Terminal audit timed out; no automatic retry or further signal')
    L.no_sim()
    prior_terminal_monitor=L.read(CASE/'supervision.monitor.json')
    prior_terminal_state=L.read(CASE/'supervision.json')
    L.immutable(HERE/'terminal_committed_cause_snapshot.json',dict(at=L.now(),
        monitor=prior_terminal_monitor,supervision_status=prior_terminal_state['status'],
        manager_stop_evidence=prior_terminal_state.get('manager_stop_evidence')))
    require(not prior_terminal_monitor['fatal_log_candidates']
            and not prior_terminal_monitor['confirmed_stop_reasons']
            and not prior_terminal_monitor['changed_frozen_inputs']
            and not prior_terminal_monitor['startup_contract']['errors']
            and not prior_terminal_state.get('manager_stop_evidence'),
            'Original committed terminal cause requires ROOT review before a new sample')
    terminal,tq=L.sample(CASE)
    require(not terminal['changed_frozen_inputs'] and not terminal['fatal_log_candidates']
            and not terminal['startup_contract']['errors'], 'New terminal fatal/frozen evidence requires ROOT review')
    manager=L.read(CASE/'manager_state.json');supervision=L.read(CASE/'supervision.json')
    require(supervision['status'] not in ('stopped_functional_guard','stopped_no_improvement'),
            'A raced original guard stop must keep its own classification')
    require(manager.get('error') is None and manager['exit_code'] in (0,-15,130,143)
            and not supervision.get('traceback') and not supervision.get('error'),
            'Unexpected exit/traceback; preserve actual state for ROOT classification')
    classification=dict(status='stopped_user_no_first_riser_progress',scope_closed=True,at=L.now(),
        case=str(CASE),decision=decision,original_source_budget=1000,natural_completion=False,
        selected_checkpoint=byte_review['checkpoint'],checkpoint_sha256=byte_review['checkpoint_sha256'],
        checkpoint_bytes_only_no_CPU=True,actual_final_footer_iteration=terminal['iteration'],
        actual_manager_status=manager['status'],actual_manager_outcome=manager['outcome'],
        actual_manager_exit_code=manager['exit_code'],actual_supervision_status=supervision['status'],
        actual_both_births_ended=True,actual_manager_birth_ended=True,actual_GPU_empty=True,
        live_work_after_saved_checkpoint='unknown; user INT does not promise whole-update or GPU RNG restoration',
        not_natural1000_not_exact_resume_not_independent_formal_failure=True,
        unconsumed_actor_raw_tail_requires_separate_ROOT_admission=True)
    L.immutable(HERE/'terminal_classification.json',classification)
    L.immutable(HERE/'terminal_monitor.json',terminal)
    snapshots={}
    for name,path in [('request',CASE/'request.json'),('launch',CASE/'launch.json'),
                      ('supervision',CASE/'supervision.json'),('manager',CASE/'manager_state.json'),
                      ('stop_policy',CASE/'stop_policy.json'),('producer_run',Path(auth['run_dir'])/'run.json')]:
        blob=path.read_bytes();target=HERE/(name+'.terminal.cached.json')
        with target.open('xb') as stream:stream.write(blob)
        snapshots[name]=dict(original=str(path),cached=str(target),sha256=hashlib.sha256(blob).hexdigest())
    L.immutable(HERE/'terminal_metadata_manifest.json',snapshots)
    append_ended_once(auth,classification,byte_review)
    updates=dict(status=classification['status'],active_simulation=False,
        current_work_status='early_progress500_stopped_ROOT_tail_review_and_repair_pending',
        producer_status=classification['status'],iteration=terminal['iteration'],
        completed_iterations=terminal['iteration']+1,completed_iterations_asof_iteration=terminal['iteration'],
        natural_completion=False,final_checkpoint=None,latest_checkpoint=byte_review['checkpoint'],
        latest_checkpoint_sha256=byte_review['checkpoint_sha256'],latest_checkpoint_iteration=500,
        latest_checkpoint_size=byte_review['identity']['st_size'],latest_checkpoint_validation=None,
        latest_checkpoint_bytes_review=str(CASE/'midpoint_500_checkpoint_byte_review.json'),
        latest_checkpoint_bytes_review_sha256=L.digest(CASE/'midpoint_500_checkpoint_byte_review.json'),
        user_early_progress_gate_status='stopped_zero_qualified_first_riser_progress',
        user_early_progress_gate_terminal_classification=str(HERE/'terminal_classification.json'),
        user_early_progress_gate_armed=False)
    old={}
    with (CASE/'monitor.write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in [L.POINTER,ROOT/'logs/stairs/campaign.json',ROOT/'logs/stairs_work/current_stair_work_state.json']:
            value=L.read(path);require(value['active_stair_job']==str(CASE), 'Wrong merge target')
            old[str(path)]={k:value.get(k) for k in updates};value.update(updates);L.write(path,value)
    L.immutable(HERE/'terminal_state_merge_review.json',dict(status='passed',oldvalues=old,
        all_other_fields_preserved=True,original_request_launch_run_supervision_stop_not_rewritten=True,
        numeric_guard_formal_failure_best_history_counts_unchanged=True))
    state('completed_stopped_user_no_progress',classification=str(HERE/'terminal_classification.json'))


def main():
    with (HERE/'operator.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        require(not (HERE/'stop_claim.once.json').exists(), 'Consumed stop claim; never restart automatically')
        auth=L.read(HERE/'human_authorization.json');bindings(auth)
        capture_window(L.read(HERE/'prearm_monitor_snapshot.json'),'prearm_snapshot')
        while True:
            if (HERE/'cancel.once.json').exists():
                state('completed_cancelled_no_signal');return
            if L.read(L.POINTER)['active_stair_job'] != str(CASE):
                state('completed_pointer_changed_no_signal');return
            bindings(auth)
            supervision=L.read(CASE/'supervision.json')
            if supervision['status'] in L.TERMINAL:
                state('producer_ended_before_gate_no_signal',actual_status=supervision['status']);return
            require(supervision['status']=='training', 'Unexpected active producer status')
            monitor=L.read(CASE/'supervision.monitor.json');capture_window(monitor,'current_monitor_JSON')
            found=[i for i in WINDOWS if (HERE/'windows'/f'window_{i:04d}.json').exists()]
            checkpoint=Path(auth['run_dir'])/'model_500.pt'
            state('waiting_for_three_windows_and_saved_M500',captured_window_iterations=found,
                  actual_monitor_iteration=monitor['iteration'],checkpoint_exists=checkpoint.exists(),signals=0,
                  checkpoint_CPU_loads=0,original_source_budget=1000)
            if monitor['iteration'] >= 500 and checkpoint.exists():
                require(monitor['iteration'] <= MAX_SIGNAL_ITERATION, 'Progress decision arrived too late')
                require(found==list(WINDOWS), 'Missing actual299/399/499 windows; no signal')
                decision=decide([L.read(HERE/'windows'/f'window_{i:04d}.json')['native'] for i in WINDOWS])
                L.immutable(HERE/'decision.json',dict(at=L.now(),**decision))
                if decision['action']=='continue_original1000':
                    state('completed_continue_original1000_no_signal',decision=decision);return
                prior_stop_priority()
                point,pending=L.sample(CASE);capture_window(point,'direct_boundary_monitor');admitted_point(point)
                stop_and_classify(auth,decision,point,pending);return
            time.sleep(3)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        L.write(HERE/'failure.json',dict(at=L.now(),error=repr(error),traceback=traceback.format_exc(),
                                      automatic_retry=False,stop_claim_exists=(HERE/'stop_claim.once.json').exists()))
        state('needs_attention_no_automatic_retry',error=repr(error))
        raise
