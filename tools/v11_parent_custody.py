"""Stdlib custody for the one measured, fully restored v10 context parent.

No tensor files are loaded here. Byte fingerprints and already-issued CPU and
formal receipts are checked independently of future model-state transfer.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from task_context_metadata import (ACTIVE_CONTEXT_PROTOCOL, ACTIVE_CONTEXT_FORMAT,
    ACTIVE_CONTEXT_PROTOCOLS, FULL_OUTWARD_CONTEXT_PROTOCOL, FULL_OUTWARD_CONTEXT_FORMAT,
    CONTEXT_PROTOCOL, CONTEXT_FORMAT, validate_task_context_plan,
    validate_task_context_metadata, validate_normalization_metadata)

PARENT_SHA = 'b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce'
PARENT_RECEIPT_SHA = '28ba2af4df557ab99550a667be960e99d4982fffade5e3d3d7d0aa2442e663de'
PARENT_RUN_SHA = 'e62568bac6945178fd7c1785b17dc3e553a2e3d613c801560d3a72eb996ff8e8'
PARENT_COMMIT = '18840a7e87558e5f664f4ee0996cacc30821b8ba'
PARENT_STATE_DIGEST = 'f1c5ae30855147af9b12c757f89831fa8af2913406d2ac3145df09e112a12f91'
PARENT_PLAN_SHA = '560c0b630ece3407ea7a82dc0a3010518291248eab91adee41c698b457faa045'
PARENT_FORMAL_REVIEW_SHA = 'f7abfcac31b31e88cbf84fe3008c1d2ffc8a42538106ecd749a14caab1088899'
PARENT_FORMAL_RESULT_SHA = 'd19f379d489cc63204d473182b0bb78eee2ffde4a4d146040671721cf55aa927'
PARENT_FORMAL_MANIFEST_SHA = '5593a3ecadef2f4c780b6332c6e255d4adc16d2b7cac786de071f34240496710'
PARENT_GUARD_SHA = '4d432ace2c78465a248e43545f3a4fb376537d3ea65fdf8d1351e113a6afd033'
PARENT_TERMINAL_SHA = 'cb82b00c5d5272f852b6afd1135595dd7bf301b8b57e466438a090c193f591bb'
PARENT_CPU_ONCE_SHA = 'd9b357a8487f6337e5e9799f17bace99a1b81aa6e7d0e97944520b97c632ee21'
PARENT_KIND = 'saved_guard_restored_v10_context_policy_transfer'
INITIALIZATION_MODE = 'transfer_verified_guard_restored_v10_policy_and_frozen_dualnorm_fresh_optimization'
SUPPORT_POLISH_POLICY = 'bounded_dual_active_set_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def reviewed_parent():
    return dict(sha=PARENT_SHA,receipt=PARENT_RECEIPT_SHA,run=PARENT_RUN_SHA,
        format=CONTEXT_FORMAT,iteration=371,updates=372,status='stopped_functional_guard',
        kind=PARENT_KIND,mode=INITIALIZATION_MODE,restored=True)


def validate_parent_records(plan, receipt, producer, guard, claim, formal, formal_review):
    """Metadata-only predicate, also usable with explicit synthetic fixtures."""
    validate_task_context_plan(plan)
    require(plan['trial_protocol'] in ACTIVE_CONTEXT_PROTOCOLS
            and plan['checkpoint_format']==(FULL_OUTWARD_CONTEXT_FORMAT if plan['trial_protocol']==FULL_OUTWARD_CONTEXT_PROTOCOL else ACTIVE_CONTEXT_FORMAT)
            and plan['parent_uncommitted_rollout'] is True
            and plan['initial_policy_CPU_receipt_sha256']==PARENT_RECEIPT_SHA
            and plan['initial_parent_run_sha256']==PARENT_RUN_SHA
            and plan['parent_formal_review_sha256']==PARENT_FORMAL_REVIEW_SHA
            and plan['parent_formal_result_sha256']==PARENT_FORMAL_RESULT_SHA
            and plan['parent_formal_artifact_manifest_sha256']==PARENT_FORMAL_MANIFEST_SHA
            and plan['parent_rollback_evidence_sha256']==PARENT_GUARD_SHA
            and plan['parent_terminal_review_sha256']==PARENT_TERMINAL_SHA
            and plan['parent_CPU_once_sha256']==PARENT_CPU_ONCE_SHA
            and plan['parent_complete_state_digest']==PARENT_STATE_DIGEST,
            'v11 is only the reviewed restored-v10 parent, not a natural or extension initializer')
    require(receipt.get('status')=='passed' and receipt.get('cpu_only') is True
            and receipt['sha256']==PARENT_SHA and receipt['checkpoint_format']==CONTEXT_FORMAT
            and receipt['trial_protocol']==CONTEXT_PROTOCOL and receipt['checkpoint_iteration']==371
            and receipt['committed_updates']==372 and receipt['finite_tensors']==255
            and receipt['uncommitted_rollout'] is True and receipt['rnd_rollout_update_counter']==8952
            and receipt['network_dimensions']==[1034,871,73,12]
            and receipt['plan_sha256']==PARENT_PLAN_SHA,
            'Restored-v10 complete CPU receipt/counters/layout differ')
    require(producer['status']=='stopped_functional_guard' and producer['git_commit']==PARENT_COMMIT
            and producer['completed_iterations']==372 and producer['total_timesteps']==9142272
            and producer['actual_sampled_environment_steps']==9166848
            and producer['discarded_uncommitted_rollout_steps']==24576,
            'Parent stopped status/sampled budget was relabeled')
    proof=receipt.get('rollback_equivalence',{})
    require(proof.get('status')=='passed' and proof.get('state_excluded_fields')==['infos']
            and proof.get('all_state_fields_equal') is True and proof.get('cuda_rng_equal') is True
            and type(proof.get('cuda_rng_tensor_count')) is int and proof['cuda_rng_tensor_count']>0
            and proof['baseline_state_digest']==proof['restored_state_digest']==PARENT_STATE_DIGEST
            and proof.get('no_simulation') is True and proof.get('CUDA_runtime_initialized') is False
            and proof.get('NNforward_calls')==proof.get('optimizer_updates')==0
            and proof.get('restored_checkpoint_loads')==proof.get('baseline_checkpoint_loads')==1
            and proof.get('candidate_checkpoint_loads')==0,
            'Independent full persistent-state/GPU RNG parent proof differs')
    require(guard.get('complete_update_restored') is True and guard.get('retry_allowed') is False
            and guard['rollback_state_digest_before']==guard['rollback_state_digest_after']==PARENT_STATE_DIGEST
            and guard['completed_updates']==guard['update_index']==372 and guard['minibatch']==19
            and guard['restored_checkpoint_sha256']==PARENT_SHA,
            'Parent actual controlled-stop rollback evidence differs')
    require(claim.get('status')=='passed' and claim['sha256']==PARENT_SHA
            and claim['receipt_sha256']==PARENT_RECEIPT_SHA
            and claim['inspector_sha256']==proof['inspector_sha256'],
            'Parent exclusive CPU-once claim is not the completed actual proof')
    for name,count,steps in (('actor',45,6640),('critic',8,7440),('rnd',4,7440)):
        require(receipt['optimizers'][name]['parameter_count']==count
                and receipt['optimizers'][name]['steps']==steps,
                'Parent exact optimizer history differs: '+name)
    require(receipt['actor_proposal_projection']['committed_minibatches']==6640,
            'Parent projection and restored Adam counters differ')
    validate_task_context_metadata(producer['runtime_flat_geometry'])
    validate_normalization_metadata(receipt['normalization_abi'],receipt['normalizer_prefix_counts'])
    require(receipt['normalization_abi']==producer['guarded_flat_learning']['normalization_abi']
            and receipt['task_context']==producer['runtime_flat_geometry']['task_context'],
            'Parent own block normalization/raw-context semantics differ')
    require(formal['status']=='completed' and formal['num_envs']==1024
            and formal.get('checkpoint_sha256')==PARENT_SHA and formal['successful_envs']==842
            and formal['duration_s']==20. and formal['required_successful_envs']==922
            and formal['simulated_steps']==1000 and formal['training_updates']==0
            and formal['flat_balance_ready'] is False and formal['formal_acceptance_passed'] is False
            and formal_review['checkpoint_sha256']==PARENT_SHA
            and formal_review['successful_envs']==842 and formal_review['required_joint']==922
            and formal_review['num_envs']==1024 and formal_review['simulated_steps']==1000
            and formal_review['simulated_duration_s']==20
            and formal_review['producer_natural_completion'] is False
            and formal_review['flat_acceptance_passed'] is False
            and formal_review['conjunction_exact_all1024'] is True
            and formal_review['manager_finished'] is True and formal_review['wrapper_exit_code']==0
            and formal_review['manager_error'] is None and formal_review['both_actual_births_ended'] is True
            and formal_review['GPU_released'] is True and formal_review['policy_and_dualnorm_unchanged'] is True
            and formal_review['all640_frozen_inputs_unchanged'] is True
            and formal_review['own_source_copy_exact'] is True,
            'Parent842 must be the ended independent original-gate measurement, not acceptance')
    return dict(parent_checkpoint_sha256=PARENT_SHA,parent_accepted_updates=372,
        parent_last_complete_iteration=371,parent_uncommitted_rollout=True,
        parent_natural_completion=False,parent_runtime_rollback_completed=True,
        parent_persistent_state_digest=PARENT_STATE_DIGEST,parent_formal_successful_envs=842,
        parent_formal_acceptance_passed=False)


def verify_parent_files(plan, root):
    """Bind actual parent files; no Torch/NumPy import or tensor deserialization."""
    root=Path(root).resolve(strict=True)
    run=root/'logs/rsl_rl/h2_main_policy_manager_based/2026-11-01_10-16-33_guarded_flat_balance_48cecc2f'
    case=root/'logs/stair_learning/flat_balance_20261101_101628_48cecc2f'
    formal_case=root/'logs/stair_learning/flat_balance_20261101_120357_50270e02'
    checkpoint=run/'model_371_guard_restored.pt';receipt_path=checkpoint.with_suffix('.validation.json')
    bindings=(('initial_policy',checkpoint,PARENT_SHA),('initial_policy_CPU_receipt',receipt_path,PARENT_RECEIPT_SHA),
        ('initial_parent_run',run/'run.json',PARENT_RUN_SHA),
        ('parent_formal_review',formal_case/'formal_result_review.json',PARENT_FORMAL_REVIEW_SHA),
        ('parent_formal_result',formal_case/'evaluation/result.json',PARENT_FORMAL_RESULT_SHA),
        ('parent_formal_artifact_manifest',formal_case/'formal_artifacts.sha256.json',PARENT_FORMAL_MANIFEST_SHA),
        ('parent_rollback_evidence',run/'functional_guard/rejected_update_0372.json',PARENT_GUARD_SHA),
        ('parent_terminal_review',case/'terminal_v10_guardstop_review.json',PARENT_TERMINAL_SHA),
        ('parent_CPU_once',checkpoint.with_suffix('.CPU_once.json'),PARENT_CPU_ONCE_SHA))
    frozen={}
    for key,path,expected in bindings:
        require(Path(plan[key]).resolve(strict=True)==path and plan[key+'_sha256']==expected and sha(path)==expected,
                'Exact restored-v10 file binding differs: '+key)
        frozen[str(path)]=expected
    receipt=read(receipt_path);producer=read(run/'run.json')
    source=Path(plan['initial_parent_source_root']).resolve(strict=True)
    require(source==run/'training_source/source'==Path(receipt['source_root']).resolve(strict=True),
            'v11 parent must retain its own1884 production source')
    guard_path=Path(producer['functional_guard_evidence']).resolve(strict=True);guard=read(guard_path)
    require(guard_path==run/'functional_guard/rejected_update_0372.json', 'Wrong parent guard event')
    claim_path=checkpoint.with_suffix('.CPU_once.json');claim=read(claim_path)
    result=validate_parent_records(plan,receipt,producer,guard,claim,read(formal_case/'evaluation/result.json'),read(formal_case/'formal_result_review.json'))
    require(receipt['producer_run_sha256']==PARENT_RUN_SHA and sha(Path(receipt['plan_path']))==PARENT_PLAN_SHA
            and Path(receipt['producer_run']).resolve(strict=True)==run/'run.json'
            and Path(receipt['checkpoint']).resolve(strict=True)==checkpoint,
            'Parent receipt path identity differs')
    proof=receipt['rollback_equivalence']
    require(proof['evidence_sha256']==sha(guard_path)
            and Path(proof['baseline']).resolve(strict=True)==Path(guard['baseline']).resolve(strict=True)
            and proof['baseline_sha256']==guard['baseline_sha256'], 'Parent independent baseline/evidence proof differs')
    require(Path(claim['checkpoint']).resolve(strict=True)==checkpoint
            and Path(claim['receipt']).resolve(strict=True)==receipt_path, 'Parent CPU claim path differs')
    actual={}
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            value=sha(path);actual['source/'+str(path.relative_to(source))]=value;frozen[str(path)]=value
    require(actual==receipt['source_sha256'], 'Parent full source tree changed')
    require(producer['training_tool_sha256']==receipt['producer_training_tool_sha256'], 'Parent tool manifest differs')
    for name,value in producer['training_tool_sha256'].items():
        path=source.parent/name;require(sha(path)==value, 'Frozen parent tool changed: '+name);frozen[str(path)]=value
    require(proof['producer_state_digest_source_sha256']==producer['training_tool_sha256']['guarded_flat_ppo.py'],
            'Parent whole-state comparator source differs')
    for path,value in receipt['external_dependency_sha256'].items():
        require(sha(path)==value, 'Parent external dependency changed');frozen[path]=value
    for key,sha_key in (('bank_path','bank_sha256'),('receipt_path','receipt_sha256'),('seal_path','seal_sha256'),('acquisition_log_path','acquisition_log_sha256')):
        bank=receipt['fixed_context_bank_custody'];path=Path(bank[key]).resolve(strict=True)
        require(sha(path)==bank[sha_key], 'Parent own bank custody changed: '+key);frozen[str(path)]=bank[sha_key]
    for key in ('baseline','candidate','prospective_adam_audit','actual_input_evidence','restored_checkpoint'):
        path=Path(guard[key]).resolve(strict=True)
        require(path.parent==run/'functional_guard' if key!='restored_checkpoint' else path==checkpoint,
                'Parent guard artifact outside original path')
        require(sha(path)==guard[key+'_sha256'], 'Parent retained failure artifact changed: '+key);frozen[str(path)]=guard[key+'_sha256']
    numerical=guard['projection_failure']['numerical_diagnostic_evidence'];path=Path(numerical['path']).resolve(strict=True)
    require(sha(path)==numerical['sha256'], 'Parent numerical evidence changed');frozen[str(path)]=numerical['sha256']
    for path in (guard_path,claim_path,run/'params/agent.yaml',Path(receipt['plan_path']),case/'request.json',case/'launch.json',case/'launch.lock',case/'terminal_v10_guardstop_review.json'):
        frozen[str(path)]=sha(path)
    # Old live ROOT paths legitimately change in this new reviewed trial. All
    # immutable input/artifact paths remain frozen; current ROOT is pinned by
    # the new plan's complete reviewed_root_source_sha256 separately.
    old_request=read(formal_case/'request.json')
    excluded={}
    formal_commit=read(formal_case/'formal_result_review.json')['launcher_commit']
    for path,value in old_request['frozen_files'].items():
        p=Path(path).resolve(strict=True)
        if p.is_relative_to(root/'source') or p.is_relative_to(root/'tools'):
            relative=p.relative_to(root)
            copies=([source/p.relative_to(root/'source')] if p.is_relative_to(root/'source') else
                    [source.parent/p.name,formal_case/'evaluation/tools_snapshot'/p.name])
            origin=next((copy for copy in copies if copy.is_file() and sha(copy)==value),None)
            if origin is not None:
                frozen[str(origin)]=value
                excluded[str(p)]=dict(sha256=value,immutable_origin=str(origin),origin_kind='frozen_own_snapshot')
            else:
                blob=subprocess.check_output(['git','show',formal_commit+':'+str(relative)],cwd=root)
                require(hashlib.sha256(blob).hexdigest()==value, 'Excluded old ROOT path lacks its exact Git source: '+str(p))
                excluded[str(p)]=dict(sha256=value,immutable_origin=formal_commit+':'+str(relative),origin_kind='git_blob')
            continue
        require(sha(p)==value,'Immutable prior formal/ancestor input changed: '+path);frozen[str(p)]=value
    for name,entry in read(formal_case/'formal_result_review.json')['artifact_hashes'].items():
        path=formal_case/'evaluation'/name
        require(sha(path)==entry['sha256'], 'Formal artifact changed: '+name);frozen[str(path)]=entry['sha256']
    result.update(frozen_files=frozen,excluded_prior_active_ROOT_hashes=excluded,parent_source_sha256=actual,
        parent_rollback_evidence=dict(path=str(guard_path),sha256=sha(guard_path),persistent_state_digest=PARENT_STATE_DIGEST),
        parent_formal_review=str(formal_case/'formal_result_review.json'),parent_formal_review_sha256=PARENT_FORMAL_REVIEW_SHA,
        parent_formal_result=str(formal_case/'evaluation/result.json'),parent_formal_result_sha256=PARENT_FORMAL_RESULT_SHA)
    if plan['trial_protocol']==FULL_OUTWARD_CONTEXT_PROTOCOL:
        archived=freeze_closed_v11_evidence(plan,root)
        result['frozen_files'].update(archived['frozen_files'])
        result['retained_v11_archive']=archived
    return result


def validate_support_projection_metadata(algorithm, projection, summary=None):
    """Check recorded bounds/counts only; not a numerical feasibility proof."""
    require(algorithm.get('actor_proposal_support_polish_policy')==SUPPORT_POLISH_POLICY
            and projection.get('support_polish_policy')==SUPPORT_POLISH_POLICY
            and projection.get('max_polish_per_QP')==1345
            and projection.get('max_support_pivots')==1344
            and projection.get('max_polish_per_projection')==37660
            and projection.get('support_pivot_limit_rule')=='2*m'
            and projection.get('polish_call_limit_per_subproblem_rule')=='2*m+1',
            'v11 must declare its bounded per-QP active-support method, not the old1SVD/0pivot solver')
    if summary is None:
        return dict(support_polish_policy=SUPPORT_POLISH_POLICY)
    p=summary.get('primal_precision_summary',{})
    require(summary.get('support_polish_policy')==p.get('support_polish_policy')==SUPPORT_POLISH_POLICY
            and p.get('direct_primal_certified') is True
            and p.get('polish_call_limit_per_subproblem')==1345
            and p.get('polish_call_limit_per_projection')==37660
            and p.get('support_pivot_limit')==1344 and p.get('polish_support_limit')==672
            and p.get('support_pivot_limit_rule')=='2*m'
            and p.get('polish_call_limit_per_subproblem_rule')=='2*m+1'
            and p.get('support_pivot_limit_scope')=='per_QP_maximum_at_m672'
            and p.get('final_same_direction_row_JVP_checked') is True,
            'v11 actual summary lacks its bounded active-support/native-direction certificate')
    records=p.get('QP_support_refinements')
    require(isinstance(records,list) and type(p.get('QP_subproblems')) is int
            and len(records)==p['QP_subproblems'] and len(records)<=28,
            'v11 must account for every actual QP subproblem')
    for item in records:
        m=item.get('QP_inequalities')
        require(type(m) is int and 1<=m<=672
                and item.get('support_pivot_limit')==2*m and item.get('polish_call_limit')==2*m+1
                and item.get('certified') is True,
                'v11 per-QP support resource/certificate differs')
        for key,limit in (('polish_calls',2*m+1),('support_pivots',2*m),
                          ('support_additions',2*m),('support_drops',2*m),
                          ('positive_support_size',m),('working_support_size',m)):
            require(type(item.get(key)) is int and 0<=item[key]<=limit,
                    'v11 per-QP bounded count differs: '+key)
        require(item['support_pivots']==item['support_additions']+item['support_drops'],
                'v11 per-QP pivot accounting differs')
    for key in ('polish_calls','support_pivots','support_additions','support_drops'):
        require(type(p.get(key)) is int and p[key]==sum(row[key] for row in records),
                'v11 active-support total differs: '+key)
    require(p['polish_calls']<=37660, 'v11 exceeded the full-projection SVD work cap')
    return dict(support_polish_policy=SUPPORT_POLISH_POLICY,QP_subproblems=len(records),
                polish_calls=p['polish_calls'],support_pivots=p['support_pivots'])


def freeze_closed_v11_evidence(plan, root):
    """Keep the completed813 comparison evidence; never use it as initializer."""
    root=Path(root).resolve(strict=True);spec=plan['retained_v11_evidence']
    case=root/'logs/stair_learning/flat_balance_20261101_162132_40277873'
    run=root/'logs/rsl_rl/h2_main_policy_manager_based/2026-11-01_13-19-17_guarded_flat_balance_e82cfb2b'
    bindings=(('checkpoint',run/'model_999.pt','71757b6b58257422b2a58c3dce74e6f551711430fb6c5ba38a6db8c95de5b15f'),
        ('CPU_receipt',run/'model_999.validation.json','d8cc754315586505a464e2240a8dc8462f03335a6ebb042b7843023bcf5dc073'),
        ('producer_run',run/'run.json','fe8588b9383170fb32e5bcbc284382bac5449c07a23dc164a2c400b48c1d5d92'),
        ('formal_result_review',case/'formal_result_review.json','35adc4d3ff4806c60ea88fbbbc6e6d9d25167f1d2c8e938b0225a17ed673aa0b'),
        ('formal_artifact_manifest',case/'formal_artifacts.sha256.json','2b5f4c347f8971db63be623d7e332125b16d0ab4fc9a62e29d6414b80e93077b'))
    require(spec.get('kind')=='closed_v11_comparison_evidence_not_initializer_v1'
            and spec.get('initialization_parent') is False
            and plan['initial_policy_sha256']==PARENT_SHA,'Closed813 comparison cannot replace the842 parent')
    frozen={}
    for key,path,value in bindings:
        require(Path(spec[key]).resolve(strict=True)==path and spec[key+'_sha256']==value and sha(path)==value,
                'Retained completedv11 identity changed: '+key)
        frozen[str(path)]=value
    receipt=read(run/'model_999.validation.json');producer=read(run/'run.json')
    formal=read(case/'evaluation/result.json');review=read(case/'formal_result_review.json')
    validate_closed_v11_records(spec,receipt,producer,formal)
    source=Path(spec['source_root']).resolve(strict=True)
    require(source==run/'training_source/source','Retainedv11 must use its original ownsource')
    actual={'source/'+str(p.relative_to(source)):sha(p) for p in source.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    require(actual==receipt['source_sha256'],'Retainedv11 ownsource changed')
    frozen.update({str(source/key.removeprefix('source/')):value for key,value in actual.items()})
    for name,value in receipt['producer_training_tool_sha256'].items():
        path=source.parent/name;require(sha(path)==value,'Retainedv11 own tool changed: '+name);frozen[str(path)]=value
    for key,sha_key in (('bank_path','bank_sha256'),('receipt_path','receipt_sha256'),('seal_path','seal_sha256'),('acquisition_log_path','acquisition_log_sha256')):
        bank=receipt['fixed_context_bank_custody'];require(sha(bank[key])==bank[sha_key],'Retainedv11 bank changed');frozen[bank[key]]=bank[sha_key]
    excluded={};request=read(case/'request.json')
    for name,value in request['frozen_files'].items():
        path=Path(name).resolve(strict=True)
        if path.is_relative_to(root/'source') or path.is_relative_to(root/'tools'):
            rel=path.relative_to(root);copies=([source/path.relative_to(root/'source')] if path.is_relative_to(root/'source') else
                [source.parent/path.name,case/'snapshot/tools'/path.name,case/'evaluation/tools_snapshot'/path.name])
            origin=next((p for p in copies if p.is_file() and sha(p)==value),None)
            if origin is not None:
                frozen[str(origin)]=value;excluded[name]=dict(sha256=value,origin=str(origin),kind='own_snapshot')
            else:
                commit=request.get('launcher_commit',request['source_commit'])
                blob=subprocess.check_output(['git','show',commit+':'+str(rel)],cwd=root)
                require(hashlib.sha256(blob).hexdigest()==value,'Retainedv11 oldROOT origin changed')
                excluded[name]=dict(sha256=value,origin=commit+':'+str(rel),kind='git_blob')
            continue
        require(sha(path)==value,'Retainedv11 old frozen input changed: '+name);frozen[str(path)]=value
    manifest_path=case/'formal_artifacts.sha256.json'
    require(Path(review['artifact_manifest']).resolve(strict=True)==manifest_path
            and review['artifact_manifest_sha256']==spec['formal_artifact_manifest_sha256'],
            'Retainedv11 review must bind its actual13-entry artifact manifest')
    entries=validate_closed_formal_artifact_manifest(read(manifest_path),case/'evaluation')
    for name,item in entries.items():
        path=Path(name).resolve(strict=True)
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],
                'Retainedv11 formal artifact changed: '+name)
        frozen[str(path)]=item['sha256']
    result_path=case/'evaluation/result.json'
    require(Path(spec['formal_result']).resolve(strict=True)==result_path and sha(result_path)==spec['formal_result_sha256'],
            'Retainedv11 formal result changed')
    for path in (case/'request.json',case/'launch.json',case/'launch.lock',run/'params/agent.yaml',Path(receipt['plan_path'])):
        frozen[str(path)]=sha(path)
    return dict(kind=spec['kind'],initialization_parent=False,frozen_files=frozen,
                excluded_prior_active_ROOT_hashes=excluded,checkpoint_sha256=spec['checkpoint_sha256'],formal_joint=813)


def validate_closed_v11_records(spec, receipt, producer, formal):
    """Archive metadata only; this record is explicitly not a policy initializer."""
    require(spec.get('kind')=='closed_v11_comparison_evidence_not_initializer_v1'
            and spec.get('initialization_parent') is False and spec['formal_joint']==813,
            'Closed813 record may not be relabeled an initialization parent')
    require(receipt['status']=='passed' and receipt['cpu_only'] is True
            and receipt['checkpoint_format']==ACTIVE_CONTEXT_FORMAT and receipt['committed_updates']==1000
            and receipt['checkpoint_iteration']==999 and receipt['uncommitted_rollout'] is False
            and receipt['sha256']==spec['checkpoint_sha256']=='71757b6b58257422b2a58c3dce74e6f551711430fb6c5ba38a6db8c95de5b15f'
            and producer['status']=='completed' and producer['completed_iterations']==1000
            and producer['total_timesteps']==24576000 and producer['git_commit']=='985ca33a82c132be7d61710006f06a3747871c77',
            'Retainedv11 must remain its completed1000 own record, not a restored or new initializer')
    require(formal['status']=='completed' and formal['successful_envs']==813
            and formal['checkpoint_sha256']==spec['checkpoint_sha256']
            and formal['formal_acceptance_passed'] is False,
            'Retainedv11 original813 formal identity changed')
    return dict(initialization_parent=False,formal_joint=813,checkpoint_sha256=spec['checkpoint_sha256'])


def validate_closed_formal_artifact_manifest(manifest, evaluation_directory):
    """Validate the pinned list schema and13 paths; do not open artifact bytes."""
    directory=Path(evaluation_directory).resolve()
    expected={'agent.yaml','checkpoint.validation.json','checkpoint_run.json','env0.mp4','environment.yaml',
        'episodes.csv','first_episode_terminal.npz','footfall_events.json','result.json','run.json',
        'runtime_physical_parameters.npz','saved_observation_normalizers.pt','trajectories.npz'}
    rows=manifest.get('artifacts')
    require(Path(manifest['case']).resolve()==directory.parent
            and isinstance(rows,list) and len(rows)==13,
            'Retainedv11 manifest must be its actual13-entry list')
    result={}
    for row in rows:
        raw=Path(row['path']);path=raw.resolve()
        require(raw.is_absolute() and path.parent==directory and path.name in expected
                and str(path) not in result, 'Retainedv11 manifest path is duplicate/outside the one evaluation directory')
        value=row['sha256']
        require(type(row.get('bytes')) is int and row['bytes']>0
                and isinstance(value,str) and len(value)==64
                and all(c in '0123456789abcdef' for c in value), 'Retainedv11 artifact size/SHA is malformed')
        result[str(path)]={'bytes':row['bytes'],'sha256':value}
    require({Path(name).name for name in result}==expected,'Retainedv11 manifest unique artifact set differs')
    return result
