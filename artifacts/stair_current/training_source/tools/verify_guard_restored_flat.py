"""One complete CPU audit and saved-state comparison for the reviewed v10 stop.

The existing complete checkpoint validator owns the single restored-policy and
required parent loads. Its optional inspector reuses that restored packet and
loads only the original whole-update baseline. No candidate is deployed or
loaded, no optimizer is run, and no physical state is resumed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
CASE=ROOT/'logs/stair_learning/flat_balance_20261101_101628_48cecc2f'
KIND='single_frozen_validation_of_controlled_guard_restored_v10_context_372'
SOURCE_COMMIT='18840a7e87558e5f664f4ee0996cacc30821b8ba'
CHECKPOINT_SHA='b7bb6198f67b033dd6c65842bafdee29666f757236e069e501a9e1ab32a2bdce'
STATE_SHA='f1c5ae30855147af9b12c757f89831fa8af2913406d2ac3145df09e112a12f91'
POSTEND_TOOLS={'tools/evaluate_flat_balance_policy.py',
               'tools/launch_flat_balance_learning.py','tools/verify_guard_restored_flat.py'}


def require(condition,message):
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


def write(path,value):
    temporary=Path(path).with_name(Path(path).name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(review):
    """Read-only admission, before any model load or exclusive audit claim."""
    import launch_flat_balance_learning as manager
    require(review.get('kind')==KIND and review.get('status')=='CPU_review_ready',
            'A specific CPU-ready restored-v10 review is required')
    pointer=read(manager.POINTER)
    require(Path(pointer['active_stair_job']).resolve()==CASE and
            Path(review['parent_case']).resolve()==CASE,'Review targets another producer')
    manager.prior_finished(pointer)
    require(not manager.gpu_snapshot()['compute_processes'] and not manager.active_simulators(),
            'An Isaac/GPU task is still active')
    require(manager.git('branch','--show-current')==manager.BRANCH and
            not manager.git('status','--porcelain=v1'),'Commit reviewed tools before CPU audit')
    _,request=manager.request_for(CASE)
    require(request['source_commit']==SOURCE_COMMIT and request['trial_protocol']=='explicit_task_context_v1',
            'Wrong producer source or protocol')
    require(sha(CASE/'launch.json')==read(CASE/'launch.identity.sha256.json')['sha256'],
            'Producer launch identity changed')
    state=read(CASE/'supervision.json');run=Path(state['run_dir']).resolve()
    checkpoint=Path(review['checkpoint']).resolve(strict=True)
    require(checkpoint==run/'model_371_guard_restored.pt'==Path(state['final_checkpoint']) and
            sha(checkpoint)==CHECKPOINT_SHA==review['checkpoint_sha256'],
            'Only the exact trusted recorded restored checkpoint is eligible')
    evidence=Path(review['rollback_evidence']).resolve(strict=True)
    require(evidence==Path(state['functional_guard_evidence'])==run/'functional_guard/rejected_update_0372.json' and
            sha(evidence)==review['rollback_evidence_sha256'],'Rollback evidence changed')
    guard=read(evidence)
    require(state['status']==guard['status']=='stopped_functional_guard' and
            state['completed_iterations']==guard['completed_updates']==guard['update_index']==372 and
            guard['minibatch']==19 and guard['complete_update_restored'] is True and
            guard['retry_allowed'] is False and guard['rollback_state_digest_before']==
            guard['rollback_state_digest_after']==STATE_SHA==review['rollback_state_digest'],
            'Actual controlled whole-update restoration is not established')
    baseline=Path(review['original_baseline_snapshot']).resolve(strict=True)
    require(baseline==Path(guard['baseline'])==run/'functional_guard/rejected_update_0372_baseline.pt' and
            sha(baseline)==guard['baseline_sha256'],'Original whole-update baseline changed')
    terminal_path=Path(review['terminal_review']).resolve(strict=True)
    terminal=read(terminal_path)
    require(terminal_path==CASE/'terminal_v10_guardstop_review.json' and
            sha(terminal_path)==review['terminal_review_sha256'] and
            terminal['manager_finished'] is True and terminal['GPU_released'] is True and
            terminal['complete_update_restored'] is True and terminal['natural_completion'] is False and
            terminal['checkpoint_sha256']==CHECKPOINT_SHA and
            terminal['rollback_evidence_sha256']==sha(evidence) and
            terminal['pre_tool_change_frozen_differences']==[] and
            terminal['terminal_commit_journal_verified'] is True,
            'Independent terminal audit does not establish the exact ended producer')
    tools=review['postend_tool_sha256']
    require(set(tools)==POSTEND_TOOLS and all(sha(ROOT/name)==value for name,value in tools.items()),
            'Postend tool scope changed')
    current=manager.root_hashes();old=request['root_hashes']
    allowed_old={str(ROOT/name) for name in POSTEND_TOOLS}
    require(set(current)==set(old)|{str(ROOT/'tools/verify_guard_restored_flat.py')} and
            all(current.get(name)==value for name,value in old.items() if name not in allowed_old),
            'An undeclared producer ROOT source/tool changed after its stop')
    for name,value in request['frozen_files'].items():
        require(sha(name)==value,'Frozen producer input changed: '+name)
    output=checkpoint.with_suffix('.validation.json')
    require(Path(review['CPU_receipt']).resolve()==output and not output.exists(),
            'Restored checkpoint already has a CPU receipt or the output identity differs')
    source=run/'training_source/source'
    return checkpoint,source,baseline,evidence,guard,output


def inspect_restored(saved,baseline,evidence,guard,source):
    """Compare actual CPU-loaded packets, including every serialized RNG field."""
    import torch
    import numpy as np
    require(not torch.cuda.is_initialized(),'CPU audit unexpectedly initialized CUDA')
    before=torch.load(baseline,map_location='cpu',weights_only=True)
    producer=source.parent/'guarded_flat_ppo.py'
    run=read(source.parent.parent/'run.json')
    require(sha(producer)==run['training_tool_sha256']['guarded_flat_ppo.py'],
            'Producer digest implementation changed')
    # Execute only the already-frozen pure digest function, not the PPO module.
    module=ast.parse(producer.read_text())
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='state_digest')
    namespace=dict(hashlib=hashlib,json=json,np=np,torch=torch)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(producer),'exec'),namespace)
    digest=namespace['state_digest']
    lhs={k:v for k,v in before.items() if k!='infos'}
    rhs={k:v for k,v in saved.items() if k!='infos'}
    require(set(lhs)==set(rhs),'Restored state fields differ from the baseline')
    tensors=0
    def exact(a,b):
        nonlocal tensors
        if isinstance(a,torch.Tensor):
            tensors+=1
            return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
        if isinstance(a,dict):
            return isinstance(b,dict) and set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):
            return type(a) is type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return type(a) is type(b) and a==b
    require(exact(lhs,rhs),'Actual complete restored learning state differs from baseline')
    left,right=digest(lhs),digest(rhs)
    require(left==right==STATE_SHA==guard['rollback_state_digest_before']==guard['rollback_state_digest_after'],
            'Independent serialized-state digests differ from actual runtime restore evidence')
    rng=saved['rng_state'];baseline_rng=before['rng_state']
    require(set(rng)=={'torch_cpu','torch_cuda','python','numpy'} and
            isinstance(rng['torch_cuda'],list) and len(rng['torch_cuda'])>0 and
            exact(rng['torch_cuda'],baseline_rng['torch_cuda']),
            'Actual GPU random state is absent or differs')
    require(not torch.cuda.is_initialized(),'Serialized GPU RNG comparison initialized CUDA')
    return dict(status='passed',baseline=str(baseline),baseline_sha256=sha(baseline),
        baseline_state_digest=left,restored_state_digest=right,state_excluded_fields=['infos'],
        all_state_fields_equal=True,compared_top_level_fields=sorted(lhs),exact_tensor_comparisons=tensors,
        per_field_state_digest={k:digest(lhs[k]) for k in sorted(lhs)},cuda_rng_equal=True,
        cuda_rng_tensor_count=len(rng['torch_cuda']),
        cuda_rng_tensors=[dict(dtype=str(x.dtype),shape=list(x.shape),sha256=digest(x)) for x in rng['torch_cuda']],
        RNG_fields_compared=sorted(rng),evidence_sha256=sha(evidence),inspector_sha256=sha(__file__),
        producer_state_digest_source_sha256=sha(producer),restored_packet_reused_from_complete_validator=True,
        restored_checkpoint_loads=1,baseline_checkpoint_loads=1,candidate_checkpoint_loads=0,
        NNforward_calls=0,optimizer_updates=0,no_simulation=True,CUDA_runtime_initialized=False,
        physical_state_rewound=False,scope='Exact persistent learning state at the pre-update snapshot after the sampled rollout; descriptive infos excluded. Not natural completion, training resume or physical acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    review=read(args.review.resolve(strict=True))
    checkpoint,source,baseline,evidence,guard,output=prepare(review)
    claim=checkpoint.with_suffix('.CPU_once.json')
    with claim.open('x') as stream:
        json.dump(dict(status='running',checkpoint=str(checkpoint),sha256=CHECKPOINT_SHA,
            inspector_sha256=sha(__file__)),stream,indent=2)
    try:
        import evaluate_flat_balance_policy as validator
        initial_files={str(p):sha(p) for p in [checkpoint,baseline,evidence,source.parent.parent/'run.json']}
        receipt=validator.validate_cpu(SimpleNamespace(checkpoint=checkpoint,source_root=source,validation_output=output),
            restored_state_inspector=lambda saved:inspect_restored(saved,baseline,evidence,guard,source))
        require(all(sha(name)==value for name,value in initial_files.items()),'An audited input changed during the CPU check')
        write(claim,dict(status='passed',checkpoint=str(checkpoint),sha256=CHECKPOINT_SHA,
            receipt=str(output),receipt_sha256=sha(output),inspector_sha256=sha(__file__)))
        print('[RESTORED_FLAT_CPU] '+json.dumps(dict(status=receipt['status'],receipt=str(output),
            finite_tensors=receipt['finite_tensors'],full_state_and_GPU_RNG_equal=True)),flush=True)
    except BaseException as error:
        write(output,dict(status='failed',cpu_only=True,no_simulation=True,
            checkpoint=str(checkpoint),sha256=CHECKPOINT_SHA,error=repr(error)))
        write(claim,dict(status='failed',checkpoint=str(checkpoint),error=repr(error),
            repeated_load_not_authorized=True))
        raise


if __name__=='__main__':
    main()
