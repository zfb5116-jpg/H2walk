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
spec=importlib.util.spec_from_file_location('h2_preserved_direct_physical',Path(__file__).with_name('evaluate_direct_initialization.py'))
physical=importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical)
physical.TASK=TASK
physical.__doc__=__doc__
require,sha,write,source_hashes=physical.require,physical.sha,physical.write,physical.source_hashes


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


def validate_cpu(args):
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
    require(saved['format']=='h2_guarded_flat_ppo_v1','Unrecognized PPO checkpoint format')
    require('optimizer_state_dict' not in saved,'A shared optimizer must not masquerade as the split optimizers')
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
    for name,digest in training_tools.items():
        require(sha(args.source_root.parent/name)==digest,'Recorded producer tool changed: '+name)
    agent=yaml.safe_load((run_dir/'params/agent.yaml').read_text())
    algo=agent['algorithm']
    require(algo['class_name']=='H2GuardedPPO' and algo['schedule']=='fixed'
            and algo['num_learning_epochs']==5 and algo['num_mini_batches']==4,'Saved algorithm contract differs')
    for key,value in (('actor_learning_rate',1e-5),('critic_learning_rate',.001),
                      ('critic_only_iterations',40),('actor_max_grad_norm',1.),('critic_max_grad_norm',1.)):
        require(float(algo[key])==value,'Saved split optimizer contract differs: '+key)
    require(algo['rnd_cfg']['weight']==0. and algo['entropy_coef']==1e-4,'Novelty/entropy setting differs')
    policy_cfg=dict(agent['policy'])
    require(policy_cfg.pop('class_name')=='H2ActorCriticOur','Original H2 network class changed')
    cls=physical.load_module(args.source_root/PACKAGE/'networks/actor_critic_our.py','flat_cpu_original_network').H2ActorCriticOur
    model=cls(1029,866,12,**policy_cfg).eval()
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
    require({name:int(gs[name+'_optimizer_steps']) for name in expected}==expected,
            'Committed split/RND update budget differs')
    require(gs['critic_only_iterations']==40 and gs['actor_updates_accepted']==max(0,count-40)
            and gs['critic_warmup_complete']==(count>=40),'Critic warm/accepted actor counters disagree')
    require(gs['actor_learning_rate']==1e-5 and gs['critic_learning_rate']==.001
            and gs['normalizers_frozen'] is True and
            gs['normalizer_counts']=={'actor':24576000,'critic':24576000},'Frozen normalization/fixed LR runtime contract differs')
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
    initial_sha='ce822132385de7b9c70315087bbcc4da1739d0da8460cf656c39e14aa1424688'
    require(sha(initial)==run['initial_policy_sha256']==initial_sha,'Initial physically evaluated policy changed')
    require(gs['initial_policy_sha256']==initial_sha,'Saved guarded state initial-policy binding differs')
    initial_saved=torch.load(initial,map_location='cpu',weights_only=True)
    for key,width in (('obs_norm_state_dict',1029),('privileged_obs_norm_state_dict',866)):
        norm=saved[key]
        require(set(norm)=={'_mean','_var','_std','count'},'Normalizer state layout differs')
        require(all(tuple(norm[n].shape)==(1,width) for n in ('_mean','_var','_std')),'Normalizer shape differs')
        require(norm['count'].numel()==1 and int(norm['count'])==24576000
                and bool((norm['_std']>=0).all()),'Frozen inherited normalization is invalid')
        require(norm.keys()==initial_saved[key].keys() and
                all(torch.equal(value,initial_saved[key][name]) for name,value in norm.items()),
                'Frozen own normalizer changed during the new PPO')
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
    protocol=run['guarded_flat_learning']
    plan_path=Path(protocol['plan_path']).resolve(strict=True)
    require(sha(plan_path)==protocol['plan_sha256'],'Authorized flat plan no longer matches producer receipt')
    plan=json.loads(plan_path.read_text())
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
        uncommitted_rollout=gs['uncommitted_rollout'],rnd_rollout_update_counter=int(gs['rnd_update_counter']),
        source_root=str(args.source_root),source_sha256=hashes,producer_run=str(run_path),producer_run_sha256=sha(run_path),
        producer_training_tool_sha256=training_tools,
        external_dependency_sha256=dependencies,normalizers_frozen_and_parent_bitwise_equal=True,
        initial_policy=str(initial),initial_policy_sha256=initial_sha,
        plan_path=str(plan_path),plan_sha256=protocol['plan_sha256'],
        optimizer_path=str(args.checkpoint),optimizer_sha256=sha(args.checkpoint),
        optimizer_path_semantics='All actual split PPO/RND optimizer tensors are inside this same complete checkpoint.',
        validator_sha256=sha(Path(__file__)),physical_validator_sha256=sha(Path(physical.__file__)),
        rnd_semantics='Weight0 means no intrinsic reward; actual fresh predictor Adam and normalization continue and are checked.',
        integrity_is_not_walking_acceptance=True)
    write(args.validation_output,receipt)
    print('[FLAT_BALANCE_CPU] '+json.dumps({key:receipt[key] for key in (
        'status','cpu_only','checkpoint','sha256','checkpoint_iteration','committed_updates',
        'finite_tensors','flat_balance_ready','source_root')}),flush=True)
    return receipt


def prepare_physical(args):
    receipt=json.loads(args.validation_output.read_text())
    require(receipt.get('status')=='passed' and receipt.get('cpu_only') is True,'Run complete split-PPO CPU validation first')
    require(receipt['checkpoint_format']=='h2_guarded_flat_ppo_v1','Wrong CPU receipt format')
    require(receipt['sha256']==sha(args.checkpoint) and receipt['source_root']==str(args.source_root)
            and receipt['source_sha256']==source_hashes(args.source_root),'Checkpoint/source changed after CPU validation')
    require(receipt['validator_sha256']==sha(Path(__file__)) and
            receipt['physical_validator_sha256']==sha(Path(physical.__file__)),'Validator changed after CPU receipt')
    require(sha(Path(receipt['producer_run']))==receipt['producer_run_sha256'],'Producer run evidence changed')
    require(sha(Path(receipt['plan_path']))==receipt['plan_sha256'],'Producer plan changed after CPU validation')
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
                 'evaluate_direct_initialization.py','evaluate_direct_initialization.sh'):
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
        num_envs=1024,duration_s=20.,speed_m_s=.3,seed=42,network_dimensions=[1029,866,73,12],
        training_updates=0,teacher_constructed=False,teacher_interventions=0,initialization_ready=False,
        flat_balance_ready=False,formal_acceptance_passed=False,stairs_acceptance_passed=False,
        validator_sha256=sha(Path(__file__)),checkpoint_format=receipt['checkpoint_format'],
        optimizer_loaded_for_control=False,physical_procedure='Unmodified evaluate_direct_initialization.evaluate with new independent TASK only')
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
