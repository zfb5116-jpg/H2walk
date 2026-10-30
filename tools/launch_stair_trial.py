"""Launch one frozen1024-env stair trial with immutable identity and source."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--height', type=float, default=.15)
    parser.add_argument('--depth', type=float, default=.35)
    parser.add_argument('--steps', type=int, default=6)
    args = parser.parse_args()
    assert 0 < args.height <= .20 and .30 <= args.depth <= .40 and args.steps == 6
    pointer_path = ROOT / 'logs/launches/current_main_training.json'
    pointer = read(pointer_path)
    checkpoint = Path(pointer['final_checkpoint'])
    validation = read(checkpoint.with_suffix('.validation.json'))
    assert validation['status'] == 'passed' and digest(checkpoint) == validation['sha256']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
    base = ROOT / 'logs/stairs'
    base.mkdir(exist_ok=True, parents=True)
    campaign_path = base / 'campaign.json'
    if campaign_path.exists():
        campaign = read(campaign_path)
        active = campaign.get('active_case')
        if active:
            assert read(Path(active) / 'state.json')['status'] == 'finished', 'A stair trial is still active'
    else:
        campaign = {'objective': 'Ascend6 stairs and reach the upper landing in Isaac Sim',
                    'authorization': '请实现机器人上楼梯的功能',
                    'minimum_successful_envs': 922, 'num_envs': 1024, 'cases': [],
                    'preserve_network_structure': True, 'flat_checkpoint': str(checkpoint)}
    source_root = ROOT / 'source'
    frozen = checkpoint.parent / 'training_source/source'
    network = Path('CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks')
    for name in ('actor_critic_our.py', 'lidar_terrain_reconstructor.py', 'terrain_reconstructor.py', '__init__.py'):
        assert digest(source_root / network / name) == digest(frozen / network / name), 'Report before changing networks'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    case = base / f'{stamp}_{args.label}'
    case.mkdir(exist_ok=False)
    snapshot = case / 'source_snapshot/source'
    shutil.copytree(source_root, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    tools = case / 'tools_snapshot'
    tools.mkdir()
    for name in ('evaluate_main_walk.py', 'evaluate_main_walk.sh', 'manage_reference_trial.py', 'launch_stair_trial.py', 'stair_entry_evidence.py', 'stair_wrench_evidence.py'):
        shutil.copy2(ROOT / 'tools' / name, tools / name)
    source_manifest = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()}
    write(case / 'source_manifest.json', source_manifest)
    command = ['bash', str(ROOT / 'tools/evaluate_main_walk.sh'), '--headless', '--device', 'cuda:0',
               '--num_envs', '1024', '--checkpoint', str(checkpoint), '--source_root', str(snapshot),
               '--terrain', 'stairs', '--duration', '90', '--speed', '.3', '--seed', '42',
               '--stair_height', str(args.height), '--stair_depth', str(args.depth),
               '--stair_steps', str(args.steps), '--stair_approach', '1.4', '--early_reject_failures', '820',
               '--video', '--output', str(case / 'evaluation')]
    request = {'case': str(case), 'label': args.label, 'command': command, 'checkpoint': str(checkpoint),
               'checkpoint_sha256': validation['sha256'], 'source_snapshot': str(snapshot),
               'source_manifest': str(case / 'source_manifest.json'), 'network_structure_changed': False,
               'controller_changed_since_flat_training': True, 'training_updates': 0, 'diagnostic_only': False,
               'terrain': 'straight_ascending_stairway', 'riser_height_m': args.height, 'tread_depth_m': args.depth,
               'steps': args.steps, 'duration_s': 90, 'num_envs': 1024, 'minimum_successful_envs': 922,
               'log': str(case / 'evaluation.log'), 'output': str(case / 'evaluation'),
               'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'development_early_rejection': 'At820 failed first episodes, or when every survivor has irreversibly held and fewer than922 are on top, the entire trial is rejected early; passage requires full90s across1024 envs.',
               'acceptance': f'Every level has both feet validly landed/loaded, full first90s survival,{args.height * args.steps:.3f}m net rise±.1m,2s upper-platform dwell; all conditions in the same>=922 of1024 envs.'}
    write(case / 'request.json', request)
    with (case / 'launch.lock').open('x') as file:
        json.dump(request, file, ensure_ascii=False, indent=2)
    write(case / 'state.json', {'status': 'starting', 'request': str(case / 'request.json')})
    with (case / 'manager.log').open('xb') as stream:
        manager = subprocess.Popen(['python3', str(ROOT / 'tools/manage_reference_trial.py'), str(case)],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    fields = Path(f'/proc/{manager.pid}/stat').read_text().rsplit(')', 1)[1].split()
    state = read(case / 'state.json')
    state['manager'] = {'pid': manager.pid, 'start_ticks': int(fields[19]), 'pgid': os.getpgid(manager.pid)}
    write(case / 'state.json', state)
    campaign['cases'].append({'case': str(case), 'label': args.label, 'status': 'running', 'geometry': [args.height, args.depth, args.steps]})
    campaign.update(active_case=str(case), status='stair_trial_running')
    write(campaign_path, campaign)
    pointer.update(active_stair_campaign=str(campaign_path), active_stair_trial=str(case),
                   runtime_state_status='stair_development_running', stairs_acceptance_passed=False,
                   latest_authorization='请实现机器人上楼梯的功能')
    write(pointer_path, pointer)
    print(json.dumps({'case': str(case), 'manager': state['manager'], 'source_files': len(source_manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
