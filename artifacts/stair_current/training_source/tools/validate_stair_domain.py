"""One short Isaac integration check of stair episode validity, never acceptance.

Eighty explicitly marked synthetic root states test the real termination/reset
chain. The other944 environments retain their normal physical states. No PPO
updates occur and none of these samples is used for learning or acceptance.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 2 <= args.steps <= 200:
    parser.error("This diagnostic is bounded to2..200 control steps")
args.checkpoint = args.checkpoint.resolve(strict=True)
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT)
app = AppLauncher(args).app


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def config(entry):
    module, name = entry.split(":")
    return getattr(importlib.import_module(module), name)()


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.modules.normalizer import EmpiricalNormalization
    sys.path.insert(0, str(ROOT/"source/CtrlLab/CtrlLab/tasks/manager_based"))
    import h2_task
    from h2_task.networks import H2ActorCriticOur

    receipt = json.loads(args.checkpoint.with_suffix(".validation.json").read_text())
    if receipt["status"] != "passed" or receipt["sha256"] != digest(args.checkpoint):
        raise RuntimeError("The diagnostic requires an already validated, unchanged PPO checkpoint")
    before = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT/"source").rglob("*")
              if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts}
    spec = gym.spec("Unitree-H2-Stair-Ascent-Policy")
    cfg, agent = config(spec.kwargs["env_cfg_entry_point"]), config(spec.kwargs["rsl_rl_cfg_entry_point"])
    cfg.scene.num_envs, cfg.seed = 1024, 42
    cfg.sim.device = args.device or "cuda:0"
    cfg.observations.policy.enable_corruption = False
    cfg.io_descriptors_output_dir = str(args.output)
    cfg.curriculum.terrain_levels = None
    dump_yaml(str(args.output/"environment.yaml"), cfg)
    env = gym.make(spec.id, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    raw, captured = env.unwrapped, {}
    try:
        if not {"stair_below_floor", "stair_out_of_tile"}.issubset(raw.termination_manager.active_terms):
            raise RuntimeError("New physical-domain termination terms are missing")
        obs, extras = wrapped.get_observations()
        if obs.shape != (1024,1029) or extras["observations"]["critic"].shape != (1024,866):
            raise RuntimeError("Unexpected diagnostic observation layout")
        policy_cfg = agent.policy.to_dict()
        policy_cfg.pop("class_name")
        actor = H2ActorCriticOur(1029,866,12,**policy_cfg).to(raw.device).eval()
        norm = EmpiricalNormalization([1029]).to(raw.device).eval()
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        actor.load_state_dict(saved["model_state_dict"],strict=True)
        norm.load_state_dict(saved["obs_norm_state_dict"],strict=True)
        actor.requires_grad_(False)
        norm.requires_grad_(False)
        robot = raw.scene["robot"]
        original_reset = raw._reset_idx

        def capture(self, env_ids):
            ids = torch.as_tensor(env_ids,device=raw.device,dtype=torch.long)
            selected = ids[ids<80]
            records = {}
            for i in selected.tolist():
                if i in captured:
                    continue
                records[i] = dict(root_before_reset=robot.data.root_pos_w[i].cpu().tolist(),
                    below_floor=bool(raw.termination_manager.get_term("stair_below_floor")[i]),
                    out_of_tile=bool(raw.termination_manager.get_term("stair_out_of_tile")[i]),
                    actual_timeout=bool(raw.termination_manager.time_outs[i]),
                    terminated=bool(raw.termination_manager.terminated[i]),
                    control_counter=int(raw.common_step_counter))
            original_reset(env_ids)
            for i, record in records.items():
                for field in ("ever_invalid","ever_out_of_tile","ever_below_floor",
                              "physical_duration_s","eligible_duration_s"):
                    value=raw.stair_progress.last_episode[field][i]
                    record[field]=float(value)
                record["root_after_reset"]=robot.data.root_pos_w[i].cpu().tolist()
                captured[i]=record
        with torch.inference_mode():
            obs,_,_,_=wrapped.step(actor.act_inference(norm(obs)))
            raw._reset_idx=types.MethodType(capture,raw)
            start=robot.data.root_pos_w.clone()
            original_origins=raw.scene.terrain.env_origins.clone()
            xmin,xmax,ymin,ymax=raw._stair_domain_bounds_local
            ids=torch.arange(80,device=raw.device)
            state=robot.data.root_state_w[ids].clone()
            state[:,7:13]=0
            state[:16,2]=original_origins[:16,2]-5.0
            state[16:32,0]=original_origins[16:32,0]+xmin-.5
            state[32:48,0]=original_origins[32:48,0]+xmax+.5
            state[48:64,1]=original_origins[48:64,1]+ymin-.5
            state[64:80,1]=original_origins[64:80,1]+ymax+.5
            injected=state.clone()
            robot.write_root_state_to_sim(state,env_ids=ids)
            obs,_,done,_=wrapped.step(actor.act_inference(norm(obs)))
            if not bool(done[ids].all()) or len(captured)!=80:
                raise RuntimeError("Some injected invalid states survived the real termination/reset chain")
            for i,record in captured.items():
                required="below_floor" if i<16 else "out_of_tile"
                if not record[required] or not record["ever_invalid"] or record["actual_timeout"]:
                    raise RuntimeError(f"Incorrect physical-domain failure label for diagnostic env{i}")
                if record["eligible_duration_s"]!=0:
                    raise RuntimeError("An invalid episode retained survival/stability-improvement credit")
            if bool(raw.termination_manager.get_term("stair_below_floor")[80:].any()) or bool(
                    raw.termination_manager.get_term("stair_out_of_tile")[80:].any()):
                raise RuntimeError("An untouched in-domain root produced a false domain failure")
            for _ in range(args.steps-2):
                obs,_,_,_=wrapped.step(actor.act_inference(norm(obs)))
        np.savez_compressed(args.output/"synthetic_injection.npz",env_ids=ids.cpu().numpy(),
            original_root_xyz=start.cpu().numpy(),origins=original_origins.cpu().numpy(),
            injected_root_state=injected.cpu().numpy())
        write(args.output/"termination_records.json",captured)
        after={name:digest(ROOT/name) for name in before}
        if before!=after or digest(args.checkpoint)!=receipt["sha256"]:
            raise RuntimeError("Source or checkpoint changed during the validity integration check")
        for module, expected in ((actor,saved["model_state_dict"]), (norm,saved["obs_norm_state_dict"])):
            if any(not torch.equal(value.cpu(),expected[key]) for key,value in module.state_dict().items()):
                raise RuntimeError("The frozen diagnostic policy or normalizer changed")
        write(args.output/"source_manifest.json",before)
        report=dict(status="passed",diagnostic_only=True,formal_acceptance_passed=False,
            num_envs=1024,synthetically_injected_envs=80,untouched_envs=944,
            control_steps=args.steps,below_floor_group=16,outside_tile_groups=64,
            every_injected_state_terminated_and_reset=True,
            invalid_episode_survival_credit_zero=True,normal_domain_false_positives=0,
            checkpoint=str(args.checkpoint),checkpoint_sha256=receipt["sha256"],
            source_root=str(ROOT/"source"),source_unchanged=True,policy_unchanged=True,
            network_dimensions=[1029,866,73,12],learning_updates=0,
            domain_bounds_local=list(raw._stair_domain_bounds_local),
            limits="Synthetic domain-failure integration only; does not establish walking, survival or stair ascent.")
        write(args.output/"result.json",report)
        print("[STAIR DOMAIN DIAGNOSTIC] "+json.dumps(report),flush=True)
    finally:
        wrapped.close()


if __name__=="__main__":
    try:
        main()
    except BaseException as error:
        write(args.output/"result.json",dict(status="failed",diagnostic_only=True,
            formal_acceptance_passed=False,error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        app.close()
