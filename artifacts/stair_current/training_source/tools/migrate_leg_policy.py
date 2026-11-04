"""Create a separate 12-action warm-start checkpoint from a 29-action H2 policy.

This changes the optimization contract and is not an exact training continuation.
The source checkpoint is never modified. No policy optimization is performed.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def policy_class():
    path = ROOT / "source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py"
    spec = importlib.util.spec_from_file_location("h2_policy_for_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.H2ActorCriticOur


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains non-finite tensors")
        return 1
    if isinstance(value, dict):
        return sum(check_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(check_finite(item) for item in value)
    return 0


def migrate(checkpoint, output):
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    original_sha = digest(checkpoint)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
                "privileged_obs_norm_state_dict", "rnd_state_dict", "rnd_optimizer_state_dict", "iter"}
    if not required.issubset(blob):
        raise ValueError(f"Missing checkpoint states: {required - blob.keys()}")
    check_finite(blob)
    actor_critic = policy_class()
    old = actor_critic(1029, 866, 29, num_joints=29)
    new = actor_critic(1029, 866, 12, num_joints=29)
    old.load_state_dict(blob["model_state_dict"], strict=True)
    migrated = deepcopy(blob)
    sliced = {"std", "actor.actor.net.6.weight", "actor.actor.net.6.bias"}
    for name in sliced:
        if blob["model_state_dict"][name].shape[0] != 29:
            raise ValueError(f"Unexpected source action parameter: {name}")
        migrated["model_state_dict"][name] = blob["model_state_dict"][name][:12].clone()
    new.load_state_dict(migrated["model_state_dict"], strict=True)

    # PyTorch optimizer parameter ids are positional. Bind them to the validated
    # model's named-parameter order before slicing only the three action tensors.
    names = [name for name, _ in old.named_parameters()]
    ids = [index for group in migrated["optimizer_state_dict"]["param_groups"] for index in group["params"]]
    if len(names) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Optimizer parameter identity/order cannot be established")
    old_params, new_params = dict(old.named_parameters()), dict(new.named_parameters())
    for name, index in zip(names, ids):
        state = migrated["optimizer_state_dict"]["state"][index]
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            if value.shape != old_params[name].shape:
                raise ValueError(f"Unexpected optimizer state shape: {name}/{key}")
            if name in sliced:
                state[key] = value[:12].clone()
            if state[key].shape != new_params[name].shape:
                raise ValueError(f"Migrated optimizer state has wrong shape: {name}/{key}")
    optimizer = torch.optim.Adam(new.parameters())
    optimizer.load_state_dict(migrated["optimizer_state_dict"])
    generator = torch.Generator().manual_seed(20261029)
    observation = torch.randn((128, 1029), generator=generator)
    with torch.inference_mode():
        maximum_error = float((old.act_inference(observation)[:, :12]
                               - new.act_inference(observation)).abs().max())
    if maximum_error != 0.0:
        raise ValueError(f"Leg policy changed during parameter slicing: {maximum_error}")
    proof = {"purpose": "frozen-policy contract verification and optional future warm start",
             "exact_resume": False, "training_updates": 0,
             "source_checkpoint": str(checkpoint), "source_sha256": original_sha,
             "source_iteration": int(blob["iter"]), "source_actions": 29, "actions": 12,
             "observed_joints": 29, "actor_observations": 1029, "critic_observations": 866,
             "sliced_parameter_names": sorted(sliced), "maximum_leg_mean_error": maximum_error,
             "optimizer_state_shapes_valid": True,
             "tensor_count": check_finite(migrated), "all_tensors_finite": True,
             "retained_states": sorted(required - {"iter"})}
    infos = migrated.get("infos")
    if infos is not None and not isinstance(infos, dict):
        raise ValueError("Cannot preserve non-dictionary checkpoint infos")
    migrated["infos"] = dict(infos or {}, action_contract_migration=proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(migrated, stream)
    round_trip = torch.load(output, map_location="cpu", weights_only=True)
    new.load_state_dict(round_trip["model_state_dict"], strict=True)
    check_finite(round_trip)
    if digest(checkpoint) != original_sha:
        raise RuntimeError("Source checkpoint changed")
    proof.update(output_checkpoint=str(output), output_sha256=digest(output), source_unchanged=True)
    output.with_suffix(".validation.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.checkpoint, args.output), indent=2))
