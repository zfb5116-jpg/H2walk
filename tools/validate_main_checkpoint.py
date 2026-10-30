"""Strict CPU validation of an H2 run's model, normalization, RND and optimizer states."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.modules.rnd import RandomNetworkDistillation


def validate_optimizer(optimizer, state):
    optimizer.load_state_dict(state)
    for parameter, values in optimizer.state.items():
        for name, value in values.items():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                raise ValueError(f"Optimizer {name} shape {tuple(value.shape)} != parameter {tuple(parameter.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.checkpoint.resolve(strict=True)
    run = path.parent
    metadata = json.loads((run / "run.json").read_text())
    cfg = yaml.safe_load((run / "params/agent.yaml").read_text())
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {"iter", "model_state_dict", "optimizer_state_dict", "obs_norm_state_dict",
                "privileged_obs_norm_state_dict", "rnd_state_dict", "rnd_optimizer_state_dict"}
    if not required.issubset(saved):
        raise ValueError(f"Missing checkpoint states: {sorted(required - saved.keys())}")
    count = 0

    def finite(value):
        nonlocal count
        if torch.is_tensor(value):
            count += 1
            if not torch.isfinite(value).all():
                raise ValueError("Checkpoint contains a non-finite tensor")
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite(child)

    finite(saved)
    actor_path = run / "training_source/source/CtrlLab/CtrlLab/tasks/manager_based/h2_task/networks/actor_critic_our.py"
    spec = importlib.util.spec_from_file_location("checkpoint_actor", actor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_cfg = dict(cfg["policy"])
    policy_cfg.pop("class_name")
    model = module.H2ActorCriticOur(1029, 866, metadata["num_actions"], **policy_cfg)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if not (model.std > 0).all():
        raise ValueError("Policy noise must be positive")
    validate_optimizer(torch.optim.Adam(model.parameters()), saved["optimizer_state_dict"])
    normalizers = {}
    for label, key, width in [("actor", "obs_norm_state_dict", 1029),
                              ("critic", "privileged_obs_norm_state_dict", 866)]:
        norm = EmpiricalNormalization([width])
        norm.load_state_dict(saved[key], strict=True)
        normalizers[label] = {k: float(v) for k, v in norm.state_dict().items() if "count" in k}
    rnd_cfg = dict(cfg["algorithm"]["rnd_cfg"])
    rnd_cfg.pop("learning_rate", None)
    rnd = RandomNetworkDistillation(num_states=73, device="cpu", **rnd_cfg)
    rnd.load_state_dict(saved["rnd_state_dict"], strict=True)
    validate_optimizer(torch.optim.Adam(rnd.predictor.parameters()), saved["rnd_optimizer_state_dict"])
    result = {"status": "passed", "checkpoint": str(path), "iteration": int(saved["iter"]),
              "completed_iterations": int(saved["iter"]) + 1, "size": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "map_location": "cpu",
              "weights_only": True, "tensor_count": count, "all_tensors_finite": True,
              "model_strict_load": True, "optimizer_state_shapes_valid": True,
              "normalizers_strict_load": True, "normalizer_counts": normalizers,
              "rnd_strict_load": True, "rnd_optimizer_state_shapes_valid": True,
              "actor_dim": 1029, "critic_dim": 866, "rnd_dim": 73, "actions": metadata["num_actions"]}
    output = args.output or path.with_suffix(".validation.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
