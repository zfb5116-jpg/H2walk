"""Reject incompatible reward, robot and policy interfaces before loading PPO."""

import json
from pathlib import Path


def validate_checkpoint_contract(checkpoint, env, *, allow_direction_change=False):
    sidecar = Path(checkpoint).with_suffix(".curriculum.json")
    if not sidecar.exists():
        raise ValueError("Checkpoint needs a versioned .curriculum.json training contract; start a new 15-action run")
    saved = json.loads(sidecar.read_text(encoding="utf-8")).get("training_contract", {})
    current = (env.env_method("get_training_contract")[0] if hasattr(env, "env_method")
               else env.unwrapped.get_training_contract())
    if allow_direction_change:
        saved, current = dict(saved), dict(current)
        saved.pop("stair_direction", None)
        current.pop("stair_direction", None)
    if saved != current:
        raise ValueError("Incompatible checkpoint: expected WM-LOCO 86 observations / 15 leg+waist actions "
                         "and the same robot/reward settings. Old 56/12 checkpoints cannot resume.")
