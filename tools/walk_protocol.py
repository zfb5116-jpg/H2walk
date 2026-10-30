"""Fixed-scale walking checkpoint contract, independent of Isaac and torch.

Only the dedicated entry uses this contract. It deliberately cannot migrate
unknown/normalized baseline checkpoints by discarding their statistics. The
marker records the producer's assertion; it is not cryptographic attestation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, TypeGuard


STAGES = ("bootstrap", "shallow", "stairs_08", "stairs_12", "stairs_16", "stairs_20")
PROTOCOL_ID = "h2_walk_fixed_scales_v2"
REWARD_VERSION = "h2_walk_rewards_v1"
FORBIDDEN_STATES = {
    "obs_norm_state_dict",
    "privileged_obs_norm_state_dict",
    "rnd_state_dict",
    "rnd_optimizer_state_dict",
}
COMPATIBILITY_KEYS = (
    "id",
    "layout",
    "actor_observations",
    "critic_observations",
    "actions",
    "empirical_normalization",
    "rnd",
    "cross_time_mode",
    "max_time_difference",
    "reward_mode",
    "reward_version",
    "configuration_sha256",
)


def configuration_fingerprint(*components: Any) -> str:
    """Hash resolved observation/action/policy semantics, preserving term order.

    Config callables use import identities, never process-specific repr().
    Unknown values are rejected rather than silently omitted. Mapping order is
    retained because observation concatenation and selected-joint order matter.
    """

    def primitive(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise ValueError(
                    "Configuration fingerprint requires string mapping keys"
                )
            return {"mapping": [[key, primitive(item)] for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return [primitive(item) for item in value]
        if isinstance(value, slice):
            return {
                "slice": [
                    primitive(value.start),
                    primitive(value.stop),
                    primitive(value.step),
                ]
            }
        if callable(value):
            module, name = getattr(value, "__module__", None), getattr(
                value, "__qualname__", None
            )
            if isinstance(module, str) and isinstance(name, str) and "<" not in name:
                return {"callable": module + ":" + name}
            raise ValueError(
                "Configuration callable must have a stable import identity"
            )
        if callable(getattr(value, "to_dict", None)):
            return primitive(value.to_dict())
        raise ValueError(
            "Unsupported configuration value for protocol fingerprint: "
            + type(value).__name__
        )

    encoded = json.dumps(
        primitive(components), separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error(message: str) -> ValueError:
    return ValueError(
        message + "; use a verified same-protocol checkpoint or an explicit fresh run"
    )


def _integer(value: Any, minimum: int = 0) -> TypeGuard[int]:
    return type(value) is int and value >= minimum


def _finite_number(value: Any, minimum: float = 0.0) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def validate_protocol(value: Any) -> None:
    """Validate marker structure without reading or applying a state dictionary."""
    if not isinstance(value, Mapping):
        raise _error("Missing walking protocol marker")
    fixed = {
        "id": PROTOCOL_ID,
        "layout": "term_major_h5_gt561_29_v1",
        "actor_observations": 1029,
        "critic_observations": 866,
        "actions": 29,
        "empirical_normalization": False,
        "rnd": None,
    }
    if any(
        type(value.get(key)) is not type(expected) or value[key] != expected
        for key, expected in fixed.items()
    ):
        raise _error("Incompatible walking observation/normalization/RND protocol")
    if value.get("reward_mode") not in ("legacy", "alternating_v1") or value.get("reward_version") != REWARD_VERSION:
        raise _error("Missing or incompatible walking reward protocol")
    if value.get("stage") not in STAGES or value.get("cross_time_mode") not in (
        "legacy",
        "bounded",
    ):
        raise _error("Unknown walking stage or timing reward mode")
    bound = value.get("max_time_difference")
    if not _finite_number(bound) or bound <= 0:
        raise _error("Invalid timing reward bound")
    if not isinstance(value.get("configuration_sha256"), str) or not re.fullmatch(
        "[0-9a-f]{64}", value["configuration_sha256"]
    ):
        raise _error("Missing resolved observation/action/policy configuration hash")
    source = value.get("source")
    if not isinstance(source, Mapping) or not re.fullmatch(
        "[0-9a-f]{64}", str(source.get("source_sha256", ""))
    ):
        raise _error("Missing producer source hash")
    commit = source.get("git_commit")
    if commit is not None and (
        not isinstance(commit, str) or not re.fullmatch("[0-9a-f]{40}", commit)
    ):
        raise _error("Invalid producer Git SHA")


def make_protocol(
    stage: str,
    cross_time_mode: str,
    max_time_difference: float,
    source: Mapping[str, Any],
    *,
    configuration_sha256: str,
    reward_mode: str = "alternating_v1",
) -> dict[str, Any]:
    value = {
        "id": PROTOCOL_ID,
        "layout": "term_major_h5_gt561_29_v1",
        "actor_observations": 1029,
        "critic_observations": 866,
        "actions": 29,
        "empirical_normalization": False,
        "rnd": None,
        "stage": stage,
        "cross_time_mode": cross_time_mode,
        "max_time_difference": max_time_difference,
        "reward_mode": reward_mode,
        "reward_version": REWARD_VERSION,
        "source": deepcopy(dict(source)),
        "configuration_sha256": configuration_sha256,
    }
    validate_protocol(value)
    return value


def validate_runner_config(cfg: Mapping[str, Any]) -> None:
    if (
        cfg.get("empirical_normalization") is not False
        or cfg.get("algorithm", {}).get("class_name") != "PPO"
        or cfg.get("algorithm", {}).get("rnd_cfg") is not None
        or cfg.get("policy", {}).get("class_name") != "H2ActorCriticOur"
    ):
        raise _error(
            "The guarded entry requires fixed-scale H2ActorCriticOur PPO without RND"
        )


def _validate_runner(runner: Any) -> None:
    if runner.empirical_normalization is not False or runner.alg.rnd is not None:
        raise _error("Runtime runner normalization/RND disagrees with walking protocol")


def validate_checkpoint(
    saved: Any,
    expected_protocol: Mapping[str, Any],
    *,
    expected_completed_iterations: int | None = None,
) -> int:
    """Return completed iterations after all protocol checks, before state mutation."""
    validate_protocol(expected_protocol)
    if not isinstance(saved, Mapping) or FORBIDDEN_STATES.intersection(saved):
        raise _error("Normalized/RND or malformed checkpoint is incompatible")
    if not _integer(saved.get("iter")):
        raise _error("Checkpoint iter must be the last completed zero-based iteration")
    for key in ("model_state_dict", "optimizer_state_dict"):
        if not isinstance(saved.get(key), Mapping):
            raise _error("Missing checkpoint " + key)
    infos = saved.get("infos")
    if not isinstance(infos, Mapping):
        raise _error("Missing walking checkpoint infos")
    marker = infos.get("h2_walk_protocol")
    if not isinstance(marker, Mapping):
        raise _error("Missing walking protocol marker")
    validate_protocol(marker)
    if any(marker[key] != expected_protocol[key] for key in COMPATIBILITY_KEYS):
        raise _error("Checkpoint input or walking reward protocol differs from this run")
    completed = saved["iter"] + 1
    progress = infos.get("h2_walk_progress")
    if (
        not isinstance(progress, Mapping)
        or not _integer(progress.get("completed_iterations"), 1)
        or progress["completed_iterations"] != completed
        or not _integer(progress.get("total_timesteps"))
        or not _finite_number(progress.get("total_time"))
    ):
        raise _error("Checkpoint completed-iteration/progress metadata is inconsistent")
    if expected_completed_iterations is not None and (
        not _integer(expected_completed_iterations, 1)
        or completed != expected_completed_iterations
    ):
        raise _error(
            f"Expected {expected_completed_iterations} completed iterations, checkpoint has {completed}"
        )
    return completed


def restore_checkpoint(
    runner: Any,
    saved: Mapping[str, Any],
    expected_protocol: Mapping[str, Any],
    *,
    load_optimizer: bool = True,
    expected_completed_iterations: int | None = None,
) -> Mapping[str, Any]:
    """Apply an already-read compatible dictionary exactly once (RSL-RL 2.3.3 API)."""
    completed = validate_checkpoint(
        saved,
        expected_protocol,
        expected_completed_iterations=expected_completed_iterations,
    )
    _validate_runner(runner)
    runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
    if load_optimizer:
        runner.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
        # RSL's adaptive learning-rate scalar must agree with restored groups.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
    runner.current_learning_iteration = completed
    progress = saved["infos"]["h2_walk_progress"]
    runner.tot_timesteps = progress["total_timesteps"]
    runner.tot_time = progress["total_time"]
    return saved["infos"]


def guarded_runner_class(base_runner: type, protocol: Mapping[str, Any]) -> type:
    """Wrap every autosave; never fall back to the base runner's unsafe load path."""
    validate_protocol(protocol)
    producer = deepcopy(dict(protocol))

    class WalkingProtocolRunner(base_runner):
        def __init__(
            self, env: Any, train_cfg: Mapping[str, Any], **kwargs: Any
        ) -> None:
            validate_runner_config(train_cfg)
            self.walk_protocol = deepcopy(producer)
            super().__init__(env, train_cfg, **kwargs)

        def save(self, path: str, infos: Mapping[str, Any] | None = None) -> None:
            _validate_runner(self)
            validate_protocol(self.walk_protocol)
            merged = dict(infos) if infos is not None else {}
            merged["h2_walk_protocol"] = deepcopy(self.walk_protocol)
            merged["h2_walk_progress"] = {
                "completed_iterations": self.current_learning_iteration + 1,
                "total_timesteps": self.tot_timesteps,
                "total_time": self.tot_time,
            }
            super().save(path, infos=merged)

        def load(self, path: str, load_optimizer: bool = True) -> Mapping[str, Any]:
            # This future user-invoked I/O path is never exercised by CPU tests.
            import torch

            saved = torch.load(path, map_location=self.device, weights_only=True)
            return restore_checkpoint(
                self, saved, self.walk_protocol, load_optimizer=load_optimizer
            )

    return WalkingProtocolRunner
