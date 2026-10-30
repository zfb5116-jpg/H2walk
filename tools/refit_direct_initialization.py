"""One frozen-data actor calibration; no simulator, teacher, PPO, or readiness claim.

Importing this file only loads the pure NumPy sampler/budget utilities. Torch,
the original actor, and its normalizer are loaded only by the explicit CLI.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

KIND = "offline_direct_imitation_refit_v1"
PACKAGE = Path("CtrlLab/CtrlLab/tasks/manager_based/h2_task")
UPDATES, PARENT_UPDATES, BATCH = 3200, 3200, 4096
START_LR, END_LR, MAX_GRAD = 1e-4, 1e-5, 1.0
HOST_LIMIT, DISK_LIMIT, GPU_LIMIT = int(1.5 * 2**30), 512 * 2**20, 11 * 2**30
EXPECTED_PHASES = (0, 1, 2, 3)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    digest = hashlib.sha256()
    flat = array.reshape(-1)
    chunk = max(1, 4 * 2**20 // array.dtype.itemsize)
    for begin in range(0, flat.size, chunk):
        digest.update(flat[begin:begin+chunk].tobytes(order="C"))
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(root):
    return {"source/" + str(path.relative_to(root)): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def learning_rate(index):
    require(0 <= index < UPDATES, "Calibration LR index outside its fixed budget")
    return END_LR + (START_LR-END_LR) * .5 * (1.0 + math.cos(math.pi * index / (UPDATES-1)))


def occupied_indices(metadata):
    capacities = np.asarray(metadata["capacities"], dtype=np.int64)
    occupied = np.asarray(metadata["occupied"], dtype=np.int64)
    require(np.all(np.asarray(metadata["closed"], dtype=bool)), "Source collection is not fully frozen")
    require(np.all((occupied >= 0) & (occupied <= capacities)), "Invalid frozen occupancy")
    offsets = np.r_[0, np.cumsum(capacities)[:-1]]
    return np.concatenate([begin + np.arange(count, dtype=np.int64)
                           for begin, count in zip(offsets, occupied)])


class FrozenDataset:
    """Read-only mappings; every old array is content-verified at open and close."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha = sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        require(self.manifest.get("immutable_after_collection") is True,
                "Dataset lacks its immutable collection receipt")
        require(self.manifest.get("status") == "completed", "Parent dataset collection is incomplete")
        self.maps = {split: {} for split in ("train", "heldout")}
        self.files = {}
        for split in self.maps:
            for name, spec in self.manifest[split]["files"].items():
                path = self.root / split / (name + ".npy")
                require(path.resolve(strict=True) == Path(spec["path"]).resolve(strict=True),
                        "Frozen dataset path binding changed")
                self.files[split + "/" + name] = (path, spec)
        self.files["parent_fit_indices"] = (self.root / "fit_indices.npy", self.manifest["fit_indices"])
        self.verifications = []

    def verify(self, label):
        require(sha(self.manifest_path) == self.manifest_sha, "Parent manifest changed")
        verified = {}
        for name, (path, spec) in self.files.items():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                require(list(array.shape) == spec["shape"] and str(array.dtype) == spec["dtype"],
                        "Parent dataset shape/dtype changed: " + name)
                digest = array_sha(array)
                require(digest == spec["array_sha256"], "Parent dataset bytes changed: " + name)
                verified[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                      bytes=int(array.nbytes), array_sha256=digest)
            finally:
                array._mmap.close()
        point = dict(label=label, checked_at_epoch=time.time(), manifest_sha256=self.manifest_sha,
                     files=verified)
        self.verifications.append(point)
        return point

    def mapping(self, split, name):
        if name not in self.maps[split]:
            self.maps[split][name] = np.load(self.root/split/(name+".npy"), mmap_mode="r", allow_pickle=False)
        return self.maps[split][name]

    def get(self, split, indices, fields):
        return {name: np.asarray(self.mapping(split, name)[indices]).copy() for name in fields}


class GlobalPhaseSampler:
    """No collection-round quotas; only eligible TRAIN UIDs enter this sampler."""
    def __init__(self, phase, uid, valid, clip, env_id, occupied, seed=42, batch_size=BATCH):
        self.phase, self.uid = phase, uid
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        occupied = np.asarray(occupied, dtype=np.int64)
        require(np.all(env_id[occupied] % 8 != 0), "Heldout lane entered the TRAIN source")
        require(len(np.unique(uid[occupied])) == len(occupied), "TRAIN UIDs are duplicated")
        eligible = valid[occupied] & ~clip[occupied].any(axis=1)
        self.eligible = occupied[eligible]
        self.phases = np.unique(phase[self.eligible])
        require(tuple(int(p) for p in self.phases) == EXPECTED_PHASES,
                "Eligible TRAIN phase set differs from the approved four phases")
        self.pools = [self.eligible[phase[self.eligible] == p] for p in self.phases]
        self.counts = np.asarray([len(pool) for pool in self.pools], dtype=np.int64)
        require(np.all(self.counts > 0), "An eligible TRAIN phase is empty")
        self.probabilities = np.sqrt(self.counts.astype(np.float64))
        self.probabilities /= self.probabilities.sum()
        self.draws = np.zeros(len(self.phases), dtype=np.int64)
        self.updates = 0

    def sample(self):
        counts = self.rng.multinomial(self.batch_size, self.probabilities)
        require(np.all(counts <= self.counts),
                "Phase allocation exceeds its distinct UID pool; do not replace or shrink the batch")
        rows = np.concatenate([self.rng.choice(pool, int(count), replace=False)
                               for pool, count in zip(self.pools, counts)])
        self.rng.shuffle(rows)
        require(len(rows) == self.batch_size and len(np.unique(self.uid[rows])) == self.batch_size,
                "A calibration batch duplicated an eligible UID")
        self.draws += counts
        self.updates += 1
        return rows, counts

    def metadata(self):
        return dict(kind="global_phase_sqrt_unique_uniform_without_replacement_per_batch",
                    phases=[int(p) for p in self.phases], eligible_unique_counts=self.counts.tolist(),
                    phase_probabilities=self.probabilities.tolist(), actual_phase_draws=self.draws.tolist(),
                    updates=self.updates, batch_size=self.batch_size, rng_state=self.rng.bit_generator.state,
                    eligible_rows=int(len(self.eligible)), holdout_fit_rows=0, round_first_quotas=False)


def resource_budget(dataset_manifest, parent_candidate_bytes, parent_optimizer_bytes, source_bytes):
    mapped_inputs = sum(int(spec.get("bytes", np.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize))
                        for split in ("train", "heldout")
                        for spec in dataset_manifest[split]["files"].values())
    parent_fit = dataset_manifest["fit_indices"]
    parent_fit_bytes = int(np.prod(parent_fit["shape"])) * np.dtype(parent_fit["dtype"]).itemsize
    fit_bytes = UPDATES * BATCH * 8 * 2  # explicit physical row indices AND stable UIDs
    heldout_rows = int(dataset_manifest["heldout"]["capacity"])
    output_predictions = 2 * heldout_rows * 4 * 12 * 4
    host = mapped_inputs + parent_fit_bytes + fit_bytes + output_predictions + 176 * 2**20
    disk = fit_bytes + output_predictions + 4 * (parent_candidate_bytes + parent_optimizer_bytes)
    disk += source_bytes + 24 * 2**20
    return dict(mapped_input_array_bytes=mapped_inputs, parent_fit_hash_mapping_bytes=parent_fit_bytes,
                new_fit_index_and_uid_bytes=fit_bytes, paired_heldout_prediction_bytes=output_predictions,
                host_staging_model_scratch_reserve_bytes=176*2**20, host_reserved_bytes=host,
                host_limit_bytes=HOST_LIMIT, disk_reserved_bytes=disk, disk_limit_bytes=DISK_LIMIT,
                gpu_limit_bytes=GPU_LIMIT, GPU_replay_bytes=0,
                scope="Extra read-only data mappings, output mappings, CPU actor/Adam, and bounded staging; not a process-RSS cap")


def validate_plan(plan):
    p = plan["protocol"]
    exact = dict(kind=KIND, physics_controls=0, additional_actor_updates=UPDATES,
                 parent_actor_updates=PARENT_UPDATES, cumulative_actor_updates=PARENT_UPDATES+UPDATES,
                 batch_size=BATCH, learning_rate_start=START_LR, learning_rate_end=END_LR,
                 max_gradient_norm=MAX_GRAD, GPU_soft_budget_GiB=11,
                 extra_dataset_and_staging_host_limit_GiB=1.5, new_output_disk_limit_MiB=512,
                 max_wall_seconds=1800, no_GPU_replay=True, no_hyperparameter_scan=True,
                 no_automatic_PPO=True)
    require(all(p.get(key) == value for key, value in exact.items()), "Approved calibration protocol changed")
    require(plan.get("network_dimensions") == [1029,866,73,12]
            and plan.get("new_policy_inputs") is False, "Network/input protocol changed")
    gate = p["quality_gate"]
    require(gate == dict(phases=list(EXPECTED_PHASES), minimum_rows_each_phase=100,
                        every_joint_mean_normalized_request_below=.05, every_joint_p95_below=.15,
                        fixed_pool_MSE_final_at_most_baseline_fraction=.5), "Quality gate changed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source_root", "initial_policy", "dataset", "output", "state"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require(args.seed == 42 and args.device == "cuda:0", "Approved calibration requires seed42 and cuda:0")
    for name in ("plan", "source_root", "initial_policy", "dataset", "state"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    case = args.state.parent
    metadata = dict(status="initializing", phase="preflight", physics_controls=0, actor_updates=0,
                    additional_actor_updates=0, cumulative_actor_updates=PARENT_UPDATES,
                    ppo_updates=0, rnd_updates=0, initialization_ready=False,
                    formal_acceptance_passed=False, stairs_acceptance_passed=False,
                    kind=KIND, simulation_started=False, started_at_epoch=time.time(), device=args.device, seed=args.seed,
                    policy_dimensions=[1029,866,73,12], output=str(args.output))
    args.output.mkdir(parents=False, exist_ok=False)

    def update(**values):
        metadata.update(values)
        state = json.loads(args.state.read_text())
        state.update(metadata)
        write(args.state, state)
        write(args.output/"run.json", metadata)

    try:
        plan, request = json.loads(args.plan.read_text()), json.loads((case/"request.json").read_text())
        validate_plan(plan)
        guard = json.loads((case/"launch.lock").read_text())
        require(request["case"] == str(case) and guard["request_sha256"] == sha(case/"request.json"),
                "Immutable calibration request/lock binding changed")
        require(request["protocol"] == plan["protocol"] and request["plan_path"] == str(args.plan)
                and request["plan_sha256"] == sha(args.plan), "Manager/approved plan binding changed")
        for key in ("parent_candidate", "parent_candidate_sha256", "parent_optimizer", "parent_optimizer_sha256",
                    "source_root", "dataset", "parent_collection"):
            require(request[key] == plan[key], "Manager/plan provenance differs: " + key)
        require(Path(request["output"]).resolve() == args.output, "Manager output binding changed")
        for key, actual in (("source_root", args.source_root), ("parent_candidate", args.initial_policy),
                            ("dataset", args.dataset)):
            require(Path(plan[key]).resolve(strict=True) == actual, "Plan/input path mismatch: " + key)
        optimizer_path = Path(plan["parent_optimizer"]).resolve(strict=True)
        require(sha(args.initial_policy) == plan["parent_candidate_sha256"], "Parent candidate changed")
        require(sha(optimizer_path) == plan["parent_optimizer_sha256"], "Parent Adam changed")
        require(Path(plan["dataset_manifest"]).resolve() == args.dataset/"manifest.json", "Dataset manifest binding changed")
        protocol = dict(kind=KIND, plan_path=str(args.plan), plan_sha256=sha(args.plan),
                        request_path=str(case/"request.json"), request_sha256=sha(case/"request.json"),
                        producer_case=str(case), parent_collection=plan["parent_collection"],
                        parent_candidate=str(args.initial_policy), parent_candidate_sha256=sha(args.initial_policy),
                        parent_optimizer=str(optimizer_path), parent_optimizer_sha256=sha(optimizer_path),
                        parent_actor_updates=PARENT_UPDATES, additional_actor_updates=UPDATES,
                        cumulative_actor_updates=PARENT_UPDATES+UPDATES, physics_controls=0,
                        batch_size=BATCH, learning_rate_start=START_LR, learning_rate_end=END_LR,
                        learning_rate_schedule="cosine_index0_to3199_inclusive", max_gradient_norm=MAX_GRAD,
                        frozen_dataset=str(args.dataset), frozen_dataset_manifest_path=str(args.dataset/"manifest.json"),
                        frozen_dataset_manifest_sha256=sha(args.dataset/"manifest.json"))
        metadata["refit_protocol"] = protocol
        dataset = FrozenDataset(args.dataset)
        budget = resource_budget(dataset.manifest, args.initial_policy.stat().st_size,
                                 optimizer_path.stat().st_size,
                                 sum(p.stat().st_size for p in args.source_root.parent.rglob("*") if p.is_file()))
        require(budget["host_reserved_bytes"] <= HOST_LIMIT and budget["disk_reserved_bytes"] <= DISK_LIMIT,
                "Calibration data/staging budget exceeds its approved limit")
        metadata["budget"] = budget
        update(phase="hashing_frozen_dataset")
        write(args.output/"dataset_content_open.json", dataset.verify("open_before_any_fit"))
        train_rows, held_rows = (occupied_indices(dataset.manifest[s]) for s in ("train", "heldout"))
        def field(split, name): return dataset.mapping(split, name)
        sampler = GlobalPhaseSampler(field("train","phase"), field("train","sample_uid"),
            field("train","valid_label"), field("train","clip"), field("train","env_id"), train_rows, args.seed)
        require(np.all(field("heldout","env_id")[held_rows] % 8 == 0), "Heldout membership changed")
        require(np.intersect1d(field("train","sample_uid")[train_rows],
                              field("heldout","sample_uid")[held_rows]).size == 0, "TRAIN/heldout UID intersection")
        np.savez_compressed(args.output/"selection.npz", eligible_train_indices=sampler.eligible,
            eligible_train_uids=field("train","sample_uid")[sampler.eligible],
            eligible_train_phase=field("train","phase")[sampler.eligible],
            heldout_indices=held_rows, heldout_uids=field("heldout","sample_uid")[held_rows],
            heldout_phase=field("heldout","phase")[held_rows], heldout_valid=field("heldout","valid_label")[held_rows],
            heldout_clip=field("heldout","clip")[held_rows])
        write(args.output/"sampling.json", sampler.metadata())
        fit_indices = np.lib.format.open_memmap(args.output/"fit_indices.npy", mode="w+", dtype="int64", shape=(UPDATES,BATCH))
        fit_uids = np.lib.format.open_memmap(args.output/"fit_uids.npy", mode="w+", dtype="int64", shape=(UPDATES,BATCH))
        fit_indices[:] = -1; fit_uids[:] = -1
        shutil.copytree(args.source_root.parent, args.output/"training_source",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        source = args.output/"training_source/source"
        refit_tools_root = args.output/"training_source/refit_tools"
        refit_tools_root.mkdir(exist_ok=False)
        refit_tools = [Path(__file__).resolve(strict=True), Path(__file__).with_suffix(".sh").resolve(strict=True)]
        refit_tools_sha256 = {path.name:sha(path) for path in refit_tools}
        for path in refit_tools:
            shutil.copy2(path, refit_tools_root/path.name)
        require(refit_tools_sha256 == {path.name:sha(refit_tools_root/path.name) for path in refit_tools},
                "Actual calibration tool copy changed")
        refit_tools_manifest = refit_tools_root/"sha256.json"
        write(refit_tools_manifest, refit_tools_sha256)
        metadata.update(refit_tools_root=str(refit_tools_root),refit_tools_sha256=refit_tools_sha256,
                        refit_tools_source_paths={path.name:str(path) for path in refit_tools},
                        refit_tools_manifest_sha256=sha(refit_tools_manifest))
        shutil.copytree(args.initial_policy.parent/"params", args.output/"params")
        params_hashes = {str(p.relative_to(args.initial_policy.parent/"params")):sha(p)
                         for p in sorted((args.initial_policy.parent/"params").rglob("*")) if p.is_file()}
        require(params_hashes == {str(p.relative_to(args.output/"params")):sha(p)
                                  for p in sorted((args.output/"params").rglob("*")) if p.is_file()},
                "Parent parameter configuration copy changed")

        import torch
        import yaml
        torch.set_num_threads(8)
        torch.manual_seed(args.seed)
        require(torch.cuda.is_available(), "Approved offline calibration CUDA device is unavailable")
        torch.cuda.set_device(args.device)
        boot = torch.load(args.initial_policy, map_location="cpu", weights_only=True)
        parent_opt = torch.load(optimizer_path, map_location="cpu", weights_only=True)
        receipt = json.loads(args.initial_policy.with_suffix(".validation.json").read_text())
        require(receipt["status"] == "passed" and receipt["sha256"] == sha(args.initial_policy), "Parent CPU receipt changed")
        hashes = source_hashes(args.source_root)
        require(hashes == source_hashes(source) == boot["source_sha256"] == receipt["source_sha256"], "Original production source changed")
        require(Path(boot["source_root"]).resolve(strict=True) == args.source_root, "Parent own source binding changed")
        require(boot["format"] == "h2_direct_policy_initialization_v1"
                and boot["policy_dimensions"] == [1029,866,73,12]
                and boot["action_semantics"] == "full_soft_tanh_position_v1"
                and boot["ppo_completed_iterations"] == 0 and boot["initialization_ready"] is False,
                "Parent initialization format/action/network differs")
        require(boot["metadata"]["optimizer_sha256"] == sha(optimizer_path), "Parent candidate/Adam binding differs")
        dependencies = boot["external_dependency_sha256"]
        for path, digest in dependencies.items():
            require(sha(Path(path).resolve(strict=True)) == digest, "External physical dependency changed")
        policy_cfg = dict(yaml.safe_load((args.output/"params/agent.yaml").read_text())["policy"])
        require(policy_cfg.pop("class_name") == "H2ActorCriticOur", "Parent actor class differs")
        actor_class = load_module(source/PACKAGE/"networks/actor_critic_our.py", "refit_own_actor").H2ActorCriticOur
        normalizer_path = next((Path(p)/"rsl_rl/modules/normalizer.py" for p in sys.path
                                if (Path(p)/"rsl_rl/modules/normalizer.py").is_file()), None)
        require(normalizer_path is not None, "Original empirical normalizer is unavailable")
        norm_class = load_module(normalizer_path, "refit_own_normalizer").EmpiricalNormalization
        model = actor_class(1029,866,12,**policy_cfg).to(args.device).eval()
        model.load_state_dict(boot["model_state_dict"], strict=True)
        model.requires_grad_(False)
        model.actor.requires_grad_(True)
        parameters = list(model.actor.named_parameters())
        require(len(parameters) == 44, "Original44 actor parameters changed")
        norm = norm_class([1029], until=1e8).to(args.device).eval().requires_grad_(False)
        critic_norm = norm_class([866], until=1e8).to(args.device).eval().requires_grad_(False)
        norm.load_state_dict(boot["obs_norm_state_dict"], strict=True)
        critic_norm.load_state_dict(boot["privileged_obs_norm_state_dict"], strict=True)

        def cpu_state(value):
            if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
            if isinstance(value, dict): return {k:cpu_state(v) for k,v in value.items()}
            if isinstance(value, list): return [cpu_state(v) for v in value]
            if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
            return value

        def equal(a,b):
            if isinstance(a, torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
            if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b

        def finite(value):
            if isinstance(value,torch.Tensor): require(bool(torch.isfinite(value).all()), "Nonfinite tensor in calibration state")
            elif isinstance(value,dict):
                for item in value.values(): finite(item)
            elif isinstance(value,(tuple,list)):
                for item in value: finite(item)
            elif isinstance(value,float): require(math.isfinite(value), "Nonfinite calibration scalar")

        def optimizer_contract(packet, expected):
            require(packet["format"] == "h2_direct_imitation_optimizer_v1" and packet["updates"] == expected,
                    "Imitation optimizer format/budget differs")
            require(packet["actor_parameter_names"] == [name for name,_ in parameters], "Adam actor parameter order differs")
            require(packet.get("ppo_updates",packet.get("PPO_updates")) == 0
                    and packet.get("rnd_updates",packet.get("RND_updates")) == 0, "PPO/RND updates are forbidden")
            state = packet["optimizer_state_dict"]
            order = [pid for group in state["param_groups"] for pid in group["params"]]
            require(len(order)==44 and len(set(order))==44 and set(order)==set(state["state"]), "Incomplete actor Adam state")
            for pid,(_,parameter) in zip(order,parameters):
                item=state["state"][pid]
                require(int(item["step"])==expected, "Adam is not continuing every original parameter step")
                require(item["exp_avg"].shape==parameter.shape and item["exp_avg_sq"].shape==parameter.shape,
                        "Adam moment shape differs")
            finite(packet)

        optimizer_contract(parent_opt,PARENT_UPDATES)
        require(parent_opt["batch_size"]==1024, "Parent optimizer was not the approved aggregation job")
        optimizer = torch.optim.Adam(model.actor.parameters(),lr=START_LR)
        optimizer.load_state_dict(parent_opt["optimizer_state_dict"])
        require(equal(cpu_state(optimizer.state_dict()), parent_opt["optimizer_state_dict"]),
                "Complete parent Adam state was not preserved before the declared LR change")
        initial_actor = {k: value for k,value in boot["model_state_dict"].items() if not k.startswith("actor.")}
        metadata.update(source_root=str(source), source_sha256=hashes, source_commit=boot["source_commit"],
            external_dependency_sha256=dependencies, actor_parameter_count=44,
            normalizer_implementation=str(normalizer_path), normalizer_implementation_sha256=sha(normalizer_path),
            parent_params_sha256=params_hashes,
            parent_Adam_moments_restored_exactly=True, frozen_nonactor_state_unchanged=True,
            frozen_dualnorm_unchanged=True, no_App=True, no_teacher=True, GPU_replay_bytes=0)

        def check_frozen():
            now=model.state_dict()
            require(all(torch.equal(now[k].detach().cpu(),v) for k,v in initial_actor.items()), "Critic/std changed")
            require(equal(cpu_state(norm.state_dict()),boot["obs_norm_state_dict"])
                    and equal(cpu_state(critic_norm.state_dict()),boot["privileged_obs_norm_state_dict"]), "Own dualnorm changed")

        def limits():
            require(time.monotonic()-started <= 1800, "Calibration exceeded its1800s bound; no continuation")
            disk=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            require(disk <= DISK_LIMIT, "Calibration output exceeded512MiB")
            allocated,reserved=torch.cuda.memory_allocated(args.device),torch.cuda.memory_reserved(args.device)
            peak_allocated=torch.cuda.max_memory_allocated(args.device)
            peak_reserved=torch.cuda.max_memory_reserved(args.device)
            require(max(allocated,reserved,peak_allocated,peak_reserved) <= GPU_LIMIT,
                    "Calibration GPU exceeds11GiB; no batch/env reduction or retry")
            return dict(allocated_bytes=allocated,reserved_bytes=reserved,
                        peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,output_disk_bytes=disk,
                        elapsed_seconds=time.monotonic()-started)

        names=json.loads((args.initial_policy.parent/"result.json").read_text())["action_joint_names"]
        valid=field("heldout","valid_label")[held_rows].copy()
        clip=field("heldout","clip")[held_rows].copy()
        phase=field("heldout","phase")[held_rows].copy()
        eligible=valid & ~clip.any(1)
        fields=("obs","position","stiffness","caps","lower","upper")

        def stage_batch(split,rows):
            batch=dataset.get(split,rows,fields)
            require(all(np.isfinite(v).all() for v in batch.values()), "Nonfinite frozen fit/evaluation data")
            require(np.all(batch["caps"]>0) and np.all(batch["stiffness"]>0)
                    and np.all(batch["lower"]<batch["upper"]), "Invalid frozen decoder/gain/cap values")
            return {key:torch.from_numpy(value).pin_memory().to(args.device,non_blocking=True)
                    for key,value in batch.items()}

        def predict(tensors):
            with torch.no_grad(): x=norm(tensors["obs"])
            raw=model.act_inference(x)
            published=raw.clamp(-100,100)
            lo,hi=tensors["lower"],tensors["upper"]
            q=(lo+hi)*.5+(hi-lo)*.5*torch.tanh(published)
            request=(q-tensors["position"])*tensors["stiffness"]/tensors["caps"]
            return raw,published,q,request

        def evaluate(label):
            output=np.lib.format.open_memmap(args.output/("heldout_predictions_"+label+".npy"),
                        mode="w+",dtype="float32",shape=(len(held_rows),4,12))
            with torch.inference_mode():
                for begin in range(0,len(held_rows),1024):
                    values=predict(stage_batch("heldout",held_rows[begin:begin+1024]))
                    for column,value in enumerate(values):
                        require(bool(torch.isfinite(value).all()), "Nonfinite heldout prediction")
                        output[begin:begin+len(value),column]=value.cpu().numpy()
                    limits()
            output.flush()
            request=np.asarray(output[:,3]); rad=np.asarray(output[:,2])-field("heldout","position")[held_rows]
            def stats(mask):
                if not mask.any(): return dict(rows=0)
                signed=request[mask];absolute=np.abs(signed);r=rad[mask]
                means=absolute.mean(0);p95=np.quantile(absolute,.95,axis=0)
                return dict(rows=int(mask.sum()),joint_order=names,signed_radian_mean=r.mean(0).tolist(),
                    absolute_radian_mean=np.abs(r).mean(0).tolist(),absolute_radian_p95=np.quantile(np.abs(r),.95,axis=0).tolist(),
                    signed_normalized_request_mean=signed.mean(0).tolist(),normalized_request_mean=means.tolist(),
                    normalized_request_p95=p95.tolist(),normalized_request_p99=np.quantile(absolute,.99,axis=0).tolist(),
                    normalized_request_max=absolute.max(0).tolist(),normalized_request_MSE=float(np.square(signed,dtype=np.float64).mean()),
                    mean_argmax_joint=names[int(means.argmax())],p95_argmax_joint=names[int(p95.argmax())])
            report=dict(label=label,heldout_rows=len(held_rows),eligible_rows=int(eligible.sum()),
                invalid_rows=int((~valid).sum()),any_clipped_rows=int(clip.any(1).sum()),
                invalid_fraction=float((~valid).mean()),per_joint_clip_fraction=clip.mean(0).tolist(),
                all_rows=stats(np.ones(len(valid),dtype=bool)),valid_no_clip=stats(eligible),
                valid_clipped=stats(valid&clip.any(1)),invalid=stats(~valid),
                by_phase={str(p):dict(all_rows=stats(phase==p),valid_no_clip=stats(eligible&(phase==p))) for p in np.unique(phase)},
                output=str(args.output/("heldout_predictions_"+label+".npy")),
                output_columns=["raw_actor","published_raw_clip100","decoded_q","signed_K_delta_q_over_cap"],
                output_array_sha256=array_sha(output),fixed_raw_obs_array_sha256=dataset.manifest["heldout"]["files"]["obs"]["array_sha256"],
                units="Decoded position and host K*position-error/cap; not measured torque or walking acceptance",
                initialization_ready=False)
            output._mmap.close()
            write(args.output/("quality_"+label+".json"),report)
            return report

        update(status="evaluating",phase="baseline_fixed_heldout",sampler=sampler.metadata(),gpu=limits())
        baseline=evaluate("baseline")

        def export(additional,quality=None):
            check_frozen()
            finite(model.state_dict())
            final=additional==UPDATES
            directory=args.output if final else args.output/"evidence_checkpoints"
            directory.mkdir(exist_ok=True)
            opt_path=directory/("imitation_optimizer_state.pt" if final else f"imitation_optimizer_update_{additional:04d}.pt")
            packet=dict(format="h2_direct_imitation_optimizer_v1",optimizer_state_dict=cpu_state(optimizer.state_dict()),
                actor_parameter_names=[name for name,_ in parameters],updates=PARENT_UPDATES+additional,batch_size=BATCH,
                learning_rate=float(optimizer.param_groups[0]["lr"]),max_gradient_norm=MAX_GRAD,ppo_updates=0,rnd_updates=0,
                dataset_manifest=str(dataset.manifest_path),refit_protocol=protocol,additional_actor_updates=additional,
                cumulative_actor_updates=PARENT_UPDATES+additional)
            optimizer_contract(packet,PARENT_UPDATES+additional)
            torch.save(packet,str(opt_path)+".tmp");Path(str(opt_path)+".tmp").replace(opt_path)
            saved=dict(format="h2_direct_policy_initialization_v1",ppo_completed_iterations=0,
                model_state_dict=cpu_state(model.state_dict()),obs_norm_state_dict=cpu_state(norm.state_dict()),
                privileged_obs_norm_state_dict=cpu_state(critic_norm.state_dict()),action_semantics=boot["action_semantics"],
                policy_dimensions=[1029,866,73,12],source_root=str(source),source_sha256=hashes,
                source_commit=boot["source_commit"],external_dependency_sha256=dependencies,
                collector_case=str(case),teacher_checkpoint_sha256=boot["teacher_checkpoint_sha256"],initialization_ready=False,
                metadata=dict(quality=quality,quality_gate_passed=bool(quality and quality["quality_gate_passed"]),
                    physics_controls=0,actor_updates=PARENT_UPDATES+additional,additional_actor_updates=additional,
                    cumulative_actor_updates=PARENT_UPDATES+additional,optimizer_path=str(opt_path),optimizer_sha256=sha(opt_path),
                    refit_protocol=protocol,refit_final_eligible=final,independent_validation_required=True,
                    refit_tools_root=str(refit_tools_root),refit_tools_sha256=refit_tools_sha256,
                    refit_tools_manifest_sha256=metadata["refit_tools_manifest_sha256"],
                    critic="Parent cold critic and std retained exactly; neither trained by calibration"))
            path=directory/("candidate_initial_policy.pt" if final else f"candidate_update_{additional:04d}.pt")
            torch.save(saved,str(path)+".tmp");Path(str(path)+".tmp").replace(path)
            return dict(candidate=str(path),candidate_sha256=sha(path),optimizer_path=str(opt_path),optimizer_sha256=sha(opt_path))

        update(status="fitting",phase="frozen_train",actor_updates=0,cumulative_actor_updates=PARENT_UPDATES)
        curves=[]
        for index in range(UPDATES):
            require(time.monotonic()-started <=1800, "Calibration reached its wall bound")
            rows,allocation=sampler.sample()
            require(np.all(field("train","valid_label")[rows]) and not field("train","clip")[rows].any()
                    and np.all(field("train","env_id")[rows]%8!=0), "Invalid/heldout label entered gradient")
            fit_indices[index]=rows;fit_uids[index]=field("train","sample_uid")[rows]
            lr=learning_rate(index)
            for group in optimizer.param_groups: group["lr"]=lr
            tensors=stage_batch("train",rows)
            _,_,_,error=predict(tensors)
            loss=error.square().mean()
            require(bool(torch.isfinite(loss)), "Nonfinite calibration loss")
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.actor.parameters(),MAX_GRAD)
            require(bool(torch.isfinite(gradient)), "Nonfinite calibration gradient")
            metadata["optimizer_step_in_progress"]=True
            optimizer.step()
            additional=index+1
            metadata.update(actor_updates=additional,additional_actor_updates=additional,
                cumulative_actor_updates=PARENT_UPDATES+additional,optimizer_step_in_progress=False)
            if additional%100==0:
                fit_indices.flush();fit_uids.flush()
                point=dict(actor_updates=additional,additional_actor_updates=additional,
                    cumulative_actor_updates=PARENT_UPDATES+additional,physics_controls=0,learning_rate=lr,
                    supervised_loss=float(loss.detach()),gradient_norm_before_clip=float(gradient),
                    sampled_phase_counts=allocation.tolist(),gpu=limits())
                curves.append(point)
                update(**point,sampler=sampler.metadata())
                print("[DIRECT_INIT_REFIT] "+json.dumps(point),flush=True)
            if additional in (800,1600,2400):
                export(additional)  # retained evidence, never evaluated or selected
        require(sampler.updates==UPDATES, "Calibration optimizer budget changed")
        check_frozen()
        update(status="evaluating",phase="final_fixed_heldout")
        final=evaluate("final")
        per_phase={}
        for p in EXPECTED_PHASES:
            point=final["by_phase"].get(str(p),{}).get("valid_no_clip",{"rows":0})
            per_phase[str(p)]=bool(point["rows"]>=100 and np.all(np.asarray(point.get("normalized_request_mean",[math.inf]))<.05)
                                 and np.all(np.asarray(point.get("normalized_request_p95",[math.inf]))<.15))
        before=baseline["valid_no_clip"]["normalized_request_MSE"]
        after=final["valid_no_clip"]["normalized_request_MSE"]
        quality=dict(quality_gate_passed=bool(all(per_phase.values()) and after<=.5*before),
            phase_quality_passed=per_phase,fixed_pool_MSE_baseline=before,fixed_pool_MSE_final=after,
            fixed_pool_MSE_ratio=None if before==0 else after/before,required_MSE_fraction=.5,
            gate=plan["protocol"]["quality_gate"],baseline=baseline,final=final,
            independent_physical_validation_required=True,initialization_ready=False)
        write(args.output/"quality.json",quality)
        final_files=export(UPDATES,quality)
        write(args.output/"fit_curve.json",curves)
        fit_indices.flush();fit_uids.flush()
        write(args.output/"sampling.json",sampler.metadata())
        write(args.output/"fit_manifest.json",dict(
            fit_indices=dict(path=str(args.output/"fit_indices.npy"),shape=[UPDATES,BATCH],dtype="int64",array_sha256=array_sha(fit_indices)),
            fit_uids=dict(path=str(args.output/"fit_uids.npy"),shape=[UPDATES,BATCH],dtype="int64",array_sha256=array_sha(fit_uids)),
            selection_sha256=sha(args.output/"selection.npz"),sampling=sampler.metadata(),refit_protocol=protocol))
        update(status="verifying",phase="closing_frozen_evidence")
        write(args.output/"dataset_content_close.json",dataset.verify("close_after_final_fit_and_evaluation"))
        require(sha(args.initial_policy)==protocol["parent_candidate_sha256"]
                and sha(optimizer_path)==protocol["parent_optimizer_sha256"]
                and sha(args.plan)==protocol["plan_sha256"]
                and sha(case/"request.json")==protocol["request_sha256"], "Parent/plan/request changed")
        require(source_hashes(args.source_root)==source_hashes(source)==hashes, "Production source changed")
        require(refit_tools_sha256 == {path.name:sha(path) for path in refit_tools}
                == {path.name:sha(refit_tools_root/path.name) for path in refit_tools}
                and sha(refit_tools_manifest)==metadata["refit_tools_manifest_sha256"],
                "Actual calibration tool or its production copy changed")
        require(params_hashes == {str(p.relative_to(args.initial_policy.parent/"params")):sha(p)
                                 for p in sorted((args.initial_policy.parent/"params").rglob("*")) if p.is_file()}
                == {str(p.relative_to(args.output/"params")):sha(p)
                    for p in sorted((args.output/"params").rglob("*")) if p.is_file()},
                "Frozen policy configuration changed")
        require(sha(normalizer_path)==metadata["normalizer_implementation_sha256"], "Normalizer implementation changed")
        for path,digest in dependencies.items(): require(sha(path)==digest,"External dependency changed")
        require(not any(name=="gymnasium" or name.startswith(("isaaclab.","isaacsim.","omni.")) for name in sys.modules),
                "Offline calibration imported a simulator/task package")
        update(status="completed",phase="finished",actor_updates=UPDATES,additional_actor_updates=UPDATES,
            cumulative_actor_updates=PARENT_UPDATES+UPDATES,physics_controls=0,quality_gate_passed=quality["quality_gate_passed"],
            finished_at_epoch=time.time(),frozen_nonactor_state_unchanged=True,frozen_dualnorm_unchanged=True,
            parent_dataset_unchanged=True,final_candidate_only=True,gpu=limits(),**final_files)
        write(args.output/"result.json",metadata)
        print("[DIRECT_INIT_REFIT] RESULT "+json.dumps(metadata),flush=True)
    except BaseException as error:
        if "dataset" in locals() and len(dataset.verifications)==1:
            try:
                write(args.output/"dataset_content_close_failed.json",dataset.verify("close_on_failed_refit"))
            except BaseException as close_error:
                metadata["closing_dataset_verification_error"]=repr(close_error)
        metadata.update(status="failed",phase="finished",initialization_ready=False,formal_acceptance_passed=False,
            finished_at_epoch=time.time(),error=repr(error),traceback=traceback.format_exc())
        write(args.output/"result.json",metadata)
        update(**metadata)
        print("[DIRECT_INIT_REFIT] FAILED "+repr(error),file=sys.stderr,flush=True)
        raise


if __name__ == "__main__":
    main()
