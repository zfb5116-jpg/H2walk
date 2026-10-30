"""Bounded CPU datasets for direct-policy aggregation; no Isaac or Torch import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

TRAIN_CAPACITY = 131072
HELDOUT_CAPACITIES = (16384, 8192, 8192, 8192, 8192)
TRAIN_CAPACITIES = (26216, 26214, 26214, 26214, 26214)
HOST_LIMIT_BYTES = int(1.5 * 2**30)
DISK_LIMIT_BYTES = int(1.2 * 2**30)
FIT_SHAPE = (4, 800, 1024)

SCHEMA = {"obs": ("float32", (1029,))}
for _name in ("position", "equivalent", "teacher_q", "teacher_v", "stiffness", "damping", "caps",
              "lower", "upper", "raw_label", "raw_teacher", "raw_student", "raw_executed", "label_error"):
    SCHEMA[_name] = ("float32", (12,))
SCHEMA.update(clip=("bool", (12,)), valid_label=("bool", ()), env_id=("int32", ()),
              episode_id=("int64", ()), control_step=("int64", ()), native_control_counter=("int64", ()),
              sample_uid=("int64", ()), phase=("int16", ()), actor_source=("uint8", ()))
HELDOUT_SCHEMA = {**SCHEMA, "critic": ("float32", (866,)), "root_state": ("float32", (13,))}


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def array_bytes(capacity, schema):
    return sum(capacity * int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
               for dtype, shape in schema.values())


def production_budget():
    arrays = array_bytes(TRAIN_CAPACITY, SCHEMA) + array_bytes(sum(HELDOUT_CAPACITIES), HELDOUT_SCHEMA)
    fit = int(np.prod(FIT_SHAPE)) * 8
    staging = 32 * 2**20
    # The journal is compact columnar data, not all-frame observations.
    journal_reserve = 32 * 2**20
    oracle_trace_reserve = 10 * 2**20
    return dict(array_bytes=arrays, fit_indices_bytes=fit, staging_reserve_bytes=staging,
                oracle_trace_reserve_bytes=oracle_trace_reserve,
                host_reserved_bytes=arrays + fit + staging + 2**20,
                disk_reserved_bytes=arrays + fit + journal_reserve + oracle_trace_reserve + 2 * 2**20,
                host_limit_bytes=HOST_LIMIT_BYTES, disk_limit_bytes=DISK_LIMIT_BYTES)


def content_sha(array):
    h = hashlib.sha256()
    flat = array.reshape(-1)
    width = max(1, (4 * 2**20) // array.dtype.itemsize)
    for start in range(0, flat.size, width):
        h.update(flat[start:start+width].tobytes(order="C"))
    return h.hexdigest()


def summarize_joint_phase_errors(error, phase, valid, clip):
    """Report every observed phase, keeping invalid labels in the denominator."""
    if (error.ndim != 3 or error.shape[1:] != (2,12) or not len(error)
            or phase.shape != (len(error),) or valid.shape != (len(error),)
            or clip.shape != (len(error),12) or not np.isfinite(error).all()):
        raise ValueError("Invalid joint/phase quality input")
    def stats(mask):
        value = error[mask]
        if not len(value): return {"rows":0}
        return dict(rows=len(value),signed_radian_mean=value[:,0].mean(0).tolist(),
            absolute_radian_p50_p95_p99_max=np.vstack([np.quantile(np.abs(value[:,0]),[.5,.95,.99],axis=0),np.abs(value[:,0]).max(0)]).tolist(),
            normalized_request_mean=value[:,1].mean(0).tolist(),
            normalized_request_p95=np.quantile(value[:,1],.95,axis=0).tolist(),
            normalized_request_p99=np.quantile(value[:,1],.99,axis=0).tolist(),
            normalized_request_max=value[:,1].max(0).tolist(),clip_fraction=clip[mask].mean(0).tolist())
    by_phase = {str(item):stats(phase==item) for item in np.unique(phase)}
    passed = {name:bool(np.all(np.asarray(point["normalized_request_mean"])<.05)
                       and np.all(np.asarray(point["normalized_request_p95"])<.15))
              for name,point in by_phase.items()}
    return dict(rows=len(error), invalid_label_fraction=float((~valid).mean()),
        per_joint=stats(np.ones(len(error),dtype=bool)), valid_label_only_per_joint=stats(valid),
        by_phase=by_phase, phase_regression_passed=passed,
        regression_quality_passed=bool(all(passed.values()) and np.all(error[:,1].mean(0)<.05)
            and np.all(np.quantile(error[:,1],.95,axis=0)<.15) and np.all(clip.mean(0)<.05) and valid.all()),
        regression_gate_scope="Every observed phase and every joint must satisfy mean<.05 and p95<.15; pooled error cannot substitute",
        initialization_ready=False)


class PartitionedReservoir:
    """Each collection round owns a reservoir that is immutable after fitting.

    Thus every recorded physical row index still refers to the same sample in
    the final files, including indices used by earlier optimization rounds.
    """
    def __init__(self, directory, capacities, schema, seed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacities = np.asarray(capacities, dtype=np.int64)
        self.offsets = np.r_[0, np.cumsum(self.capacities)[:-1]]
        self.schema, self.capacity = schema, int(self.capacities.sum())
        self.seen = np.zeros(len(capacities), dtype=np.int64)
        self.closed = np.zeros(len(capacities), dtype=bool)
        self.rngs = [np.random.default_rng(seed + 1009*i) for i in range(len(capacities))]
        self.arrays = {name:np.lib.format.open_memmap(self.directory/(name+".npy"), mode="w+",
                       dtype=dtype, shape=(self.capacity, *shape)) for name, (dtype, shape) in schema.items()}
        self.arrays["valid_label"][:] = False

    def add(self, partition, batch):
        if not 0 <= partition < len(self.capacities) or self.closed[partition]:
            raise ValueError("Collection partition is absent or already frozen")
        n = len(batch["env_id"])
        if not n:
            return
        for name, (dtype, shape) in self.schema.items():
            value = np.asarray(batch[name])
            if value.shape != (n, *shape) or value.dtype != np.dtype(dtype):
                raise ValueError(f"Dataset field mismatch: {name}/{value.shape}/{value.dtype}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite dataset field: {name}")
        cap, offset = int(self.capacities[partition]), int(self.offsets[partition])
        ordinal = self.seen[partition] + np.arange(n, dtype=np.int64)
        destination = np.where(ordinal < cap, ordinal, self.rngs[partition].integers(0, ordinal+1))
        keep = np.flatnonzero(destination < cap)
        # Sequential reservoir semantics: the last incoming replacement wins
        # when several samples in this vectorized append select the same row.
        _, reverse_index = np.unique(destination[keep][::-1], return_index=True)
        selected = keep[len(keep)-1-reverse_index]
        rows = offset + destination[selected]
        for name, array in self.arrays.items():
            array[rows] = batch[name][selected]
        self.seen[partition] += n

    def close(self, partition):
        self.closed[partition] = True
        self.flush()

    def indices(self, valid_only=False):
        rows = np.concatenate([self.offsets[i] + np.arange(min(int(seen), int(self.capacities[i])))
                               for i, seen in enumerate(self.seen)])
        return rows[self.arrays["valid_label"][rows]] if valid_only else rows

    def get(self, rows, fields=None):
        return {name:np.asarray(self.arrays[name][rows]).copy() for name in (fields or self.schema)}

    def flush(self):
        for array in self.arrays.values():
            array.flush()

    def metadata(self, hashes=False):
        result = dict(capacity=self.capacity, capacities=self.capacities.tolist(), seen=self.seen.tolist(),
            closed=self.closed.tolist(), occupied=[min(int(n),int(c)) for n,c in zip(self.seen,self.capacities)],
            rng_states=[rng.bit_generator.state for rng in self.rngs], files={})
        for name, array in self.arrays.items():
            result["files"][name] = dict(path=str(self.directory/(name+".npy")), shape=list(array.shape),
                dtype=str(array.dtype), bytes=array.nbytes)
            if hashes:
                result["files"][name]["array_sha256"] = content_sha(array)
        return result


class InitializationDataset:
    def __init__(self, directory, seed=42):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        budget = production_budget()
        if budget["host_reserved_bytes"] > HOST_LIMIT_BYTES or budget["disk_reserved_bytes"] > DISK_LIMIT_BYTES:
            raise MemoryError("Declared initialization dataset exceeds its approved memory/disk budget")
        self.train = PartitionedReservoir(self.directory/"train", TRAIN_CAPACITIES, SCHEMA, seed+20000)
        self.heldout = PartitionedReservoir(self.directory/"heldout", HELDOUT_CAPACITIES, HELDOUT_SCHEMA, seed+40000)
        self.fit_indices = np.lib.format.open_memmap(self.directory/"fit_indices.npy", mode="w+",
                                                    dtype="int64", shape=FIT_SHAPE)
        self.fit_indices[:] = -1
        self.fit_updates = np.zeros(4, dtype=np.int64)
        self.fit_rng = np.random.default_rng(seed+60000)
        self.counts = {"all_rows":0,"invalid_label_rows":0,"clip_rows":0,"train_seen":0,"heldout_seen":0}
        self.last_control_step = 0
        write_json(self.directory/"schema.json", dict(version=1, fields=SCHEMA, heldout_fields=HELDOUT_SCHEMA,
            holdout_membership="env_id%8==0; excluded from every fit;128/1024 fixed lanes",
            stages={"0":"150-control admission", "1":"teacher episode probability.75",
                    "2":"teacher probability.5", "3":"teacher probability.25", "4":"pure student"},
            action_source={"0":"student", "1":"teacher-equivalent position"}, budget=budget,
            observation_semantics="Raw fp32 original layouts; teacher labels never enter current observation",
            index_semantics="fit_indices are physical rows in final train files; a stage is frozen before first fit"))
        self.progress()

    def add(self, stage, batch):
        if not 0 <= stage < 5 or self.train.closed[stage] or self.heldout.closed[stage]:
            raise RuntimeError("Dataset collection stage is absent or already frozen")
        env_ids = batch["env_id"]
        holdout = env_ids % 8 == 0
        if stage > 0 and np.any(batch["actor_source"][holdout] != 0):
            raise RuntimeError("Heldout lanes must actually execute the student during aggregation")
        if not np.array_equal(env_ids, np.arange(1024, dtype=np.int32)):
            raise ValueError("Every physical collection step must retain all1024 lanes in its counters")
        clock = np.unique(batch["control_step"])
        if len(clock) != 1 or int(clock[0]) != self.last_control_step+1:
            raise RuntimeError("Dataset control steps must be consecutive and appended exactly once")
        expected_uid = (int(clock[0])-1)*1024 + env_ids.astype(np.int64)
        if not np.array_equal(batch["sample_uid"],expected_uid):
            raise RuntimeError("Dataset sample UID does not bind control step and environment")
        self.counts["all_rows"] += len(env_ids)
        self.counts["invalid_label_rows"] += int((~batch["valid_label"]).sum())
        self.counts["clip_rows"] += int(batch["clip"].any(-1).sum())
        for reservoir, mask, key in ((self.train,~holdout,"train_seen"),(self.heldout,holdout,"heldout_seen")):
            reservoir.add(stage, {name:value[mask] for name,value in batch.items() if name in reservoir.schema})
            self.counts[key] += int(mask.sum())
        self.last_control_step = int(clock[0])

    def close_stage(self, stage):
        self.train.close(stage); self.heldout.close(stage); self.progress()

    def training_batch(self, round_index):
        update = int(self.fit_updates[round_index])
        if update >= 800 or not self.train.closed[round_index+1]:
            raise RuntimeError("Fit outside frozen round or its800-update budget")
        rows = self.train.indices(valid_only=True)
        if len(rows) < 1024:
            raise RuntimeError("Insufficient valid expert labels; do not silently shrink the batch")
        partitions = [i for i in range(round_index+2) if self.train.closed[i]]
        if len(partitions) != round_index+2:
            raise RuntimeError("All preceding aggregate rounds must be frozen before fitting")
        def counts(total, groups, rotation):
            result = np.full(groups,total//groups,dtype=np.int64)
            result[(np.arange(total%groups)+rotation)%groups] += 1
            return result
        selected, balance = [], []
        for partition, allocation in zip(partitions,counts(1024,len(partitions),update)):
            begin = int(self.train.offsets[partition])
            length = min(int(self.train.seen[partition]),int(self.train.capacities[partition]))
            pool = np.arange(begin,begin+length,dtype=np.int64)
            pool = pool[self.train.arrays["valid_label"][pool]]
            phases = np.unique(self.train.arrays["phase"][pool])
            if not len(phases):
                raise RuntimeError("A closed aggregate round has no valid expert phase; do not silently omit it")
            for phase, amount in zip(phases,counts(int(allocation),len(phases),update+partition)):
                phase_pool = pool[self.train.arrays["phase"][pool]==phase]
                selected.append(self.fit_rng.choice(phase_pool,size=int(amount),replace=True))
                balance.append(dict(partition=partition,phase=int(phase),samples=int(amount),available_valid=len(phase_pool)))
        selected = np.concatenate(selected)
        self.fit_rng.shuffle(selected)
        self.last_training_batch_balance = balance
        if np.any(self.train.arrays["env_id"][selected] % 8 == 0):
            raise RuntimeError("Heldout sample leaked into training")
        self.fit_indices[round_index, update] = selected
        self.fit_updates[round_index] += 1
        return self.train.get(selected, ["obs","position","stiffness","caps","lower","upper"])

    def progress(self):
        self.fit_indices.flush()
        disk = sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())
        if disk > DISK_LIMIT_BYTES:
            raise OSError("Initialization dataset exceeded1.2GiB disk budget")
        value = dict(counts=self.counts, last_control_step=self.last_control_step,
                     fit_updates=self.fit_updates.tolist(), fit_rng_state=self.fit_rng.bit_generator.state,
                     sampling_scheme="Equal closed-round allocation, equal valid-phase allocation within each round; rotating remainders",
                     last_training_batch_balance=getattr(self,"last_training_batch_balance",[]),
                     train=self.train.metadata(), heldout=self.heldout.metadata(), dataset_disk_bytes=disk,
                     budget=production_budget())
        write_json(self.directory/"progress.json", value)
        return value

    def finalize(self, status):
        self.train.flush(); self.heldout.flush()
        value = self.progress()
        value.update(status=status, train=self.train.metadata(hashes=True), heldout=self.heldout.metadata(hashes=True),
                     fit_indices=dict(path=str(self.directory/"fit_indices.npy"), shape=list(FIT_SHAPE),
                                      dtype="int64", array_sha256=content_sha(self.fit_indices)),
                     immutable_after_collection=True, initialization_ready=False)
        write_json(self.directory/"manifest.json", value)
        return value
