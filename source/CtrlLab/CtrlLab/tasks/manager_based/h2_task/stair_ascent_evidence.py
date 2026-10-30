"""Compact pre-reset episode evidence, buffered on device and flushed in chunks."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class StairEpisodeEvidence:
    """One row per ended episode, including invalid and bootstrap-partial rows.

    A single device matrix avoids dozens of tiny per-episode CPU transfers.
    Complete chunks and manifest appends are fsynced. Up to one partial chunk
    is buffered; it flushes every100 control epochs and on normal env.close().
    """

    def __init__(self, directory, metadata, *, chunk_size=1024, flush_interval=100):
        self.directory = Path(directory) if directory is not None else None
        self.metadata = metadata
        self.chunk_size, self.flush_interval = int(chunk_size), int(flush_interval)
        self.schema, self.buffer = None, None
        self.pending, self.written, self.chunk_index, self.last_flush_counter = 0, 0, 0, 0
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            if (self.directory/"schema.json").exists() or (self.directory/"manifest.jsonl").exists():
                raise FileExistsError(f"Episode evidence directory must be new: {self.directory}")

    def append(self, fields):
        if self.directory is None:
            return
        first = next(iter(fields.values()))
        n = first.shape[0]
        if not n:
            return
        specifications, columns, cursor = [], [], 0
        for name, value in fields.items():
            if value.shape[0] != n:
                raise ValueError("Episode evidence fields have different batch sizes")
            width = value[0].numel()
            specifications.append({"name": name, "shape": list(value.shape[1:]), "dtype": str(value.dtype),
                                   "start": cursor, "stop": cursor+width})
            columns.append(value.reshape(n, width).to(torch.float64))
            cursor += width
        if self.schema is None:
            self.schema = specifications
            self.buffer = torch.empty(self.chunk_size, cursor, dtype=torch.float64, device=first.device)
            with (self.directory/"schema.json").open("x") as stream:
                json.dump({"format": "h2_stair_episode_matrix_v1", "fields": specifications,
                           "metadata": self.metadata, "sampling": "same pre-reset post-control state, not a5ms wrench measurement",
                           "partial_chunk_flush_interval_control_steps": self.flush_interval,
                           "abrupt_SIGKILL_can_lose_only_the_unflushed_partial_chunk": True}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        elif self.schema != specifications:
            raise RuntimeError("Episode evidence schema changed during a run")
        matrix = torch.cat(columns, dim=-1)
        start = 0
        while start < n:
            count = min(n-start, self.chunk_size-self.pending)
            self.buffer[self.pending:self.pending+count].copy_(matrix[start:start+count])
            self.pending += count
            start += count
            if self.pending == self.chunk_size:
                self.flush()

    def maybe_flush(self, counter):
        if self.pending and int(counter)-self.last_flush_counter >= self.flush_interval:
            self.flush(counter)

    def flush(self, counter=None):
        if self.directory is None or not self.pending:
            return
        array = self.buffer[:self.pending].detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite stair episode evidence")
        name = f"episodes_{self.chunk_index:06d}.npz"
        path = self.directory/name
        with path.open("xb") as stream:
            np.savez_compressed(stream, matrix=array)
            stream.flush()
            os.fsync(stream.fileno())
        columns = {item["name"]: item["start"] for item in self.schema}
        end_counter = int(array[:, columns["end_control_counter"]].max())
        point = {"file": name, "rows": self.pending, "total_rows": self.written+self.pending,
                 "end_control_counter": end_counter, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "invalid_rows": int(array[:, columns["ever_invalid"]].sum()),
                 "true_timeout_rows": int(array[:, columns["true_timeout"]].sum())}
        with (self.directory/"manifest.jsonl").open("a") as stream:
            stream.write(json.dumps(point)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.written += self.pending
        self.pending = 0
        self.chunk_index += 1
        self.last_flush_counter = int(counter) if counter is not None else end_counter

    def close(self):
        self.flush()

