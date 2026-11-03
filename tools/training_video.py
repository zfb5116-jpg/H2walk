"""Bounded, labelled env0 RGB from the existing training simulation only."""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

FFMPEG = "/home/nsgm/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def video_contract():
    return dict(kind="same_sim_env0_three_bounded_training_windows_v1", enabled=True,
        env_id=0, resolution=[512, 384], fps=10, control_dt_s=0.02, capture_every_controls=5,
        windows=[[0, 10, 500, 100], [480, 500, 1000, 200], [979, 999, 1000, 200]],
        maximum_frames=500, maximum_manual_renders=504, static_warmup_renders=4,
        queue_frames=2, cpu_codec="libx264", encoder_binary=FFMPEG,
        eye=[1.75, -4.5, 2.2], lookat=[1.75, 0.0, 1.35],
        viewer_origin="env", viewer_env_index=0, camera_sensor_added=False,
        expected_has_rtx_sensors=False, render_recompute=False,
        training_not_formal=True, reset_frames_labelled=True)


def video_config_errors(value):
    expected = video_contract()
    if type(value) is not dict or set(value) != set(expected):
        return ["training video requires exact declared fields"]
    def exact(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(b, list):
            return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return a == b
    return ["training video changed " + k for k in expected if not exact(value[k], expected[k])]


def _birth(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=pid, start_ticks=int(stat[19]), pgid=os.getpgid(pid),
                argv=(root / "cmdline").read_bytes().split(b"\0")[:-1])


class _Encoder:
    def __init__(self, path, binary):
        self.error = None
        self.written = 0
        self.closed = False
        self.queue = queue.Queue(maxsize=2)
        self.stderr = path.with_suffix(".stderr").open("xb")
        self.argv = [binary, "-n", "-nostdin", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "512x384", "-r", "10", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)]
        self.process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self.stderr, start_new_session=True)
        b = _birth(self.process.pid)
        b["argv"] = [x.decode() for x in b["argv"]]
        self.birth = b
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                self.process.stdin.write(frame)
                self.written += 1
        except BaseException as error:
            self.error = repr(error)
        finally:
            try:
                self.process.stdin.close()
            except BaseException as error:
                self.error = self.error or repr(error)

    def add(self, frame):
        if self.error or not self.thread.is_alive():
            raise RuntimeError("Training encoder failed: " + str(self.error))
        try:
            self.queue.put(frame, timeout=5)
        except queue.Full as error:
            raise RuntimeError("Bounded training video queue stalled") from error

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.thread.is_alive():
                self.queue.put(None, timeout=5)
                self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Training encoder writer did not close")
            code = self.process.wait(timeout=10)
            if code != 0 or self.error:
                raise RuntimeError("Training encoder failed: " + str((code, self.error)))
        except (RuntimeError, subprocess.TimeoutExpired, queue.Full):
            # This child is still held by this Popen object; additionally verify
            # the complete birth. Never signal a stale/reused PID or a Sim.
            if self.process.poll() is None:
                fresh = _birth(self.process.pid)
                if (fresh["start_ticks"] == self.birth["start_ticks"]
                        and fresh["pgid"] == self.birth["pgid"]
                        and [x.decode() for x in fresh["argv"]] == self.birth["argv"]):
                    self.process.terminate()
                    self.process.wait(timeout=3)
            raise
        finally:
            self.stderr.close()


class TrainingVideo:
    def __init__(self, raw, runner, directory, binding):
        if video_config_errors(binding) or raw.num_envs != 1024 or raw.step_dt != 0.02:
            raise RuntimeError("Training video admission differs")
        if raw.sim.has_gui() or raw.sim.has_rtx_sensors():
            raise RuntimeError("Training video requires the reviewed offscreen viewer-only render path")
        self.raw, self.runner, self.contract = raw, runner, binding
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.sidecar = (self.directory / "frames.jsonl").open("x")
        self.windows = [dict(index=i, completed_update_range=w[:2], max_controls=w[2], max_frames=w[3],
            controls=0, attempts=0, accepted_frames=0, status="not_reached")
            for i, w in enumerate(binding["windows"])]
        self.encoder = None
        self.active = None
        self.manual_renders = 0
        self.closed = False
        self.resource_events = []
        self.previous_generation = int(raw.initialization_reset_generation[0])
        self.binary_sha256 = hashlib.sha256(Path(FFMPEG).read_bytes()).hexdigest()
        from isaaclab.sensors import RayCaster
        lidar = raw.scene.sensors["lidar"]
        if (type(lidar) is not RayCaster or lidar.cfg.mesh_prim_paths != ["/World/ground"]
                or lidar.cfg.update_period != raw.step_dt):
            raise RuntimeError("The reviewed ground-only Warp LiDAR path/cadence changed")
        self.lidar_descriptor = dict(class_name=type(lidar).__module__+"."+type(lidar).__name__,
            prim_path=lidar.cfg.prim_path, mesh_prim_paths=list(lidar.cfg.mesh_prim_paths),
            update_period_s=float(lidar.cfg.update_period))
        from pxr import UsdGeom
        for i in range(1, 1024):
            prim = raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}")
            if not prim.IsValid():
                raise RuntimeError("Training clone visual prim absent")
            UsdGeom.Imageable(prim).MakeInvisible()
        # Rendering creates no control, reset, policy query or physics step.
        self.initial_control_counter = int(raw.common_step_counter)
        self.initial_physics_counter = int(raw._sim_step_counter)
        for _ in range(4):
            self._render()
        if (int(raw.common_step_counter) != self.initial_control_counter
                or int(raw._sim_step_counter) != self.initial_physics_counter):
            raise RuntimeError("Static training video initialization advanced physics")
        self._resources("initialized_after_static_render")
        self._manifest()

    def _resources(self, event):
        import torch
        from launch_stair_evaluation import gpu_snapshot
        self.resource_events.append(dict(event=event, control=int(self.raw.common_step_counter),
            Torch_allocated=int(torch.cuda.memory_allocated()), Torch_reserved=int(torch.cuda.memory_reserved()),
            Torch_lifetime_peak_allocated=int(torch.cuda.max_memory_allocated()),
            actual_GPU_snapshot=gpu_snapshot(), peak_reset=False,
            GPU_snapshot_not_a_per_kernel_peak_or_complete_nonTorch_inventory=True))

    def _render(self):
        if self.manual_renders >= 504 or self.raw.sim.has_rtx_sensors():
            raise RuntimeError("Training render path or bound changed")
        self.manual_renders += 1
        control, physics = int(self.raw.common_step_counter), int(self.raw._sim_step_counter)
        result = self.raw.render(recompute=False)
        if (int(self.raw.common_step_counter) != control or int(self.raw._sim_step_counter) != physics):
            raise RuntimeError("An observational render advanced a physics/control epoch")
        return result

    def observe(self):
        c = int(self.runner.alg.completed_updates)
        selected = next((w for w in self.windows if w["completed_update_range"][0] <= c
            <= w["completed_update_range"][1] and w["controls"] < w["max_controls"]), None)
        if self.active is not None and self.active is not selected:
            self._close_window("complete" if self.active["controls"] == self.active["max_controls"] else "range_ended")
        if selected is None:
            return
        self.active = selected
        if selected["status"] == "not_reached":
            selected["status"] = "recording"
            self.encoder = _Encoder(self.directory / f"env0_window_{selected['index']}.mp4", FFMPEG)
            selected["encoder_birth"] = self.encoder.birth
            self._manifest()
        selected["controls"] += 1
        if selected["controls"] % 5 == 0:
            selected["attempts"] += 1
            frame = self._render()
            import numpy as np
            if frame.shape != (384, 512, 3) or frame.dtype != np.uint8:
                raise RuntimeError("Training RGB is empty or has an undeclared format")
            if frame.min() == frame.max():
                raise RuntimeError("Training RGB contains only a constant placeholder")
            if sum(w["accepted_frames"] for w in self.windows) == 0:
                self._resources("first_actual_RGB")
            from PIL import Image, ImageDraw
            generation = int(self.raw.initialization_reset_generation[0])
            label = dict(window=selected["index"], control=int(self.raw.common_step_counter),
                completed_updates=c, env_id=0, actual_row=int(self.raw.scene.terrain.terrain_levels[0]),
                actual_riser_height_m=float(self.raw._stair_heights()[0]),
                episode_reset_generation=generation, cut_reset=generation != self.previous_generation,
                frame_is_post_step_may_be_after_reset=True, training_mixed_weights_not_formal=True)
            observer = self.raw.stair_training_observer
            label["pre_reset_generation_at_this_control"] = int(observer.pre_generation[0])
            label["reset_in_this_control"] = generation != label["pre_reset_generation_at_this_control"]
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 512, 36), fill=(0, 0, 0))
            draw.text((4, 3), f"TRAIN env0 h=.15 update={c} control={label['control']}", fill="white")
            draw.text((4, 19), f"episode-generation={generation} " + ("CUT RESET" if label["cut_reset"] else ""), fill="white")
            self.encoder.add(np.asarray(image).tobytes())
            selected["accepted_frames"] += 1
            self.sidecar.write(json.dumps(label) + "\n")
            self.sidecar.flush()
            self.previous_generation = generation
        if selected["controls"] >= selected["max_controls"]:
            self._close_window("complete")

    def _close_window(self, status):
        window, encoder = self.active, self.encoder
        self.active = self.encoder = None
        if encoder is not None:
            try:
                encoder.close()
                window.update(status=status, written_frames=encoder.written,
                    encoder_returncode=encoder.process.returncode)
            except BaseException as error:
                window.update(status="encoder_failed", error=repr(error), written_frames=encoder.written)
                self._manifest()
                raise
        self._manifest()

    def snapshot(self):
        # Only owner-thread schedule counters enter packets. Asynchronous pipe
        # progress is in the separate manifest, never a racy packet/native bind.
        return dict(contract=self.contract, manual_render_calls=self.manual_renders,
            attempts=sum(w["attempts"] for w in self.windows),
            accepted_frames=sum(w["accepted_frames"] for w in self.windows),
            observed_controls=[w["controls"] for w in self.windows],
            encoder_binary_sha256=self.binary_sha256,
            original_ground_only_lidar=self.lidar_descriptor,
            resource_events=list(self.resource_events),
            has_rtx_sensors=bool(self.raw.sim.has_rtx_sensors()))

    def _manifest(self):
        value = dict(snapshot=self.snapshot(), windows=self.windows,
            training_not_formal=True, same_simulation=True)
        p = self.directory / "manifest.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(p)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.active is not None:
                self._close_window("partial_producer_ended")
            for window in self.windows:
                if window["status"] == "not_reached":
                    window["status"] = "skipped_not_reached"
            self._manifest()
        finally:
            self.sidecar.close()
