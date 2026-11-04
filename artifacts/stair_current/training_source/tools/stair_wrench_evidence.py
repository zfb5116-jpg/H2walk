"""Read-only H2 evaluator contact evidence.

No SimulationApp, view creation, stepping, configuration or state writes.  Call once after a completed physics tick,
BEFORE any reset, with the matching sole/root poses.  A control-boundary capture
is the last 5 ms physics snapshot, NOT a 20 ms control average.  Returned arrays
are owned CPU NumPy data in canonical env, left/right order.

    reader = build_foot_wrench_reader(raw, robot)
    sample = reader.capture(sole_positions_w, sole_quaternions_w,
        physics_step_index=raw._sim_step_counter,
        control_step_index=raw.common_step_counter,
        root_quat_w=same_frame_root_quat, no_push_window_confirmed=True)

Native normal/friction point sets have independent count/start arrays.  The
normal tuple is cloned AND CUDA-synchronized before the friction getter, which
reuses those native indexing buffers.  Friction is a signed force-on-body vector;
the normal-channel sign calibration must never be applied to it.

The whole-body force residual is diagnostic only: the installed documentation
defines world COM acceleration but does not certify its averaging alignment with
the contact-impulse window.  No CPU assertion, normal control or small residual
automatically validates friction.  All friction/net physics_valid flags remain
False until independently validated.  Reported force-at-point moments also do not
recover any independent unreported torsional/rolling free couple.
"""
from __future__ import annotations

import re
import numpy as np
import torch


def _raise_if_fatal(error):
    message = str(error).lower()
    if isinstance(error, (MemoryError, AssertionError, torch.cuda.OutOfMemoryError)) or any(token in message for token in (
            "out of memory", "device-side assert", "device side assert", "illegal memory access", "illegal address",
            "illegal instruction", "cuda", "cublas", "cudnn", "cusolver", "cufft")):
        raise error


def _owned(values):
    values = tuple(value.detach().clone() for value in values)
    devices = {value.device for value in values if value.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)
    return values


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def _rz(quaternion):
    """World Z dot body Z; scalar-first quaternion. Reject nonunit input outside."""
    return 1 - 2 * (quaternion[..., 1].square() + quaternion[..., 2].square())


class _PackedReadError(ValueError):
    def __init__(self, message, stats, count):
        super().__init__(message)
        self.stats, self.count = stats, count


def _segments(counts, starts, n, capacity):
    """Separate channel ownership; positive-length intervals must not overlap."""
    if counts.shape != (n, 1) or starts.shape != counts.shape:
        raise ValueError("count/start shape is not N x one ground filter")
    count, start = counts[:, 0].long(), starts[:, 0].long()
    total = int(count.sum())
    active = count > 0
    valid = (count >= 0) & ((count == 0) | ((start >= 0) & (start + count <= capacity)))
    stats = {
        "capacity": capacity, "used": total, "headroom": capacity - total,
        "max_count": int(count.max()) if n else 0,
        "max_end": int((start[active] + count[active]).max()) if bool(active.any()) else 0,
        "at_capacity": total >= capacity,
    }
    if not bool(valid.all()):
        raise _PackedReadError("packed segment outside native capacity", stats, count)
    if total >= capacity:
        raise _PackedReadError("native buffer at capacity; completeness unproven", stats, count)
    begin = start[active]
    if begin.numel():
        order = begin.argsort()
        begin, end = begin[order], (begin + count[active])[order]
        if begin.numel() > 1 and bool((begin[1:] < end.cummax(0).values[:-1]).any()):
            raise _PackedReadError("overlapping packed contact segments", stats, count)
    owner = torch.repeat_interleave(torch.arange(n, device=count.device), count)
    block = count.cumsum(0) - count
    index = start[owner] + torch.arange(total, device=count.device) - block[owner]
    return owner, index, count, stats


def _sum_rows(values, owner, n):
    out = values.new_zeros((n,) + values.shape[1:])
    out.index_add_(0, owner, values)
    return out


def _minmax(values, owner, n):
    low, high = values.new_full((n,), float("inf")), values.new_full((n,), -float("inf"))
    low.scatter_reduce_(0, owner, values, reduce="amin", include_self=True)
    high.scatter_reduce_(0, owner, values, reduce="amax", include_self=True)
    return low, high


class FootGroundWrenchReader:
    """Aggregate existing filtered foot views; all validity scopes are explicit."""

    def __init__(self, raw, robot=None, *, expected_mesh="/World/ground/terrain/mesh"):
        self.raw = raw
        self.robot = raw.scene["robot"] if robot is None else robot
        self.n = int(raw.num_envs)
        self.dt = float(raw.cfg.sim.dt)
        if abs(self.dt - .005) > 1e-9:
            raise ValueError("This H2 draft requires the actual 5 ms physics timestep")
        self.mesh = expected_mesh
        self.feet = tuple(getattr(raw.cfg, "feet_body_names",
            ("left_ankle_pitch_link", "right_ankle_pitch_link")))
        if len(self.feet) != 2 or len(set(self.feet)) != 2:
            raise ValueError("Require exactly two distinct ordered foot body names")
        self.sensors = [raw.scene.sensors.get(name)
            for name in ("stair_left_ground", "stair_right_ground")]
        self.unfiltered = raw.scene.sensors.get("contact_forces")
        self.unfiltered_ids = None
        self.unfiltered_covers_robot = False
        if self.unfiltered is not None:
            try:
                ids, names = self.unfiltered.find_bodies(self.feet, preserve_order=True)
                if tuple(names) == self.feet and len(set(ids)) == 2:
                    self.unfiltered_ids = list(ids)
                self.unfiltered_covers_robot = (
                    set(self.unfiltered.body_names) == set(self.robot.body_names))
            except (AttributeError, TypeError, ValueError):
                pass
        self.normal_sign = [0, 0]
        self.first_normal_positive_control = [None, None]
        self.last_physics_step_index = None
        self.frames = 0
        self.errors = [[], []]
        self.last_errors = ["", ""]

    def _layout(self, sensor, foot, device):
        if sensor is None:
            raise ValueError("existing filtered foot sensor missing")
        view = sensor.contact_physx_view
        capacity = int(view.max_contact_data_count)
        if int(view.sensor_count) != self.n or int(view.filter_count) != 1 or capacity <= 0:
            raise ValueError("unexpected foot/filter count or detailed contacts disabled")
        paths, filters = list(view.sensor_paths), list(view.filter_paths)
        if len(paths) != self.n or len(filters) != self.n:
            raise ValueError("incomplete native sensor/filter identity table")
        ids = []
        for path, pair in zip(paths, filters):
            match = re.fullmatch(r"/World/envs/env_(\d+)/Robot/" + re.escape(self.feet[foot]), path)
            if match is None or list(pair) != [self.mesh]:
                raise ValueError("foot body or exact terrain collision mesh does not match")
            ids.append(int(match.group(1)))
        if sorted(ids) != list(range(self.n)):
            raise ValueError("native rows do not uniquely cover canonical env IDs")
        return view, capacity, torch.tensor(ids, dtype=torch.long, device=device)

    @torch.inference_mode()
    def capture(self, sole_pos_w, sole_quat_w, *, physics_step_index,
                control_step_index=-1, root_quat_w=None,
                no_push_window_confirmed=False):
        """Capture all envs before reset. Never reuse another env/time's evidence.

        Caller guarantees the simulator cannot advance during this function and
        supplied poses match this tick.  Per-env resets require a pre-reset hook;
        reading reset poses after a step cannot reconstruct terminal wrenches.
        no_push_window_confirmed is an explicit caller claim, default False.
        """
        step = int(physics_step_index)
        if step < 0 or (self.last_physics_step_index is not None and step <= self.last_physics_step_index):
            raise ValueError("Capture once per strictly increasing completed physics tick")
        pos = torch.as_tensor(sole_pos_w, device=getattr(self.raw, "device", None)).detach().clone()
        quat = torch.as_tensor(sole_quat_w, device=pos.device, dtype=pos.dtype).detach().clone()
        if pos.shape != (self.n, 2, 3) or quat.shape != (self.n, 2, 4):
            raise ValueError("Require same-frame canonical N x 2 sole positions and scalar-first quaternions")
        n, device, dtype = self.n, pos.device, pos.dtype
        output = {}
        def floating(name, tail=()):
            output[name] = torch.full((n, 2) + tail, float("nan"), device=device, dtype=dtype)
        def flag(name):
            output[name] = torch.zeros(n, 2, device=device, dtype=torch.bool)
        for key in ("normal_force_w_N", "normal_moment_about_sole_w_Nm", "friction_force_w_N",
                    "friction_moment_about_sole_w_Nm", "net_force_w_N", "net_moment_about_sole_w_Nm",
                    "force_weighted_normal_cop_w_m"):
            floating(key, (3,))
        for key in ("normal_wrench_about_sole_w", "friction_wrench_about_sole_w", "net_wrench_about_sole_w"):
            floating(key, (6,))
        for key in ("filtered_normal_fz_N", "unfiltered_foot_normal_fz_N", "normal_sum_error_N",
                    "same_foot_fz_error_N", "contact_plane_zspread_m", "minimum_separation_m",
                    "maximum_separation_m"):
            floating(key)
        for key in ("normal_read_valid", "friction_read_valid", "normal_aggregate_check",
                    "same_foot_normal_check", "normal_positive_control_established", "normal_physics_valid",
                    "normal_cop_valid", "net_wrench_read_valid", "friction_physics_valid", "net_wrench_physics_valid"):
            flag(key)
        for channel in ("normal", "friction"):
            for suffix in ("count", "buffer_capacity", "buffer_used", "buffer_headroom", "buffer_max_count", "buffer_max_end"):
                output[f"{channel}_{suffix}"] = torch.full((n, 2), -1, device=device, dtype=torch.long)
            flag(f"{channel}_buffer_at_capacity")
        output["sole_pos_w_m"] = pos
        output["sole_quat_w_wxyz"] = quat
        pose_ok = (torch.isfinite(pos).all(-1) & torch.isfinite(quat).all(-1)
                   & ((quat.norm(dim=-1) - 1).abs() < .002))
        tick_before = getattr(self.raw, "_sim_step_counter", None)
        unfiltered = None
        try:
            unfiltered = self.unfiltered.data.net_forces_w.detach().clone()
            if (self.unfiltered_ids is None or unfiltered.ndim != 3 or unfiltered.shape[0] != n
                    or unfiltered.shape[-1] != 3 or max(self.unfiltered_ids) >= unfiltered.shape[1]):
                unfiltered = None
        except (AttributeError, RuntimeError, ValueError) as error:
            _raise_if_fatal(error)
        if unfiltered is not None:
            output["unfiltered_foot_normal_fz_N"] = unfiltered[:, self.unfiltered_ids, 2].clone()
        self.last_errors = ["", ""]
        def packed(channel, counts, starts, ids, foot, capacity):
            try:
                result = _segments(counts, starts, n, capacity)
                stats, count = result[3], result[2]
            except _PackedReadError as error:
                stats, count = error.stats, error.count
                for key, value in stats.items():
                    if f"{channel}_buffer_{key}" in output:
                        output[f"{channel}_buffer_{key}"][:, foot] = value
                output[f"{channel}_count"][ids, foot] = count
                raise
            for key, value in stats.items():
                if f"{channel}_buffer_{key}" in output:
                    output[f"{channel}_buffer_{key}"][:, foot] = value
            output[f"{channel}_count"][ids, foot] = count
            return result
        for foot, sensor in enumerate(self.sensors):
            try:
                view, capacity, ids = self._layout(sensor, foot, device)
                # Backend buffers alias: complete clone AND sync before friction.
                magnitude, pn, normals, separation, cn, sn = _owned(view.get_contact_data(dt=self.dt))
                friction_error = None
                try:
                    ft_buf, pt, ct, st = _owned(view.get_friction_data(dt=self.dt))
                except Exception as error:
                    _raise_if_fatal(error)
                    friction_error = str(error)
                matrix, = _owned((view.get_contact_force_matrix(dt=self.dt),))
                if (magnitude.shape != (capacity, 1) or pn.shape != (capacity, 3)
                        or normals.shape != pn.shape or separation.shape != (capacity, 1)
                        or matrix.shape != (n, 1, 3)):
                    raise ValueError("unexpected detailed normal/matrix native shape")
                owner, ix, count, stats = packed("normal", cn, sn, ids, foot, capacity)
                pn, nv, sep = pn[ix], normals[ix], separation[ix, 0]
                fn_unsigned = magnitude[ix] * nv
                finite = (torch.isfinite(pn).all(-1) & torch.isfinite(fn_unsigned).all(-1)
                          & torch.isfinite(nv).all(-1) & torch.isfinite(sep))
                bad = _sum_rows((~finite).long(), owner, n) > 0
                clean = torch.where(finite[:, None], fn_unsigned, 0)
                aggregate = _sum_rows(clean, owner, n)
                expected = matrix[:, 0]
                tolerance = 1 + .02 * expected.norm(dim=-1)
                original = output["unfiltered_foot_normal_fz_N"][ids, foot]
                same_foot = (torch.isfinite(original) & torch.isfinite(expected).all(-1)
                             & ((original - expected[:, 2]).abs() <= 1 + .02 * expected[:, 2].abs()))
                low, high = _minmax(pn[:, 2], owner, n)
                unit_good = ((nv.norm(dim=-1) - 1).abs() < .05) & (nv[:, 2].abs() > .98) & finite
                all_points_good = _sum_rows((~unit_good).long(), owner, n) == 0
                depth_good = (sep >= -.010) & (sep <= .002) & finite
                all_depth_good = _sum_rows((~depth_good).long(), owner, n) == 0
                calibration = ((count >= 3) & ~bad & all_points_good & all_depth_good
                    & (high - low < .004) & (expected[:, 2] >= 20)
                    & pose_ok[ids, foot] & (_rz(quat[ids, foot]) > .995) & same_foot)
                if not self.normal_sign[foot]:
                    direct = calibration & ((aggregate - expected).norm(dim=-1) <= tolerance)
                    reverse = calibration & ((-aggregate - expected).norm(dim=-1) <= tolerance)
                    if bool(direct.any()) != bool(reverse.any()):
                        self.normal_sign[foot] = 1 if bool(direct.any()) else -1
                        row = int((direct if self.normal_sign[foot] == 1 else reverse).nonzero()[0, 0])
                        self.first_normal_positive_control[foot] = {
                            "env_id": int(ids[row]), "physics_step_index": step,
                            "control_step_index": int(control_step_index), "normal_sign": self.normal_sign[foot],
                            "body_name": self.feet[foot], "mesh": self.mesh,
                            "raw_fz_N": float(self.normal_sign[foot] * aggregate[row, 2]),
                            "filtered_fz_N": float(expected[row, 2]), "unfiltered_fz_N": float(original[row]),
                            "normal_point_count": int(count[row]),
                        }
                sign = self.normal_sign[foot]
                fn = clean * sign
                nf = _sum_rows(fn, owner, n)
                nm = _sum_rows(torch.cross(pn - pos[ids[owner], foot], fn, dim=-1), owner, n)
                valid = ~bad & pose_ok[ids, foot] & bool(sign)
                aggregate_check = (nf - expected).norm(dim=-1) <= tolerance
                physical = valid & aggregate_check & same_foot & (self.first_normal_positive_control[foot] is not None)
                for name, value in (("normal_force_w_N", nf), ("normal_moment_about_sole_w_Nm", nm),
                        ("normal_wrench_about_sole_w", torch.cat((nf, nm), -1))):
                    output[name][ids, foot] = torch.where(valid[:, None], value, float("nan"))
                for name, value in (("normal_read_valid", valid), ("normal_aggregate_check", aggregate_check),
                        ("same_foot_normal_check", same_foot), ("normal_physics_valid", physical)):
                    output[name][ids, foot] = value
                output["normal_positive_control_established"][:, foot] = self.first_normal_positive_control[foot] is not None
                output["filtered_normal_fz_N"][ids, foot] = expected[:, 2]
                output["normal_sum_error_N"][ids, foot] = (nf - expected).norm(dim=-1)
                output["same_foot_fz_error_N"][ids, foot] = (original - expected[:, 2]).abs()
                output["contact_plane_zspread_m"][ids, foot] = torch.where(count > 0, high - low, float("nan"))
                smin, smax = _minmax(sep, owner, n)
                output["minimum_separation_m"][ids, foot] = torch.where(count > 0, smin, float("nan"))
                output["maximum_separation_m"][ids, foot] = torch.where(count > 0, smax, float("nan"))
                positive_z = fn[:, 2] > 0
                weights = torch.where(positive_z, fn[:, 2], 0)
                cop = _sum_rows(pn * weights[:, None], owner, n) / _sum_rows(weights, owner, n).clamp_min(1e-12)[:, None]
                upward = _sum_rows((~positive_z).long(), owner, n) == 0
                cop_valid = (physical & (count > 0) & (nf[:, 2] >= 20) & all_points_good & all_depth_good
                             & upward & (high - low < .004) & (_rz(quat[ids, foot]) > .995))
                output["normal_cop_valid"][ids, foot] = cop_valid
                output["force_weighted_normal_cop_w_m"][ids, foot] = torch.where(cop_valid[:, None], cop, float("nan"))
                if friction_error is not None:
                    raise ValueError("friction getter unavailable: " + friction_error)
                if ft_buf.shape != (capacity, 3) or pt.shape != ft_buf.shape:
                    raise ValueError("unexpected friction force/anchor native shape")
                towner, tix, tcount, tstats = packed("friction", ct, st, ids, foot, capacity)
                ft, pt = ft_buf[tix], pt[tix]
                tfine = torch.isfinite(ft).all(-1) & torch.isfinite(pt).all(-1)
                tvalid = (_sum_rows((~tfine).long(), towner, n) == 0) & pose_ok[ids, foot]
                ft = torch.where(tfine[:, None], ft, 0)
                tf = _sum_rows(ft, towner, n)
                tm = _sum_rows(torch.cross(pt - pos[ids[towner], foot], ft, dim=-1), towner, n)
                for name, value in (("friction_force_w_N", tf), ("friction_moment_about_sole_w_Nm", tm),
                        ("friction_wrench_about_sole_w", torch.cat((tf, tm), -1))):
                    output[name][ids, foot] = torch.where(tvalid[:, None], value, float("nan"))
                output["friction_read_valid"][ids, foot] = tvalid
                net_valid = valid & tvalid
                output["net_wrench_read_valid"][ids, foot] = net_valid
                for name, value in (("net_force_w_N", nf + tf), ("net_moment_about_sole_w_Nm", nm + tm),
                        ("net_wrench_about_sole_w", torch.cat((nf + tf, nm + tm), -1))):
                    output[name][ids, foot] = torch.where(net_valid[:, None], value, float("nan"))
            except Exception as error:
                _raise_if_fatal(error)
                self.last_errors[foot] = type(error).__name__ + ": " + str(error)
                self.errors[foot].append({"physics_step_index": step, "reason": self.last_errors[foot]})

        # Latest native world link-COM acceleration; no finite-difference claim.
        nan3 = pos.new_full((n, 3), float("nan"))
        mass, com_a = pos.new_full((n,), float("nan")), nan3.clone()
        dynamics_valid = torch.zeros(n, device=device, dtype=torch.bool)
        dynamics_error = ""
        try:
            accel, = _owned((self.robot.root_physx_view.get_link_accelerations(),))
            masses, = _owned((self.robot.root_physx_view.get_masses(),))
            accel, masses = accel.to(device=device, dtype=dtype), masses.to(device=device, dtype=dtype)
            if masses.ndim != 2 or masses.shape[0] != n or accel.shape != masses.shape + (6,):
                raise ValueError("native mass/link COM acceleration layout differs")
            dynamics_valid = (torch.isfinite(masses).all(-1) & (masses >= 0).all(-1)
                              & torch.isfinite(accel).all(dim=(-1, -2)) & (masses.sum(-1) > 0))
            mass = masses.sum(-1)
            com_a = (masses[..., None] * accel[..., :3]).sum(1) / mass[:, None]
        except Exception as error:
            _raise_if_fatal(error)
            dynamics_error = type(error).__name__ + ": " + str(error)
        nonfoot = pos.new_full((n,), float("nan"))
        if unfiltered is not None:
            mask = torch.ones(unfiltered.shape[1], device=device, dtype=torch.bool)
            mask[self.unfiltered_ids] = False
            nonfoot = unfiltered[:, mask].norm(dim=-1).sum(-1)
        gravity = pos.new_tensor(getattr(self.raw.cfg.sim, "gravity", (0., 0., -9.81)))
        total_n = output["normal_force_w_N"].sum(1)
        total_f = output["friction_force_w_N"].sum(1)
        normal_residual = total_n + mass[:, None] * gravity - mass[:, None] * com_a
        net_residual = normal_residual + total_f
        root_upright = torch.zeros(n, device=device, dtype=torch.bool)
        if root_quat_w is not None:
            rq = torch.as_tensor(root_quat_w, device=device, dtype=dtype).detach().clone()
            if rq.shape != (n, 4):
                raise ValueError("root_quat_w must be matching canonical N x 4")
            root_upright = (torch.isfinite(rq).all(-1) & ((rq.norm(dim=-1) - 1).abs() < .002)
                            & (_rz(rq) > .984807753))  # 10 degrees: diagnostic window only.
        no_push = torch.as_tensor(no_push_window_confirmed, device=device, dtype=torch.bool)
        if no_push.ndim == 0:
            no_push = no_push.expand(n)
        if no_push.shape != (n,):
            raise ValueError("no_push_window_confirmed must be scalar or N")
        frame_stable = tick_before is not None and tick_before == getattr(self.raw, "_sim_step_counter", None)
        candidate = (dynamics_valid & output["normal_physics_valid"].all(-1)
            & output["net_wrench_read_valid"].all(-1) & torch.isfinite(nonfoot)
            & (nonfoot <= torch.maximum(mass * .0981, mass.new_full((n,), 5.)))
            & self.unfiltered_covers_robot & root_upright & no_push & frame_stable)
        if not frame_stable:
            for key in tuple(output):
                if key.endswith("valid") or key.endswith("check"):
                    output[key].zero_()
        output.update({
            "normal_sign": torch.tensor(self.normal_sign, device=device).expand(n, -1),
            "whole_body_mass_kg": mass, "mass_weighted_com_acceleration_w_m_s2": com_a,
            "body_com_acceleration_read_valid": dynamics_valid,
            "nonfoot_normal_force_norm_sum_N": nonfoot,
            "normal_only_force_residual_w_N": normal_residual,
            "normal_plus_friction_force_residual_w_N": net_residual,
            "friction_resultant_w_N": total_f,
            "nonzero_friction_resultant": total_f.norm(dim=-1) >= 5.,
            "friction_closure_candidate_window": candidate,
            # Deliberately never promoted by synthetic algebra or one snapshot.
            "acceleration_impulse_window_alignment_validated": torch.zeros(n, device=device, dtype=torch.bool),
        })
        result = {name: _numpy(value) for name, value in output.items()}
        result.update({
            "physics_step_index": np.int64(step), "control_step_index": np.int64(control_step_index),
            "window_duration_s": np.float64(self.dt), "same_frame_counter_unchanged": np.bool_(frame_stable),
            "same_frame_counter_available": np.bool_(tick_before is not None),
            "all_robot_bodies_in_unfiltered_sensor": np.bool_(self.unfiltered_covers_robot),
            "channel_error": np.asarray(self.last_errors), "dynamics_error": np.asarray(dynamics_error),
        })
        self.last_physics_step_index, self.frames = step, self.frames + 1
        return result

    def summary(self):
        return {
            "diagnostic_measurement": True, "friction_physically_validated": False,
            "foot_order": list(self.feet), "mesh_filter": self.mesh,
            "wrench_component_order": ["Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm"],
            "moment_reference": "matching actual sole_pos_w_m, world axes",
            "time_scope": "latest completed 5ms physics snapshot at a control boundary, never 20ms average",
            "friction_validation": "UNVALIDATED; physics_valid remains false, including after CPU mock assertions",
            "acceleration_scope": "native world link-COM instantaneous/backend acceleration; impulse-window alignment unvalidated",
            "buffer_scope": "bounds/headroom/segment checks are necessary, not proof that native friction lost no samples",
            "unreported_free_couples": "no reconstruction guarantee",
            "frames": self.frames, "normal_sign": list(self.normal_sign),
            "first_normal_positive_control": self.first_normal_positive_control,
            "unfiltered_foot_body_ids": self.unfiltered_ids,
            "unfiltered_covers_all_robot_bodies": self.unfiltered_covers_robot,
            "read_errors": self.errors,
        }


def build_foot_wrench_reader(raw, robot=None, **kwargs):
    return FootGroundWrenchReader(raw, robot, **kwargs)
