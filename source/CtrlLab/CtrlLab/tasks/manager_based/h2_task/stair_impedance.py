"""Smooth native drive handover after the flat approach has finished.

Only leg impedance changes. Native effort, velocity, friction, armature and
sampled setpoint delay remain owned by the original actuators.
"""

import torch


def stair_gain_factor(stiffness,inertia,frequency):
    """Hip/knee bandwidth plus ankle-roll preload headroom, in leg order."""
    factor=(inertia*frequency**2/stiffness).clamp(1,8)
    mask=torch.tensor([True,True,False,True,False,False]*2,device=stiffness.device)
    factor=torch.where(mask,factor,1)
    # At loaded split-height poses the original80N m/rad roll drive can need
    # q+tau/K outside its soft setpoint interval even at only4--6N m. The
    # measured18.5mrad headroom minus4mrad margin needs >=285N m/rad for
    # 4.13N m. Four times the original80 meets this with the same19N m cap.
    factor[:,[4,10]]=4
    return factor


class StairImpedance:
    def __init__(self, gait):
        self.gait, self.robot = gait, gait.robot
        self.base_k = self.base_d = None
        self.age = torch.zeros_like(gait.time)
        self.latched = torch.zeros_like(gait.approach)
        self.factor = torch.ones_like(gait.q)
        self.base_ratio = torch.ones_like(gait.q)
        self.first_verification = None
        self.groups = []
        for actuator in self.robot.actuators.values():
            native = (list(range(self.robot.num_joints)) if isinstance(actuator.joint_indices, slice)
                      else [int(j) for j in actuator.joint_indices])
            actions, local = [], []
            for action, joint in enumerate(gait.joint_ids):
                if int(joint) in native:
                    actions.append(action)
                    local.append(native.index(int(joint)))
            if actions:
                if not actuator.is_implicit_model:
                    raise ValueError("Stair impedance requires native implicit leg drives")
                self.groups.append((actuator, actions, local))

    def _write(self, ids, k, d):
        # Articulation.write_* updates native/data buffers, not actuator caches.
        # The latter also supply gravity preload and the effort estimator.
        for actuator, actions, local in self.groups:
            actuator.stiffness[ids[:, None], local] = k[:, actions]
            actuator.damping[ids[:, None], local] = d[:, actions]
        self.robot.write_joint_stiffness_to_sim(k, joint_ids=self.gait.joint_ids, env_ids=ids)
        self.robot.write_joint_damping_to_sim(d, joint_ids=self.gait.joint_ids, env_ids=ids)
        # Nominal planning/cache gains above remain intact. A received support
        # role owns the final native gains/cap until its own delayed handback.
        for actuator,_,_ in self.groups:
            if getattr(actuator,"support_priority_buffer",None) is not None:
                actuator.apply_support_native_budget(ids)
                break  # The writer aggregates every enabled group atomically.

    def reset(self, ids=None):
        ids = (torch.arange(self.gait.n, device=self.gait.device) if ids is None
               or isinstance(ids, slice) else ids)
        if self.base_k is not None:
            self._write(ids, self.base_k[ids], self.base_d[ids])
        self.age[ids], self.latched[ids] = 0, False
        self.factor[ids], self.base_ratio[ids] = 1, 1

    @torch.no_grad()
    def update(self):
        gait = self.gait
        if self.base_k is None:
            # Lazy initialization occurs after startup domain randomization.
            self.base_k = self.robot.data.joint_stiffness[:, gait.joint_ids].clone()
            self.base_d = self.robot.data.joint_damping[:, gait.joint_ids].clone()
            if not bool(torch.isfinite(self.base_k).all() & torch.isfinite(self.base_d).all()
                        & (self.base_k > 0).all() & (self.base_d > 0).all()):
                raise ValueError("Leg impedance must start with finite positive K/D")
        active = ~gait.approach
        entering = (active & ~self.latched).nonzero().flatten()
        if entering.numel():
            matrix = self.robot.root_physx_view.get_generalized_mass_matrices()
            size = self.robot.num_joints + 6
            if matrix.shape != (gait.n, size, size):
                raise ValueError("Stair impedance requires floating-base native mass matrices")
            joint = torch.as_tensor(gait.joint_ids, device=gait.device) + 6
            inertia = matrix[entering[:, None], joint[None], joint[None]]
            if not bool(torch.isfinite(inertia).all() & (inertia > 0).all()):
                raise RuntimeError("Non-finite or non-positive native leg inertia")
            frequency = gait.options["stair_drive_frequency_rad_s"]
            factor = stair_gain_factor(self.base_k[entering],inertia,frequency)
            self.factor[entering] = factor.reshape(-1, 2, 6)
            self.latched[entering] = True
        ids = (active & (self.age < gait.options["stair_drive_handover_s"])).nonzero().flatten()
        if not ids.numel():
            return
        self.age[ids] += gait.env.step_dt
        u = (self.age[ids] / gait.options["stair_drive_handover_s"]).clamp(0, 1)
        u = u**3 * (10 - 15*u + 6*u*u)
        factor = 1 + u[:, None] * (self.factor[ids].reshape(-1, 12) - 1)
        k, d = self.base_k[ids] * factor, self.base_d[ids] * factor.sqrt()
        self._write(ids, k, d)
        self.base_ratio[ids] = (self.base_k[ids]/k).reshape(-1, 2, 6)
        finished = ids[self.age[ids] >= gait.options["stair_drive_handover_s"]]
        if self.first_verification is None and finished.numel():
            actual_k = self.robot.root_physx_view.get_dof_stiffnesses().to(gait.device)
            actual_d = self.robot.root_physx_view.get_dof_dampings().to(gait.device)
            expected_k = self.robot.data.joint_stiffness[finished][:, gait.joint_ids]
            expected_d = self.robot.data.joint_damping[finished][:, gait.joint_ids]
            solver_k,solver_d = expected_k.clone(),expected_d.clone()
            for actuator,actions,local in self.groups:
                if getattr(actuator,"support_priority_buffer",None) is not None:
                    fraction = 1-actuator._support_effective_role[finished[:,None],local]
                    solver_k[:,actions] *= fraction
                    solver_d[:,actions] *= fraction
            k_error = float((actual_k[finished][:, gait.joint_ids]-solver_k).abs().max())
            d_error = float((actual_d[finished][:, gait.joint_ids]-solver_d).abs().max())
            cache_k_error = cache_d_error = 0.
            for actuator,actions,local in self.groups:
                cache_k_error = max(cache_k_error,float((actuator.stiffness[finished[:,None],local]-expected_k[:,actions]).abs().max()))
                cache_d_error = max(cache_d_error,float((actuator.damping[finished[:,None],local]-expected_d[:,actions]).abs().max()))
            if max(k_error,d_error,cache_k_error,cache_d_error) > .001:
                raise RuntimeError("Native stair impedance does not match requested K/D")
            sample = int(finished[0])
            self.first_verification = dict(env_id=sample, native_K_error=k_error,
                native_D_error=d_error, actuator_K_error=cache_k_error,actuator_D_error=cache_d_error,
                base_K=self.base_k[sample].cpu().tolist(),
                base_D=self.base_d[sample].cpu().tolist(),
                requested_K=solver_k[0].cpu().tolist(), requested_D=solver_d[0].cpu().tolist(),
                nominal_K=expected_k[0].cpu().tolist(), nominal_D=expected_d[0].cpu().tolist(),
                frequency_rad_s=gait.options["stair_drive_frequency_rad_s"],
                handover_s=gait.options["stair_drive_handover_s"],
                physical_effort_velocity_delay_limits_unchanged=True)
