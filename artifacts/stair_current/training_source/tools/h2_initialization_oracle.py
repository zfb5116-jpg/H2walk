"""Query-only accepted flat teacher on a real direct-policy plant.

No teacher ActionManager is installed in the plant. process_actions is invoked
on a separate term; apply_actions, actuator compute and native setters are never
called. Teacher phase/history/perception are counterfactual query memory when
the student controls. They are not observable extra inputs to the student.
Importing this module does not import or launch Isaac Sim.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import types

import torch

from h2_bootstrap_contract import DirectActionHistory, teacher_position_label, assert_observation_layout


class OracleEpochCache:
    """Per-environment once/counter guard, including same-counter true resets."""
    def __init__(self, n, device):
        self.generation = torch.zeros(n, device=device, dtype=torch.long)
        self.updated_generation = torch.full_like(self.generation, -1)
        self.updated_counter = torch.full_like(self.generation, -1)
        self.last_counter = None

    def fresh(self, counter):
        counter = int(counter)
        if counter < 0 or (self.last_counter is not None and
                (counter < self.last_counter or counter > self.last_counter+1)):
            raise RuntimeError("Oracle must be queried at every real control epoch, without clock jumps")
        return (self.updated_counter != counter) | (self.updated_generation != self.generation)

    def commit(self, counter, fresh):
        self.updated_counter[fresh] = int(counter)
        self.updated_generation[fresh] = self.generation[fresh]
        self.last_counter = int(counter)

    def reset(self, ids):
        self.generation[ids] += 1


def snapshot_plant(raw):
    """Read command ports, real native parameters and transport counters."""
    robot = raw.scene["robot"]
    values = {"data_"+name: getattr(robot.data, name).clone() for name in
        ("joint_pos_target", "joint_vel_target", "joint_effort_target", "joint_stiffness",
         "joint_damping", "joint_effort_limits", "joint_armature")}
    view = robot.root_physx_view
    for name in ("masses", "dof_stiffnesses", "dof_dampings", "dof_max_forces",
                 "dof_max_velocities", "dof_armatures", "dof_friction_properties", "material_properties"):
        values["native_"+name] = getattr(view, "get_"+name)().clone()
    for group, actuator in robot.actuators.items():
        for kind in ("positions", "velocities", "efforts"):
            delay = getattr(actuator, kind+"_delay_buffer", None)
            if delay is None:
                raise RuntimeError("The original delayed native path is missing")
            prefix = group+"_"+kind
            values[prefix+"_lag"] = delay.time_lags.clone()
            ring = delay._circular_buffer
            values[prefix+"_pushes"] = ring._num_pushes.clone()
    return values


def require_unchanged(before, after):
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed or before.keys() != after.keys():
        raise RuntimeError("Query-only teacher changed plant ports/physical parameters/transport: "+str(changed))


class _TeacherProxy:
    def __init__(self, raw, cfg, history):
        self._raw, self.cfg, self.unwrapped = raw, cfg, self
        self.action_manager = types.SimpleNamespace(action=history.latest(), get_term=lambda _: self.term)
        original = getattr(raw, "_h2_lidar_reconstructor", None)
        if original is None:
            raise RuntimeError("Construct the oracle after the true reset/raw observation has loaded frozen LiDAR")
        self._h2_lidar_reconstructor = copy.deepcopy(original).eval().requires_grad_(False)
        self._h2_lidar_reconstructor.reset_temporal_state()
        self._h2_lidar_recon_hidden = None
        self._h2_lidar_recon_cache = None
        self._h2_lidar_ray_cache = None
        if any(a.data_ptr() == b.data_ptr() for a, b in
               zip(original.parameters(), self._h2_lidar_reconstructor.parameters())):
            raise RuntimeError("Teacher and student perception parameters share storage")

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def scan(self):
        term = self.cfg.observations.policy.height_scanner
        value = term.func(self, **term.params).clone()
        if term.clip is not None:
            value.clamp_(*term.clip)
        if term.scale is not None:
            value.mul_(term.scale)
        return value


class FlatInitializationOracle:
    """One original-controller query per actual Direct plant control epoch.

    query(raw_policy1029) returns labels only. The collector selects complete
    raw12 vectors and submits them through the plant DirectLegPositionAction.
    Neither queried residual history nor labels are written into student obs.
    """
    def __init__(self, raw, teacher_checkpoint, teacher_cfg=None, teacher_agent=None):
        from h2_task.assets import H2_JOINT_NAMES, H2_LEG_JOINT_NAMES
        from h2_task.direct_leg_action import DirectLegPositionAction
        from h2_task.manager.leg_action import BoundedJointPositionAction
        from h2_task.networks import H2ActorCriticOur
        from h2_task.robot_env_cfg import RobotStairsEnvCfg
        from h2_task.agents.rsl_rl_ppo_cfg import H2StairsPPORunnerCfg
        from rsl_rl.modules.normalizer import EmpiricalNormalization

        self.raw, self.n, self.device = raw, raw.num_envs, raw.device
        if not 0 < self.n <= 1024 or not hasattr(raw, "add_reset_callback"):
            raise ValueError("Require the bounded DirectInitializationEnv/reset API")
        plant_term = raw.action_manager.get_term("JointPositionAction")
        if type(plant_term) is not DirectLegPositionAction or plant_term.gait_reference is not None:
            raise ValueError("The plant must have no teacher/reference action path")
        self.ids = [int(i) for i in plant_term._joint_ids]
        if list(plant_term._joint_names) != list(H2_LEG_JOINT_NAMES):
            raise ValueError("Teacher/student action joint order changed")
        self.layout = assert_observation_layout(raw.observation_manager, plant_term)
        before = snapshot_plant(raw)
        cfg = copy.deepcopy(teacher_cfg if teacher_cfg is not None else RobotStairsEnvCfg())
        cfg.scene.num_envs = self.n
        cfg.observations.policy.enable_corruption = False
        cfg.observations.policy.height_scanner.params["side_noise_probability"] = 0.0
        if cfg.contact_gait_reference is None or cfg.stair_reference is not None:
            raise ValueError("The oracle must be the accepted flat ContactWalkingReference controller")
        self.history = DirectActionHistory(self.n, self.device, H2_LEG_JOINT_NAMES, H2_JOINT_NAMES)
        self.proxy = _TeacherProxy(raw, cfg, self.history)
        self.term = BoundedJointPositionAction(copy.deepcopy(cfg.actions.JointPositionAction), self.proxy)
        self.proxy.term = self.term
        if self.term._support_priority_groups or self.term.gait_reference is None:
            raise RuntimeError("Unexpected teacher force-port attachment or absent reference")
        for actuator in raw.scene["robot"].actuators.values():
            if getattr(actuator, "support_priority_buffer", None) is not None:
                raise RuntimeError("Effort-priority control is not part of this initializer")
        agent = teacher_agent if teacher_agent is not None else H2StairsPPORunnerCfg()
        pcfg = copy.deepcopy(agent["policy"] if isinstance(agent, dict) else agent.policy.to_dict())
        pcfg.pop("class_name")
        cp = Path(teacher_checkpoint).resolve(strict=True)
        self.checkpoint_sha256 = hashlib.sha256(cp.read_bytes()).hexdigest()
        if self.checkpoint_sha256 != "2b06db2e09061cc660d46578b112ea29d0e601c3e75ac2163f8425aa24553fcb":
            raise RuntimeError("Oracle checkpoint is not the retained accepted flat residual")
        saved = torch.load(cp, map_location="cpu", weights_only=True)
        self.actor = H2ActorCriticOur(1029, 866, 12, **pcfg).to(self.device)
        self.norm = EmpiricalNormalization([1029], until=1e8).to(self.device)
        self.actor.load_state_dict(saved["model_state_dict"], strict=True)
        self.norm.load_state_dict(saved["obs_norm_state_dict"], strict=True)
        for obj in (self.actor, self.norm):
            obj.eval().requires_grad_(False)
        self.epoch = OracleEpochCache(self.n, self.device)
        self._cache = None
        self.audit_summary = dict(queries=0, cache_hits=0, nonintrusive_audits=0, reset_rows=0,
            teacher_apply_calls=0, native_setter_calls=0, student_history_writes=0)
        robot = raw.scene["robot"]
        self._receipt_ratio = (robot.data.joint_damping[:, self.ids]/robot.data.joint_stiffness[:, self.ids]).clone()
        self.metadata = dict(checkpoint=str(cp), checkpoint_sha256=self.checkpoint_sha256,
            teacher_action_history="counterfactual queried residuals; not actual executed student actions",
            student_action_history="plant ActionManager only; no oracle rewrite",
            perception="independent deep-copied frozen parameters and reset temporal/GRU/cache state",
            once_clock="actual common_step_counter plus real reset generation",
            unobserved_state_risk="Teacher phase/anchors/gains are not extra student inputs; inconsistent labels must be reported, not repaired with hidden input channels",
            equivalence="q_teacher + D/K * v_teacher, zero external leg force; clipped labels are explicitly not exact",
            force_semantics="Published request algebra, not isolated measured native motor torque")
        require_unchanged(before, snapshot_plant(raw))
        raw.add_reset_callback(self.reset, when="after")

    def reset(self, env_ids=None):
        ids = (torch.arange(self.n, device=self.device) if env_ids is None else
               torch.as_tensor(env_ids, device=self.device, dtype=torch.long).unique())
        self.term.reset(ids)
        self.history.reset(ids)
        self.history._last_step[ids] = -1
        self.epoch.reset(ids)
        self.proxy._h2_lidar_reconstructor.reset_temporal_state(ids)
        hidden = self.proxy._h2_lidar_recon_hidden
        if hidden is not None:
            hidden[:, ids] = 0
        self.proxy._h2_lidar_recon_cache = None
        self.proxy._h2_lidar_ray_cache = None
        robot = self.raw.scene["robot"]
        self._receipt_ratio[ids] = robot.data.joint_damping[ids][:, self.ids]/robot.data.joint_stiffness[ids][:, self.ids]
        if hasattr(self, "audit_summary"):
            self.audit_summary["reset_rows"] += int(ids.numel())

    def _keep_nonfresh(self, ids):
        """Rare same-counter partial reset: do not double-advance other lanes."""
        gait = self.term.gait_reference
        tensors = [(gait, name, value[ids].clone()) for name, value in vars(gait).items()
                   if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.n
                   and name not in ("masses", "total_mass")]
        tensors += [(self.term, name, getattr(self.term, name)[ids].clone())
                    for name in ("_raw_actions", "_processed_actions")]
        model = self.proxy._h2_lidar_reconstructor
        for name in ("_tok_buf", "_tok_ptr", "_tok_fill", "_pose_rng", "_pose_mask",
                     "_pose_pos", "_pose_quat", "_pose_ptr", "_pose_fill"):
            value = getattr(model, name)
            if value is not None:
                tensors.append((model, name, value[ids].clone()))
        hidden = self.proxy._h2_lidar_recon_hidden
        return tensors, None if hidden is None else hidden[:, ids].clone()

    @torch.no_grad()
    def query(self, obs):
        counter = int(self.raw.common_step_counter)
        fresh = self.epoch.fresh(counter)
        if not bool(fresh.any()):
            self.audit_summary["cache_hits"] += 1
            return {**{name: value.clone() for name, value in self._cache.items()}, "new_query_mask": fresh}
        if obs.shape != (self.n, 1029) or not bool(torch.isfinite(obs).all()):
            raise ValueError("Oracle requires the real raw policy observation[N,1029]")
        before = snapshot_plant(self.raw)
        keep = (~fresh).nonzero().flatten()
        retained = self._keep_nonfresh(keep) if keep.numel() else None
        teacher_obs, _ = self.history.rewrite(obs)
        self.proxy.action_manager.action = self.history.latest()
        teacher_obs[:, 468:1029] = self.proxy.scan()
        teacher_raw_unclipped = self.actor.act_inference(self.norm(teacher_obs))
        teacher_raw = teacher_raw_unclipped.clamp(-100.0, 100.0)
        self.term.process_actions(teacher_raw)
        robot = self.raw.scene["robot"]
        K = robot.data.joint_stiffness[:, self.ids].clone()
        D = robot.data.joint_damping[:, self.ids].clone()
        if not torch.equal(D/K, self._receipt_ratio):
            raise RuntimeError("Teacher position/velocity equivalence lost its D/K-at-receipt precondition")
        if bool((robot.data.joint_effort_target[:, self.ids] != 0).any()):
            raise RuntimeError("Direct plant has a nonzero external leg force port")
        gait = self.term.gait_reference
        teacher_q = self.term._processed_actions.clone()
        teacher_v = gait.qdot.reshape(self.n, 12).clone()
        limits = robot.data.soft_joint_pos_limits[:, self.ids]
        labels = teacher_position_label(teacher_q, teacher_v, K, D, limits[..., 0], limits[..., 1])
        finite = torch.isfinite(teacher_raw).all(-1) & torch.isfinite(labels["raw"]).all(-1)
        valid = finite & (gait.failure == 0)
        values = dict(**labels, teacher_q=teacher_q, teacher_v=teacher_v,
            teacher_raw=teacher_raw.clone(), teacher_raw_unclipped=teacher_raw_unclipped.clone(),
            teacher_obs=teacher_obs, state=gait.state.clone(), valid=valid,
            invalid_reason_bits=(~finite).long()+2*(gait.failure != 0).long(),
            reference_failure=gait.failure.clone(), reference_time=gait.time.clone(),
            reference_steps=gait.steps.clone(), reference_stance=gait.stance.clone(),
            reference_anchors=gait.anchors.clone(), reference_q=gait.q.clone(),
            reference_qdot=gait.qdot.clone(), reference_ik_error=gait.ik_error.clone(),
            reference_foot_target=gait.foot_target.clone(), reference_load=gait.load.clone(),
            K=K, D=D, caps=robot.data.joint_effort_limits[:, self.ids].clone(),
            control_step=torch.full((self.n,), counter, device=self.device, dtype=torch.long),
            reset_generation=self.epoch.generation.clone())
        if retained is not None:
            tensors, hidden = retained
            for obj, name, old in tensors:
                getattr(obj, name)[keep] = old
            if hidden is not None:
                self.proxy._h2_lidar_recon_hidden[:, keep] = hidden
        # Commit only future teacher observations. Student obs(t) is untouched.
        self.history.append(teacher_raw, torch.zeros(self.n, device=self.device, dtype=torch.bool), step=counter)
        self.epoch.commit(counter, fresh)
        if self._cache is None:
            self._cache = {name: value.clone() for name, value in values.items()}
        else:
            for name, value in values.items():
                self._cache[name][fresh] = value[fresh]
        require_unchanged(before, snapshot_plant(self.raw))
        self.audit_summary["queries"] += 1
        self.audit_summary["nonintrusive_audits"] += 1
        return {**{name: value.clone() for name, value in self._cache.items()}, "new_query_mask": fresh}
