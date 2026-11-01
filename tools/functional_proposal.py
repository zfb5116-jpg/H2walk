"""Prospective Adam and a finite-real-bank simultaneous output projection.

This module does not step an optimizer, run its hooks, or launch an environment.
The projection is first order only. The caller must retain its exact nonlinear
bank check, coherent whole-update rollback, and one explicit std hook on commit.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jacrev, jvp


KIND = "simultaneous_actual_bank_linear_v1"
RESERVED_KIND = "simultaneous_actual_bank_reserved_linear_v1"
LINEAR_TOLERANCE_RAD = 2e-7
ROW_VALUE_TOLERANCE_RAD = 5e-6
MAX_BANK_ROWS, NUM_OUTPUTS, NUM_MEAN_PARAMETERS = 5120, 12, 44
GRAM_COLUMN_CHUNK = 65536
MAX_CPU_JACOBIAN_ARENA_BYTES = 2**30
JACOBIAN_RESOURCE_KIND = "single_cpu_float32_jacobian_arena_v1"
PRIMAL_PRECISION_POLICY = "direct_fp64_primal_envelope_v1"
MAX_KKT_POLISH_CALLS = 28
MAX_KKT_POLISH_SUPPORT = 672
FIXED_SUPPORT_POLISH_POLICY = "fixed_positive_support_v1"
BOUNDED_SUPPORT_POLISH_POLICY = "bounded_dual_active_set_v1"
MAX_BOUNDED_POLISH_CALLS_PER_QP = 2 * MAX_KKT_POLISH_SUPPORT + 1
MAX_BOUNDED_POLISH_CALLS_PER_PROJECTION = MAX_KKT_POLISH_CALLS * MAX_BOUNDED_POLISH_CALLS_PER_QP
BOUNDED_BOUNDARY_ARITHMETIC = "exact_rational_from_represented_fp64_endpoints_then_fp64"


def _metadata_value(value):
    """Diagnostic packets remain JSON/weights_only-safe, even on failure."""
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite:" + str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Projection diagnostic metadata contains " + type(value).__name__)


class ProposalProjectionError(RuntimeError):
    """A bounded proposal/projection could not produce its declared certificate."""

    def __init__(self, metadata, *, diagnostic_evidence=None):
        self.metadata = _metadata_value(dict(metadata))
        # Arrays are deliberately NOT metadata/checkpoint fields. The caller
        # may persist this separate, non-deployable NPZ only on rejection.
        self.diagnostic_evidence = diagnostic_evidence
        self.metadata.setdefault("feasible", False)
        super().__init__(str(self.metadata.get("reason", "Proposal projection failed")))


def _require(condition, reason, **metadata):
    if not condition:
        raise ProposalProjectionError(dict(reason=reason, **metadata))


def _finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _version(value):
    try:
        return value._version
    except RuntimeError:
        return None


@dataclass
class AdamProposal:
    """A one-use Adam transition, with independent tensors for all new state.

    ``commit`` changes the real parameters and moments exactly once, but never
    calls optimizer.step or any registered hook. The caller owns its std hook.
    Unchanged original deltas use the prospective torch.addcdiv target directly
    to preserve the native Adam floating-point result, including for std.
    """

    parameters: tuple[torch.Tensor, ...]
    deltas: tuple[torch.Tensor, ...]
    new_states: tuple[dict, ...]
    _optimizer: torch.optim.Optimizer = field(repr=False)
    _targets: tuple[torch.Tensor, ...] = field(repr=False)
    _parameter_versions: tuple = field(repr=False)
    _state_versions: tuple = field(repr=False)
    _delta_versions: tuple = field(repr=False)
    _state_sources: tuple = field(repr=False)
    _configuration: tuple = field(repr=False)
    _committed: bool = field(default=False, init=False, repr=False)

    @property
    def committed(self):
        return self._committed

    def commit(self, deltas):
        _require(not self._committed, "Adam proposal was already committed")
        ds = tuple(deltas)
        _require(len(ds) == len(self.parameters), "Adam commit requires all45 deltas")
        _require(_adam_configuration(self._optimizer) == self._configuration,
                 "Adam configuration changed after proposal")
        for i, (p, d, expected_version) in enumerate(zip(self.parameters, ds, self._parameter_versions)):
            _require(isinstance(d, torch.Tensor) and d.shape == p.shape
                     and d.dtype == p.dtype and d.device == p.device and _finite(d),
                     "Invalid Adam commit delta", parameter=i)
            _require(_version(p) == expected_version, "Real parameter changed after proposal", parameter=i)
            _require(self._optimizer.state.get(p) is self._state_sources[i],
                     "Real Adam state was replaced after proposal", parameter=i)
            source = self._state_sources[i]
            versions = tuple((key, _version(value)) for key, value in source.items()
                             if isinstance(value, torch.Tensor)) if source is not None else ()
            _require(versions == self._state_versions[i], "Real Adam state changed after proposal", parameter=i)
            _require(_version(self.deltas[i]) == self._delta_versions[i],
                     "Original proposal delta was mutated", parameter=i)
        # All validation precedes the first mutation. No real optimizer hook is
        # dispatched here: the original std projection is explicit at the caller.
        with torch.no_grad():
            for i, (p, d, state) in enumerate(zip(self.parameters, ds, self.new_states)):
                if d is self.deltas[i]:
                    p.copy_(self._targets[i])
                else:
                    p.add_(d)
                if state:
                    self._optimizer.state[p] = state
        self._committed = True


def _adam_configuration(optimizer):
    _require(type(optimizer) is torch.optim.Adam, "Only the current torch.optim.Adam is supported")
    _require(len(optimizer.param_groups) == 1, "Require one unchanged45-parameter Adam group")
    group = optimizer.param_groups[0]
    expected = {"lr": 1e-5, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0,
                "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}
    for key, value in expected.items():
        _require(group.get(key, False if isinstance(value, bool) else None) == value,
                 "Unsupported Adam configuration", field=key, actual=str(group.get(key)))
    _require(group.get("fused") in (None, False) and group.get("foreach") is False,
             "Prospective Adam requires explicit foreach=False (single-tensor path)")
    _require("decoupled_weight_decay" not in group or not group["decoupled_weight_decay"],
             "Decoupled weight decay is unsupported")
    return tuple((key, group.get(key)) for key in
                 ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable",
                  "differentiable", "fused", "foreach", "decoupled_weight_decay"))


def adam_proposal(optimizer):
    """Compute the fixed Adam transition without live parameter/state/RNG edits.

    Gradients are the caller's already clipped gradients. State tensors are
    cloned before the same single-tensor Adam moment operations. A missing
    gradient behaves exactly like Adam: no moment/step update for that parameter.
    """
    configuration = _adam_configuration(optimizer)
    group = optimizer.param_groups[0]
    parameters = tuple(group["params"])
    _require(len(parameters) == 45 and len({id(p) for p in parameters}) == 45,
             "Require the original44 actor parameters followed by std")
    _require(parameters[-1].shape == (12,), "The45th original Adam parameter must be std12")
    targets, deltas, new_states, sources, source_versions = [], [], [], [], []
    beta1, beta2 = group["betas"]
    with torch.no_grad():
        for index, p in enumerate(parameters):
            _require(p.is_floating_point() and not p.is_complex() and p.dtype in (torch.float32, torch.float64),
                     "Only the current real floating Adam parameters are supported", parameter=index)
            source = optimizer.state.get(p)
            sources.append(source)
            source_versions.append(tuple((key, _version(value)) for key, value in source.items()
                                         if isinstance(value, torch.Tensor)) if source is not None else ())
            state = {key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                     for key, value in source.items()} if source else {}
            grad = p.grad
            if grad is None:
                target = p.detach().clone()
            else:
                _require(not grad.is_sparse and grad.shape == p.shape and grad.dtype == p.dtype
                         and grad.device == p.device and _finite(grad), "Invalid dense Adam gradient", parameter=index)
                if not state:
                    step_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
                    state = {"step": torch.tensor(0., dtype=step_dtype),
                             "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                             "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format)}
                _require(set(state) == {"step", "exp_avg", "exp_avg_sq"},
                         "Unexpected original Adam state keys", parameter=index)
                _require(isinstance(state["step"], torch.Tensor) and state["step"].numel() == 1
                         and state["step"].device.type == "cpu" and _finite(state["step"]),
                         "Require the original noncapturable CPU Adam step", parameter=index)
                for key in ("exp_avg", "exp_avg_sq"):
                    v = state[key]
                    _require(v.shape == p.shape and v.dtype == p.dtype and v.device == p.device and _finite(v),
                             "Invalid Adam moment", parameter=index, state_key=key)
                _require(bool((state["exp_avg_sq"] >= 0).all()), "Negative Adam second moment", parameter=index)
                state["step"].add_(1)
                step = float(state["step"])
                _require(step >= 1 and step.is_integer(), "Invalid prospective Adam step", parameter=index)
                state["exp_avg"].lerp_(grad, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = state["exp_avg_sq"].sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                target = torch.addcdiv(p, state["exp_avg"], denominator,
                                      value=-group["lr"] / (1 - beta1 ** step))
            delta = target - p
            _require(_finite(target) and _finite(delta), "Nonfinite prospective Adam target", parameter=index)
            targets.append(target.detach())
            deltas.append(delta.detach())
            new_states.append(state)
    return AdamProposal(parameters, tuple(deltas), tuple(new_states), optimizer, tuple(targets),
                        tuple(_version(p) for p in parameters), tuple(source_versions),
                        tuple(_version(d) for d in deltas), tuple(sources), configuration)


def _hildreth(gram, rhs, sweeps, tolerance):
    """Dual coordinate descent for min .5||d-dA||², A d <= c.

    ``rhs`` is A dA-c and gram is A A.T, both small CPU float64 arrays.
    Convergence includes complementarity, not merely primal feasibility.
    """
    count = len(rhs)
    dual = np.zeros(count, dtype=np.float64)
    diagonal = np.diag(gram)
    residual = rhs.copy()
    kkt = np.maximum(residual, 0.)
    for sweep in range(1, sweeps + 1):
        for i in range(count):
            if diagonal[i] <= np.finfo(np.float64).tiny:
                if residual[i] > tolerance:
                    return None, dict(converged=False, reason="zero_gradient_infeasible_constraint",
                                      solver_sweeps=sweep, max_residual_rad=float(residual.max()))
                continue
            value = max(0., dual[i] + residual[i] / diagonal[i])
            change = value - dual[i]
            if change:
                dual[i] = value
                residual -= change * gram[:, i]
        # Recompute to prevent an accumulated coordinate-update roundoff drift.
        residual = rhs - gram @ dual
        kkt = np.where(dual > 0., np.abs(residual), np.maximum(residual, 0.))
        if float(residual.max(initial=-math.inf)) <= tolerance and float(kkt.max(initial=0.)) <= tolerance:
            return dual, dict(converged=True, solver_sweeps=sweep,
                              max_residual_rad=float(residual.max()), max_KKT_residual_rad=float(kkt.max()),
                              nonzero_dual_constraints=int(np.count_nonzero(dual)))
    return None, dict(converged=False, reason="solver_sweep_cap", solver_sweeps=sweeps,
                      max_residual_rad=float(residual.max()), max_KKT_residual_rad=float(kkt.max()))


def _gram_and_direction(jacobian, proposed, upper, lower, sweeps, tolerance):
    """Use one positive J per output; ± rows share its small double Gram."""
    rows, width = jacobian.shape
    positive_gram = np.zeros((rows, rows), dtype=np.float64)
    jd = np.zeros(rows, dtype=np.float64)
    for start in range(0, width, GRAM_COLUMN_CHUNK):
        stop = min(start + GRAM_COLUMN_CHUNK, width)
        block = jacobian[:, start:stop].astype(np.float64)
        positive_gram += block @ block.T
        jd += block @ proposed[start:stop]
    gram = np.block([[positive_gram, -positive_gram], [-positive_gram, positive_gram]])
    gram = (gram + gram.T) * .5
    rhs = np.r_[jd - upper, -jd + lower]
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        return None, dict(converged=False, reason="nonfinite_Gram_or_dual_rhs", solver_sweeps=0)
    dual, metadata = _hildreth(gram, rhs, sweeps, tolerance)
    if dual is None:
        return None, metadata
    coefficients = dual[:rows] - dual[rows:]
    result = proposed.copy()
    for start in range(0, width, GRAM_COLUMN_CHUNK):
        stop = min(start + GRAM_COLUMN_CHUNK, width)
        result[start:stop] -= coefficients @ jacobian[:, start:stop].astype(np.float64)
    metadata.update(objective_half_squared_change=float(.5 * np.dot(result - proposed, result - proposed)),
                    constraints=2 * rows, gram_dtype="float64", jacobian_storage_dtype=str(jacobian.dtype))
    return result, metadata


def _rounding_constants(operations):
    """Conservative IEEE64 operation count, not a learned/input error model."""
    unit = np.float64(np.finfo(np.float64).eps * .5)
    product = np.nextafter(np.float64(int(operations)) * unit, np.inf)
    denominator = np.nextafter(np.float64(1.) - product, -np.inf)
    if not 0 < product < 1 or denominator <= 0.:
        raise ArithmeticError("Unsupported FP64 rounding operation count")
    gamma = np.nextafter(product / denominator, np.inf)
    if gamma >= 1.:
        raise ArithmeticError("Unsupported FP64 sum-envelope operation count")
    underflow = np.nextafter(np.float64(int(operations)) * np.nextafter(0., 1.), np.inf)
    return gamma, underflow


def _up_add_nonnegative(left, right):
    return np.nextafter(np.asarray(left,dtype=np.float64)+np.asarray(right,dtype=np.float64),np.inf)


def _up_mul_nonnegative(left, right):
    return np.nextafter(np.asarray(left,dtype=np.float64)*np.asarray(right,dtype=np.float64),np.inf)


def _up_nonnegative_sum(computed, operations):
    gamma, underflow = _rounding_constants(operations)
    # The BLAS/chunk sum itself is enclosed by gamma_k. Also round every
    # operation constructing its enclosure outward, not only the final result.
    numerator = _up_add_nonnegative(computed,underflow)
    denominator = np.nextafter(np.float64(1.)-gamma,-np.inf)
    return np.nextafter(numerator/denominator,np.inf)


def _reconstruct_and_envelope(jacobian, proposed, dual):
    """Reconstruct d64 and bound rounding in this represented d0-J.T*dual path."""
    rows, width = jacobian.shape
    operations = 3 * rows + 4
    gamma, underflow = _rounding_constants(operations)
    coefficient = dual[:rows] - dual[rows:]
    dual_magnitude = np.abs(dual[:rows]) + np.abs(dual[rows:])
    direction = proposed.copy()
    error = np.empty(width, dtype=np.float64)
    for start in range(0, width, GRAM_COLUMN_CHUNK):
        stop = min(start+GRAM_COLUMN_CHUNK,width)
        block = jacobian[:,start:stop].astype(np.float64)
        direction[start:stop] -= coefficient @ block
        np.abs(block,out=block)
        sum_abs = dual_magnitude @ block
        upper_sum = _up_nonnegative_sum(sum_abs,operations)
        scale = _up_add_nonnegative(np.abs(proposed[start:stop]),upper_sum)
        error[start:stop] = _up_add_nonnegative(_up_mul_nonnegative(gamma,scale),underflow)
    return direction,error,operations


def _direct_primal_envelope(jacobian, direction, reconstruction_error, upper, lower, tolerance):
    """A posteriori arithmetic envelope for direct A*d64-c.

    Float32 Jacobian coefficients are treated as exact represented inputs.
    eta includes ONLY known FP64 reconstruction/dot/subtraction rounding. It
    does not bound reverse/forward AD, batch effects, input error or curvature.
    """
    rows,width=jacobian.shape
    blocks=(width+GRAM_COLUMN_CHUNK-1)//GRAM_COLUMN_CHUNK
    operations=2*width+2*blocks+4
    gamma,underflow=_rounding_constants(operations)
    dot=np.zeros(rows,dtype=np.float64);sum_abs=np.zeros_like(dot);reconstruction=np.zeros_like(dot)
    for start in range(0,width,GRAM_COLUMN_CHUNK):
        stop=min(start+GRAM_COLUMN_CHUNK,width)
        block=jacobian[:,start:stop].astype(np.float64)
        dot+=block@direction[start:stop]
        np.abs(block,out=block)
        sum_abs+=block@np.abs(direction[start:stop])
        reconstruction+=block@reconstruction_error[start:stop]
    constants=np.r_[upper,-lower]
    residual=np.r_[dot-upper,-dot+lower]
    sum_up=np.r_[_up_nonnegative_sum(sum_abs,operations),_up_nonnegative_sum(sum_abs,operations)]
    reconstruction_up=_up_nonnegative_sum(reconstruction,operations)
    dot_scale=_up_add_nonnegative(sum_up,np.abs(constants))
    dot_error=_up_add_nonnegative(_up_mul_nonnegative(gamma,dot_scale),underflow)
    eta=_up_add_nonnegative(np.r_[reconstruction_up,reconstruction_up],dot_error)
    residual_upper=np.nextafter(residual+eta,np.inf)
    finite=all(np.isfinite(a).all() for a in (dot,residual,eta,residual_upper))
    certified=bool(finite and np.all(residual<=eta) and np.all(eta<tolerance)
                   and np.all(residual_upper<tolerance))
    summary=dict(certified=certified,direct_primal_max_residual_rad=float(residual.max()),
        direct_primal_max_eta64_rad=float(eta.max()),direct_primal_max_upper_bound_rad=float(residual_upper.max()),
        direct_dot_operation_count=int(operations),direct_dot_gamma64=float(gamma),
        numerical_reserve_remaining_rad=float(tolerance-max(0.,float(residual_upper.max()))))
    return summary,dict(primal_residual_direct64=residual,primal_eta64=eta,
                       primal_residual_upper_bound64=residual_upper,positive_row_dot64=dot)


def _hildreth_with_evidence(gram,rhs,sweeps,tolerance):
    """Same coordinate equations; preserve the last dual even on strict failure."""
    dual=np.zeros(len(rhs),dtype=np.float64);diagonal=np.diag(gram);residual=rhs.copy()
    kkt=np.maximum(residual,0.)
    for sweep in range(1,sweeps+1):
        for i in range(len(rhs)):
            if diagonal[i]<=np.finfo(np.float64).tiny:
                if residual[i]>tolerance:
                    return dual,dict(converged=False,reason="zero_gradient_infeasible_constraint",solver_sweeps=sweep,
                                     max_residual_rad=float(residual.max()))
                continue
            value=max(0.,dual[i]+residual[i]/diagonal[i]);change=value-dual[i]
            if change:
                dual[i]=value;residual-=change*gram[:,i]
        residual=rhs-gram@dual
        kkt=np.where(dual>0.,np.abs(residual),np.maximum(residual,0.))
        if not np.isfinite(dual).all() or not np.isfinite(residual).all():
            return dual,dict(converged=False,reason="nonfinite_Hildreth_iterate",solver_sweeps=sweep)
        if float(residual.max())<=tolerance and float(kkt.max())<=tolerance:
            return dual,dict(converged=True,solver_sweeps=sweep,max_residual_rad=float(residual.max()),
                max_KKT_residual_rad=float(kkt.max()),nonzero_dual_constraints=int(np.count_nonzero(dual)))
    return dual,dict(converged=False,reason="solver_sweep_cap",solver_sweeps=sweeps,
                     max_residual_rad=float(residual.max()),max_KKT_residual_rad=float(kkt.max()))


def _polish_positive_support(gram,rhs,dual,tolerance):
    """One rank-revealing solve, no support threshold search, pivots or retries."""
    support=np.flatnonzero(dual>0.)
    evidence=dict(positive_dual_indices=support.astype(np.int64))
    meta=dict(polish_calls=1,support_pivots=0,positive_support_size=int(len(support)))
    if not 0<len(support)<=MAX_KKT_POLISH_SUPPORT:
        return None,dict(meta,reason="KKT_polish_support_limit"),evidence
    small=gram[np.ix_(support,support)]
    try:
        left,singular,right=np.linalg.svd(small,full_matrices=False)
    except np.linalg.LinAlgError:
        return None,dict(meta,reason="KKT_polish_factorization_failed"),evidence
    threshold=len(support)*np.finfo(np.float64).eps*float(singular[0])
    rank=int(np.count_nonzero(singular>threshold))
    evidence.update(support_Gram=small,support_singular_values=singular)
    meta.update(support_rank=rank,rank_threshold=float(threshold),
                support_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else "singular")
    if rank!=len(support):
        return None,dict(meta,reason="KKT_polish_rank_deficient"),evidence
    solution=right.T@((left.T@rhs[support])/singular)
    if not np.isfinite(solution).all() or np.any(solution<0.):
        evidence["support_dual_solution"]=solution
        return None,dict(meta,reason="KKT_polish_nonfinite_or_negative_dual"),evidence
    polished=np.zeros_like(dual);polished[support]=solution
    residual=rhs-gram@polished
    kkt=np.where(polished>0.,np.abs(residual),np.maximum(residual,0.))
    evidence.update(polished_dual=polished,polished_Gram_residual=residual)
    meta.update(polished_max_KKT_residual_rad=float(kkt.max()))
    if not np.isfinite(kkt).all() or float(kkt.max())>tolerance:
        return None,dict(meta,reason="KKT_polish_optimality_certificate_failed"),evidence
    return polished,meta,evidence


def _strict_gram_and_direction(jacobian,proposed,upper,lower,sweeps,tolerance,polish_budget):
    """Same QP, separate KKT accuracy and direct primal arithmetic precision."""
    rows,width=jacobian.shape
    positive=np.zeros((rows,rows),dtype=np.float64);jd=np.zeros(rows,dtype=np.float64)
    for start in range(0,width,GRAM_COLUMN_CHUNK):
        stop=min(start+GRAM_COLUMN_CHUNK,width);block=jacobian[:,start:stop].astype(np.float64)
        positive+=block@block.T;jd+=block@proposed[start:stop]
    gram=np.block([[positive,-positive],[-positive,positive]]);gram=(gram+gram.T)*.5
    rhs=np.r_[jd-upper,-jd+lower]
    evidence=dict(signed_Gram=gram,rhs=rhs,active_upper=upper.copy(),active_lower=lower.copy())
    precision=dict(policy=PRIMAL_PRECISION_POLICY,certified=False,polish_calls=0,
                   KKT_optimality_tolerance_rad=float(tolerance),support_pivots=0)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        return None,dict(converged=False,reason="nonfinite_Gram_or_dual_rhs",solver_sweeps=0,primal_precision=precision),evidence
    dual,meta=_hildreth_with_evidence(gram,rhs,sweeps,tolerance)
    evidence.update(dual_before_polish=dual.copy(),dual=dual.copy(),primal_residual_Gram=rhs-gram@dual)
    if not meta["converged"]:
        return None,dict(meta,primal_precision=precision),evidence
    direction,error,reconstruction_operations=_reconstruct_and_envelope(jacobian,proposed,dual)
    initial,detail=_direct_primal_envelope(jacobian,direction,error,upper,lower,tolerance)
    evidence.update(detail,d64_before_polish=direction.copy(),reconstruction_error64=error)
    evidence.update(primal_residual_direct64_before_polish=detail["primal_residual_direct64"].copy(),
                    primal_eta64_before_polish=detail["primal_eta64"].copy())
    precision.update(initial_direct_primal=initial,reconstruction_operation_count=int(reconstruction_operations))
    if not initial["certified"]:
        if polish_budget<1:
            return None,dict(meta,reason="precision_polish_call_cap",primal_precision=precision),evidence
        polished,polish,extra=_polish_positive_support(gram,rhs,dual,tolerance)
        precision.update(polish);evidence.update(extra)
        if polished is None:
            return None,dict(meta,reason=polish["reason"],primal_precision=precision),evidence
        dual=polished
        direction,error,reconstruction_operations=_reconstruct_and_envelope(jacobian,proposed,dual)
        final,detail=_direct_primal_envelope(jacobian,direction,error,upper,lower,tolerance)
    else:
        final=initial
    evidence.update(detail,dual=dual,positive_dual_indices=np.flatnonzero(dual>0.).astype(np.int64),
                    d64=direction,reconstruction_error64=error,primal_residual_Gram=rhs-gram@dual)
    precision.update(final)
    final_residual=rhs-gram@dual
    final_kkt=np.where(dual>0.,np.abs(final_residual),np.maximum(final_residual,0.))
    meta.update(hildreth_max_residual_rad=meta.get("max_residual_rad"),
                hildreth_max_KKT_residual_rad=meta.get("max_KKT_residual_rad"),
                max_residual_rad=float(final_residual.max()),max_KKT_residual_rad=float(final_kkt.max()),
                nonzero_dual_constraints=int(np.count_nonzero(dual)),
                primal_precision=precision,objective_half_squared_change=float(.5*np.dot(direction-proposed,direction-proposed)),
                constraints=2*rows,gram_dtype="float64",jacobian_storage_dtype=str(jacobian.dtype))
    if not final["certified"]:
        return None,dict(meta,reason="direct_FP64_primal_precision_uncertified"),evidence
    return direction,meta,evidence


def _native_active_primal(jacobian,direction,upper,lower):
    """Evaluate the FINAL native direction on every stored row; no NN call."""
    rows,width=jacobian.shape;dot=np.zeros(rows,dtype=np.float64)
    for start in range(0,width,GRAM_COLUMN_CHUNK):
        stop=min(start+GRAM_COLUMN_CHUNK,width)
        dot+=jacobian[:,start:stop].astype(np.float64)@direction[start:stop]
    return dot,np.r_[dot-upper,-dot+lower]


def _bounded_dual_active_support(gram,rhs,dual,tolerance,certify,*,remaining_svd_calls):
    """Refine a converged nonnegative dual on explicitly retained faces.

    New zero-valued constraints remain in ``working`` until its next SVD.
    Negative face solutions move along the exact represented-endpoint segment
    to a nonnegative boundary. No negative multiplier is clipped to zero.
    Only scalar boundary arithmetic uses Fraction; Gram/SVD and the unchanged
    reconstruction/direct-primal certificate remain CPU FP64.
    """
    m=len(rhs)
    meta=dict(support_polish_policy=BOUNDED_SUPPORT_POLISH_POLICY,QP_inequalities=int(m),
        support_pivot_limit=2*m,polish_call_limit=2*m+1,polish_calls=0,
        support_pivots=0,support_additions=0,support_drops=0,
        positive_support_size=0,working_support_size=0,certified=False,
        boundary_arithmetic=BOUNDED_BOUNDARY_ARITHMETIC,
        support_addition_ids=[],support_drop_ids=[],boundary_steps=[],direct_certificate_checks=0)
    evidence={};current=np.array(dual,dtype=np.float64,copy=True);working=set()
    last_checked=None;last_summary=None;last_detail=None

    def check_current():
        nonlocal last_checked,last_summary,last_detail
        if last_checked is None or last_checked.tobytes()!=current.tobytes():
            last_summary,last_detail=certify(current)
            last_checked=current.copy();meta["direct_certificate_checks"]+=1
        evidence.update(last_detail)
        return last_summary,last_detail

    def rejected(reason,*,inspect_current=True,**details):
        meta.update(details,reason=reason,certified=False,
                    positive_support_size=int(np.count_nonzero(current>0.)),working_support_size=len(working))
        evidence.update(dual=current.copy(),working_support_indices=np.asarray(sorted(working),dtype=np.int64))
        if inspect_current:
            check_current()
        return None,meta,evidence

    if (not 0<m<=MAX_KKT_POLISH_SUPPORT or gram.shape!=(m,m) or rhs.shape!=(m,)
            or current.shape!=(m,) or not np.isfinite(gram).all() or not np.isfinite(rhs).all()
            or not np.isfinite(current).all() or np.any(current<0.)):
        return rejected("bounded_active_support_invalid_input",inspect_current=False)
    if type(remaining_svd_calls) is not int or remaining_svd_calls<0:
        return rejected("bounded_active_support_invalid_SVD_budget",inspect_current=False)
    working=set(np.flatnonzero(current>0.).tolist())
    call_limit=min(meta["polish_call_limit"],remaining_svd_calls)
    meta.update(remaining_projection_SVD_budget=remaining_svd_calls,effective_polish_call_limit=call_limit)

    while True:
        indices=np.asarray(sorted(working),dtype=np.int64)
        meta.update(working_support_size=len(working),positive_support_size=int(np.count_nonzero(current>0.)))
        if len(indices):
            if meta["polish_calls"]>=call_limit:
                return rejected("bounded_active_support_SVD_limit")
            small=gram[np.ix_(indices,indices)]
            evidence.update(working_support_indices=indices.copy(),support_Gram=small.copy())
            meta["polish_calls"]+=1
            try:
                left,singular,right=np.linalg.svd(small,full_matrices=False)
            except np.linalg.LinAlgError:
                return rejected("bounded_active_support_factorization_failed")
            if not all(np.isfinite(item).all() for item in (left,singular,right)):
                return rejected("bounded_active_support_nonfinite_factorization")
            threshold=len(indices)*np.finfo(np.float64).eps*float(singular[0])
            rank=int(np.count_nonzero(singular>threshold))
            evidence["support_singular_values"]=singular.copy()
            meta.update(support_rank=rank,rank_threshold=float(threshold),
                support_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else "singular")
            if rank!=len(indices):
                return rejected("bounded_active_support_rank_deficient")
            solution=right.T@((left.T@rhs[indices])/singular)
            evidence["support_dual_solution"]=solution.copy()
            if not np.isfinite(solution).all():
                return rejected("bounded_active_support_nonfinite_face_solution")
            negative=np.flatnonzero(solution<0.)
            if len(negative):
                # Ratios/interpolation are exact for the represented FP64
                # endpoints, so tied boundary multipliers become exactly0.
                ratios=[(Fraction.from_float(float(current[indices[j]]))/
                         (Fraction.from_float(float(current[indices[j]]))-
                          Fraction.from_float(float(solution[j]))),int(indices[j])) for j in negative]
                alpha,blocking=min(ratios,key=lambda item:(item[0],item[1]))
                if not Fraction(0)<=alpha<=Fraction(1):
                    return rejected("bounded_active_support_invalid_boundary_ratio")
                candidate=current.copy();zeros=[]
                for j,index in enumerate(indices):
                    old=Fraction.from_float(float(current[index]));target=Fraction.from_float(float(solution[j]))
                    value=old+alpha*(target-old)
                    if value<0:
                        return rejected("bounded_active_support_negative_exact_boundary")
                    represented=float(value)
                    if not math.isfinite(represented) or (value>0 and represented==0.):
                        return rejected("bounded_active_support_boundary_range_failure")
                    candidate[index]=represented
                    if value==0:zeros.append(int(index))
                if blocking not in zeros or not zeros:
                    return rejected("bounded_active_support_boundary_not_reached")
                if meta["support_pivots"]+len(zeros)>meta["support_pivot_limit"]:
                    evidence["attempted_boundary_dual"]=candidate
                    return rejected("bounded_active_support_pivot_limit",attempted_drop_ids=zeros)
                if np.any(candidate<0.) or not np.isfinite(candidate).all():
                    return rejected("bounded_active_support_nonnegative_boundary_uncertified")
                current=candidate;working.difference_update(zeros)
                meta["support_pivots"]+=len(zeros);meta["support_drops"]+=len(zeros)
                meta["support_drop_ids"].extend(zeros)
                meta["boundary_steps"].append(dict(alpha=float(alpha),alpha_numerator=str(alpha.numerator),
                    alpha_denominator=str(alpha.denominator),blocking_constraint=blocking,dropped_constraints=zeros,
                    minimum_dual_after=float(current.min()),negative_multiplier_clipping=False))
                continue
            current=np.zeros_like(current);current[indices]=solution

        summary,detail=check_current()
        residual=detail["primal_residual_direct64"];eta=detail["primal_eta64"]
        gram_residual=rhs-gram@current
        kkt=np.where(current>0.,np.abs(gram_residual),np.maximum(gram_residual,0.))
        evidence.update(dual=current.copy(),positive_dual_indices=np.flatnonzero(current>0.).astype(np.int64),
            working_support_indices=np.asarray(sorted(working),dtype=np.int64),
            polished_dual=current.copy(),polished_Gram_residual=gram_residual.copy())
        if not all(np.isfinite(value).all() for value in (residual,eta,gram_residual,kkt)):
            return rejected("bounded_active_support_nonfinite_certificate")
        meta["polished_max_KKT_residual_rad"]=float(kkt.max())
        if summary["certified"]:
            if float(kkt.max())>tolerance:
                return rejected("bounded_active_support_KKT_uncertified")
            meta.update(certified=True,positive_support_size=int(np.count_nonzero(current>0.)),
                        working_support_size=len(working))
            return current.copy(),meta,evidence

        outside=np.ones(m,dtype=bool)
        if working:outside[np.asarray(sorted(working),dtype=np.int64)]=False
        candidates=np.flatnonzero((residual>eta)&outside)
        if not len(candidates):
            reason=("bounded_active_support_unresolved_active_primal" if np.any(residual>eta)
                    else "bounded_active_support_direct_envelope_uncertified")
            return rejected(reason)
        # Stable signed-index ordering resolves exact residual ties.
        chosen=int(candidates[int(np.argmax(residual[candidates]))])
        if meta["support_pivots"]>=meta["support_pivot_limit"]:
            return rejected("bounded_active_support_pivot_limit",attempted_add_constraint=chosen)
        if len(working)>=MAX_KKT_POLISH_SUPPORT:
            return rejected("bounded_active_support_size_limit",attempted_add_constraint=chosen)
        working.add(chosen)
        # Keep this zero-valued new member explicitly for the NEXT SVD.
        if current[chosen]!=0.:
            return rejected("bounded_active_support_inconsistent_working_face")
        meta["support_pivots"]+=1;meta["support_additions"]+=1
        meta["support_addition_ids"].append(chosen)


def _active_support_gram_and_direction(jacobian,proposed,upper,lower,sweeps,tolerance,polish_budget):
    """The opt-in bounded polish, only after the original Hildreth converges."""
    rows,width=jacobian.shape;m=2*rows
    precision=dict(policy=PRIMAL_PRECISION_POLICY,certified=False,polish_calls=0,
        KKT_optimality_tolerance_rad=float(tolerance),support_polish_policy=BOUNDED_SUPPORT_POLISH_POLICY,
        QP_inequalities=m,support_pivot_limit=2*m,polish_call_limit=2*m+1,
        support_pivots=0,support_additions=0,support_drops=0,
        positive_support_size=0,working_support_size=0,boundary_arithmetic=BOUNDED_BOUNDARY_ARITHMETIC)
    evidence=dict(active_upper=upper.copy(),active_lower=lower.copy())
    if not 0<m<=MAX_KKT_POLISH_SUPPORT:
        return None,dict(converged=False,reason="bounded_active_support_inequality_limit",
                         solver_sweeps=0,primal_precision=precision),evidence
    positive=np.zeros((rows,rows),dtype=np.float64);jd=np.zeros(rows,dtype=np.float64)
    for start in range(0,width,GRAM_COLUMN_CHUNK):
        stop=min(start+GRAM_COLUMN_CHUNK,width);block=jacobian[:,start:stop].astype(np.float64)
        positive+=block@block.T;jd+=block@proposed[start:stop]
    gram=np.block([[positive,-positive],[-positive,positive]]);gram=(gram+gram.T)*.5
    rhs=np.r_[jd-upper,-jd+lower]
    evidence.update(signed_Gram=gram,rhs=rhs)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        return None,dict(converged=False,reason="nonfinite_Gram_or_dual_rhs",solver_sweeps=0,
                         primal_precision=precision),evidence
    dual,meta=_hildreth_with_evidence(gram,rhs,sweeps,tolerance)
    evidence.update(dual_before_polish=dual.copy(),dual=dual.copy(),primal_residual_Gram=rhs-gram@dual)
    if not meta["converged"]:
        return None,dict(meta,primal_precision=precision),evidence
    state={}

    def certify(value):
        direction,error,operations=_reconstruct_and_envelope(jacobian,proposed,value)
        summary,detail=_direct_primal_envelope(jacobian,direction,error,upper,lower,tolerance)
        state.update(dual=value.copy(),direction=direction,error=error,operations=operations,
                     summary=summary,detail=detail)
        return summary,detail

    initial,detail=certify(dual)
    evidence.update(detail,d64_before_polish=state["direction"].copy(),
        primal_residual_direct64_before_polish=detail["primal_residual_direct64"].copy(),
        primal_eta64_before_polish=detail["primal_eta64"].copy())
    precision.update(initial_direct_primal=initial,reconstruction_operation_count=state["operations"],
        positive_support_size=int(np.count_nonzero(dual>0.)),working_support_size=int(np.count_nonzero(dual>0.)))
    if not initial["certified"]:
        polished,refinement,extra=_bounded_dual_active_support(gram,rhs,dual,tolerance,certify,
                                                             remaining_svd_calls=polish_budget)
        precision.update(refinement);evidence.update(extra)
        if polished is None:
            precision.update(state["summary"]);precision["certified"]=False
            evidence.update(d64=state["direction"],reconstruction_error64=state["error"],
                            primal_residual_Gram=rhs-gram@state["dual"])
            return None,dict(meta,reason=refinement["reason"],primal_precision=precision),evidence
        dual=polished
    final=state["summary"]
    evidence.update(state["detail"],dual=dual.copy(),positive_dual_indices=np.flatnonzero(dual>0.).astype(np.int64),
                    d64=state["direction"],reconstruction_error64=state["error"],primal_residual_Gram=rhs-gram@dual)
    precision.update(final)
    residual=rhs-gram@dual;kkt=np.where(dual>0.,np.abs(residual),np.maximum(residual,0.))
    meta.update(hildreth_max_residual_rad=meta.get("max_residual_rad"),
        hildreth_max_KKT_residual_rad=meta.get("max_KKT_residual_rad"),max_residual_rad=float(residual.max()),
        max_KKT_residual_rad=float(kkt.max()),nonzero_dual_constraints=int(np.count_nonzero(dual)),
        primal_precision=precision,objective_half_squared_change=float(.5*np.dot(state["direction"]-proposed,state["direction"]-proposed)),
        constraints=m,gram_dtype="float64",jacobian_storage_dtype=str(jacobian.dtype))
    if not final["certified"] or float(kkt.max())>tolerance:
        return None,dict(meta,reason="bounded_active_support_final_certificate_failed"),evidence
    return state["direction"],meta,evidence


def project_actual_bank(actor_module, actor_normalizer, bank, baseline_q, budget, mean_deltas, *,
                        interior_fraction=.5, max_active_rows=8, max_solver_sweeps=2048,
                        working_budget_fraction=None, jacobian_arena_budget_bytes=None,
                        primal_precision_policy=None, support_polish_policy=FIXED_SUPPORT_POLISH_POLICY):
    """Project44 Adam directions into simultaneous actual-output linear bounds.

    The fixed bounds are ``-(b+e)/2 <= Jd <= (b-e)/2``, where ``e=q-q0``.
    Every certification JVP covers the complete5120x12 actual bank. A failing
    row contributes all12 output Jacobians and both signs to the convex QP.
    The sweep cap is TOTAL over this call, not a hidden8x multiplier.

    Numerical tolerances describe only this first-order certificate. They are
    never applied to the caller's strict nonlinear post-commit hard budget.
    The optional reserved mode intersects the original midpoint with a fixed
    whole-update ``abs(e+Jd)<=.5*b`` domain, then reserves the same solver
    tolerance inward on BOTH sides of that intersection. ``None`` is legacy.
    Optional CPU-arena mode requires ``max_active_rows=None``. Its finite row
    ceiling is derived from bytes, actual44-parameter width and float32 storage;
    it changes compute coverage/storage only, never the constraint boundaries.
    """
    reserved = working_budget_fraction is not None
    resource_mode = jacobian_arena_budget_bytes is not None
    strict_primal = primal_precision_policy is not None
    bounded_support = support_polish_policy == BOUNDED_SUPPORT_POLISH_POLICY
    precision_evidence = None
    metadata = dict(kind=RESERVED_KIND if reserved else KIND, feasible=False, interior_fraction=.5,
                    max_active_rows=8, max_solver_sweeps=2048,
                    tolerance_rad=LINEAR_TOLERANCE_RAD,
                    row_value_tolerance_rad=ROW_VALUE_TOLERANCE_RAD,
                    fullbank_jvp_calls=0, row_jacobian_calls=0, solver_sweeps=0,
                    active_row_ids=[], active_rows=0, solver_records=[], direction_changed=False,
                    nonlinear_hard_gate_unchanged=True, no_backtracking_or_retry=True,
                    std_direction_not_in_mean_constraints=True,
                    interpretation="First-order finite actual-bank certificate, not nonlinear/unseen-state/physical safety")

    def fail(reason, **more):
        metadata.update(more)
        raise ProposalProjectionError(dict(metadata, reason=reason, feasible=False),
            diagnostic_evidence=precision_evidence if strict_primal else None)

    if support_polish_policy not in (FIXED_SUPPORT_POLISH_POLICY,BOUNDED_SUPPORT_POLISH_POLICY):
        fail("Undeclared support polish policy")
    if bounded_support:
        if not strict_primal or not resource_mode or not reserved:
            fail("Bounded support polish requires strict resource/reserved mode")
        metadata.update(support_polish_policy=BOUNDED_SUPPORT_POLISH_POLICY,
                        boundary_arithmetic=BOUNDED_BOUNDARY_ARITHMETIC)
    if interior_fraction != .5 or max_solver_sweeps != 2048:
        fail("Fixed prospective projection protocol changed")
    if resource_mode:
        if max_active_rows is not None:
            fail("Byte-derived coverage requires max_active_rows=None")
        if type(jacobian_arena_budget_bytes) is not int or not 0 < jacobian_arena_budget_bytes <= MAX_CPU_JACOBIAN_ARENA_BYTES:
            fail("CPU Jacobian arena requires a positive integer byte budget at most1GiB")
        if not reserved:
            fail("Byte-derived coverage requires the declared reserved working mode")
        metadata.update(resource_kind=JACOBIAN_RESOURCE_KIND, configured_max_active_rows=None,
                        max_active_rows=None,
                        jacobian_storage_location="cpu_numpy", jacobian_storage_dtype="float32",
                        jacobian_arena_budget_bytes=jacobian_arena_budget_bytes,
                        jacobian_arena_allocated_bytes=0, jacobian_used_rows=0,
                        jacobian_used_bytes=0, jacobian_storage_bytes=0)
    elif max_active_rows != 8:
        fail("Fixed prospective projection protocol changed")
    if reserved:
        if working_budget_fraction != .5:
            fail("Reserved working budget fraction must be exactly.5")
        metadata.update(working_budget_fraction=.5, solver_tolerance_reserved_inward=True,
                        whole_update_fixed_working_bound=True,
                        working_reference="original_whole_update_q0",
                        original_nonlinear_hard_budget_unchanged=True)
    if strict_primal:
        if primal_precision_policy!=PRIMAL_PRECISION_POLICY or not resource_mode or not reserved:
            fail("Strict direct-primal policy requires its declared resource/reserved mode")
        precision_summary=dict(direct_primal_certified=True,QP_subproblems=0,
            KKT_optimality_tolerance_rad=LINEAR_TOLERANCE_RAD,direct_primal_max_eta64_rad=0.,
            direct_primal_max_upper_bound_rad=0.,direct_primal_max_residual_rad=0.,polish_calls=0,
            polish_call_limit_per_subproblem=1,polish_call_limit_per_projection=MAX_KKT_POLISH_CALLS,
            support_pivot_limit=0,polish_support_limit=MAX_KKT_POLISH_SUPPORT,
            final_same_direction_row_JVP_checked=True,final_same_direction_active_rows=0,
            final_same_direction_row_JVP_maxdiff_rad=0.,final_active_native_primal_max_residual_rad=0.,
            no_active_rows_vacuous=True,
            scope="eta64 is FP64 arithmetic only; final same-direction JVP difference is an observation,not a global NN bound")
        metadata.update(primal_precision_policy=PRIMAL_PRECISION_POLICY,primal_precision_summary=precision_summary)
        if bounded_support:
            precision_summary.update(support_polish_policy=BOUNDED_SUPPORT_POLISH_POLICY,
                polish_call_limit_per_subproblem=MAX_BOUNDED_POLISH_CALLS_PER_QP,
                polish_call_limit_per_projection=MAX_BOUNDED_POLISH_CALLS_PER_PROJECTION,
                support_pivot_limit=2*MAX_KKT_POLISH_SUPPORT,support_pivot_limit_rule="2*m",
                polish_call_limit_per_subproblem_rule="2*m+1",support_pivot_limit_scope="per_QP_maximum_at_m672",
                support_pivots=0,support_additions=0,support_drops=0,QP_support_refinements=[],
                boundary_arithmetic=BOUNDED_BOUNDARY_ARITHMETIC)
    if not isinstance(bank, Mapping) or any(k not in bank for k in ("obs", "lower", "upper")):
        fail("Incomplete actual bank")
    if actor_normalizer.training:
        fail("Actor normalizer must remain eval/frozen")
    if tuple(actor_module.buffers()):
        fail("The original row-local actor has no mutable buffers")
    if any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.RNNBase)) for m in actor_module.modules()):
        fail("Batch-coupled or stochastic actor is unsupported")
    named = tuple(actor_module.named_parameters())
    parameters = tuple(p for _, p in named)
    directions = tuple(mean_deltas)
    if len(parameters) != NUM_MEAN_PARAMETERS or len(directions) != NUM_MEAN_PARAMETERS:
        fail("Require all44 original actor parameter directions")
    device, dtype = parameters[0].device, parameters[0].dtype
    if dtype not in (torch.float32, torch.float64):
        fail("Unsupported actor dtype")
    if resource_mode and dtype != torch.float32:
        fail("CPU Jacobian arena requires the unchanged float32 actor dtype")
    for i, (p, d) in enumerate(zip(parameters, directions)):
        if p.device != device or p.dtype != dtype or not isinstance(d, torch.Tensor) or d.shape != p.shape \
                or d.device != device or d.dtype != dtype or not _finite(p) or not _finite(d):
            fail("Invalid actor proposal tensor", parameter=i)
    versions = tuple(_version(p) for p in parameters)
    normalizer_versions = tuple((name, id(b), _version(b)) for name, b in actor_normalizer.named_buffers())
    param_values = tuple(p.detach() for p in parameters)
    sizes = [p.numel() for p in parameters]
    offsets = np.cumsum([0] + sizes)
    proposed = np.concatenate([d.detach().cpu().reshape(-1).double().numpy() for d in directions])
    metadata.update(mean_parameter_tensors=len(parameters), mean_parameter_elements=len(proposed),
                    original_direction_norm=float(np.linalg.norm(proposed)),
                    bank_rows=MAX_BANK_ROWS, output_coordinates=NUM_OUTPUTS)
    if resource_mode:
        row_bytes = NUM_OUTPUTS * len(proposed) * np.dtype(np.float32).itemsize
        max_active_rows = min(MAX_BANK_ROWS, jacobian_arena_budget_bytes // row_bytes)
        metadata.update(max_active_rows=int(max_active_rows), jacobian_bytes_per_row=int(row_bytes),
                        jacobian_derived_max_active_rows=int(max_active_rows),
                        jacobian_arena_capacity_bytes=int(max_active_rows * row_bytes))
        if max_active_rows < 1:
            fail("CPU Jacobian arena cannot fit one complete actual row", required_next_row_bytes=int(row_bytes))

    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=dtype).detach()

    raw = tensor(bank["obs"])
    lo, hi = tensor(bank["lower"]), tensor(bank["upper"])
    reference = tensor(baseline_q)
    limits = tensor(budget)
    context_width = getattr(actor_module, "extra_context_dim", 0)
    if type(context_width) is not int or context_width not in (0, 5):
        fail("Actor must declare the legacy or explicit five-value context ABI")
    if context_width == 5 and getattr(actor_module, "obs_dim", None) != 1034:
        fail("Explicit task context requires the declared1034 actor input")
    if raw.shape != (MAX_BANK_ROWS, 1029 + context_width) or lo.shape != hi.shape or lo.shape != (MAX_BANK_ROWS, NUM_OUTPUTS) \
            or reference.shape != limits.shape or reference.shape != lo.shape:
        fail("Actual bank dimensions changed")
    if not all(_finite(x) for x in (raw, lo, hi, reference, limits)) or not bool((lo < hi).all()) \
            or not bool((limits > 0).all()):
        fail("Invalid finite actual bank/limits")
    with torch.no_grad():
        normalized = actor_normalizer(raw)
    if normalized.shape != raw.shape or not _finite(normalized):
        fail("Invalid frozen normalized actual bank")
    reference_array = reference.cpu().numpy()
    limit_array = limits.cpu().numpy()

    def decoder(values, inputs, low, high):
        replacements = {name: value for (name, _), value in zip(named, values)}
        action = functional_call(actor_module, replacements, (inputs,), strict=True).clamp(-100, 100)
        return (low + high) * .5 + (high - low) * .5 * torch.tanh(action)

    def unpack(flat):
        return tuple(torch.from_numpy(flat[offsets[i]:offsets[i+1]].reshape(p.shape)).to(device=device, dtype=dtype)
                     for i, p in enumerate(parameters))

    def scan(ds):
        values, tangents = [], []
        with torch.no_grad():
            for start in range(0, MAX_BANK_ROWS, 1024):
                stop = start + 1024
                fn = lambda ps: decoder(ps, normalized[start:stop], lo[start:stop], hi[start:stop])
                value, tangent = jvp(fn, (param_values,), (ds,))
                if value.shape != (1024, NUM_OUTPUTS) or tangent.shape != value.shape \
                        or not _finite(value) or not _finite(tangent):
                    fail("Nonfinite or malformed actual full-bank JVP")
                values.append(value.detach().cpu().numpy())
                tangents.append(tangent.detach().cpu().numpy())
        metadata["fullbank_jvp_calls"] += 1
        return np.concatenate(values), np.concatenate(tangents).astype(np.float64)

    qcur, tangent = scan(directions)
    error = (qcur - reference_array).astype(np.float64)
    if np.any(np.abs(qcur - reference_array) > limit_array):
        fail("Current actor already violates the original nonlinear budget",
             max_current_nonlinear_ratio=float(np.max(np.abs(qcur-reference_array)/limit_array)))
    upper = .5 * (limit_array.astype(np.float64) - error)
    lower = -.5 * (limit_array.astype(np.float64) + error)
    if reserved:
        working_bound = .5 * limit_array.astype(np.float64)
        unreserved_upper = np.minimum(upper, working_bound-error)
        unreserved_lower = np.maximum(lower, -working_bound-error)
        if not np.isfinite(unreserved_upper).all() or not np.isfinite(unreserved_lower).all() \
                or np.any(unreserved_lower > unreserved_upper):
            fail("empty_or_nonfinite_working_midpoint_intersection")
        lower = unreserved_lower + LINEAR_TOLERANCE_RAD
        upper = unreserved_upper - LINEAR_TOLERANCE_RAD
        if np.any(lower > upper):
            fail("empty_solver_reserved_intersection")
        metadata.update(initial_current_max_hard_ratio=float(np.max(np.abs(error)/limit_array)),
                        initial_current_max_working_ratio=float(np.max(np.abs(error)/working_bound)),
                        initial_linearized_working_max_ratio=float(np.max(np.abs(error+tangent)/working_bound)))
    metadata.update(initial_max_linear_violation_rad=float(np.maximum(tangent-upper,lower-tangent).max()),
                    initial_linearized_total_max_ratio=float(np.max(np.abs(error+tangent)/limit_array)))
    jacobians = []
    arena = None
    direction = directions
    active_matrix = None
    row_value_max, row_jvp_max = 0., 0.

    for iteration in range(max_active_rows + 1):
        if strict_primal and jacobians:
            native=np.concatenate([d.detach().cpu().reshape(-1).double().numpy() for d in direction])
            active_ids=metadata["active_row_ids"]
            direct,active_residual=_native_active_primal(active_matrix,native,
                upper[active_ids].reshape(-1),lower[active_ids].reshape(-1))
            fullbank_active=tangent[active_ids].reshape(-1)
            disagreement=float(np.max(np.abs(direct-fullbank_active)))
            precision_summary.update(final_same_direction_row_JVP_checked=True,
                final_same_direction_active_rows=len(jacobians),no_active_rows_vacuous=False,
                final_same_direction_row_JVP_maxdiff_rad=disagreement,
                final_active_native_primal_max_residual_rad=float(active_residual.max()))
            precision_evidence.update(active_row_ids=np.asarray(active_ids,dtype=np.int64),
                native_direction32=native.astype(np.float32),primal_residual_after_native_unpack=active_residual,
                native_positive_row_Jd=direct,final_fullbank_positive_row_Jd=fullbank_active,
                active_qcur=qcur[active_ids],active_q0=reference_array[active_ids],active_budget=limit_array[active_ids])
            if not np.isfinite(active_residual).all() or not math.isfinite(disagreement):
                fail("Nonfinite final native active-row primal probe")
            if disagreement>LINEAR_TOLERANCE_RAD:
                fail("Final native active-row Jacobian/full-bank JVP disagreement")
        violation = np.maximum(tangent - upper, lower - tangent)
        worst = np.unravel_index(int(np.argmax(violation)), violation.shape)
        maximum = float(violation[worst])
        metadata.update(max_linear_violation_rad=maximum, worst_row=int(worst[0]), worst_joint=int(worst[1]),
                        active_rows=len(jacobians), row_value_max_disagreement_rad=row_value_max,
                        row_JVP_max_disagreement_rad=row_jvp_max)
        if reserved:
            unreserved_max = float(np.maximum(tangent-unreserved_upper,unreserved_lower-tangent).max())
            metadata.update(max_unreserved_linear_violation_rad=unreserved_max,
                            max_working_bound_violation_rad=float(np.max(np.abs(error+tangent)-working_bound)),
                            linearized_working_max_ratio=float(np.max(np.abs(error+tangent)/working_bound)))
        # The direct-return path preserves the original tensor identities and
        # exact prospective Adam commit, without invoking a solver at all.
        if maximum <= (0. if not jacobians else LINEAR_TOLERANCE_RAD):
            if reserved and unreserved_max > 0.:
                fail("Unreserved working/midpoint certificate failed after solver tolerance reserve")
            metadata.update(feasible=True, reason="unmodified_proposal_feasible" if not jacobians else "projected_proposal_feasible",
                            direction_changed=bool(jacobians), projected_direction_norm=float(math.sqrt(sum(float(d.double().square().sum()) for d in direction))),
                            parameter_delta_change_norm=float(math.sqrt(sum(float((d.double()-o.double()).square().sum()) for d,o in zip(direction,directions)))),
                            linearized_total_max_ratio=float(np.max(np.abs(error+tangent)/limit_array)),
                            max_absolute_Jd_rad=float(np.max(np.abs(tangent))),
                            signed_midpoint_bounds_verified_rows=MAX_BANK_ROWS,
                            signed_midpoint_bounds_verified_coordinates=MAX_BANK_ROWS*NUM_OUTPUTS)
            if tuple(_version(p) for p in parameters) != versions \
                    or tuple((name,id(b),_version(b)) for name,b in actor_normalizer.named_buffers()) != normalizer_versions:
                fail("Projection mutated real actor/normalizer")
            return direction, _metadata_value(metadata)
        if iteration == max_active_rows:
            if resource_mode:
                fail("jacobian_arena_capacity", attempted_new_row=int(worst[0]),
                     required_next_row_bytes=int(row_bytes))
            fail("active_row_cap", attempted_new_row=int(worst[0]))
        row = int(worst[0])
        if row in metadata["active_row_ids"]:
            fail("Active-row full-bank JVP certificate failed after QP")
        one = lambda ps: decoder(ps, normalized[row:row+1], lo[row:row+1], hi[row:row+1])[0]
        with torch.no_grad():
            one_value = one(param_values).detach().cpu().numpy()
        row_value_max = max(row_value_max, float(np.max(np.abs(one_value-qcur[row]))))
        if row_value_max > ROW_VALUE_TOLERANCE_RAD:
            fail("Single-row/full-bank primal disagreement", row=row,
                 row_value_max_disagreement_rad=row_value_max)
        row_derivatives = jacrev(one)(param_values)
        metadata["row_jacobian_calls"] += 1
        matrix = np.concatenate([d.detach().reshape(NUM_OUTPUTS,-1).cpu().numpy() for d in row_derivatives], axis=1)
        if matrix.shape != (NUM_OUTPUTS,len(proposed)) or not np.isfinite(matrix).all():
            fail("Invalid actual-row full Jacobian", row=row)
        current_flat = np.concatenate([d.detach().cpu().reshape(-1).double().numpy() for d in direction])
        row_jvp = np.zeros(NUM_OUTPUTS, dtype=np.float64)
        for start in range(0, len(proposed), GRAM_COLUMN_CHUNK):
            stop = min(start+GRAM_COLUMN_CHUNK,len(proposed))
            row_jvp += matrix[:,start:stop].astype(np.float64) @ current_flat[start:stop]
        row_jvp_max = max(row_jvp_max, float(np.max(np.abs(row_jvp-tangent[row]))))
        if row_jvp_max > LINEAR_TOLERANCE_RAD:
            fail("Single-row Jacobian/full-bank JVP disagreement", row=row,
                 row_JVP_max_disagreement_rad=row_jvp_max)
        if resource_mode:
            if matrix.dtype != np.float32:
                fail("CPU Jacobian row dtype differs from declared arena")
            if arena is None:
                try:
                    arena = np.empty((max_active_rows*NUM_OUTPUTS,len(proposed)),dtype=np.float32)
                except MemoryError:
                    fail("jacobian_arena_allocation_failed", attempted_new_row=row)
                metadata["jacobian_arena_allocated_bytes"] = int(arena.nbytes)
            arena_start = len(jacobians)*NUM_OUTPUTS
            arena[arena_start:arena_start+NUM_OUTPUTS] = matrix
            jacobians.append(arena[arena_start:arena_start+NUM_OUTPUTS])
            metadata["active_row_ids"].append(row)
            del row_derivatives,matrix
            active_matrix = arena[:len(jacobians)*NUM_OUTPUTS]
            metadata.update(jacobian_used_rows=len(jacobians),
                            jacobian_used_bytes=int(active_matrix.nbytes),
                            jacobian_storage_bytes=int(active_matrix.nbytes))
        else:
            jacobians.append(matrix)
            metadata["active_row_ids"].append(row)
            del row_derivatives
            active_matrix = np.concatenate(jacobians, axis=0)
        remaining = max_solver_sweeps - metadata["solver_sweeps"]
        if remaining <= 0:
            fail("solver_sweep_cap")
        active_ids = metadata["active_row_ids"]
        if strict_primal:
            if bounded_support:
                projected,solve,precision_evidence=_active_support_gram_and_direction(active_matrix,proposed,
                    upper[active_ids].reshape(-1),lower[active_ids].reshape(-1),remaining,LINEAR_TOLERANCE_RAD,
                    MAX_BOUNDED_POLISH_CALLS_PER_PROJECTION-precision_summary["polish_calls"])
            else:
                projected,solve,precision_evidence=_strict_gram_and_direction(active_matrix,proposed,
                    upper[active_ids].reshape(-1),lower[active_ids].reshape(-1),remaining,LINEAR_TOLERANCE_RAD,
                    MAX_KKT_POLISH_CALLS-precision_summary["polish_calls"])
            precision=solve["primal_precision"]
            precision_evidence["active_row_ids"]=np.asarray(active_ids,dtype=np.int64)
            precision_summary["QP_subproblems"]+=1
            precision_summary["polish_calls"]+=precision.get("polish_calls",0)
            precision_summary["direct_primal_certified"] &= bool(precision.get("certified",False))
            if bounded_support:
                fields=("QP_inequalities","support_pivot_limit","polish_call_limit","polish_calls", "support_pivots",
                        "support_additions","support_drops","positive_support_size","working_support_size","certified")
                precision_summary["QP_support_refinements"].append({key:precision[key] for key in fields})
                for key in ("support_pivots","support_additions","support_drops"):
                    precision_summary[key]+=precision[key]
            for key in ("direct_primal_max_eta64_rad","direct_primal_max_upper_bound_rad","direct_primal_max_residual_rad"):
                precision_summary[key]=max(precision_summary[key],precision.get(key,0.))
        else:
            projected, solve = _gram_and_direction(active_matrix, proposed,
                upper[active_ids].reshape(-1), lower[active_ids].reshape(-1), remaining, LINEAR_TOLERANCE_RAD)
        metadata["solver_sweeps"] += solve["solver_sweeps"]
        metadata["solver_records"].append(solve)
        metadata["jacobian_storage_bytes"] = sum(j.nbytes for j in jacobians)
        if projected is None or not np.isfinite(projected).all():
            fail(solve.get("reason", "nonfinite_QP_solution"))
        direction = unpack(projected)
        check_qcur, tangent = scan(direction)
        if not np.array_equal(check_qcur,qcur):
            fail("Actor primal changed between direction-only bank scans")
    fail("Unreachable bounded projection exit")
