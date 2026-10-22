"""Offline height-map diagnostics; no simulator, checkpoints, or training imports.

Heights and edge thresholds are in metres. Edge F1 compares adjacent-cell
height jumps at exactly matching grid boundaries (no spatial tolerance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _maps(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or min(array.shape) < 1 or min(array.shape[-2:]) < 2:
        raise ValueError(f"{name} must have shape (H,W) or (B,H,W), H,W >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite heights")
    return array


def terrain_quality(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    observed: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    edge_threshold_m: float = 0.015,
) -> dict:
    """Measure map error and boundaries, including error hidden by global MAE.

    ``observed`` is an optional binary hit mask. ``confidence`` is an optional
    reliability score in [0,1]; its bins report error, not probability calibration.
    Empty subsets return ``None`` instead of a misleading zero error.
    """
    if not np.isfinite(edge_threshold_m) or edge_threshold_m <= 0:
        raise ValueError("edge_threshold_m must be positive and finite")
    pred = _maps(prediction, "prediction")
    gt = _maps(target, "target")
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {gt.shape}")
    error = np.abs(pred - gt)
    pred_grad = np.concatenate([np.diff(pred, axis=1).ravel(), np.diff(pred, axis=2).ravel()])
    gt_grad = np.concatenate([np.diff(gt, axis=1).ravel(), np.diff(gt, axis=2).ravel()])
    pred_edge = np.abs(pred_grad) >= edge_threshold_m
    gt_edge = np.abs(gt_grad) >= edge_threshold_m
    tp = int(np.count_nonzero(pred_edge & gt_edge))
    fp = int(np.count_nonzero(pred_edge & ~gt_edge))
    fn = int(np.count_nonzero(~pred_edge & gt_edge))
    result = {
        "maps": int(pred.shape[0]),
        "mae_m": float(error.mean()),
        "p95_abs_error_m": float(np.quantile(error, 0.95)),
        "max_abs_error_m": float(error.max()),
        "gradient_mae_m": float(np.abs(pred_grad - gt_grad).mean()),
        "edge_threshold_m": float(edge_threshold_m),
        "edge_precision": tp / (tp + fp) if tp + fp else None,
        "edge_recall": tp / (tp + fn) if tp + fn else None,
        "edge_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "edge_true_positive": tp,
        "edge_false_positive": fp,
        "edge_false_negative": fn,
        "edge_sign_accuracy": (
            float(np.mean(np.sign(pred_grad[pred_edge & gt_edge]) == np.sign(gt_grad[pred_edge & gt_edge])))
            if tp else None
        ),
    }
    if observed is not None:
        hit = _maps(observed, "observed")
        if hit.shape != gt.shape or not np.isin(hit, [0, 1]).all():
            raise ValueError("observed must be a binary mask matching target shape")
        hit = hit.astype(bool)
        result.update({
            "observed_fraction": float(hit.mean()),
            "observed_mae_m": float(error[hit].mean()) if hit.any() else None,
            "unobserved_mae_m": float(error[~hit].mean()) if (~hit).any() else None,
        })
    if confidence is not None:
        score = _maps(confidence, "confidence")
        if score.shape != gt.shape or np.any((score < 0) | (score > 1)):
            raise ValueError("confidence must match target shape and lie in [0,1]")
        bins = []
        for lo, hi in zip([0.0, 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.0]):
            selected = (score >= lo) & ((score <= hi) if hi == 1 else (score < hi))
            bins.append({
                "lower": lo,
                "upper": hi,
                "count": int(selected.sum()),
                "mae_m": float(error[selected].mean()) if selected.any() else None,
            })
        result["reliability_error_bins"] = bins
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with prediction, target; optional observed, confidence")
    parser.add_argument("--edge-threshold-m", type=float, default=0.015)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        report = terrain_quality(
            data["prediction"], data["target"],
            observed=data["observed"] if "observed" in data else None,
            confidence=data["confidence"] if "confidence" in data else None,
            edge_threshold_m=args.edge_threshold_m,
        )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
