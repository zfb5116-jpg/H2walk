"""Measure one actual output row without changing its original decoder block."""
import os
import numpy as np
import torch
from torch.func import jacrev, jvp

SELECTED_ROW_BLOCK_POLICY = "original_1024_block_selected_output_v1"
BLOCK_ROWS = 1024
OUTPUTS = 12


def selected_block_row_derivatives(decoder, values, direction, normalized, lower, upper,
                                   row, expected_value, expected_tangent, metadata, fail,
                                   *, primal_tolerance, tangent_tolerance):
    """Return only the selected12-output parameter Jacobian and measured primals.

    All decoder calls keep the original contiguous1024-row slice. Selecting an
    output before jacrev avoids constructing a1024x12-output Jacobian. The
    forward-mode and reverse-mode primals are independently observed; equal
    block shape alone never substitutes for either original numerical gate.
    """
    if type(row) is not int or not 0 <= row < normalized.shape[0] \
            or normalized.shape[0] % BLOCK_ROWS or primal_tolerance != 5e-6 \
            or tangent_tolerance != 2e-7:
        fail("Invalid original-block selected-row admission", row=row)
    start = (row // BLOCK_ROWS) * BLOCK_ROWS
    stop = start + BLOCK_ROWS
    offset = row-start
    metadata.update(selected_row_block_start=start, selected_row_block_stop=stop,
                    selected_row_local_index=offset)

    def block(ps):
        return decoder(ps, normalized[start:stop], lower[start:stop], upper[start:stop])

    metadata["selected_block_jvp_calls"] += 1
    with torch.no_grad():
        primal, tangent = jvp(block, (values,), (direction,))
    if primal.shape != (BLOCK_ROWS, OUTPUTS) or tangent.shape != primal.shape \
            or not bool(torch.isfinite(primal).all()) or not bool(torch.isfinite(tangent).all()):
        fail("Nonfinite or malformed selected original-block JVP", row=row)
    measured = primal[offset].detach().cpu().numpy()
    measured_tangent = tangent[offset].detach().cpu().numpy().astype(np.float64)
    primal_diff = float(np.max(np.abs(measured-expected_value)))
    tangent_diff = float(np.max(np.abs(measured_tangent-expected_tangent)))
    metadata["selected_primal_maxdiff_rad"] = max(metadata["selected_primal_maxdiff_rad"], primal_diff)
    metadata["selected_block_tangent_maxdiff_rad"] = max(metadata["selected_block_tangent_maxdiff_rad"], tangent_diff)
    metadata["row_value_max_disagreement_rad"] = max(metadata["row_value_max_disagreement_rad"], primal_diff)
    if primal_diff > primal_tolerance:
        fail("Selected original-block/full-bank primal disagreement", row=row)
    if tangent_diff > tangent_tolerance:
        fail("Selected original-block/full-bank JVP disagreement", row=row)

    def selected_with_aux(ps):
        selected = block(ps)[offset]
        return selected, selected

    metadata["selected_block_jacobian_calls"] += 1
    metadata["row_jacobian_calls"] += 1
    derivatives, jacobian_primal = jacrev(selected_with_aux, has_aux=True, chunk_size=1)(values)
    # This is the declared12 reverse seeds for this completed jacrev call,
    # not a measurement of CUDA kernels or independent per-kernel IDs.
    metadata["selected_reverse_seeds_expected"] += OUTPUTS
    if jacobian_primal.shape != (OUTPUTS,) or not bool(torch.isfinite(jacobian_primal).all()):
        fail("Nonfinite selected Jacobian-forward primal", row=row)
    reverse_value = jacobian_primal.detach().cpu().numpy()
    reverse_diff = max(float(np.max(np.abs(reverse_value-expected_value))),
                       float(np.max(np.abs(reverse_value-measured))))
    metadata["jacobian_forward_primal_maxdiff_rad"] = max(metadata["jacobian_forward_primal_maxdiff_rad"], reverse_diff)
    metadata["row_value_max_disagreement_rad"] = max(metadata["row_value_max_disagreement_rad"], reverse_diff)
    if reverse_diff > primal_tolerance:
        fail("Selected Jacobian-forward/full-bank primal disagreement", row=row)
    return derivatives


def selected_row_block_contract():
    return dict(kind=SELECTED_ROW_BLOCK_POLICY, decoder_block_rows=BLOCK_ROWS,
                selected_output_coordinates=OUTPUTS, select_after_decoder=True,
                independent_block_jvp=True, jacobian_forward_primal_aux=True,
                jacrev_chunk_size=1, primal_tolerance_rad=5e-6,
                tangent_tolerance_rad=2e-7, full_block_Jacobian_materialized=False,
                expected_reverse_seeds_are_not_kernel_counts=True)


def selected_row_backend_metadata():
    """Record configured arithmetic context without changing backend policy."""
    return dict(torch_version=torch.__version__, float32_matmul_precision=torch.get_float32_matmul_precision(),
                CUDA_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                CUDA_autocast_enabled=bool(torch.is_autocast_enabled()),
                TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
                actual_BLAS_kernel_not_profiled=True, backend_settings_not_modified=True)
