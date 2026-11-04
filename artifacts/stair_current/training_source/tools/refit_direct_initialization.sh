#!/usr/bin/env bash
set -euo pipefail
H2_REFIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_REFIT_RUNTIME="${H2_ISAAC_SIM:-/home/nsgm/isaac-sim}"
H2_REFIT_PACKAGES="${H2_RL_PACKAGES:-/home/nsgm/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages}"
# Only Torch/YAML/NumPy and standalone frozen modules are needed. No AppLauncher.
export PYTHONPATH="${H2_REFIT_PACKAGES}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd -- "${H2_REFIT_ROOT}"
exec "${H2_REFIT_RUNTIME}/python.sh" -B "${H2_REFIT_ROOT}/tools/refit_direct_initialization.py" "$@"
