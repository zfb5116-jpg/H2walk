#!/usr/bin/env bash
set -euo pipefail
H2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_ISAAC_SIM="${H2_ISAAC_SIM:-/home/nsgm/isaac-sim}"
H2_ISAAC_LAB="${H2_ISAAC_LAB:-${H2_ROOT}/../IsaacLab}"
H2_RL_PACKAGES="${H2_RL_PACKAGES:-/home/nsgm/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages}"
export PYTHONPATH="${H2_ISAAC_LAB}/source/isaaclab:${H2_ISAAC_LAB}/source/isaaclab_rl:${H2_ISAAC_LAB}/source/isaaclab_assets:${H2_ISAAC_LAB}/source/isaaclab_tasks:${H2_RL_PACKAGES}"
export PYTHONUNBUFFERED=1
export TZ=Asia/Shanghai
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd -- "${H2_ROOT}"
exec "${H2_ISAAC_SIM}/python.sh" "${H2_ROOT}/tools/evaluate_main_walk.py" "$@"
