#!/usr/bin/env bash
set -euo pipefail

H2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
H2_FLAT_CHECKPOINT="${H2_FLAT_CHECKPOINT:-${H2_ROOT}/artifacts/flat_forward/model_999.pt}"
H2_REPLAY_OUTPUT="${1:-${H2_ROOT}/logs/evaluations/flat_$(date +%Y%m%d_%H%M%S)}"
export CTRLLAB_H2_USD_PATH="${CTRLLAB_H2_USD_PATH:-${H2_ROOT}/assets/isaaclab/unitree_robots/h2/h2.usd}"
export CTRLLAB_H2_LIDAR_RECON_CHECKPOINT="${CTRLLAB_H2_LIDAR_RECON_CHECKPOINT:-${H2_ROOT}/logs/rsl_rl/stairs_l0b2_20261020/lidar_reconstructor_phase1.pt}"

exec bash "${H2_ROOT}/tools/evaluate_main_walk.sh" \
  --headless --device cuda:0 --num_envs 1024 \
  --checkpoint "${H2_FLAT_CHECKPOINT}" --source_root "${H2_ROOT}/source" \
  --terrain flat --duration 20 --speed 0.3 --seed 42 --video \
  --output "${H2_REPLAY_OUTPUT}"
