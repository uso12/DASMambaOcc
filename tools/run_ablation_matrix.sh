#!/usr/bin/env bash
set -euo pipefail

GPUS=${1:-8}
RUN_ROOT=${2:-work_dirs/ablation_matrix}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PORT_BASE=${PORT_BASE:-29700}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CONFIG_ROOT="${PROJECT_ROOT}/configs/nuscenes/occ3d/ablations"

EXTRA_ARGS=("${@:3}")

NAMES=(baseline lift_only full)
CONFIGS=(
  "${CONFIG_ROOT}/dasmambaocc_occ3d_nus_ablation_baseline.yaml"
  "${CONFIG_ROOT}/dasmambaocc_occ3d_nus_ablation_lift_only.yaml"
  "${CONFIG_ROOT}/dasmambaocc_occ3d_nus_ablation_full.yaml"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  run_dir="${RUN_ROOT}/${name}"
  port=$((PORT_BASE + i))

  echo "[DASMambaOcc][Ablation] start ${name}"
  NNODES="${NNODES}" NODE_RANK="${NODE_RANK}" MASTER_ADDR="${MASTER_ADDR}" PORT="${port}" \
    bash "${SCRIPT_DIR}/dist_train.sh" "${cfg}" "${GPUS}" --run-dir "${run_dir}" "${EXTRA_ARGS[@]}"
done
