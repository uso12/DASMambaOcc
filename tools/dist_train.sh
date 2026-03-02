#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?"config path required"}
GPUS=${2:?"num gpus required"}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DAOCC_ROOT=${DAOCC_ROOT:-/home/ruiyu12/DAOcc}

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${DAOCC_ROOT}:${PYTHONPATH:-}"
export CC="${CC:-/usr/bin/gcc-10}"
export CXX="${CXX:-/usr/bin/g++-10}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

python -m torch.distributed.launch \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  "${SCRIPT_DIR}/train.py" \
  "${CONFIG}" \
  --launcher pytorch \
  ${@:3}
