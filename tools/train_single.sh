#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?"config path required"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DAOCC_ROOT=${DAOCC_ROOT:-/home/ruiyu12/DAOcc}

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${DAOCC_ROOT}:${PYTHONPATH:-}"
export CC="${CC:-/usr/bin/gcc-10}"
export CXX="${CXX:-/usr/bin/g++-10}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

python "${SCRIPT_DIR}/train.py" "${CONFIG}" --launcher none ${@:2}
