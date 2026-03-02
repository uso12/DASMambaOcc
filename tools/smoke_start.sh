#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DAOCC_ROOT=${DAOCC_ROOT:-/home/ruiyu12/DAOcc}

if [[ ! -d "${DAOCC_ROOT}" ]]; then
  echo "[ERROR] DAOCC_ROOT does not exist: ${DAOCC_ROOT}"
  echo "Set DAOCC_ROOT to your DAOcc repo path."
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${DAOCC_ROOT}:${PYTHONPATH:-}"
export CC="${CC:-/usr/bin/gcc-10}"
export CXX="${CXX:-/usr/bin/g++-10}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "[ERROR] Python env is missing torch. Activate your DASMambaOcc env first."
  echo "Example:"
  echo "  source /home/ruiyu12/miniconda/etc/profile.d/conda.sh"
  echo "  conda activate dasmambaocc"
  exit 3
fi

python "${SCRIPT_DIR}/smoke_start.py" "$@"
