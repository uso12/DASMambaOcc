#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ENV_NAME=${ENV_NAME:-dasmambaocc}
BASE_ENV=${BASE_ENV:-daocc}
CONDA_BASE=$(conda info --base)
BASE_ENV_PATH="${CONDA_BASE}/envs/${BASE_ENV}"
TARGET_ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists, skipping create."
else
  if [[ -d "${BASE_ENV_PATH}" ]]; then
    echo "[INFO] Creating '${ENV_NAME}' by copying base env '${BASE_ENV}'"
    cp -a "${BASE_ENV_PATH}" "${TARGET_ENV_PATH}"
  else
    echo "[INFO] Base env '${BASE_ENV}' not found. Falling back to YAML create."
    conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/env/environment.yml"
    conda run -n "${ENV_NAME}" pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
      "torch==1.10.2+cu113" \
      "torchvision==0.11.3+cu113" \
      -f "https://download.pytorch.org/whl/cu113/torch_stable.html"
    conda run -n "${ENV_NAME}" pip install \
      "mmcv-full==1.4.0" \
      -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"
    conda run -n "${ENV_NAME}" pip install "mmdet==2.20.0"
    conda run -n "${ENV_NAME}" pip install -r "${PROJECT_ROOT}/env/requirements.txt"
  fi
fi

# Optional dependency for OccMamba-style refinement. Fallback mixer is used if install fails.
conda run -n "${ENV_NAME}" pip install "causal-conv1d==1.2.0.post2" "mamba-ssm==1.2.0.post1" \
  || echo "[WARN] mamba-ssm install failed. Falling back to MLP mixer refinement."
if conda run -n "${ENV_NAME}" python -c "import mamba_ssm" >/dev/null 2>&1; then
  echo "[OK] mamba-ssm import check passed"
else
  echo "[WARN] mamba-ssm is not importable in '${ENV_NAME}'."
  echo "       Run: bash tools/smoke_start.sh --require-mamba to enforce strict runtime check."
fi

echo "[OK] Environment '${ENV_NAME}' is ready for DASMambaOcc"
