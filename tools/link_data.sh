#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/link_data.sh <external_data_root>"
  exit 1
fi

EXT_ROOT=$(realpath "$1")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATA_DIR="${PROJECT_ROOT}/data"
NU_DIR="${DATA_DIR}/nuscenes"

safe_link() {
  local src="$1"
  local dst="$2"
  local src_real dst_real
  src_real=$(realpath "${src}")
  dst_real=$(realpath -m "${dst}")
  if [[ "${src_real}" == "${dst_real}" ]]; then
    echo "[SKIP] ${dst} already resolves to ${src_real}"
    return 0
  fi
  ln -sfn "${src}" "${dst}"
}

mkdir -p "${DATA_DIR}"

if [[ -d "${EXT_ROOT}/nuscenes" ]]; then
  ln -sfn "${EXT_ROOT}/nuscenes" "${NU_DIR}"
else
  echo "[ERROR] Missing ${EXT_ROOT}/nuscenes"
  exit 2
fi

if [[ -d "${EXT_ROOT}/nuscenes/gts" ]]; then
  safe_link "${EXT_ROOT}/nuscenes/gts" "${NU_DIR}/gts"
elif [[ -d "${EXT_ROOT}/gts" ]]; then
  safe_link "${EXT_ROOT}/gts" "${NU_DIR}/gts"
fi

for PKL in nuscenes_infos_train_w_3occ.pkl nuscenes_infos_val_w_3occ.pkl; do
  if [[ -f "${EXT_ROOT}/nuscenes/${PKL}" ]]; then
    safe_link "${EXT_ROOT}/nuscenes/${PKL}" "${NU_DIR}/${PKL}"
  elif [[ -f "${EXT_ROOT}/${PKL}" ]]; then
    safe_link "${EXT_ROOT}/${PKL}" "${NU_DIR}/${PKL}"
  fi
done

echo "[OK] Dataset symlinks ready under ${DATA_DIR}"
ls -la "${DATA_DIR}"
