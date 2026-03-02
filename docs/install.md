# Installation

## 1. Create Environment

```bash
cd /home/ruiyu12/DASMambaOcc
bash tools/create_conda_env.sh
conda activate dasmambaocc
```

By default this creates `dasmambaocc` by copying the existing `daocc` env.
Fallback to full YAML/pip install is used only if `daocc` is missing.

## 2. Compiler/CUDA Build Env

```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST=8.6+PTX
```

## 3. Set DAOcc Base Path

```bash
export DAOCC_ROOT=/home/ruiyu12/DAOcc
```

## 4. Verify Entry Points

```bash
python tools/train.py -h
python tools/test.py -h
bash tools/smoke_start.sh --skip-model-build
# Optional strict check (fails if Mamba fallback is active):
bash tools/smoke_start.sh --require-mamba
```
