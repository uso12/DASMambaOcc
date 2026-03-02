# Training and Evaluation

## Preflight Smoke Start

```bash
cd /home/ruiyu12/DASMambaOcc
conda activate dasmambaocc
export DAOCC_ROOT=/home/ruiyu12/DAOcc
bash tools/smoke_start.sh
# Optional strict mode for full OccMamba verification:
bash tools/smoke_start.sh --require-mamba
```

## 8-GPU Training

```bash
cd /home/ruiyu12/DASMambaOcc
conda activate dasmambaocc
export DAOCC_ROOT=/home/ruiyu12/DAOcc

bash tools/dist_train.sh configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask.yaml 8 \
  --run-dir work_dirs/dasmambaocc_occ3d_8gpu
```

## 1-GPU Long Training

```bash
bash tools/dist_train.sh configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask_1gpu_long.yaml 1 \
  --run-dir work_dirs/dasmambaocc_occ3d_1gpu_long
```

## 8-GPU Evaluation

```bash
bash tools/dist_test.sh configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask.yaml \
  /path/to/checkpoint.pth 8
```

## Single-GPU Evaluation

```bash
python tools/test.py configs/nuscenes/occ3d/dasmambaocc_occ3d_nus_w_mask.yaml \
  /path/to/checkpoint.pth --launcher none --eval bbox
```

## Optional ablation matrix

```bash
bash tools/run_ablation_matrix.sh 8 work_dirs/ablation_matrix_8gpu
```
