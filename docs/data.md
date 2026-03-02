# Data Preparation and Symlink Workflow

`DASMambaOcc` follows DAOcc-style external dataset symlinks.

## Create project-local symlinks

```bash
cd /home/ruiyu12/DASMambaOcc
bash tools/link_data.sh /path/to/external_root
```

Expected structure under external root:

```text
<EXTERNAL_ROOT>/
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   ├── v1.0-trainval/
│   ├── gts/
│   ├── nuscenes_infos_train_w_3occ.pkl
│   └── nuscenes_infos_val_w_3occ.pkl
```

Quick check:

```bash
ls -la data
ls -la data/nuscenes | head
```
