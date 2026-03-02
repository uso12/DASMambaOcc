import os
import sys
from pathlib import Path


def bootstrap_paths():
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    daocc_root = Path(os.environ.get("DAOCC_ROOT", "/home/ruiyu12/DAOcc")).expanduser().resolve()

    if not daocc_root.exists():
        raise FileNotFoundError(
            f"DAOCC_ROOT does not exist: {daocc_root}. Set DAOCC_ROOT to your DAOcc path."
        )

    for path in (str(src_root), str(project_root), str(daocc_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    return project_root, daocc_root
