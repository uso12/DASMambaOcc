#!/usr/bin/env python3
import argparse
import os
import runpy
import sys
from pathlib import Path

from bootstrap_paths import bootstrap_paths


def _resolve_to_abs(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return str(path)


def _normalize_option_path(flag: str) -> None:
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if arg == flag and i + 1 < len(sys.argv):
            sys.argv[i + 1] = _resolve_to_abs(sys.argv[i + 1])
        elif arg.startswith(f"{flag}="):
            _, value = arg.split("=", 1)
            sys.argv[i] = f"{flag}={_resolve_to_abs(value)}"


def _normalize_cli_paths() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", nargs="?")
    parser.add_argument("--run-dir")
    parser.add_argument("--launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    args, _ = parser.parse_known_args(sys.argv[1:])

    if not args.config:
        return

    abs_config = _resolve_to_abs(args.config)
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == args.config:
            sys.argv[i] = abs_config
            break

    _normalize_option_path("--run-dir")


def main():
    os.environ["CC"] = "/usr/bin/gcc-10"
    os.environ["CXX"] = "/usr/bin/g++-10"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"

    _normalize_cli_paths()

    _, daocc_root = bootstrap_paths()

    import dasmambaocc  # noqa: F401

    target = Path(daocc_root) / "tools" / "dist_train.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing DAOcc train launcher: {target}")

    prev_cwd = os.getcwd()
    try:
        os.chdir(str(daocc_root))
        runpy.run_path(str(target), run_name="__main__")
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
