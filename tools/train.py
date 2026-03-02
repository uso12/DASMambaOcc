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


def _install_train_shuffle_patch() -> None:
    import torch
    import mmdet3d.apis as apis_pkg
    import mmdet3d.apis.train as train_api_mod
    from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
    from mmcv.runner import (
        DistSamplerSeedHook,
        EpochBasedRunner,
        GradientCumulativeFp16OptimizerHook,
        Fp16OptimizerHook,
        OptimizerHook,
        build_optimizer,
        build_runner,
    )
    from mmdet.core import DistEvalHook, EvalHook
    from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
    from mmdet3d.utils import get_root_logger

    def _resolve_train_shuffle(cfg) -> bool:
        configured = cfg.data.get("train_shuffle", None)
        if configured is not None:
            return bool(configured)
        occ_cfg = cfg.model.get("heads", {}).get("occ", {})
        use_temporal_memory = bool(occ_cfg.get("use_temporal_memory", False))
        # For temporal memory, default to deterministic sequential traversal.
        return not use_temporal_memory

    def patched_train_model(
        model,
        dataset,
        cfg,
        distributed=False,
        validate=False,
        timestamp=None,
    ):
        logger = get_root_logger()

        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        train_shuffle = _resolve_train_shuffle(cfg)
        logger.info(f"[DASMambaOcc] train dataloader shuffle={train_shuffle}")

        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                num_gpus=None if distributed else 1,
                dist=distributed,
                shuffle=train_shuffle,
                seed=cfg.seed,
            )
            for ds in dataset
        ]

        find_unused_parameters = cfg.get("find_unused_parameters", False)
        if distributed:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            model = MMDataParallel(
                model.cuda(),
                device_ids=[0],
            )

        optimizer = build_optimizer(model, cfg.optimizer)
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.run_dir,
                logger=logger,
                meta={},
            ),
        )

        if hasattr(runner, "set_dataset"):
            runner.set_dataset(dataset)

        runner.timestamp = timestamp

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            if "cumulative_iters" in cfg.optimizer_config:
                optimizer_config = GradientCumulativeFp16OptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=distributed
                )
            else:
                optimizer_config = Fp16OptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=distributed
                )
        elif distributed and "type" not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        runner.register_training_hooks(
            cfg.lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get("momentum_config", None),
            custom_hooks_config=cfg.get("custom_hooks", None),
        )
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

        if validate:
            val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
            if val_samples_per_gpu > 1:
                cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=val_samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
            )
            eval_cfg = cfg.get("evaluation", {})
            eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run(data_loaders, [("train", 1)])

    train_api_mod.train_model = patched_train_model
    apis_pkg.train_model = patched_train_model


def main():
    os.environ["CC"] = "/usr/bin/gcc-10"
    os.environ["CXX"] = "/usr/bin/g++-10"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6+PTX"

    _normalize_cli_paths()

    _, daocc_root = bootstrap_paths()

    import dasmambaocc  # noqa: F401
    _install_train_shuffle_patch()

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
