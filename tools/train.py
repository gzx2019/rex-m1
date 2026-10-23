# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings
from typing import Any, List, Optional

import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


def prepare(cfg: DictConfig):
    from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
    from lightning.pytorch.loggers import WandbLogger
    from mmengine import DATASETS
    from mibot.models import MIMODEL
    from mibot.utils.cfg_utils import prepare_training_config
    from omegaconf import OmegaConf
    import mibot.data
    import wandb

    logged = OmegaConf.to_container(cfg, resolve=True)
    cfg = prepare_training_config(cfg)
    datamodule = DATASETS.build(cfg.data)
    model = MIMODEL.build(cfg.model)
    cfg.trainer["callbacks"] = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            save_top_k=-1,
            save_last="link",
            every_n_train_steps=cfg.trainer.pop("save_interval", 10000),
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ),
    ]
    resume_interval = cfg.trainer.pop("resume_interval", None)
    if resume_interval:
        # Rolling resume point: keeps only the newest one (the new file is written before the old is removed).
        cfg.trainer["callbacks"].append(ModelCheckpoint(
            save_top_k=1,
            monitor="step",
            mode="max",
            filename="resume-{step}",
            every_n_train_steps=resume_interval,
            dirpath=cfg.trainer.default_root_dir,
            enable_version_counter=False,
        ))

    logger: List[Any] = [
        WandbLogger(
            project=cfg.trainer.pop("project"),
            name=cfg.trainer.pop("exp_name"),
            config=logged,
            save_dir=cfg.trainer.default_root_dir,
            mode=os.environ.get("WANDB_MODE", "online"),
        )
    ]
    if int(os.environ.get("RANK", "0")) == 0:
        logger[0].experiment
        config_root = Path(__file__).resolve().parents[1]
        for path in sorted((config_root / "configs").rglob("*.yaml")):
            wandb.save(str(path), base_path=str(config_root), policy="now")

    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

    return (cfg, datamodule, model, logger)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if 'LOCAL_RANK' not in os.environ:
        launch = cfg.get('launch', {})
        nodes, gpus = int(launch.get('nodes', 1)), int(launch.get('gpus', 1))
        if nodes < 1 or gpus < 1:
            raise ValueError('launch.nodes and launch.gpus must be positive')
        node_rank = int(os.environ.get('NODE_RANK', launch.get('node_rank', 0)))
        if not 0 <= node_rank < nodes:
            raise ValueError('launch.node_rank / NODE_RANK must be in [0, launch.nodes)')
        host = os.environ.get('MASTER_ADDR', str(launch.get('master_addr', 'localhost')))
        if nodes > 1 and host in {'localhost', '127.0.0.1'}:
            raise ValueError('Multi-node launch requires a reachable master address (MASTER_ADDR or launch.master_addr)')
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes', str(nodes),
                   '--nproc_per_node', str(gpus), '--node_rank', str(node_rank),
                   '--master_addr', host, '--master_port', str(launch.get('master_port', 29500)),
                   str(Path(__file__).resolve()), *sys.argv[1:]]
        import signal
        process = subprocess.Popen(command, start_new_session=True)
        try:
            raise SystemExit(process.wait())
        finally:
            if process.poll() is None:
                import psutil
                # torchrun workers may create separate sessions; retain their identities
                # before the launcher exits so timeout cleanup cannot orphan a GPU worker.
                children = psutil.Process(process.pid).children(recursive=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                _, alive = psutil.wait_procs(children, timeout=1)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
    import faulthandler
    import signal
    # torchrun SIGTERMs the surviving ranks after a collective timeout; dump where each one is stuck.
    faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)
    from lightning import Trainer
    cfg, datamodule, model, logger = prepare(cfg)
    ckpt_path: Optional[str] = cfg.trainer.pop("ckpt_path", None)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
