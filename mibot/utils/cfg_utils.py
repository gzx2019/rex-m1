# Copyright (C) 2026 Xiaomi Corporation.
# Modified by the Rex-M1-preview authors (2026).
"""Resolve training configuration and persist it only in the run directory."""
import hashlib
import json
import multiprocessing.util
import os
from pathlib import Path
import tempfile

from lightning import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from mmengine import Config
from omegaconf import OmegaConf


def prepare_training_config(cfg):
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    save_run_config(cfg)
    cfg.pop('launch', None)
    seed_everything(cfg.trainer.pop('seed', 42) + int(os.environ.get('RANK', 0)), workers=True)
    strategy = cfg.trainer.strategy
    if strategy.type == 'deepspeed':
        cfg.trainer.strategy = DeepSpeedStrategy(**strategy.params)
    elif strategy.type == 'auto':
        cfg.trainer.strategy = 'auto'
    else:
        raise ValueError('Unsupported strategy')
    cfg.model.params.optimizer = cfg.trainer.pop('optimizer')
    cfg.model.params.scheduler = cfg.trainer.pop('scheduler')
    return cfg


def save_run_config(cfg):
    root = Path(cfg.trainer.default_root_dir).expanduser().absolute()
    root = root / f'project_{cfg.trainer.project}' / cfg.trainer.exp_name
    cfg.trainer.default_root_dir = str(root)
    cache = root / 'cache'
    temporary = cache / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    # Lightning's atomic checkpoint writer uses tempfile before renaming.
    # Keep staging on the checkpoint filesystem, including on multi-node runs.
    # DataLoader workers share tensors over AF_UNIX sockets in multiprocessing's temp dir (107-byte
    # path limit); bind that dir to the system tmp first so long run names cannot overflow it.
    multiprocessing.util.get_temp_dir()
    tempfile.tempdir = str(temporary)
    os.environ.update(TMPDIR=str(temporary), CUDA_CACHE_PATH=str(cache / 'cuda'),
                      TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'))
    if cfg.trainer.num_nodes <= 0:
        cfg.trainer.num_nodes = int(cfg.get('launch', {}).get('nodes', 1))
    if int(os.environ.get('RANK', 0)) != 0:
        return
    config_file = root / 'config.py'
    if config_file.exists() and not cfg.trainer.get('ckpt_path'):
        raise FileExistsError(f'Run already exists: {root}; choose trainer.exp_name or an explicit resume checkpoint')
    # Preserve the original configuration when resuming.
    if not config_file.exists():
        cfg.dump(str(root / 'config.yaml'))
        cfg.dump(str(config_file))
    model_root = Path(__file__).resolve().parents[2]
    source = {str(p.relative_to(model_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in model_root.rglob('*.py') if '__pycache__' not in p.parts}
    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                h.update(block)
        return h.hexdigest()
    data_files = []
    for value in cfg.data.params.train_datasets.paths:
        path = Path(value)
        data_files.extend([path] if path.is_file() else sorted(path.rglob('*.json')))
    if not data_files:
        raise ValueError('Training data has no JSON trajectories')
    data_identity = {str(p.absolute()): digest(p) for p in data_files}
    evidence = {'source_sha256': source, 'data_json_sha256': data_identity,
                'pretrained_sha256': digest(cfg.model.params.pretrained) if cfg.model.params.pretrained else None, 'data_paths': list(cfg.data.params.train_datasets.paths),
                'pretrained': cfg.model.params.pretrained,
                'config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest()}
    (root / ('resume_provenance.json' if cfg.trainer.get('ckpt_path') else 'provenance.json')).write_text(json.dumps(evidence, indent=2)+'\n')
