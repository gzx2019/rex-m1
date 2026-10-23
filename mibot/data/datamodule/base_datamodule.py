# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as distributed

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


class ResumeSampler(DistributedSampler):
    """DistributedSampler whose first pass skips ``skip`` indices already consumed before a mid-epoch resume.

    Lightning restarts a non-stateful loader from batch 0 on resume; the length stays unchanged so the
    restored batch counter still ends the epoch on time.
    """
    skip = 0

    def __iter__(self):
        indices = list(super().__iter__())[self.skip:]
        self.skip = 0
        return iter(indices)


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate(**params.get("collate", {}))
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            self.train_set = JsonDataset(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        distributed_run = distributed.is_available() and distributed.is_initialized()
        sampler = ResumeSampler(self.train_set, shuffle=True, seed=42,
                                num_replicas=distributed.get_world_size() if distributed_run else 1,
                                rank=distributed.get_rank() if distributed_run else 0)
        # Called after Lightning restored the checkpoint, so global_step is the resumed step.
        sampler.skip = self.trainer.global_step * self.batch_size % len(sampler) if self.trainer else 0
        workers = int(self.params.get("num_workers", 2))
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=workers,
            prefetch_factor=2 if workers else None,
            collate_fn=self.collate_fn,
            persistent_workers=workers > 0,
            pin_memory=True,
        )
