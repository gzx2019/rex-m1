# Copyright (C) 2026 Xiaomi Corporation.
# Modified by the Rex-M1-preview authors (2026).
"""One registry and one model for training and inference."""
from mmengine import Registry

MIMODEL = Registry("MIMODEL")

from mibot.models.runner.base_runner import BaseRunner
from mibot.models.VLA.xr1 import xr1

__all__ = ["MIMODEL", "BaseRunner", "xr1"]
