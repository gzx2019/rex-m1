# Copyright (C) 2026 Xiaomi Corporation.
from functools import wraps
from typing import Any, Callable

import torch


def auto_cast(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(model_self: Any, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        dtype = model_self.compute_dtype
        batch = {key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                 for key, value in batch.items()}
        return func(model_self, batch, *args, **kwargs)

    return wrapper
