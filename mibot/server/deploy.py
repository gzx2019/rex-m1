# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import argparse
import sys

import torch
from mmengine import Config

from mibot.models import MIMODEL
from mibot.server.runtime.server import Server
from mibot.utils.io import build_action_mask, validate_quantiles, validate_stats


def load_model(checkpoint, device, *, dtype="bf16", num_steps=5, backbone_path=None):
    from mibot.checkpoints import load_model_state, resolve_checkpoint, inference_model_options
    weight, config = resolve_checkpoint(checkpoint)
    cfg = Config.fromfile(str(config))
    options = inference_model_options(cfg.model.params.model)
    options.update(dtype=dtype, num_steps=num_steps)
    if backbone_path is not None:
        options['backbone_path'] = backbone_path
    from accelerate import init_empty_weights
    with init_empty_weights():
        model = MIMODEL.build(options)
    print(model.load_state_dict(load_model_state(weight), strict=True, assign=True))
    return cfg, model.eval().to(device=device, dtype=model.compute_dtype)


def load_stats(cfg, device):
    data = cfg.data.params.train_datasets
    action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = validate_stats(data.mean, data.std, action_length)
    q01, q99 = validate_quantiles(data.q01, data.q99)
    return (
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.tensor(q01, device=device),
        torch.tensor(q99, device=device),
        torch.from_numpy(build_action_mask(action_length)).to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0"
    cfg, model = load_model(args.model, device)
    mean, std, q01, q99, action_mask = load_stats(cfg, device)

    try:
        server = Server(args.host, args.port, model, mean, std, q01, q99, action_mask, device)
        print(f"Starting server on {args.host}:{args.port}")
        server.run()
    except OSError as error:
        if error.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
