"""Explicit checkpoint paths and shared strict-loading state for Rex."""
from pathlib import Path


def inference_model_options(options):
    """Model options for inference (architecture keys of the training config)."""
    return dict(options)


def load_model_state(path):
    """Read base, DeepSpeed or Lightning weights with the same model key contract."""
    import torch

    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    state = payload.get('module', payload.get('state_dict', payload))
    normalized = {key.removeprefix('model.'): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError('Checkpoint contains colliding model-prefixed and plain parameter names')
    return normalized


def resolve_checkpoint(value):
    path = Path(value).expanduser().absolute()
    if any(part.startswith('.incoming-') for part in path.resolve().parts):
        raise ValueError('Checkpoint transfer has not been verified and published')
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        weight = path
    else:
        candidates = [path / 'mp_rank_00_model_states.pt',
                      path / 'checkpoint/mp_rank_00_model_states.pt',
                      path / 'last.ckpt/checkpoint/mp_rank_00_model_states.pt', path / 'last.ckpt']
        matches = [p for p in candidates if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f'Expected one explicit checkpoint in {path}; specify the checkpoint file or step directory')
        weight = matches[0]
    for parent in list(weight.parents)[:4]:
        config = parent / 'config.py'
        if config.is_file():
            return weight, config
    raise FileNotFoundError(f'No training config.py adjacent to {weight} or its run directory')
