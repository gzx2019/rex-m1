"""Checkpoint-owned image/state alignment, shared by launcher and adapter."""
from pathlib import PurePosixPath

# Audited dataset releases: v2 shifts images forward once, leaving states unchanged.
DATASET_LAGS = {'RoboDojo-all-arx_x5-ee': 1, 'RoboDojo-all-arx_x5-ee-v2': 0}


def observation_contract(train_data, requested='auto'):
    paths = train_data.get('paths', [])
    if isinstance(paths, str):
        paths = [paths]
    identities = []
    for path in paths:
        matches = set(PurePosixPath(str(path)).parts) & DATASET_LAGS.keys()
        if len(matches) != 1:
            identities.append(None)
        else:
            identities.append(matches.pop())
    declared = train_data.get('observation_image_lag')
    if declared is not None and (type(declared) is not int or declared not in (0, 1)):
        raise ValueError('observation_image_lag must be integer 0 or 1')
    known = {DATASET_LAGS[name] for name in identities if name is not None}
    if len(known) > 1 or (declared is not None and known and known != {declared}):
        raise ValueError('Conflicting training image lag contracts')
    if declared is None and (not identities or None in identities):
        raise ValueError('Unknown training image lag; record observation_image_lag in the training configuration')
    lag = declared if declared is not None else known.pop()
    if requested != 'auto' and requested is not None:
        if not isinstance(requested, (int, bool)) or requested not in (0, 1):
            raise ValueError('obs_delay must be auto, 0 or 1')
        if int(requested) != lag:
            raise ValueError(f'obs_delay={requested} conflicts with training_image_lag={lag}')
    return dict(image_lag=lag, datasets=identities, paths=list(paths),
                source='checkpoint_metadata' if declared is not None else 'audited_dataset_release')
