"""Portable, pickle-free caches of selected transport validation rows."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np

FEATURES = ['pdg', 'process', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'n_descendants', 'own_hit_mask', 'subtree_hit_mask']


def save_cache(path, truth, score, group, metadata, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, truth=truth, score=score, group=group,
                                metadata=np.array(json.dumps(metadata, allow_nan=False)),
                                **{'feature_'+name: values for name, values in features.items()})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata'].item()))
        if meta.get('version') != 1:
            raise ValueError('Unsupported validation-cache version')
        truth, score, group = (data[name] for name in ['truth', 'score', 'group'])
        features = {name: data['feature_'+name] for name in FEATURES if 'feature_'+name in data}
    if any(a.ndim != 1 or len(a) != len(truth) for a in [truth,score,group,*features.values()]):
        raise ValueError('Cache columns have incompatible dimensions')
    if not len(truth) or not np.isin(truth, [0,1]).all():
        raise ValueError('Cache has no rows or invalid truth labels')
    if not np.issubdtype(group.dtype, np.integer) or np.any(group < 0) or np.any(group >= len(meta['event_keys'])):
        raise ValueError('Invalid cache event groups')
    return truth, score, group, meta, features
