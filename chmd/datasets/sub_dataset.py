import numpy as np
from chainer.datasets import split_dataset


def split_dataset_by_key(dataset, keys):
    n = len(dataset)
    not_keys = np.array([i for i in range(n) if i not in keys])
    order = np.concatenate([keys, not_keys])
    assert len(keys) < n
    return split_dataset(dataset, len(keys), order=order)
