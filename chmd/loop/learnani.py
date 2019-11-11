import json
from chainer import ChainList, Optimizer
from chainer.datasets import open_pickle_dataset
from chmd.models.ani import ANI1, ANI1Batch

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key in self.__annotations__ or key[0] == '_':
                setattr(self, key, val)
            else:
                raise KeyError()


class LearningManager(Struct):
    manager_path: str
    dataset_path: str
    params: dict
    out: str
    load: str
    purpose: float
    batch_size: int
    max_epoch: int
    device_id: int
    optimizer: Optimizer
    n_agent: int


def learn(learn):
    with open(learn.manager_path) as f:
        manager = json.load(f)
    train_keys = manager['train']
    models = ChainList(*[ANI1(**learn.params) for _ in range(learn.n_agent)])
    with open_pickle_dataset(learn.dataset_path) as all_dataset:
        train_dataset = sp
