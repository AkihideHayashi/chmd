"""Functions for learning loop."""
import os
import json
from chainer.serializers import load_npz
from chainer.datasets import open_pickle_dataset
from chainer.datasets import split_dataset_random
from chmd.datasets import split_dataset_by_key


def initialize_train(maninp, manout, ratio):
    """Initialize data set by moving some remain data to train."""
    with open(maninp) as f:
        data = json.load(f)
    assert not data['train']
    n = int(len(data['remain']) * ratio)
    train, remain = split_dataset_random(data['remain'], n)
    data['train'] = sorted(train)
    data['remain'] = sorted(remain)
    with open(manout, 'w') as f:
        json.dump(data, f)



def learn(man_path, dataset_path, model_class, parameters,
          device_id, out, load, batch_size, max_epoch):
    with open(man_path) as f:
        manager = json.load(f)
    train_keys = manager['train']
    model = model_class(**parameters)
    if load and os.path.isfile(load):
        load_npz(load, model)
    
    with open_pickle_dataset(dataset_path) as pdataset:
        dataset = split_dataset_by_key(pdataset, train_keys)
        

