"""Learn."""
import os
import json
import argparse
import numpy as np
import chainer
from chainer.datasets import open_pickle_dataset
from chainer.training import updaters, triggers, extensions
from chmd.models.ani import ANI1
from chmd.functions.activations import gaussian
from chmd.links.loss import EnergyGradLoss, EachEnergyGradLoss
from chmd.datasets import split_dataset_by_key
from chmd.dataset.convert import concat_converter
from chmd.training.triggers.purpose_trigger import PurposeTrigger


def main():
    params = {
        "num_elements": 3,
        "aev_params": {
            "radial": {
                "cutoff": 9.0,
                "head": 0.7,
                "tail": 9.0,
                "step": 0.25,
                "sigma": 0.25
            },
            "angular": {
                "cutoff": 3.5,
                "head": 0.7,
                "tail": 3.5,
                "step": 0.4,
                "sigma": 0.4,
                "ndiv": 9,
                "zeta": 32.0
            }
        },
        "nn_params": {
            "n_layers": [[128, 128, 1], [128, 128, 1], [128, 128, 1]],
            "act": gaussian
        }
    }


def search_fail_cases(manager_path, dataset_path, key, params, tol,
                      load, batch_size, device_id):
    """Calculate Loss for all data belong to key."""
    with open(manager_path) as f:
        manager = json.load(f)
    predict_keys = manager[key]
    model = ANI1(**params)
    chainer.serializers.load_npz(load, model)
    loss_fun = EachEnergyGradLoss(model, 1.0, 1.0)
    loss_fun.to_device(device_id)
    all_loss = []
    with open_pickle_dataset(dataset_path) as all_dataset:
        dataset = split_dataset_by_key(all_dataset, predict_keys)[0]
        iterator = chainer.iterators.SerialIterator(
            dataset, batch_size, repeat=False, shuffle=False)
        for it in iterator:
            data = concat_converter(it, device=device_id, padding=None)
            all_loss.extend(loss_fun(data))
    all_loss = np.array(all_loss)
    return predict_keys[all_loss > tol]


def learn(manager_path, dataset_path, params, out, load,
          purpose, batch_size, max_epoch, device_id):
    """Learn paramter from data."""
    with open(manager_path) as f:
        manager = json.load(f)
    train_keys = manager['train']
    model = ANI1(**params)
    with open_pickle_dataset(dataset_path) as all_dataset:
        train_dataset = split_dataset_by_key(all_dataset, train_keys)[0]
        if load and os.path.isfile(load):
            chainer.serializers.load_npz(load, model)
        else:
            model.shift.setup(
                elements=[d['elements'] for d in train_dataset],
                energies=[d['energy'] for d in train_dataset]
            )
        loss = EnergyGradLoss(model, 1.0, 1.0)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        train_iter = chainer.iterators.SerialIterator(
            train_dataset, batch_size)
        updater = updaters.StandardUpdater(iterator=train_iter,
                                           optimizer=optimizer,
                                           converter=concat_converter,
                                           device=device_id,
                                           loss_func=loss)
        trigger_early_stopping = PurposeTrigger(
            purpose,
            monitor='main/loss',
            check_trigger=(1, 'epoch'),
            max_trigger=(max_epoch, 'epoch')
        )
        trainer = chainer.training.Trainer(updater,
                                           trigger_early_stopping,
                                           out=out)
        trigger_min = triggers.MinValueTrigger(
            'main/loss',
            trigger=(1, 'epoch'))
        trainer.extend(
            extensions.snapshot_object(model, filename='best_model'),
            trigger=trigger_min)
        trainer.extend(
            extensions.LogReport(trigger=(1, 'epoch'), log_name='log'))
        trainer.extend(
            extensions.PrintReport(['epoch', 'iteration',
                                    'main/loss', 'main/loss_e', 'main/loss_f',
                                    'elapsed_time']))
        trainer.run()
