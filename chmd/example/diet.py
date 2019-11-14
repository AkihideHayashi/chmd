"""Learn."""
import os
import json
import argparse
import numpy as np
import chainer
from chainer.optimizers import Adam
from chainer.datasets import open_pickle_dataset, open_pickle_dataset_writer
from chainer.training import updaters, triggers, extensions
from chmd.models.ani import ANI1, ANI1EachEnergyGradLoss, ANI1EnergyGradLoss
from chmd.functions.activations import gaussian
from chmd.datasets import split_dataset_by_key
from chmd.dataset.convert import converter_concat_neighbors
from chmd.training.triggers.purpose_trigger import PurposeTrigger


def main():
    """Train and extend loop."""
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
        },
        "cutoff": 9.0,
        "pbc": [True, True, True],
        "n_agents": 4,
        "order": ["H", "C", "Pt"]
    }

    dataset_dir = '../../../data/'
    dataset_prefix = 'processed'
    out = 'result'
    load = 'result/best_model'
    purpose = 0.004
    batch_size = 500
    max_epoch = 100000
    device_id = -1
    for i in range(10):
        inp_dataset_path = os.path.join(
            dataset_dir, f'{dataset_prefix}_{i}.pkl')
        out_dataset_path = os.path.join(
            dataset_dir, f'{dataset_prefix}_{i+1}.pkl')
        learn(inp_dataset_path, params, out, purpose, batch_size,
              max_epoch, device_id, Adam())
        fail = search_fail_cases(inp_dataset_path, params, 'train',
                                 purpose * 2, load, batch_size, device_id)
        print(f'Cycle {i}: {len(fail)} fail cases.', flush=True)
        fail = np.random.choice(fail, len(fail) * 0.01, replace=False)
        copy_and_renew(inp_dataset_path, out_dataset_path, fail, 'train')


def copy_and_renew(inp, out, trigger, key):
    """Copy. However renew status."""
    with open_pickle_dataset(inp) as fi:
        with open_pickle_dataset_writer(out) as fo:
            for i, data in enumerate(fi):
                if i in trigger:
                    data['status'] = key
                fo.write(data)


def search_fail_cases(dataset_path, params, key, tol,
                      load, batch_size, device_id):
    """Calculate Loss for all data belong to key."""
    model = ANI1(**params)
    chainer.serializers.load_npz(load, model)
    loss_fun = ANI1EachEnergyGradLoss(model, 1.0, 1.0)
    loss_fun.to_device(device_id)
    all_loss = []
    with open_pickle_dataset(dataset_path) as all_dataset:
        all_dataset = list(all_dataset)
        print('Selecting train keys.', flush=True)
        predict_keys = np.array([i for i, data in enumerate(all_dataset)
                                 if data['status'] == key and i != 0])
        print("Number of train keys:", len(predict_keys), flush=True)
        dataset, _ = split_dataset_by_key(all_dataset, predict_keys)
        iterator = chainer.iterators.SerialIterator(
            dataset, batch_size, repeat=False, shuffle=False)
        for it in iterator:
            data = converter_concat_neighbors(it, device=device_id)
            l = loss_fun(**data)
            all_loss.extend(l)
    all_loss = np.array(all_loss)
    return predict_keys[all_loss > tol]


def learn(dataset_path, params, out, purpose, batch_size,
          max_epoch, device_id, optimizer):
    """Learn paramter from data."""
    model = ANI1(**params)
    with open_pickle_dataset(dataset_path) as all_dataset:
        all_dataset = list(all_dataset)
        print('Selecting train keys.', flush=True)
        train_keys = [i for i, data in enumerate(all_dataset)
                      if data['status'] == 'train' and i != 0]
        print("Number of train keys:", len(train_keys), flush=True)
        train_dataset, _ = split_dataset_by_key(all_dataset, train_keys)
        model.shift.setup(
            elements=[d['elements'] for d in train_dataset],
            energies=[d['energy'] for d in train_dataset]
        )
        loss = ANI1EnergyGradLoss(model, ce=1.0, cf=1.0)
        optimizer.setup(loss)
        train_iter = chainer.iterators.MultiprocessIterator(
            train_dataset, batch_size)
        updater = updaters.StandardUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            converter=converter_concat_neighbors,
            device=device_id)
        trigger_early_stopping = PurposeTrigger(
            purpose,
            monitor='main/loss_f',
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
        log_report_extension = extensions.LogReport(
            trigger=(1, 'epoch'), log_name='log')
        trainer.extend(log_report_extension)
        trainer.extend(
            extensions.PrintReport(['epoch', 'iteration',
                                    'main/loss', 'main/loss_e', 'main/loss_f',
                                    'elapsed_time']))
        trainer.run()

if __name__ == "__main__":
    main()
