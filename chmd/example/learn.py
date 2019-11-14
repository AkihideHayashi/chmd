"""Learn."""
import os
import json
import chainer
from chainer.datasets import open_pickle_dataset
from chainer.training import updaters, triggers, extensions
from chmd.models.ani import ANI1, ANI1EnergyGradLoss
from chmd.functions.activations import gaussian
from chmd.dataset.convert import converter_concat_neighbors
from chmd.datasets import split_dataset_by_key
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
        },
        "cutoff": 9.0,
        "pbc": [True, True, True],
        "n_agents": 4,
        "order": ["H", "C", "Pt"]
    }
    # dataset_path = 'files/direct.pkl'
    dataset_path = '../../../data/processed.pkl'
    out = 'result'
    purpose = 0.003
    batch_size = 5
    max_epoch = 1000
    device_id = -1
    optimizer = chainer.optimizers.Adam()
    learn(dataset_path, params, out, purpose, batch_size, max_epoch, device_id, optimizer)


def learn(dataset_path, params, out, purpose, batch_size, max_epoch, device_id, optimizer):
    """Learn paramter from data."""
    model = ANI1(**params)
    with open_pickle_dataset(dataset_path) as all_dataset:
        all_dataset = list(all_dataset)
        print('Selecting valid keys.', flush=True)
        train_keys = [i for i, data in enumerate(all_dataset) if data['status'] == 'train' and i != 0]
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
        trainer.extend(extensions.ProgressBar())
        trainer.run()


if __name__ == "__main__":
    main()