"""Learn."""
import json
import chainer
from chainer.datasets import open_pickle_dataset
from chainer.training import updaters, triggers, extensions
from chmd.models.ani import (concat_aev,
                             ANI1AEV2EnergyWithShifter,
                             ANI1AEV2EnergyNet,
                             ANI1EnergyLoss,
                             ANI1Preprocessor,
                             ANI1WeightedEnergyLoss,
                             )
from chmd.links.ensemble import EnsemblePredictor
from chmd.datasets import split_dataset_by_key
from chmd.training.triggers.purpose_trigger import PurposeTrigger
from chmd.database.pickle import tarfile_to_pickle, merge_pickle
from chmd.preprocess import preprocess


def main():
    with open('ani.json') as f:
        params = json.load(f)
    dataset_path = '../../../note/processed.pkl'
    out = '../../../note/result'
    tarfile_path = '../../../note/vaspruns.tar'
    tmp_path = '../../../note/tmp.pkl'
    merge_path = '../../../note/merge.pkl'
    seed_path = '../../../note/seed.pkl'
    processed_path = '../../../note/processed.pkl'
    purpose = 0.01
    batch_size = 5
    max_epoch = 1000
    device_id = -1
    tarfile_to_pickle(tarfile_path, tmp_path)
    merge_pickle([seed_path, tmp_path], merge_path)
    preprocess(merge_path, processed_path, batch_size, device_id,
               ANI1Preprocessor(params))
    optimizer = chainer.optimizers.Adam()
    learn(dataset_path, params, out, purpose, batch_size, max_epoch, device_id, optimizer)


def learn(dataset_path, params, out, purpose, batch_size, max_epoch, device_id, optimizer):
    """Learn paramter from data."""
    model = EnsemblePredictor(
        params['n_ensemble'],
        lambda: ANI1AEV2EnergyNet(params['nn_params'], params['energy_params'])
        )
    # model = EnsemblePredictor(
    #     params['n_ensemble'],
    #     lambda: ANI1AEV2EnergyWithShifter(params['nn_params'], params['shift_params'])
    #     )
    with open_pickle_dataset(dataset_path) as all_dataset:
        all_dataset = list(all_dataset)
        print('Selecting valid keys.', flush=True)
        train_keys = [i for i, data in enumerate(all_dataset) if data['status'] == 'train' and i != 0]
        print("Number of train keys:", len(train_keys), flush=True)
        train_dataset, _ = split_dataset_by_key(all_dataset, train_keys)
        # loss = ANI1EnergyLoss(model)
        kb = 11604.59
        loss = ANI1WeightedEnergyLoss(model, temperature=300 * kb)
        optimizer.setup(loss)
        train_iter = chainer.iterators.MultiprocessIterator(
            train_dataset, batch_size)
        updater = updaters.StandardUpdater(
            iterator=train_iter,
            optimizer=optimizer,
            converter=concat_aev,
            device=device_id)
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
        log_report_extension = extensions.LogReport(
            trigger=(1, 'epoch'), log_name='log')
        trainer.extend(log_report_extension)
        trainer.extend(
            extensions.PrintReport(['epoch', 'iteration',
                                    'main/loss',
                                    'elapsed_time']))
        trainer.run()


if __name__ == "__main__":
    main()
