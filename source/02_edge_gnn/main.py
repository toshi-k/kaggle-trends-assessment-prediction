import time
import random
import argparse
import traceback
from pathlib import Path
from logging import getLogger

from distutils.util import strtobool

import numpy as np
import pandas as pd

import chainer
from chainer import optimizers, training
from chainer.datasets.dict_dataset import DictDataset
from chainer.training import make_extension, triggers

from lib.model import EdgeUpdateNet
from lib.evaluator import TreNDSEvaluator
from lib.log import init_logger
from lib.fnc import get_fnc


def load_dataset(dir_dataset: Path):

    train_scores = pd.read_csv(dir_dataset / 'train_scores.csv')
    sample_submission = pd.read_csv(dir_dataset / 'sample_submission.csv')
    train_ids_all = train_scores['Id'].tolist()
    test_ids = sample_submission['Id'].tolist()
    test_ids = sorted(list(set([int(s[:5]) for s in test_ids])))

    random.shuffle(train_ids_all)

    num_train = int(len(train_ids_all) * 0.9)

    train_ids = sorted(train_ids_all[:num_train])
    valid_ids = sorted(train_ids_all[num_train:])

    return train_ids, valid_ids, test_ids


def load_spatial_map(train_ids, valid_ids):

    train_ids_all = sorted(train_ids + valid_ids)
    spatial_map_all = np.load('../../input/spatial_map_train.npy')

    spatial_map_train = spatial_map_all[[(s in train_ids) for s in train_ids_all]]
    spatial_map_valid = spatial_map_all[[(s in valid_ids) for s in train_ids_all]]

    return spatial_map_train, spatial_map_valid


def run(dir_dataset: Path, batch_size: int, epochs: int, alpha: float, seed: int, debug: bool):

    tic = time.time()

    logger = getLogger('root')

    np.random.seed(seed)
    random.seed(seed)

    model = EdgeUpdateNet()
    model.to_gpu(device=0)

    train_ids, valid_ids, test_ids = load_dataset(dir_dataset)

    logger.info(f'train_ids: {train_ids[:5]} ... {train_ids[-5:]}')
    logger.info(f'valid_ids: {valid_ids[:5]} ... {valid_ids[-5:]}')
    logger.info(f' test_ids: {test_ids[:5]} ... {test_ids[-5:]}')

    train_scores = pd.read_csv(dir_dataset / 'train_scores.csv')
    train_scores.index = train_scores['Id']

    target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    train_target = train_scores.loc[train_ids][target_cols].values.astype(np.float32)
    valid_target = train_scores.loc[valid_ids][target_cols].values.astype(np.float32)
    test_target = np.zeros((len(test_ids), len(target_cols)), dtype=np.float32)

    loading = pd.read_csv(dir_dataset / 'loading.csv')
    loading.index = loading['Id']

    loading_train = loading.loc[train_ids].iloc[:, 1:].values.astype(np.float32)
    loading_valid = loading.loc[valid_ids].iloc[:, 1:].values.astype(np.float32)
    loading_test = loading.loc[test_ids].iloc[:, 1:].values.astype(np.float32)

    fnc_train, fnc_valid, fnc_test = get_fnc(dir_dataset, train_ids, valid_ids, test_ids, alpha)

    logger.info(f'fnc train: {fnc_train.shape}')
    logger.info(f'fnc valid: {fnc_valid.shape}')
    logger.info(f'fnc  test: {fnc_test.shape}')

    icn_numbers = pd.read_csv('../../input/ICN_numbers.csv')
    feature = np.zeros((53, len(icn_numbers['net_type'].unique())), dtype=np.float32)
    feature[range(len(feature)), icn_numbers['net_type_code']] = 1.0

    net_type_train = np.tile(np.expand_dims(feature, 0), (len(train_ids), 1, 1))
    net_type_valid = np.tile(np.expand_dims(feature, 0), (len(valid_ids), 1, 1))
    net_type_test = np.tile(np.expand_dims(feature, 0), (len(test_ids), 1, 1))

    spatial_map_train, spatial_map_valid = load_spatial_map(train_ids, valid_ids)
    spatial_map_test = np.load('../../input/spatial_map_test.npy')

    train_dataset = DictDataset(
        loading=loading_train,
        fnc=fnc_train,
        net_type=net_type_train,
        spatial_map=spatial_map_train,
        targets=train_target,
        Id=train_ids
    )

    valid_dataset = DictDataset(
        loading=loading_valid,
        fnc=fnc_valid,
        net_type=net_type_valid,
        spatial_map=spatial_map_valid,
        targets=valid_target,
        Id=valid_ids
    )

    test_dataset = DictDataset(
        loading=loading_test,
        fnc=fnc_test,
        net_type=net_type_test,
        spatial_map=spatial_map_test,
        targets=test_target,
        Id=test_ids
    )

    train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(valid_dataset, batch_size,
                                                  shuffle=False, repeat=False)
    test_iter = chainer.iterators.SerialIterator(test_dataset, batch_size,
                                                 shuffle=False, repeat=False)

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out="result")

    trainer.extend(training.extensions.LogReport(filename=f'seed{seed}.log'))

    trainer.extend(training.extensions.ExponentialShift('alpha', 0.99999))
    trainer.extend(
        training.extensions.observe_value(
            'alpha', lambda tr: tr.updater.get_optimizer('main').alpha))

    def stop_train_mode(trigger):
        @make_extension(trigger=trigger)
        def _stop_train_mode(_):
            logger.debug('turn off training mode')
            chainer.config.train = False

        return _stop_train_mode

    trainer.extend(stop_train_mode(trigger=(1, 'epoch')))

    trainer.extend(training.extensions.PrintReport(
        ['epoch', 'elapsed_time', 'main/loss', 'valid/main/All', 'alpha']))

    trainer.extend(
        TreNDSEvaluator(iterator=valid_iter, target=model,
                        name='valid', device=0, is_validate=True))

    trainer.extend(
        TreNDSEvaluator(iterator=test_iter, target=model,
                        name='test', device=0, is_submit=True,
                        submission_name=f'submit_seed{seed}.csv'),
        trigger=triggers.MinValueTrigger('valid/main/All'))

    chainer.config.train = True
    trainer.run()

    trained_result = pd.DataFrame(trainer.get_extension('LogReport').log)
    best_score = np.min(trained_result['valid/main/All'])
    logger.info(f'validation score: {best_score: .4f} (seed: {seed})')

    elapsed_time = time.time() - tic
    logger.info(f'elapsed time: {elapsed_time / 60.0: .1f} [min]')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, type=strtobool)

    parser.add_argument('--dir_dataset', type=Path, default='../../dataset')

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--alpha', type=float, default=-1)

    parser.add_argument('--seed', type=int, default=1048)

    params = parser.parse_args()

    if params.debug:
        logger = init_logger('_log/debug.log', level=10)
    else:
        logger = init_logger('_log/main.log', level=20)

    logger.info(vars(params))

    try:
        run(**vars(params))
    except Exception:
        logger.info(traceback.format_exc())


if __name__ == '__main__':
    main()
