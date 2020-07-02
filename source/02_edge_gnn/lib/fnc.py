import pickle
from pathlib import Path
from logging import getLogger
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.covariance import graphical_lasso


def get_fnc_matrix(fnc):

    arr = np.ones((len(fnc), 53, 53), dtype=np.float32)

    len_seq = 52
    col_start = 1
    count = 0

    while len_seq != 0:

        arr[:, count, (count + 1):] = fnc.iloc[:, col_start:col_start + len_seq].values
        arr[:, (count + 1):, count] = fnc.iloc[:, col_start:col_start + len_seq].values

        col_start += len_seq
        len_seq -= 1
        count += 1

    return arr


def graphical_lasso_wrap(emp_cov, alpha, max_iter):

    try:
        _, precision = graphical_lasso(emp_cov[0], alpha=alpha, max_iter=max_iter)
        return precision
    except FloatingPointError:
        return graphical_lasso_wrap(emp_cov, alpha=alpha * 1.1, max_iter=max_iter)


def get_fnc(dir_dataset, train_ids, valid_ids, test_ids, alpha: float):

    logger = getLogger('root')

    fnc = pd.read_csv(dir_dataset / 'fnc.csv')
    fnc.index = fnc['Id']

    if alpha < 0:
        logger.info('alpha < 0, raw fnc is used')
        fnc_train = get_fnc_matrix(fnc.loc[train_ids])
        fnc_valid = get_fnc_matrix(fnc.loc[valid_ids])
        fnc_test = get_fnc_matrix(fnc.loc[test_ids])
        return fnc_train, fnc_valid, fnc_test

    logger.info(f'alpha = {alpha}, graphical lasso is used')

    graphical_lasso_init = partial(graphical_lasso_wrap, alpha=alpha, max_iter=300)

    dir_temp = Path('_temp')

    if (dir_temp / f'fnc_alpha{alpha:.2f}.pickle').exists():
        logger.info('load preprocessed data')

        with open(str(dir_temp / f'fnc_alpha{alpha:.2f}.pickle'), 'rb') as f:
            fnc_matrix_dict = pickle.load(f)

    else:
        logger.info('preprocess fnc data')
        fnc_matrix = get_fnc_matrix(fnc)

        with Pool(10) as p:
            fnc_matrix = list(tqdm(p.imap(graphical_lasso_init, np.split(fnc_matrix, len(fnc_matrix))),
                                   total=len(fnc_matrix)))

        fnc_matrix_dict = {_id: mat for _id, mat in zip(fnc['Id'], fnc_matrix)}

        dir_temp.mkdir(exist_ok=True)

        with open(str(dir_temp / f'fnc_alpha{alpha:.2f}.pickle'), 'wb') as f:
            pickle.dump(fnc_matrix_dict, f)

    fnc_train = np.stack([fnc_matrix_dict[_id] for _id in train_ids], axis=0)
    fnc_valid = np.stack([fnc_matrix_dict[_id] for _id in valid_ids], axis=0)
    fnc_test = np.stack([fnc_matrix_dict[_id] for _id in test_ids], axis=0)

    fnc_train_non_zero = np.sum(np.abs(fnc_train) >= 1e-5)
    fnc_train_zero = np.sum(np.abs(fnc_train) < 1e-5)

    logger.info(f'fnc train non zero rate {fnc_train_non_zero / (fnc_train_non_zero + fnc_train_zero): .3f}')

    return fnc_train, fnc_valid, fnc_test
