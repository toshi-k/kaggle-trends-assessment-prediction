import argparse

import numpy as np
import pandas as pd


def run(seeds: str):

    list_seeds = seeds.split(',')

    print(list_seeds)

    list_submit = list()

    for i, seed in enumerate(list_seeds):

        submit_seed = pd.read_csv(f'submit_seed{seed}.csv')
        if i == 0:
            submit_ensemble = submit_seed[['Id']]

        list_submit.append(submit_seed['Predicted'].values)

    submit_ensemble['Predicted'] = np.mean(np.stack(list_submit, 1), 1)

    submit_ensemble.to_csv(f'submit_ensemble_{len(list_seeds)}models.csv',
                           index=False, float_format='%.5f')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seeds',
                        type=str,
                        default='1048,1049,1050,1051,1052'
                                ',1053,1054,1055,1056,1057')

    params = parser.parse_args()

    run(**vars(params))


if __name__ == '__main__':
    main()
