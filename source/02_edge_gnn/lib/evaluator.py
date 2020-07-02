from logging import getLogger

import numpy as np
import pandas as pd

import chainer
from chainer import cuda, reporter
from chainer.training.extensions import Evaluator

from lib.loss import normalized_absolute_error


class TreNDSEvaluator(Evaluator):

    def __init__(self, iterator, target, device, name,
                 is_validate=False, is_submit=False, submission_name=''):
        super().__init__(iterator, target, device=device)

        self.is_validate = is_validate
        self.is_submit = is_submit
        self.name = name

        if submission_name == '':
            self.submission_name = 'submission.csv'
        else:
            self.submission_name = submission_name

        self.features = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
        self.loss_weights = [.3, .175, .175, .175, .175]

    def calc_score(self, y_truth, y_pred):
        score = 0
        metrics = {}

        for i, (feature, weight) in enumerate(zip(self.features, self.loss_weights)):
            loss_feature = normalized_absolute_error(y_pred[:, i], y_truth[:, i])
            metrics[feature] = loss_feature
            score += weight * loss_feature

        metrics['All'] = score

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(metrics, self._targets['main'])

        return observation

    def generate_submission(self, submit_id, y_pred):
        logger = getLogger('root')

        y_pred_df = pd.DataFrame(y_pred, columns=self.features)
        submit_id = pd.DataFrame(submit_id, columns=['Id'])
        y_pred_df = pd.concat([submit_id, y_pred_df], axis=1)

        submit_df = pd.melt(y_pred_df,
                            id_vars=['Id'],
                            var_name='feature',
                            value_name='Predicted')
        submit_df = submit_df.sort_values(['Id', 'feature'])

        submit_df['Id'] = submit_df.apply(
            lambda row: '{}_{}'.format(row['Id'], row['feature']),
            axis=1)

        submit_df = submit_df.drop('feature', axis=1)
        logger.info('save submission')
        submit_df.to_csv(self.submission_name, index=False, float_format='%.5f')

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self._targets['main']

        iterator.reset()
        it = iterator

        y_total = []
        t_total = []
        submit_id = []

        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func.predict(**in_arrays)

            y_data = cuda.to_cpu(y.data)
            y_total.append(y_data)
            t_total.extend([d['targets'] for d in batch])
            submit_id.extend([d['Id'] for d in batch])

        y_truth = np.stack(t_total, axis=0)
        y_pred = np.concatenate(y_total)

        if self.is_submit:
            self.generate_submission(submit_id, y_pred)

        if self.is_validate:
            return self.calc_score(y_truth, y_pred)

        return {}
