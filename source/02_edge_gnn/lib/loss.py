import numpy as np
import chainer
from chainer import functions as F
from typing import List


def normalized_absolute_error(y_pred: chainer.Variable, t: np.ndarray):
    """
    TReNDS：Simple NN Baseline (Tawara)
    https://www.kaggle.com/ttahara/trends-simple-nn-baseline
    \sum_{i} |y_pred_{i} - t_{i}| / \sum_{i} t_{i}
    """
    y_pred_rm_nan = y_pred[~np.isnan(t)]
    t_not_rm_nan = t[~np.isnan(t)]

    return F.sum(F.absolute(y_pred_rm_nan - t_not_rm_nan)) / F.sum(t_not_rm_nan)


class WeightedNormalizedAbsoluteError:
    """
    TReNDS：Simple NN Baseline (Tawara)
    https://www.kaggle.com/ttahara/trends-simple-nn-baseline
    Metric for this competition
    """

    def __init__(self, weights: List[float] = [.3, .175, .175, .175, .175]):
        """Initialize."""
        self.weights = weights

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray) -> chainer.Variable:
        """Forward."""
        loss = 0
        for i, weight in enumerate(self.weights):
            loss += weight * normalized_absolute_error(y_pred[:, i], t[:, i])

        return loss
