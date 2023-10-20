# coding=utf-8
import numpy as np

from core.utils.data import ReinitDataIterator


def load_data(n_te_points=225):
    lo = -1.
    hi = 1.
    train_data = {0: (np.array([[x]]) for x in [[lo, lo], [hi, hi]]),
                  1: (np.array([[x]]) for x in [[hi, lo], [lo, hi]]),
                  'examples_per_class': {0: 2, 1: 2}}

    points_per_rowcol = int(np.sqrt(n_te_points))
    x = np.linspace(lo - 0.2, hi + 0.2, points_per_rowcol)
    y = np.linspace(lo - 0.2, hi + 0.2, points_per_rowcol)
    xy = np.reshape(np.concatenate([xx[..., None] for xx in np.meshgrid(x, y)], axis=-1), [x.shape[0] * y.shape[0], -1])
    test_data = {None: ReinitDataIterator(xy[:, None, None, :])}
    return train_data, test_data